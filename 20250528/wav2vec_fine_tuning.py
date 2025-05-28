"""
wav2vec 모델 파인튜닝 예제

이 스크립트는 사전 학습된 wav2vec 모델을 특정 음성 데이터셋으로 파인튜닝하는 방법을 보여줍니다.
"""

import torch
import numpy as np
import pandas as pd
from datasets import load_dataset, load_metric
from transformers import (
    Wav2Vec2CTCTokenizer,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Processor,
    Wav2Vec2ForCTC,
    TrainingArguments,
    Trainer
)
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

# 데이터 준비 헬퍼 함수
def prepare_dataset(batch):
    audio = batch["audio"]
    
    # 오디오 배열 정규화
    batch["input_values"] = processor(
        audio["array"], 
        sampling_rate=audio["sampling_rate"]
    ).input_values[0]
    
    # 레이블을 토큰 ID로 변환
    with processor.as_target_processor():
        batch["labels"] = processor(batch["text"]).input_ids
        
    return batch

# 데이터 로딩을 위한 커스텀 데이터 클래스
@dataclass
class DataCollatorCTC:
    """
    데이터 로더를 위한 데이터 콜레이터
    음성 입력값을 패딩하고 레이블을 처리합니다.
    """
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True
    
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # 입력값과 레이블 리스트 준비
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        
        # 패딩
        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )
        
        # 레이블 패딩
        with self.processor.as_target_processor():
            labels_batch = self.processor.pad(
                label_features,
                padding=self.padding,
                return_tensors="pt",
            )
            
        # 레이블 추가
        batch["labels"] = labels_batch["input_ids"].masked_fill(labels_batch["attention_mask"].ne(1), -100)
        
        return batch

# 평가 지표 계산을 위한 함수
def compute_metrics(pred):
    pred_logits = pred.predictions
    pred_ids = np.argmax(pred_logits, axis=-1)
    
    pred.label_ids[pred.label_ids == -100] = processor.tokenizer.pad_token_id
    
    pred_str = processor.batch_decode(pred_ids)
    label_str = processor.batch_decode(pred.label_ids)
    
    wer = wer_metric.compute(predictions=pred_str, references=label_str)
    
    return {"wer": wer}

def main():
    # 훈련에 사용할 장치 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"사용 장치: {device}")
    
    # 데이터셋 로드 (예: Common Voice 한국어)
    # 실제로는 이 부분을 자신의 데이터셋에 맞게 수정해야 합니다
    try:
        dataset = load_dataset("mozilla-foundation/common_voice_11_0", "ko", split="train+validation")
        print(f"데이터셋 로드 완료: {len(dataset)} 샘플")
    except Exception as e:
        print(f"데이터셋 로드 실패: {e}")
        print("데이터셋 로드에 실패했습니다. 샘플 데이터로 진행합니다.")
        
        # 샘플 데이터셋 생성 (실제 훈련용으로는 적합하지 않음)
        sample_data = {
            "audio": [{"array": np.zeros(16000), "sampling_rate": 16000}] * 5,
            "text": ["안녕하세요", "반갑습니다", "wav2vec 테스트", "음성 인식", "한국어 예제"]
        }
        dataset = pd.DataFrame(sample_data)
    
    # 데이터셋 분할
    train_size = int(0.9 * len(dataset))
    train_dataset = dataset.select(range(train_size))
    eval_dataset = dataset.select(range(train_size, len(dataset)))
    
    # 토크나이저 및 프로세서 설정
    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained("kresnik/wav2vec2-large-xlsr-korean")
    feature_extractor = Wav2Vec2FeatureExtractor(
        feature_size=1, 
        sampling_rate=16000,
        padding_value=0.0,
        do_normalize=True,
        return_attention_mask=True
    )
    
    global processor
    processor = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
    
    # 데이터셋 전처리
    train_dataset = train_dataset.map(prepare_dataset, remove_columns=train_dataset.column_names)
    eval_dataset = eval_dataset.map(prepare_dataset, remove_columns=eval_dataset.column_names)
    
    # 데이터 콜레이터 설정
    data_collator = DataCollatorCTC(processor=processor, padding=True)
    
    # 모델 로드
    model = Wav2Vec2ForCTC.from_pretrained(
        "facebook/wav2vec2-large-xlsr-53",
        attention_dropout=0.1,
        hidden_dropout=0.1,
        feat_proj_dropout=0.0,
        mask_time_prob=0.05,
        layerdrop=0.1,
        ctc_loss_reduction="mean",
        pad_token_id=processor.tokenizer.pad_token_id,
        vocab_size=len(processor.tokenizer)
    )
    
    # 중요: 특성 추출기 동결 (낮은 레이어는 학습하지 않음)
    model.freeze_feature_extractor()
    
    # 평가 지표 설정
    global wer_metric
    wer_metric = load_metric("wer")
    
    # 훈련 인자 설정
    training_args = TrainingArguments(
        output_dir="./results",
        group_by_length=True,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=2,
        evaluation_strategy="steps",
        num_train_epochs=30,
        fp16=True,
        save_steps=500,
        eval_steps=500,
        logging_steps=100,
        learning_rate=1e-4,
        warmup_steps=500,
        save_total_limit=2,
    )
    
    # 트레이너 설정
    trainer = Trainer(
        model=model,
        data_collator=data_collator,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=processor.feature_extractor,
    )
    
    # 모델 훈련
    print("모델 훈련 시작...")
    trainer.train()
    
    # 모델 저장
    model.save_pretrained("./wav2vec2-finetuned-korean")
    processor.save_pretrained("./wav2vec2-finetuned-korean")
    print("훈련된 모델 저장 완료!")

if __name__ == "__main__":
    main() 