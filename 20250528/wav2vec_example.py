"""
wav2vec 모델을 사용한 음성 인식 예제

이 스크립트는 사전 학습된 wav2vec 모델을 사용하여 음성 파일을 텍스트로 변환하는 방법을 보여줍니다.
"""

import torch
import librosa
import numpy as np
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

def load_audio(file_path, sample_rate=16000):
    """
    오디오 파일을 로드하고 적절한 형식으로 변환합니다.
    
    Args:
        file_path (str): 오디오 파일 경로
        sample_rate (int): 샘플링 레이트
        
    Returns:
        numpy.ndarray: 오디오 데이터
    """
    audio, sr = librosa.load(file_path, sr=sample_rate)
    return audio

def transcribe_audio(audio_data, processor, model):
    """
    오디오 데이터를 텍스트로 변환합니다.
    
    Args:
        audio_data (numpy.ndarray): 오디오 데이터
        processor (Wav2Vec2Processor): wav2vec 프로세서
        model (Wav2Vec2ForCTC): wav2vec 모델
        
    Returns:
        str: 변환된 텍스트
    """
    # 입력을 모델 형식에 맞게 변환
    inputs = processor(audio_data, sampling_rate=16000, return_tensors="pt", padding=True)
    
    # 모델에 입력 전달 (CTC 디코딩)
    with torch.no_grad():
        logits = model(inputs.input_values).logits
    
    # 확률이 가장 높은 토큰 가져오기
    predicted_ids = torch.argmax(logits, dim=-1)
    
    # 텍스트로 디코딩
    transcription = processor.batch_decode(predicted_ids)
    
    return transcription[0]

def main():
    # 모델 및 프로세서 로드 (한국어 모델 사용)
    model_name = "kresnik/wav2vec2-large-xlsr-korean"
    processor = Wav2Vec2Processor.from_pretrained(model_name)
    model = Wav2Vec2ForCTC.from_pretrained(model_name)
    
    # 테스트할 오디오 파일 경로
    audio_file = "sample.wav"  # 이 파일은 따로 준비해야 합니다
    
    try:
        # 오디오 로드
        audio_data = load_audio(audio_file)
        
        # 텍스트 변환
        transcription = transcribe_audio(audio_data, processor, model)
        
        print(f"변환 결과: {transcription}")
    except FileNotFoundError:
        print(f"파일을 찾을 수 없습니다: {audio_file}")
        print("샘플 오디오 파일을 다운로드하거나 경로를 수정해주세요.")

if __name__ == "__main__":
    main() 