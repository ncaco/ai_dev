"""
Word2Vec 모델 구현 및 사용 예제

이 스크립트는 Word2Vec 모델을 구현하고 간단한 한국어 텍스트에 적용하는 방법을 보여줍니다.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec
import logging
import os
from konlpy.tag import Mecab, Okt

# 로깅 설정
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

class EpochLogger(CallbackAny2Vec):
    """Word2Vec 학습 진행 상황을 모니터링하기 위한 콜백"""
    def __init__(self):
        self.epoch = 0
        self.loss_previous_step = 0
        
    def on_epoch_end(self, model):
        loss = model.get_latest_training_loss()
        loss_delta = loss - self.loss_previous_step
        self.loss_previous_step = loss
        print(f'에포크: {self.epoch}, 손실: {loss_delta}')
        self.epoch += 1

def tokenize_text(text_data, use_mecab=False):
    """
    텍스트를 토큰화합니다.
    
    Args:
        text_data (list): 문장 리스트
        use_mecab (bool): Mecab 사용 여부 (False면 Okt 사용)
        
    Returns:
        list: 토큰화된 문장 리스트
    """
    tokenized_data = []
    
    if use_mecab:
        try:
            mecab = Mecab()
            for text in text_data:
                tokens = mecab.morphs(text)
                tokenized_data.append(tokens)
        except Exception as e:
            print(f"Mecab 초기화 실패: {e}")
            print("Okt 토크나이저로 대체합니다.")
            use_mecab = False
    
    if not use_mecab:
        okt = Okt()
        for text in text_data:
            tokens = okt.morphs(text)
            tokenized_data.append(tokens)
    
    return tokenized_data

def train_word2vec(tokenized_data, vector_size=100, window=5, min_count=1, sg=0, epochs=100):
    """
    Word2Vec 모델을 학습합니다.
    
    Args:
        tokenized_data (list): 토큰화된 문장 리스트
        vector_size (int): 단어 벡터 차원
        window (int): 컨텍스트 윈도우 크기
        min_count (int): 최소 단어 빈도
        sg (int): 학습 알고리즘 (0=CBOW, 1=Skip-gram)
        epochs (int): 학습 에포크 수
        
    Returns:
        Word2Vec: 학습된 Word2Vec 모델
    """
    epoch_logger = EpochLogger()
    
    model = Word2Vec(
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=4,
        sg=sg,
        epochs=epochs,
        compute_loss=True,
        callbacks=[epoch_logger]
    )
    
    # 어휘 구축
    model.build_vocab(tokenized_data)
    
    # 모델 학습
    model.train(
        tokenized_data,
        total_examples=model.corpus_count,
        epochs=model.epochs
    )
    
    return model

def visualize_embeddings(model, words=None, n_words=50):
    """
    Word2Vec 임베딩을 시각화합니다.
    
    Args:
        model (Word2Vec): 학습된 Word2Vec 모델
        words (list): 시각화할 단어 리스트 (None이면 가장 빈도가 높은 n_words개 단어 사용)
        n_words (int): 시각화할 단어 수
    """
    if words is None:
        words = [word for word, _ in model.wv.most_common(n_words)]
    
    # 단어 벡터 가져오기
    word_vectors = np.array([model.wv[word] for word in words])
    
    # t-SNE로 차원 축소
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(words)-1))
    embedded = tsne.fit_transform(word_vectors)
    
    # 시각화
    plt.figure(figsize=(14, 10))
    for i, word in enumerate(words):
        plt.scatter(embedded[i, 0], embedded[i, 1])
        plt.annotate(word, xy=(embedded[i, 0], embedded[i, 1]), xytext=(5, 2),
                    textcoords='offset points', ha='right', va='bottom',
                    fontsize=12, fontproperties={'family': 'Malgun Gothic'})
    
    plt.title('Word2Vec 임베딩 시각화 (t-SNE)', fontproperties={'family': 'Malgun Gothic'})
    plt.tight_layout()
    plt.savefig('word2vec_visualization.png', dpi=300, bbox_inches='tight')
    plt.show()

def find_similar_words(model, word, topn=10):
    """
    단어와 가장 유사한 단어들을 찾습니다.
    
    Args:
        model (Word2Vec): 학습된 Word2Vec 모델
        word (str): 기준 단어
        topn (int): 반환할 유사 단어 수
        
    Returns:
        list: (단어, 유사도) 튜플 리스트
    """
    try:
        return model.wv.most_similar(word, topn=topn)
    except KeyError:
        print(f"'{word}'는 어휘에 없습니다.")
        return []

def word_arithmetic(model, pos_words, neg_words, topn=5):
    """
    단어 벡터 연산을 수행합니다 (예: king - man + woman = queen).
    
    Args:
        model (Word2Vec): 학습된 Word2Vec 모델
        pos_words (list): 더할 단어 리스트
        neg_words (list): 뺄 단어 리스트
        topn (int): 반환할 결과 단어 수
        
    Returns:
        list: (단어, 유사도) 튜플 리스트
    """
    try:
        return model.wv.most_similar(positive=pos_words, negative=neg_words, topn=topn)
    except KeyError as e:
        print(f"단어 연산 실패: {e}")
        return []

def prepare_sample_data():
    """
    샘플 한국어 데이터를 준비합니다.
    """
    return [
        "인공지능은 컴퓨터 과학의 한 분야입니다.",
        "딥러닝은 인공지능의 하위 분야 중 하나입니다.",
        "자연어 처리는 기계가 인간의 언어를 이해하고 처리하는 기술입니다.",
        "Word2Vec은 단어를 벡터로 표현하는 알고리즘입니다.",
        "임베딩은 고차원 데이터를 저차원 공간에 매핑하는 것입니다.",
        "한국어는 형태소 분석이 중요한 언어입니다.",
        "언어 모델은 자연어의 패턴을 학습합니다.",
        "기계 번역은 자연어 처리의 응용 분야입니다.",
        "신경망은 인간의 뇌를 모방한 학습 알고리즘입니다.",
        "컴퓨터 비전은 컴퓨터가 이미지를 이해하는 기술입니다.",
        "강화 학습은 시행착오를 통해 학습하는 방법입니다.",
        "음성 인식은 오디오를 텍스트로 변환하는 기술입니다.",
        "추천 시스템은 사용자의 취향을 예측합니다.",
        "자율 주행 자동차는 인공지능 기술을 활용합니다.",
        "데이터 마이닝은 대용량 데이터에서 패턴을 찾는 과정입니다.",
        "빅데이터는 기존 방식으로 처리하기 어려운 대규모 데이터입니다.",
        "클라우드 컴퓨팅은 인터넷을 통해 컴퓨팅 자원을 제공합니다.",
        "블록체인은 분산 원장 기술의 한 종류입니다.",
        "사물인터넷은 기기들이 인터넷으로 연결되는 기술입니다.",
        "가상현실은 컴퓨터로 생성된 환경을 체험하는 기술입니다."
    ]

def load_text_data(file_path):
    """
    텍스트 파일에서 데이터를 로드합니다.
    
    Args:
        file_path (str): 텍스트 파일 경로
        
    Returns:
        list: 문장 리스트
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = [line.strip() for line in f if line.strip()]
        return data
    except Exception as e:
        print(f"파일 로드 실패: {e}")
        return []

def main():
    print("Word2Vec 모델 예제를 시작합니다...")
    
    # 1. 데이터 준비
    print("\n1. 데이터 준비 중...")
    data_file = "korean_text.txt"  # 한국어 텍스트 파일
    
    if os.path.exists(data_file):
        sentences = load_text_data(data_file)
        print(f"'{data_file}' 파일에서 {len(sentences)}개 문장을 로드했습니다.")
    else:
        print(f"'{data_file}' 파일이 없습니다. 샘플 데이터를 사용합니다.")
        sentences = prepare_sample_data()
        print(f"샘플 데이터: {len(sentences)}개 문장")
    
    # 2. 텍스트 토큰화
    print("\n2. 텍스트 토큰화 중...")
    try:
        tokenized_sentences = tokenize_text(sentences, use_mecab=True)
    except:
        tokenized_sentences = tokenize_text(sentences, use_mecab=False)
    
    print(f"토큰화 완료: {len(tokenized_sentences)}개 문장")
    print(f"예시: {tokenized_sentences[0]}")
    
    # 3. Word2Vec 모델 학습 (CBOW 방식)
    print("\n3. Word2Vec 모델 학습 중 (CBOW)...")
    cbow_model = train_word2vec(
        tokenized_sentences,
        vector_size=100,
        window=5,
        min_count=1,
        sg=0,  # CBOW
        epochs=20
    )
    
    # 4. Word2Vec 모델 학습 (Skip-gram 방식)
    print("\n4. Word2Vec 모델 학습 중 (Skip-gram)...")
    sg_model = train_word2vec(
        tokenized_sentences,
        vector_size=100,
        window=5,
        min_count=1,
        sg=1,  # Skip-gram
        epochs=20
    )
    
    # 5. 학습된 모델 평가 및 활용
    print("\n5. 학습된 모델 평가 및 활용:")
    
    # 어휘 크기 확인
    vocab_size = len(cbow_model.wv)
    print(f"어휘 크기: {vocab_size}개 단어")
    
    # 유사 단어 찾기
    target_word = "인공지능"
    if target_word in cbow_model.wv:
        print(f"\n'{target_word}'과(와) 가장 유사한 단어 (CBOW):")
        similar_words = find_similar_words(cbow_model, target_word)
        for word, similarity in similar_words:
            print(f"  {word}: {similarity:.4f}")
        
        print(f"\n'{target_word}'과(와) 가장 유사한 단어 (Skip-gram):")
        similar_words = find_similar_words(sg_model, target_word)
        for word, similarity in similar_words:
            print(f"  {word}: {similarity:.4f}")
    else:
        print(f"'{target_word}'는 어휘에 없습니다.")
    
    # 단어 벡터 연산
    try:
        print("\n단어 벡터 연산 (컴퓨터 + 언어 - 기술):")
        results = word_arithmetic(sg_model, ["컴퓨터", "언어"], ["기술"])
        for word, similarity in results:
            print(f"  {word}: {similarity:.4f}")
    except:
        print("단어 벡터 연산에 필요한 단어가 어휘에 없습니다.")
    
    # 6. 모델 저장 및 로드
    print("\n6. 모델 저장 중...")
    cbow_model.save("word2vec_cbow.model")
    sg_model.save("word2vec_skipgram.model")
    print("모델 저장 완료")
    
    # 7. 임베딩 시각화
    print("\n7. 임베딩 시각화 중...")
    try:
        # 가장 빈도가 높은 30개 단어 시각화
        visualize_embeddings(sg_model, n_words=30)
        print("시각화 완료. 'word2vec_visualization.png' 파일을 확인하세요.")
    except Exception as e:
        print(f"임베딩 시각화 실패: {e}")

if __name__ == "__main__":
    main() 