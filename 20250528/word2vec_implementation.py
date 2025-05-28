"""
Word2Vec 모델의 기본 구현

이 스크립트는 Word2Vec 모델(CBOW 및 Skip-gram)을 PyTorch를 사용하여 처음부터 구현합니다.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from collections import Counter
import random
import time
import os
from konlpy.tag import Okt

# 시드 설정
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# 장치 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Word2VecDataset(Dataset):
    """Word2Vec 학습을 위한 데이터셋 클래스"""
    
    def __init__(self, text_data, window_size=2, mode='cbow'):
        """
        Args:
            text_data (list): 토큰화된 문장 리스트
            window_size (int): 컨텍스트 윈도우 크기
            mode (str): 'cbow' 또는 'skipgram'
        """
        self.window_size = window_size
        self.mode = mode
        
        # 어휘 구축
        self.vocab = self._build_vocab(text_data)
        self.word_to_idx = {word: idx for idx, (word, _) in enumerate(self.vocab)}
        self.idx_to_word = {idx: word for word, idx in self.word_to_idx.items()}
        self.vocab_size = len(self.vocab)
        
        # 데이터 준비
        self.data = self._prepare_data(text_data)
        
    def _build_vocab(self, text_data):
        """어휘 구축"""
        word_counts = Counter()
        for sentence in text_data:
            word_counts.update(sentence)
        return word_counts.most_common()
    
    def _prepare_data(self, text_data):
        """학습 데이터 준비"""
        data = []
        
        for sentence in text_data:
            word_indices = [self.word_to_idx[word] for word in sentence if word in self.word_to_idx]
            
            for i, word_idx in enumerate(word_indices):
                # 윈도우 범위 내의 컨텍스트 단어
                context_indices = []
                for j in range(i - self.window_size, i + self.window_size + 1):
                    if j != i and 0 <= j < len(word_indices):
                        context_indices.append(word_indices[j])
                
                if context_indices:
                    if self.mode == 'cbow':
                        # CBOW: 컨텍스트(x)로 타겟 단어(y) 예측
                        data.append((context_indices, word_idx))
                    else:
                        # Skip-gram: 타겟 단어(x)로 컨텍스트 단어(y) 예측
                        for context_idx in context_indices:
                            data.append((word_idx, context_idx))
        
        return data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        x, y = self.data[idx]
        
        if self.mode == 'cbow' and isinstance(x, list):
            # CBOW: 컨텍스트 단어 인덱스를 one-hot 벡터의 평균으로 변환
            x_tensor = torch.zeros(self.vocab_size)
            for context_idx in x:
                x_tensor[context_idx] += 1
            x_tensor = x_tensor / len(x)  # 평균
        else:
            # Skip-gram: 타겟 단어 인덱스를 one-hot 벡터로 변환
            x_tensor = torch.zeros(self.vocab_size)
            x_tensor[x] = 1
        
        # 타겟을 one-hot 벡터로 변환
        y_tensor = torch.zeros(self.vocab_size)
        y_tensor[y] = 1
        
        return x_tensor, y_tensor

class CBOW(nn.Module):
    """Continuous Bag of Words (CBOW) 모델"""
    
    def __init__(self, vocab_size, embedding_dim):
        """
        Args:
            vocab_size (int): 어휘 크기
            embedding_dim (int): 임베딩 차원
        """
        super(CBOW, self).__init__()
        
        # 입력층 -> 은닉층 가중치
        self.embeddings = nn.Linear(vocab_size, embedding_dim, bias=False)
        
        # 은닉층 -> 출력층 가중치
        self.output_layer = nn.Linear(embedding_dim, vocab_size, bias=False)
        
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): 컨텍스트 단어의 one-hot 벡터 평균 [batch_size, vocab_size]
            
        Returns:
            torch.Tensor: 각 단어의 확률 [batch_size, vocab_size]
        """
        # 임베딩 계산 [batch_size, embedding_dim]
        hidden = self.embeddings(x)
        
        # 출력층 계산 [batch_size, vocab_size]
        output = self.output_layer(hidden)
        
        return torch.log_softmax(output, dim=1)

class SkipGram(nn.Module):
    """Skip-gram 모델"""
    
    def __init__(self, vocab_size, embedding_dim):
        """
        Args:
            vocab_size (int): 어휘 크기
            embedding_dim (int): 임베딩 차원
        """
        super(SkipGram, self).__init__()
        
        # 입력층 -> 은닉층 가중치
        self.embeddings = nn.Linear(vocab_size, embedding_dim, bias=False)
        
        # 은닉층 -> 출력층 가중치
        self.output_layer = nn.Linear(embedding_dim, vocab_size, bias=False)
        
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): 타겟 단어의 one-hot 벡터 [batch_size, vocab_size]
            
        Returns:
            torch.Tensor: 각 컨텍스트 단어의 확률 [batch_size, vocab_size]
        """
        # 임베딩 계산 [batch_size, embedding_dim]
        hidden = self.embeddings(x)
        
        # 출력층 계산 [batch_size, vocab_size]
        output = self.output_layer(hidden)
        
        return torch.log_softmax(output, dim=1)

def train_model(model, dataloader, epochs, learning_rate):
    """
    Word2Vec 모델 학습
    
    Args:
        model (nn.Module): 학습할 모델 (CBOW 또는 SkipGram)
        dataloader (DataLoader): 학습 데이터 로더
        epochs (int): 학습 에포크 수
        learning_rate (float): 학습률
        
    Returns:
        nn.Module: 학습된 모델
        list: 에포크별 손실값
    """
    model.to(device)
    
    # 손실 함수 및 옵티마이저 설정
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 학습 과정 기록
    epoch_losses = []
    
    for epoch in range(epochs):
        start_time = time.time()
        total_loss = 0
        
        for x_batch, y_batch in dataloader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            
            # 모델 출력
            log_probs = model(x_batch)
            
            # 손실 계산 (NLL 손실 함수는 one-hot이 아닌 클래스 인덱스를 기대)
            _, y_indices = torch.max(y_batch, dim=1)
            loss = criterion(log_probs, y_indices)
            
            # 역전파 및 가중치 업데이트
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # 에포크 평균 손실
        avg_loss = total_loss / len(dataloader)
        epoch_losses.append(avg_loss)
        
        # 진행 상황 출력
        elapsed = time.time() - start_time
        print(f'에포크 {epoch+1}/{epochs}, 손실: {avg_loss:.4f}, 시간: {elapsed:.2f}초')
    
    return model, epoch_losses

def get_word_vector(model, word, word_to_idx, vocab_size):
    """
    학습된 모델에서 단어 벡터 추출
    
    Args:
        model (nn.Module): 학습된 모델
        word (str): 벡터를 추출할 단어
        word_to_idx (dict): 단어-인덱스 매핑
        vocab_size (int): 어휘 크기
        
    Returns:
        numpy.ndarray: 단어 벡터
    """
    if word not in word_to_idx:
        return None
    
    # 단어의 one-hot 벡터 생성
    word_idx = word_to_idx[word]
    one_hot = torch.zeros(vocab_size)
    one_hot[word_idx] = 1
    
    # 임베딩 계산
    model.eval()
    with torch.no_grad():
        word_vector = model.embeddings(one_hot.to(device))
        
    return word_vector.cpu().numpy()

def get_similar_words(model, word, word_to_idx, idx_to_word, vocab_size, top_k=5):
    """
    단어와 가장 유사한 단어 추출
    
    Args:
        model (nn.Module): 학습된 모델
        word (str): 기준 단어
        word_to_idx (dict): 단어-인덱스 매핑
        idx_to_word (dict): 인덱스-단어 매핑
        vocab_size (int): 어휘 크기
        top_k (int): 반환할 유사 단어 개수
        
    Returns:
        list: (단어, 유사도) 튜플 리스트
    """
    if word not in word_to_idx:
        return []
    
    # 모든 단어의 임베딩 계산
    embeddings = []
    words = []
    
    for idx in range(vocab_size):
        w = idx_to_word[idx]
        vector = get_word_vector(model, w, word_to_idx, vocab_size)
        embeddings.append(vector)
        words.append(w)
    
    # 타겟 단어 벡터
    target_vector = get_word_vector(model, word, word_to_idx, vocab_size)
    
    # 유사도 계산 (코사인 유사도)
    similarities = []
    for i, vector in enumerate(embeddings):
        if words[i] != word:  # 자기 자신 제외
            similarity = np.dot(target_vector, vector) / (np.linalg.norm(target_vector) * np.linalg.norm(vector))
            similarities.append((words[i], similarity))
    
    # 유사도 기준 정렬
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    return similarities[:top_k]

def plot_loss(losses, title="Word2Vec 학습 손실"):
    """손실 그래프 그리기"""
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title(title)
    plt.xlabel('에포크')
    plt.ylabel('손실')
    plt.grid(True)
    plt.savefig('word2vec_loss.png', dpi=300, bbox_inches='tight')
    plt.show()

def prepare_sample_data():
    """샘플 한국어 데이터 준비"""
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
        "데이터 마이닝은 대용량 데이터에서 패턴을 찾는 과정입니다."
    ]

def tokenize_text(text_data):
    """텍스트 토큰화"""
    tokenized_data = []
    
    okt = Okt()
    for text in text_data:
        tokens = okt.morphs(text)
        tokenized_data.append(tokens)
    
    return tokenized_data

def main():
    print("Word2Vec 모델 구현 예제를 시작합니다...")
    
    # 하이퍼파라미터 설정
    embedding_dim = 50
    window_size = 2
    batch_size = 32
    epochs = 100
    learning_rate = 0.001
    
    # 1. 데이터 준비
    print("\n1. 데이터 준비 중...")
    sentences = prepare_sample_data()
    tokenized_sentences = tokenize_text(sentences)
    
    print(f"샘플 데이터: {len(sentences)}개 문장")
    print(f"토큰화 예시: {tokenized_sentences[0]}")
    
    # 2. CBOW 모델 학습
    print("\n2. CBOW 모델 학습 중...")
    
    # CBOW 데이터셋 생성
    cbow_dataset = Word2VecDataset(tokenized_sentences, window_size, mode='cbow')
    cbow_dataloader = DataLoader(cbow_dataset, batch_size=batch_size, shuffle=True)
    
    # CBOW 모델 초기화
    cbow_model = CBOW(cbow_dataset.vocab_size, embedding_dim)
    
    # CBOW 모델 학습
    print(f"어휘 크기: {cbow_dataset.vocab_size}개 단어")
    print(f"학습 데이터: {len(cbow_dataset)}개 샘플")
    
    cbow_model, cbow_losses = train_model(
        cbow_model, 
        cbow_dataloader,
        epochs,
        learning_rate
    )
    
    # 손실 그래프 그리기
    plot_loss(cbow_losses, "CBOW 모델 학습 손실")
    
    # 3. Skip-gram 모델 학습
    print("\n3. Skip-gram 모델 학습 중...")
    
    # Skip-gram 데이터셋 생성
    sg_dataset = Word2VecDataset(tokenized_sentences, window_size, mode='skipgram')
    sg_dataloader = DataLoader(sg_dataset, batch_size=batch_size, shuffle=True)
    
    # Skip-gram 모델 초기화
    sg_model = SkipGram(sg_dataset.vocab_size, embedding_dim)
    
    # Skip-gram 모델 학습
    print(f"학습 데이터: {len(sg_dataset)}개 샘플")
    
    sg_model, sg_losses = train_model(
        sg_model, 
        sg_dataloader,
        epochs,
        learning_rate
    )
    
    # 손실 그래프 그리기
    plot_loss(sg_losses, "Skip-gram 모델 학습 손실")
    
    # 4. 학습된 모델 활용
    print("\n4. 학습된 모델 활용:")
    
    # 유사 단어 찾기
    target_word = "인공지능"
    if target_word in cbow_dataset.word_to_idx:
        # CBOW 모델로 유사 단어 찾기
        print(f"\n'{target_word}'과(와) 가장 유사한 단어 (CBOW):")
        similar_words = get_similar_words(
            cbow_model,
            target_word,
            cbow_dataset.word_to_idx,
            cbow_dataset.idx_to_word,
            cbow_dataset.vocab_size,
            top_k=5
        )
        
        for word, similarity in similar_words:
            print(f"  {word}: {similarity:.4f}")
        
        # Skip-gram 모델로 유사 단어 찾기
        print(f"\n'{target_word}'과(와) 가장 유사한 단어 (Skip-gram):")
        similar_words = get_similar_words(
            sg_model,
            target_word,
            sg_dataset.word_to_idx,
            sg_dataset.idx_to_word,
            sg_dataset.vocab_size,
            top_k=5
        )
        
        for word, similarity in similar_words:
            print(f"  {word}: {similarity:.4f}")
    else:
        print(f"'{target_word}'는 어휘에 없습니다.")
    
    # 5. 모델 저장
    print("\n5. 모델 저장 중...")
    
    # 모델 상태 저장
    torch.save(cbow_model.state_dict(), "cbow_model.pt")
    torch.save(sg_model.state_dict(), "skipgram_model.pt")
    
    # 어휘 정보 저장
    vocab_info = {
        "word_to_idx": cbow_dataset.word_to_idx,
        "idx_to_word": cbow_dataset.idx_to_word,
        "vocab_size": cbow_dataset.vocab_size
    }
    
    torch.save(vocab_info, "word2vec_vocab.pt")
    
    print("모델 저장 완료")
    print("\nWord2Vec 모델 구현 예제를 완료했습니다!")

if __name__ == "__main__":
    main() 