from collections import Counter
import math
import random

class NGramClassifier:
    def __init__(self, n=2):
        """
        간단한 n-gram 기반 텍스트 분류기를 초기화합니다.
        
        Args:
            n (int): 사용할 n-gram의 크기
        """
        self.n = n
        self.class_profiles = {}
    
    def _extract_ngrams(self, text):
        """텍스트에서 문자 단위 n-gram을 추출합니다"""
        text = text.lower()
        ngrams = []
        
        for i in range(len(text) - self.n + 1):
            ngram = text[i:i+self.n]
            ngrams.append(ngram)
        
        return Counter(ngrams)
    
    def train(self, texts, labels):
        """
        해당 레이블이 있는 텍스트 세트에서 분류기를 훈련합니다.
        
        Args:
            texts (list): 텍스트 샘플 목록
            labels (list): 해당 레이블 목록
        """
        # 레이블별로 텍스트 그룹화
        grouped_texts = {}
        for text, label in zip(texts, labels):
            if label not in grouped_texts:
                grouped_texts[label] = []
            grouped_texts[label].append(text)
        
        # 각 클래스에 대한 n-gram 프로필 생성
        for label, class_texts in grouped_texts.items():
            combined_counter = Counter()
            
            for text in class_texts:
                ngrams = self._extract_ngrams(text)
                combined_counter.update(ngrams)
            
            # 카운트를 정규화하여 프로필 생성
            total = sum(combined_counter.values())
            profile = {ngram: count/total for ngram, count in combined_counter.items()}
            
            self.class_profiles[label] = profile
    
    def predict(self, text):
        """
        주어진 텍스트의 클래스를 예측합니다.
        
        Args:
            text (str): 분류할 텍스트
            
        Returns:
            str: 예측된 클래스 레이블
        """
        text_ngrams = self._extract_ngrams(text)
        
        best_score = float('-inf')
        best_label = None
        
        for label, profile in self.class_profiles.items():
            score = 0
            
            # n-gram 오버랩을 기반으로 점수 계산
            for ngram, count in text_ngrams.items():
                if ngram in profile:
                    # 언더플로우를 방지하기 위해 로그 확률 추가
                    score += count * math.log(profile[ngram] + 0.0001)
                else:
                    # 보이지 않는 n-gram에 대한 스무딩
                    score += count * math.log(0.0001)
            
            if score > best_score:
                best_score = score
                best_label = label
        
        return best_label

# 사용 예시
if __name__ == "__main__":
    # 샘플 데이터셋: 언어 식별
    english_texts = [
        "The quick brown fox jumps over the lazy dog",
        "To be or not to be, that is the question",
        "Life is like a box of chocolates",
        "All that glitters is not gold",
    ]
    
    spanish_texts = [
        "El rápido zorro marrón salta sobre el perro perezoso",
        "Ser o no ser, esa es la cuestión",
        "La vida es como una caja de chocolates",
        "No todo lo que brilla es oro",
    ]
    
    korean_texts = [
        "빠른 갈색 여우가 게으른 개를 뛰어넘습니다",
        "사느냐 죽느냐, 그것이 문제로다",
        "인생은 초콜릿 상자와 같습니다",
        "반짝이는 모든 것이 금은 아닙니다",
    ]
    
    # 훈련 및 테스트 데이터 준비
    all_texts = english_texts + spanish_texts + korean_texts
    all_labels = ["영어"] * len(english_texts) + ["스페인어"] * len(spanish_texts) + ["한국어"] * len(korean_texts)
    
    # 데이터 섞기
    combined = list(zip(all_texts, all_labels))
    random.shuffle(combined)
    all_texts, all_labels = zip(*combined)
    
    # 훈련 및 테스트 세트로 분할 (70% 훈련, 30% 테스트)
    split_idx = int(len(all_texts) * 0.7)
    train_texts, train_labels = all_texts[:split_idx], all_labels[:split_idx]
    test_texts, test_labels = all_texts[split_idx:], all_labels[split_idx:]
    
    # 분류기 훈련
    classifier = NGramClassifier(n=3)  # 트라이그램 사용
    classifier.train(train_texts, train_labels)
    
    # 분류기 테스트
    correct = 0
    print("\n언어 식별 결과:")
    print("-" * 40)
    
    for text, true_label in zip(test_texts, test_labels):
        predicted = classifier.predict(text)
        is_correct = predicted == true_label
        if is_correct:
            correct += 1
        
        print(f"텍스트: {text[:50]}{'...' if len(text) > 50 else ''}")
        print(f"실제 레이블: {true_label}, 예측: {predicted}, 정확함: {is_correct}")
        print("-" * 40)
    
    accuracy = correct / len(test_labels) * 100
    print(f"\n정확도: {accuracy:.2f}%")
    
    # 새로운 텍스트로 시도
    new_texts = [
        "Hello world, this is a test",
        "Hola mundo, esto es una prueba",
        "안녕 세상아, 이것은 테스트야"
    ]
    
    print("\n새 텍스트에 대한 예측:")
    for text in new_texts:
        predicted = classifier.predict(text)
        print(f"텍스트: {text}")
        print(f"예측된 언어: {predicted}")
        print("-" * 40) 