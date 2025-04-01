import numpy as np
from collections import defaultdict, Counter
import math
import random

class NGramLanguageModel:
    """
    N-gram 기반 언어 모델 클래스.
    각종 스무딩(평활화) 기법 구현을 포함합니다.
    """
    
    def __init__(self, n=2):
        """
        N-gram 언어 모델 초기화
        
        Args:
            n (int): n-gram의 크기
        """
        self.n = n
        self.ngram_counts = defaultdict(Counter)
        self.context_counts = Counter()
        self.vocab = set()
        
    def fit(self, sentences):
        """
        말뭉치에서 n-gram 빈도를 학습합니다.
        
        Args:
            sentences (list): 문장 목록
        """
        for sentence in sentences:
            # 문장 시작과 끝 표시 추가
            tokens = ['<s>'] * (self.n-1) + sentence.lower().split() + ['</s>']
            self.vocab.update(tokens)
            
            # n-gram 빈도 계산
            for i in range(len(tokens) - self.n + 1):
                ngram = tuple(tokens[i:i+self.n])
                context = tuple(tokens[i:i+self.n-1])
                
                self.ngram_counts[context][ngram[-1]] += 1
                self.context_counts[context] += 1
    
    def mle_prob(self, context, word):
        """
        최대 우도 추정(MLE) 확률 계산: 평활화 없음
        
        Args:
            context (tuple): 이전 단어들의 튜플 (n-1개 단어)
            word (str): 현재 단어
            
        Returns:
            float: P(word|context)의 MLE 확률
        """
        if context in self.context_counts and self.context_counts[context] > 0:
            return self.ngram_counts[context][word] / self.context_counts[context]
        return 0.0
    
    def laplace_smoothing(self, context, word, alpha=1.0):
        """
        라플라스(애드원) 스무딩으로 확률 계산
        
        Args:
            context (tuple): 이전 단어들의 튜플 (n-1개 단어)
            word (str): 현재 단어
            alpha (float): 평활화 파라미터 (기본값: 1)
            
        Returns:
            float: P(word|context)의 평활화된 확률
        """
        vocab_size = len(self.vocab)
        
        # 해당 컨텍스트의 횟수
        context_count = self.context_counts[context]
        
        # 해당 컨텍스트에서 단어의 횟수
        word_count = self.ngram_counts[context][word]
        
        # 라플라스 스무딩 적용
        return (word_count + alpha) / (context_count + alpha * vocab_size)
    
    def add_k_smoothing(self, context, word, k=0.1):
        """
        Add-k 스무딩으로 확률 계산 (k < 1인 경우를 위한 라플라스 변형)
        
        Args:
            context (tuple): 이전 단어들의 튜플 (n-1개 단어)
            word (str): 현재 단어
            k (float): 평활화 파라미터 (기본값: 0.1)
            
        Returns:
            float: P(word|context)의 평활화된 확률
        """
        # 라플라스 스무딩과 동일하나 k가 다름
        return self.laplace_smoothing(context, word, alpha=k)
    
    def jelinek_mercer_smoothing(self, context, word, lambda_=0.8):
        """
        Jelinek-Mercer 스무딩으로 확률 계산
        상위 수준 모델(unigram)과 하위 수준 모델(bigram) 보간
        
        Args:
            context (tuple): 이전 단어들의 튜플 (n-1개 단어)
            word (str): 현재 단어
            lambda_ (float): 보간 가중치 (0~1 사이)
            
        Returns:
            float: P(word|context)의 평활화된 확률
        """
        # bigram 확률 (또는 더 높은 차수 n-gram)
        p_mle = self.mle_prob(context, word)
        
        # unigram 확률 (또는 더 낮은 차수 n-gram)
        total_words = sum(self.ngram_counts[()].values())
        p_unigram = self.ngram_counts[()][word] / total_words if total_words > 0 else 0
        
        # 보간법 적용
        return lambda_ * p_mle + (1 - lambda_) * p_unigram
    
    def witten_bell_smoothing(self, context, word):
        """
        Witten-Bell 스무딩으로 확률 계산
        
        Args:
            context (tuple): 이전 단어들의 튜플 (n-1개 단어)
            word (str): 현재 단어
            
        Returns:
            float: P(word|context)의 평활화된 확률
        """
        # 해당 컨텍스트에서 나타난 고유 단어 수
        if context in self.ngram_counts:
            unique_words = len(self.ngram_counts[context])
        else:
            unique_words = 0
        
        # 해당 컨텍스트 출현 횟수
        context_count = self.context_counts[context]
        
        # 해당 단어 출현 횟수
        word_count = self.ngram_counts[context][word]
        
        if context_count == 0 or unique_words == 0:
            # 컨텍스트가 보이지 않은 경우 unigram 확률 사용
            total_words = sum(self.ngram_counts[()].values())
            return self.ngram_counts[()][word] / total_words if total_words > 0 else 1/len(self.vocab)
        
        # Witten-Bell 스무딩 파라미터
        lambda_ = context_count / (context_count + unique_words)
        
        # 고차 모델 확률
        p_mle = word_count / context_count
        
        # 백오프 확률 (unigram)
        total_words = sum(self.ngram_counts[()].values())
        p_unigram = self.ngram_counts[()][word] / total_words if total_words > 0 else 0
        
        # Witten-Bell 스무딩 적용
        return lambda_ * p_mle + (1 - lambda_) * p_unigram
    
    def kneser_ney_smoothing(self, context, word, delta=0.75):
        """
        Kneser-Ney 스무딩으로 확률 계산
        
        Args:
            context (tuple): 이전 단어들의 튜플 (n-1개 단어)
            word (str): 현재 단어
            delta (float): 할인 파라미터
            
        Returns:
            float: P(word|context)의 평활화된 확률
        """
        # 컨텍스트가 없는 경우 기본값 반환
        if context not in self.context_counts or self.context_counts[context] == 0:
            # 단어의 "다양성" 측정
            num_contexts = sum(1 for cont in self.ngram_counts if word in self.ngram_counts[cont])
            total_contexts = sum(len(self.ngram_counts[cont]) for cont in self.ngram_counts)
            return num_contexts / total_contexts if total_contexts > 0 else 1/len(self.vocab)
        
        # 해당 컨텍스트에서 고유한 단어 수
        unique_words = len(self.ngram_counts[context])
        
        # 컨텍스트 빈도
        context_count = self.context_counts[context]
        
        # 단어 빈도
        word_count = self.ngram_counts[context][word]
        
        # 할인된 확률 계산
        if word_count > 0:
            discount_prob = max(0, word_count - delta) / context_count
        else:
            discount_prob = 0
        
        # 정규화 질량
        alpha = (delta * unique_words) / context_count
        
        # 낮은 차수의 분포 (백오프)
        num_contexts = sum(1 for cont in self.ngram_counts if word in self.ngram_counts[cont])
        total_contexts = sum(len(self.ngram_counts[cont]) for cont in self.ngram_counts)
        continuation_prob = num_contexts / total_contexts if total_contexts > 0 else 1/len(self.vocab)
        
        # Kneser-Ney 스무딩 적용
        return discount_prob + alpha * continuation_prob
    
    def perplexity(self, test_sentences, smoothing_method="laplace", **kwargs):
        """
        테스트 문장들에 대한 퍼플렉시티 계산
        
        Args:
            test_sentences (list): 테스트 문장 목록
            smoothing_method (str): 사용할 스무딩 방법
            **kwargs: 스무딩 함수에 전달할 추가 인자
            
        Returns:
            float: 테스트 문장들에 대한 퍼플렉시티
        """
        log_prob_sum = 0.0
        token_count = 0
        
        for sentence in test_sentences:
            # 문장 시작과 끝 표시 추가
            tokens = ['<s>'] * (self.n-1) + sentence.lower().split() + ['</s>']
            token_count += len(tokens) - (self.n - 1)  # 시작 태그 제외
            
            for i in range(len(tokens) - self.n + 1):
                context = tuple(tokens[i:i+self.n-1])
                word = tokens[i+self.n-1]
                
                # 선택한 스무딩 방법 적용
                if smoothing_method == "mle":
                    prob = self.mle_prob(context, word)
                elif smoothing_method == "laplace":
                    prob = self.laplace_smoothing(context, word, **kwargs)
                elif smoothing_method == "add_k":
                    prob = self.add_k_smoothing(context, word, **kwargs)
                elif smoothing_method == "jelinek_mercer":
                    prob = self.jelinek_mercer_smoothing(context, word, **kwargs)
                elif smoothing_method == "witten_bell":
                    prob = self.witten_bell_smoothing(context, word)
                elif smoothing_method == "kneser_ney":
                    prob = self.kneser_ney_smoothing(context, word, **kwargs)
                else:
                    prob = self.laplace_smoothing(context, word)  # 기본값
                
                # 확률이 0인 경우 처리 (로그 계산을 위해)
                if prob <= 0:
                    prob = 1e-10
                
                log_prob_sum += math.log2(prob)
        
        # 퍼플렉시티 계산: 2^(-평균 로그 확률)
        return math.pow(2, -log_prob_sum / token_count if token_count > 0 else 0)
    
    def generate_sentence(self, max_length=20, smoothing_method="laplace", **kwargs):
        """
        언어 모델을 사용하여 문장 생성
        
        Args:
            max_length (int): 최대 문장 길이
            smoothing_method (str): 사용할 스무딩 방법
            **kwargs: 스무딩 함수에 전달할 추가 인자
            
        Returns:
            str: 생성된 문장
        """
        # 문장 시작 토큰
        sentence = ['<s>'] * (self.n-1)
        
        # 종료 토큰이 나오거나 최대 길이에 도달할 때까지 단어 생성
        while sentence[-1] != '</s>' and len(sentence) < max_length + (self.n-1):
            context = tuple(sentence[-(self.n-1):])
            
            # 다음 단어의 확률 계산
            word_probs = {}
            for word in self.vocab:
                if smoothing_method == "mle":
                    prob = self.mle_prob(context, word)
                elif smoothing_method == "laplace":
                    prob = self.laplace_smoothing(context, word, **kwargs)
                elif smoothing_method == "add_k":
                    prob = self.add_k_smoothing(context, word, **kwargs)
                elif smoothing_method == "jelinek_mercer":
                    prob = self.jelinek_mercer_smoothing(context, word, **kwargs)
                elif smoothing_method == "witten_bell":
                    prob = self.witten_bell_smoothing(context, word)
                elif smoothing_method == "kneser_ney":
                    prob = self.kneser_ney_smoothing(context, word, **kwargs)
                else:
                    prob = self.laplace_smoothing(context, word)  # 기본값
                
                word_probs[word] = prob
            
            # 확률에 따라 다음 단어 선택
            total_prob = sum(word_probs.values())
            if total_prob == 0:
                # 알려진 컨텍스트가 없는 경우 랜덤하게 선택
                next_word = random.choice(list(self.vocab))
            else:
                # 확률에 따라 선택
                rand_prob = random.random() * total_prob
                cumulative_prob = 0.0
                next_word = None
                
                for word, prob in word_probs.items():
                    cumulative_prob += prob
                    if cumulative_prob >= rand_prob:
                        next_word = word
                        break
                
                if next_word is None:
                    next_word = random.choice(list(self.vocab))
            
            sentence.append(next_word)
        
        # 시작 및 종료 태그 제거
        result = ' '.join(sentence[(self.n-1):])
        if result.endswith(' </s>'):
            result = result[:-5]
        
        return result


# 학습 데이터
train_sentences = [
    "나는 오늘 학교에 갔다",
    "그는 어제 영화를 보았다",
    "우리는 내일 공원에 갈 것이다",
    "그녀는 책을 읽고 있다",
    "학생들은 시험을 준비하고 있다",
    "나는 커피를 마시고 싶다",
    "그는 음악을 들으며 운동한다",
    "우리는 저녁을 먹으러 식당에 갔다",
    "그녀는 친구와 쇼핑을 하고 있다",
    "우리는 주말에 여행을 계획하고 있다",
    "학생들이 교실에서 공부하고 있다",
    "그는 매일 아침 운동을 한다",
    "나는 파이썬 프로그래밍을 배우고 있다",
    "그녀는 주말에 책을 읽는 것을 좋아한다",
    "우리는 함께 프로젝트를 진행하고 있다"
]

# 테스트 데이터
test_sentences = [
    "나는 학교에 갔다",
    "그는 영화를 보았다",
    "우리는 공원에 갈 것이다",
    "그녀는 책을 읽는다",
    "학생들은 시험을 준비한다"
]

# 메인 실행 코드
if __name__ == "__main__":
    print("N-gram 언어 모델과 다양한 스무딩 기법 예제\n")
    
    # 바이그램 모델 생성 및 학습
    bigram_model = NGramLanguageModel(n=2)
    bigram_model.fit(train_sentences)
    
    # 트라이그램 모델 생성 및 학습
    trigram_model = NGramLanguageModel(n=3)
    trigram_model.fit(train_sentences)
    
    print(f"학습 데이터 크기: {len(train_sentences)} 문장")
    print(f"테스트 데이터 크기: {len(test_sentences)} 문장")
    print(f"어휘 크기: {len(bigram_model.vocab)} 단어\n")
    
    # 스무딩 비교를 위한 컨텍스트와 단어
    context = ('나는',)
    word = '학교에'
    
    print(f"컨텍스트 '{' '.join(context)}'에서 '{word}'의 확률 비교:")
    print(f"최대 우도 추정 (MLE): {bigram_model.mle_prob(context, word):.6f}")
    print(f"라플라스 스무딩: {bigram_model.laplace_smoothing(context, word):.6f}")
    print(f"Add-k 스무딩 (k=0.1): {bigram_model.add_k_smoothing(context, word, k=0.1):.6f}")
    print(f"Jelinek-Mercer 스무딩: {bigram_model.jelinek_mercer_smoothing(context, word):.6f}")
    print(f"Witten-Bell 스무딩: {bigram_model.witten_bell_smoothing(context, word):.6f}")
    print(f"Kneser-Ney 스무딩: {bigram_model.kneser_ney_smoothing(context, word):.6f}\n")
    
    # 보이지 않은 컨텍스트-단어 쌍
    unknown_context = ('학생이',)
    unknown_word = '춤을'
    
    print(f"보이지 않은 컨텍스트 '{' '.join(unknown_context)}'에서 '{unknown_word}'의 확률 비교:")
    print(f"최대 우도 추정 (MLE): {bigram_model.mle_prob(unknown_context, unknown_word):.6f}")
    print(f"라플라스 스무딩: {bigram_model.laplace_smoothing(unknown_context, unknown_word):.6f}")
    print(f"Add-k 스무딩 (k=0.1): {bigram_model.add_k_smoothing(unknown_context, unknown_word, k=0.1):.6f}")
    print(f"Jelinek-Mercer 스무딩: {bigram_model.jelinek_mercer_smoothing(unknown_context, unknown_word):.6f}")
    print(f"Witten-Bell 스무딩: {bigram_model.witten_bell_smoothing(unknown_context, unknown_word):.6f}")
    print(f"Kneser-Ney 스무딩: {bigram_model.kneser_ney_smoothing(unknown_context, unknown_word):.6f}\n")
    
    # 다양한 스무딩 방법을 사용한 퍼플렉시티 계산
    print("바이그램 모델의 테스트 데이터에 대한 퍼플렉시티:")
    print(f"최대 우도 추정 (MLE): {bigram_model.perplexity(test_sentences, 'mle'):.2f}")
    print(f"라플라스 스무딩: {bigram_model.perplexity(test_sentences, 'laplace'):.2f}")
    print(f"Add-k 스무딩 (k=0.1): {bigram_model.perplexity(test_sentences, 'add_k', k=0.1):.2f}")
    print(f"Jelinek-Mercer 스무딩: {bigram_model.perplexity(test_sentences, 'jelinek_mercer'):.2f}")
    print(f"Witten-Bell 스무딩: {bigram_model.perplexity(test_sentences, 'witten_bell'):.2f}")
    print(f"Kneser-Ney 스무딩: {bigram_model.perplexity(test_sentences, 'kneser_ney'):.2f}\n")
    
    print("트라이그램 모델의 테스트 데이터에 대한 퍼플렉시티:")
    print(f"라플라스 스무딩: {trigram_model.perplexity(test_sentences, 'laplace'):.2f}")
    print(f"Add-k 스무딩 (k=0.1): {trigram_model.perplexity(test_sentences, 'add_k', k=0.1):.2f}")
    print(f"Jelinek-Mercer 스무딩: {trigram_model.perplexity(test_sentences, 'jelinek_mercer'):.2f}\n")
    
    # 문장 생성 예제
    print("다양한 스무딩 방법을 사용한 문장 생성 (바이그램 모델):")
    print(f"라플라스 스무딩: {bigram_model.generate_sentence(smoothing_method='laplace')}")
    print(f"Add-k 스무딩: {bigram_model.generate_sentence(smoothing_method='add_k', k=0.1)}")
    print(f"Jelinek-Mercer 스무딩: {bigram_model.generate_sentence(smoothing_method='jelinek_mercer')}")
    print(f"Witten-Bell 스무딩: {bigram_model.generate_sentence(smoothing_method='witten_bell')}")
    print(f"Kneser-Ney 스무딩: {bigram_model.generate_sentence(smoothing_method='kneser_ney')}\n")
    
    print("트라이그램 모델을 사용한 문장 생성:")
    print(f"라플라스 스무딩: {trigram_model.generate_sentence(smoothing_method='laplace')}")
    print(f"Add-k 스무딩: {trigram_model.generate_sentence(smoothing_method='add_k', k=0.1)}")
    print(f"Jelinek-Mercer 스무딩: {trigram_model.generate_sentence(smoothing_method='jelinek_mercer')}") 