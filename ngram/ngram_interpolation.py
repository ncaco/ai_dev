import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
import pandas as pd
import random
from collections import Counter

def demonstrate_interpolation():
    """다양한 보간법 기법을 시각화하여 비교합니다."""
    
    # 샘플 데이터 포인트
    x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    y = np.array([1, 3, 2, 5, 7, 8, 8, 9, 10, 12, 14])
    
    # 보간할 x 포인트 (더 촘촘하게)
    x_new = np.linspace(0, 10, 100)
    
    # 다양한 보간법 적용
    # 1. 선형 보간법
    f_linear = interpolate.interp1d(x, y, kind='linear')
    y_linear = f_linear(x_new)
    
    # 2. 최근접 이웃 보간법
    f_nearest = interpolate.interp1d(x, y, kind='nearest')
    y_nearest = f_nearest(x_new)
    
    # 3. 3차 스플라인 보간법
    f_cubic = interpolate.interp1d(x, y, kind='cubic')
    y_cubic = f_cubic(x_new)
    
    # 4. 다항식 보간법 (5차)
    poly_coeffs = np.polyfit(x, y, 5)
    p = np.poly1d(poly_coeffs)
    y_poly = p(x_new)
    
    # 5. 자연 스플라인
    spl = interpolate.splrep(x, y, k=3)
    y_spline = interpolate.splev(x_new, spl)
    
    # 결과 시각화
    plt.figure(figsize=(10, 6))
    
    # 원본 데이터 포인트
    plt.scatter(x, y, c='black', label='원본 데이터')
    
    # 다양한 보간법 결과
    plt.plot(x_new, y_linear, 'r-', label='선형 보간법')
    plt.plot(x_new, y_nearest, 'g--', label='최근접 이웃 보간법')
    plt.plot(x_new, y_cubic, 'b-', label='3차 스플라인 보간법')
    plt.plot(x_new, y_poly, 'm--', label='5차 다항식 보간법')
    plt.plot(x_new, y_spline, 'c-', label='자연 스플라인')
    
    plt.title('다양한 보간법 비교')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()
    plt.grid(True)
    
    # 그래프 저장
    plt.savefig('interpolation_comparison.png')
    print("보간법 비교 그래프가 'interpolation_comparison.png'로 저장되었습니다.")
    

class BackoffInterpolationModel:
    """
    n-gram 언어 모델에서 백오프(backoff)와 보간법(interpolation)을 구현하는 클래스
    """
    
    def __init__(self, n=3):
        """
        모델 초기화
        
        Args:
            n (int): 최대 n-gram 크기
        """
        self.n = n
        self.ngram_counts = {}
        self.context_counts = {}
        self.vocab = set()
        self.lambdas = None  # 보간을 위한 가중치
        
    def fit(self, sentences, held_out_sentences=None):
        """
        말뭉치에서 n-gram 모델 학습
        
        Args:
            sentences (list): 학습 문장 목록
            held_out_sentences (list): 보간 가중치 학습을 위한 별도 문장 목록
        """
        # n-gram 카운트 초기화
        for i in range(1, self.n + 1):
            self.ngram_counts[i] = Counter()
            self.context_counts[i] = Counter()
        
        # 훈련 데이터에서 n-gram 카운트 계산
        for sentence in sentences:
            tokens = ['<s>'] * (self.n - 1) + sentence.lower().split() + ['</s>']
            self.vocab.update(tokens)
            
            # 모든 차수의 n-gram 계산
            for i in range(1, self.n + 1):
                for j in range(len(tokens) - i + 1):
                    ngram = tuple(tokens[j:j+i])
                    self.ngram_counts[i][ngram] += 1
                    
                    if i > 1:
                        context = ngram[:-1]
                        self.context_counts[i][context] += 1
        
        # 보간 가중치 설정
        if held_out_sentences:
            self.set_lambdas_with_held_out(held_out_sentences)
        else:
            # 간단한 기본 가중치 설정
            self.lambdas = {i: 1.0 / self.n for i in range(1, self.n + 1)}
    
    def set_lambdas_with_held_out(self, held_out_sentences):
        """
        별도 검증 데이터를 사용하여 보간 가중치 학습
        
        Args:
            held_out_sentences (list): 검증 문장 목록
        """
        # 초기화
        lambda_counts = {i: 0 for i in range(1, self.n + 1)}
        total_count = 0
        
        # 검증 데이터에서 각 n-gram 모델의 예측 성공 횟수 계산
        for sentence in held_out_sentences:
            tokens = ['<s>'] * (self.n - 1) + sentence.lower().split() + ['</s>']
            
            for j in range(self.n - 1, len(tokens)):
                word = tokens[j]
                best_model = None
                highest_prob = -1
                
                # 각 차수의 n-gram 모델 확인
                for i in range(1, self.n + 1):
                    if j >= i - 1:  # 충분한 컨텍스트가 있는지 확인
                        context_start = j - (i - 1)
                        context = tuple(tokens[context_start:j])
                        
                        if i == 1 or context in self.context_counts[i]:
                            # n-gram 확률 계산
                            prob = self.conditional_prob(word, context, i)
                            
                            if prob > highest_prob:
                                highest_prob = prob
                                best_model = i
                
                if best_model:
                    lambda_counts[best_model] += 1
                    total_count += 1
        
        # 가중치 정규화
        if total_count > 0:
            self.lambdas = {i: count / total_count for i, count in lambda_counts.items()}
        else:
            # 균등 가중치 설정
            self.lambdas = {i: 1.0 / self.n for i in range(1, self.n + 1)}
    
    def conditional_prob(self, word, context, n):
        """
        단일 n-gram 모델에 대한 조건부 확률 계산
        
        Args:
            word (str): 대상 단어
            context (tuple): 컨텍스트 (n-1개 단어)
            n (int): n-gram 차수
            
        Returns:
            float: P(word|context) 확률
        """
        if n == 1:  # unigram
            unigram = (word,)
            total_count = sum(self.ngram_counts[1].values())
            return self.ngram_counts[1][unigram] / total_count if total_count > 0 else 0
        
        ngram = context + (word,)
        
        # 컨텍스트가 훈련 데이터에 있는 경우
        if context in self.context_counts[n]:
            return self.ngram_counts[n][ngram] / self.context_counts[n][context]
        
        return 0
    
    def interpolated_prob(self, word, context):
        """
        보간법을 사용한 확률 계산
        
        Args:
            word (str): 대상 단어
            context (tuple): 최대 컨텍스트 (n-1개 단어)
            
        Returns:
            float: 보간된 P(word|context) 확률
        """
        prob = 0.0
        
        # 모든 n-gram 차수에 대한 가중 합계
        for i in range(1, self.n + 1):
            if i == 1:
                # unigram은 컨텍스트가 없음
                curr_context = ()
            else:
                # 컨텍스트 길이 제한 (n-1 단어까지만)
                curr_context = context[-(i-1):] if len(context) >= i - 1 else context
            
            # 현재 차수의 조건부 확률
            curr_prob = self.conditional_prob(word, curr_context, i)
            
            # 가중치 적용
            prob += self.lambdas[i] * curr_prob
        
        return prob
    
    def backoff_prob(self, word, context):
        """
        백오프(back-off) 방식을 사용한 확률 계산
        
        Args:
            word (str): 대상 단어
            context (tuple): 최대 컨텍스트 (n-1개 단어)
            
        Returns:
            float: 백오프된 P(word|context) 확률
        """
        # 최고 차수부터 시작하여 백오프
        for i in range(self.n, 0, -1):
            if i == 1:
                # unigram은 컨텍스트가 없음
                curr_context = ()
            else:
                # 컨텍스트 길이 제한 (n-1 단어까지만)
                curr_context = context[-(i-1):] if len(context) >= i - 1 else context
            
            # 현재 차수에서 단어가 관찰된 경우
            ngram = curr_context + (word,)
            if i == 1 or (curr_context in self.context_counts[i] and self.ngram_counts[i][ngram] > 0):
                return self.conditional_prob(word, curr_context, i)
        
        # 모든 n-gram에서 단어가 관찰되지 않은 경우
        return 1.0 / len(self.vocab) if len(self.vocab) > 0 else 0
    
    def score_sentence(self, sentence, method='interpolation'):
        """
        문장에 대한 로그 확률 계산
        
        Args:
            sentence (str): 평가할 문장
            method (str): 'interpolation' 또는 'backoff'
            
        Returns:
            float: 문장의 로그 확률
        """
        tokens = ['<s>'] * (self.n - 1) + sentence.lower().split() + ['</s>']
        log_prob = 0.0
        
        for i in range(self.n - 1, len(tokens)):
            word = tokens[i]
            context = tuple(tokens[max(0, i - (self.n - 1)):i])
            
            if method == 'interpolation':
                prob = self.interpolated_prob(word, context)
            else:  # backoff
                prob = self.backoff_prob(word, context)
            
            # 확률이 0인 경우 처리 (로그 계산을 위해)
            if prob <= 0:
                prob = 1e-10
            
            log_prob += np.log(prob)
        
        return log_prob
    
    def perplexity(self, sentences, method='interpolation'):
        """
        문장 집합에 대한 퍼플렉시티 계산
        
        Args:
            sentences (list): 평가할 문장 목록
            method (str): 'interpolation' 또는 'backoff'
            
        Returns:
            float: 퍼플렉시티
        """
        total_log_prob = 0.0
        total_tokens = 0
        
        for sentence in sentences:
            tokens = sentence.lower().split()
            total_tokens += len(tokens) + 1  # +1 for </s>
            
            log_prob = self.score_sentence(sentence, method)
            total_log_prob += log_prob
        
        # 퍼플렉시티 계산: 2^(-평균 로그 확률)
        return np.exp(-total_log_prob / total_tokens)
    
    def generate_sentence(self, max_length=20, method='interpolation'):
        """
        언어 모델을 사용하여 문장 생성
        
        Args:
            max_length (int): 최대 문장 길이
            method (str): 'interpolation' 또는 'backoff'
            
        Returns:
            str: 생성된 문장
        """
        # 문장 시작 토큰
        sentence = ['<s>'] * (self.n - 1)
        
        # 종료 토큰이 나오거나 최대 길이에 도달할 때까지 단어 생성
        while sentence[-1] != '</s>' and len(sentence) < max_length + (self.n - 1):
            context = tuple(sentence[-(self.n-1):])
            
            # 가능한 모든 단어에 대한 확률 계산
            word_probs = {}
            for word in self.vocab:
                if method == 'interpolation':
                    prob = self.interpolated_prob(word, context)
                else:  # backoff
                    prob = self.backoff_prob(word, context)
                
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


def compare_interpolation_vs_backoff():
    """보간법과 백오프 방식의 성능을 비교합니다."""
    
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
    
    # 검증 데이터
    held_out_sentences = [
        "나는 내일 학교에 갈 것이다",
        "그는 오늘 영화를 본다",
        "우리는 어제 공원에 갔다"
    ]
    
    # 테스트 데이터
    test_sentences = [
        "나는 학교에 갔다",
        "그는 영화를 보았다",
        "우리는 공원에 갈 것이다",
        "그녀는 책을 읽는다",
        "학생들은 시험을 준비한다"
    ]
    
    # 미관찰 데이터 (OOV)
    oov_sentences = [
        "교수님은 강의를 하고 계신다",
        "개발자들이 새로운 기술을 학습한다",
        "회사원들은 회의에 참석한다"
    ]
    
    # 모델 초기화 및 학습
    model = BackoffInterpolationModel(n=3)
    model.fit(train_sentences, held_out_sentences)
    
    print("N-gram 보간법과 백오프 비교 예제\n")
    print(f"학습 데이터 크기: {len(train_sentences)} 문장")
    print(f"검증 데이터 크기: {len(held_out_sentences)} 문장")
    print(f"테스트 데이터 크기: {len(test_sentences)} 문장")
    print(f"OOV 데이터 크기: {len(oov_sentences)} 문장")
    print(f"어휘 크기: {len(model.vocab)} 단어\n")
    
    # 학습된 보간 가중치 출력
    print("학습된 보간 가중치:")
    for i, weight in model.lambdas.items():
        if i == 1:
            print(f"  유니그램: {weight:.3f}")
        elif i == 2:
            print(f"  바이그램: {weight:.3f}")
        elif i == 3:
            print(f"  트라이그램: {weight:.3f}")
    print()
    
    # 퍼플렉시티 비교
    print("테스트 데이터에 대한 퍼플렉시티:")
    interp_ppl = model.perplexity(test_sentences, 'interpolation')
    backoff_ppl = model.perplexity(test_sentences, 'backoff')
    print(f"  보간법: {interp_ppl:.2f}")
    print(f"  백오프: {backoff_ppl:.2f}\n")
    
    print("OOV 데이터에 대한 퍼플렉시티:")
    interp_ppl_oov = model.perplexity(oov_sentences, 'interpolation')
    backoff_ppl_oov = model.perplexity(oov_sentences, 'backoff')
    print(f"  보간법: {interp_ppl_oov:.2f}")
    print(f"  백오프: {backoff_ppl_oov:.2f}\n")
    
    # 예제 문장에 대한 확률 비교
    example_sentence = "나는 학교에 갔다"
    print(f"예제 문장: '{example_sentence}'")
    
    interp_log_prob = model.score_sentence(example_sentence, 'interpolation')
    backoff_log_prob = model.score_sentence(example_sentence, 'backoff')
    
    print(f"  보간법 로그 확률: {interp_log_prob:.4f}")
    print(f"  백오프 로그 확률: {backoff_log_prob:.4f}\n")
    
    # 각 단어별 확률 분석
    tokens = ['<s>'] * (model.n - 1) + example_sentence.lower().split() + ['</s>']
    
    print("단어별 확률 비교:")
    print(f"{'단어':<10} {'보간법 확률':<15} {'백오프 확률':<15}")
    print("-" * 40)
    
    for i in range(model.n - 1, len(tokens)):
        word = tokens[i]
        context = tuple(tokens[max(0, i - (model.n - 1)):i])
        
        interp_prob = model.interpolated_prob(word, context)
        backoff_prob = model.backoff_prob(word, context)
        
        print(f"{word:<10} {interp_prob:<15.6f} {backoff_prob:<15.6f}")
    print()
    
    # 문장 생성 비교
    print("보간법으로 생성된 문장:")
    for _ in range(3):
        print(f"  {model.generate_sentence(method='interpolation')}")
    
    print("\n백오프로 생성된 문장:")
    for _ in range(3):
        print(f"  {model.generate_sentence(method='backoff')}")


if __name__ == "__main__":
    # 수치적 보간법 예제 실행
    try:
        demonstrate_interpolation()
    except Exception as e:
        print(f"수치적 보간법 예제 실행 중 오류: {e}")
        print("matplotlib 또는 scipy가 설치되지 않았을 수 있습니다.")
    
    print("\n" + "="*50 + "\n")
    
    # 언어 모델 보간법 예제 실행
    compare_interpolation_vs_backoff() 