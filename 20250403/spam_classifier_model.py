import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 스팸 분류기 예제 확률 테이블
# 특징 변수 X = [Viagra, lottery]
# 타겟 변수 Y = [spam, ham]
# 조건부 확률 P(Y|X)

# 조건부 확률 테이블 정의
spam_prob_table = {
    # (Viagra, lottery): (P(spam|Viagra,lottery), P(ham|Viagra,lottery))
    (0, 0): (0.31, 0.69),  # Viagra=0, lottery=0
    (0, 1): (0.65, 0.35),  # Viagra=0, lottery=1
    (1, 0): (0.80, 0.20),  # Viagra=1, lottery=0
    (1, 1): (0.40, 0.60)   # Viagra=1, lottery=1
}

# 단일 특징 확률 (prior probabilities)
feature_priors = {
    'viagra': 0.1,  # P(Viagra=1) = 0.1, P(Viagra=0) = 0.9
    'lottery': 0.3   # 예시 값, 실제로는 데이터에서 추정해야 함
}

def print_spam_prob_table():
    """조건부 확률 테이블을 출력합니다."""
    print("\n스팸 분류 조건부 확률 테이블:")
    print("="*70)
    print(f"{'Viagra':<10} | {'lottery':<10} | {'P(spam|Viagra,lottery)':<25} | {'P(ham|Viagra,lottery)':<25}")
    print("-"*70)
    
    for (viagra, lottery), (p_spam, p_ham) in spam_prob_table.items():
        print(f"{viagra:<10} | {lottery:<10} | {p_spam:<25.2f} | {p_ham:<25.2f}")

def predict_spam_probability(viagra, lottery):
    """
    주어진 특징 변수 값에 대한 스팸 확률을 계산합니다.
    
    Args:
        viagra (int): Viagra 언급 여부 (0 또는 1)
        lottery (int): 복권 언급 여부 (0 또는 1)
    
    Returns:
        tuple: (스팸 확률, 햄 확률)
    """
    feature_key = (viagra, lottery)
    if feature_key in spam_prob_table:
        return spam_prob_table[feature_key]
    else:
        print(f"경고: 입력 특징 ({viagra}, {lottery})에 대한 확률이 정의되지 않았습니다.")
        return (0, 0)

def predict_spam_with_missing_values(known_features):
    """
    일부 특징 값이 누락된 경우에도 스팸 확률을 계산합니다.
    
    Args:
        known_features (dict): 알려진 특징 값의 딕셔너리 (예: {'viagra': 0, 'lottery': None})
    
    Returns:
        tuple: (스팸 확률, 햄 확률)
    """
    # 누락된 값 확인
    viagra = known_features.get('viagra')
    lottery = known_features.get('lottery')
    
    # 모든 값이 알려진 경우 직접 테이블에서 확률 가져오기
    if viagra is not None and lottery is not None:
        return predict_spam_probability(viagra, lottery)
    
    # Viagra 값만 누락된 경우 (lottery만 알려진 경우)
    elif viagra is None and lottery is not None:
        # P(Y|lottery) = P(Y|Viagra=0,lottery)P(Viagra=0) + P(Y|Viagra=1,lottery)P(Viagra=1)
        p_viagra_0 = 1 - feature_priors['viagra']  # P(Viagra=0)
        p_viagra_1 = feature_priors['viagra']      # P(Viagra=1)
        
        p_spam_given_viagra_0_lottery = spam_prob_table[(0, lottery)][0]
        p_spam_given_viagra_1_lottery = spam_prob_table[(1, lottery)][0]
        
        p_spam = p_spam_given_viagra_0_lottery * p_viagra_0 + p_spam_given_viagra_1_lottery * p_viagra_1
        p_ham = 1 - p_spam
        
        return (p_spam, p_ham)
    
    # lottery 값만 누락된 경우 (Viagra만 알려진 경우)
    elif viagra is not None and lottery is None:
        # P(Y|Viagra) = P(Y|Viagra,lottery=0)P(lottery=0) + P(Y|Viagra,lottery=1)P(lottery=1)
        p_lottery_0 = 1 - feature_priors['lottery']  # P(lottery=0)
        p_lottery_1 = feature_priors['lottery']      # P(lottery=1)
        
        p_spam_given_viagra_lottery_0 = spam_prob_table[(viagra, 0)][0]
        p_spam_given_viagra_lottery_1 = spam_prob_table[(viagra, 1)][0]
        
        p_spam = p_spam_given_viagra_lottery_0 * p_lottery_0 + p_spam_given_viagra_lottery_1 * p_lottery_1
        p_ham = 1 - p_spam
        
        return (p_spam, p_ham)
    
    # 모든 값이 누락된 경우 - 사전 확률 사용
    else:
        # 전체 테이블에서 평균 스팸 확률 계산 (더 정교한 방법도 가능)
        p_spam = sum(p[0] for p in spam_prob_table.values()) / len(spam_prob_table)
        p_ham = 1 - p_spam
        return (p_spam, p_ham)

def classify_email(viagra, lottery, threshold=0.5):
    """
    이메일을 스팸 또는 햄으로 분류합니다.
    
    Args:
        viagra (int): Viagra 언급 여부 (0 또는 1)
        lottery (int): 복권 언급 여부 (0 또는 1)
        threshold (float): 스팸으로 분류하기 위한 확률 임계값
    
    Returns:
        str: 'spam' 또는 'ham'
    """
    p_spam, p_ham = predict_spam_probability(viagra, lottery)
    return "spam" if p_spam >= threshold else "ham"

def classify_email_with_missing_values(known_features, threshold=0.5):
    """
    일부 특징 값이 누락된 경우에도 이메일을 분류합니다.
    
    Args:
        known_features (dict): 알려진 특징 값의 딕셔너리
        threshold (float): 스팸으로 분류하기 위한 확률 임계값
    
    Returns:
        str: 'spam' 또는 'ham'
    """
    p_spam, p_ham = predict_spam_with_missing_values(known_features)
    return "spam" if p_spam >= threshold else "ham"

def demonstrate_example_emails():
    """예제 이메일을 통한 스팸 분류 데모"""
    test_cases = [
        {"viagra": 0, "lottery": 0, "description": "Viagra와 복권이 모두 언급되지 않은 이메일"},
        {"viagra": 0, "lottery": 1, "description": "Viagra는 언급되지 않고 복권이 언급된 이메일"},
        {"viagra": 1, "lottery": 0, "description": "Viagra가 언급되고 복권은 언급되지 않은 이메일"},
        {"viagra": 1, "lottery": 1, "description": "Viagra와 복권이 모두 언급된 이메일"}
    ]
    
    print("\n예제 이메일 분류 결과:")
    print("="*100)
    print(f"{'설명':<50} | {'특징 (Viagra, lottery)':<25} | {'P(spam|X)':<10} | {'P(ham|X)':<10} | {'분류 결과':<10}")
    print("-"*100)
    
    for case in test_cases:
        viagra = case["viagra"]
        lottery = case["lottery"]
        description = case["description"]
        
        p_spam, p_ham = predict_spam_probability(viagra, lottery)
        classification = classify_email(viagra, lottery)
        
        print(f"{description:<50} | ({viagra}, {lottery}){'':>16} | {p_spam:<10.2f} | {p_ham:<10.2f} | {classification:<10}")

def demonstrate_missing_values():
    """누락된 값이 있는 경우의 예제"""
    # 슬라이드 예제: lottery는 알고 있지만 viagra 상태는 모르는 경우
    known_features = {'viagra': None, 'lottery': 1}
    
    p_spam, p_ham = predict_spam_with_missing_values(known_features)
    classification = classify_email_with_missing_values(known_features)
    
    print("\n누락된 값이 있는 경우의 예제 (P(Viagra=1) = 0.1):")
    print("="*80)
    print("Lottery=1이 있음을 알지만 Viagra 상태는 알 수 없는 경우")
    print(f"P(spam|lottery=1) = P(spam|Viagra=0,lottery=1)P(Viagra=0) + P(spam|Viagra=1,lottery=1)P(Viagra=1)")
    print(f"P(spam|lottery=1) = 0.65 * 0.9 + 0.40 * 0.1 = 0.625")
    print(f"계산된 스팸 확률: {p_spam:.3f}, 햄 확률: {p_ham:.3f}")
    print(f"분류 결과: {classification}")
    print("\n이는 Viagra=0, lottery=1인 경우(0.65)보다 약간 낮은 확률입니다.")
    print("Viagra=1일 확률이 낮기 때문에, 결과는 Viagra=0일 경우에 더 가깝습니다.")

def explain_probability_model():
    """확률적 모델의 개념을 설명합니다."""
    print("\n확률적 모델 설명:")
    print("="*70)
    print("1. 기본 가정:")
    print("   - 특징 변수 X와 타겟 변수 Y의 값을 생성하는 기본 확률 프로세스가 존재합니다.")
    print("   - X: 인스턴스의 특징 값 (예: 이메일에 'Viagra'와 'lottery' 단어가 존재하는지 여부)")
    print("   - Y: 타겟 변수 (예: 이메일이 'spam'인지 'ham'인지 여부)")
    print("\n2. 목표:")
    print("   - 사후 확률 P(Y|X)를 계산하는 것이 목표입니다.")
    print("   - 즉, 특정 특징을 가진 이메일이 스팸일 확률을 알고 싶습니다.")
    print("\n3. 테이블 크기 문제:")
    print("   - 이 예제에서는 특징이 2개뿐이라 테이블 크기가 작습니다.")
    print("   - 하지만 실제로는 특징이 n개일 때 테이블 크기는 O(2^n)으로 급격히 증가합니다.")
    print("   - 이는 모든 가능한 특징 조합에 대한 확률을 저장해야 하기 때문입니다.")
    print("\n4. 누락된 값 처리:")
    print("   - 일부 특징 값이 누락된 경우에도 확률적 추론이 가능합니다.")
    print("   - 누락된 변수를 주변화(marginalization)하여 처리합니다.")
    print("   - 예: P(Y|lottery) = P(Y|Viagra=0,lottery)P(Viagra=0) + P(Y|Viagra=1,lottery)P(Viagra=1)")

def main():
    try:
        print("확률적 모델 - 스팸 분류기 예제")
        
        # 확률적 모델 개념 설명
        explain_probability_model()
        
        # 확률 테이블 출력
        print_spam_prob_table()
        
        # 예제 이메일 분류 데모
        demonstrate_example_emails()
        
        # 누락된 값이 있는 경우 예제
        demonstrate_missing_values()
        
        print("\n결론:")
        print("이 간단한 예제에서 볼 수 있듯이, 확률적 모델은 주어진 특징 X에 대해")
        print("타겟 변수 Y의 확률 분포를 제공합니다. 이를 통해 분류 결정을 내릴 수 있습니다.")
        print("게다가, 일부 특징 값이 누락된 경우에도 확률적 추론을 통해 의사결정이 가능합니다.")
        print("그러나 특징의 수가 증가하면 테이블 크기가 기하급수적으로 증가하는 문제가 있습니다.")
        
        # stdout 버퍼 강제 flush
        import sys
        sys.stdout.flush()
    except Exception as e:
        print(f"오류 발생: {e}")

if __name__ == "__main__":
    main() 