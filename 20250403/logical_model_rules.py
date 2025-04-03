"""
논리 모델(Logical Models)의 규칙 집합 문제점 구현

이 프로그램은 특징 리스트(Feature List) 모델에서 규칙 집합이
불완전(incomplete)하거나 비일관적(inconsistent)일 때 발생하는 문제를 시연합니다.

슬라이드 내용:
- 규칙 집합은 불일관적(inconsistent)이거나 불완전(incomplete)할 수 있음
- 불완전성은 기본 규칙(default rule)을 추가하여 해결할 수 있음
- 비일관성(모호성)은 규칙의 순서를 정하여 해결할 수 있음
"""

def classify_email(features):
    """
    이메일 분류 함수 - 규칙 집합에 따라 이메일을 분류
    
    Args:
        features: 이메일의 특징들을 담은 사전 (dictionary)
                 {'lottery': 0 또는 1, 'peter': 0 또는 1}
    
    Returns:
        분류 결과: 'spam', 'ham', 또는 None (분류 불가)
    """
    
    # 규칙 집합 정의
    # 규칙 1: lottery = 1 이면 spam
    # 규칙 2: peter = 1 이면 ham
    
    # 적용 가능한 규칙 저장
    applicable_rules = []
    
    # 규칙 1 검사
    if features.get('lottery') == 1:
        applicable_rules.append(('spam', 1))  # (결과, 규칙 우선순위)
    
    # 규칙 2 검사
    if features.get('peter') == 1:
        applicable_rules.append(('ham', 2))  # (결과, 규칙 우선순위)
    
    # 결과 분석
    if not applicable_rules:
        return None  # 불완전성(incompleteness): 적용 가능한 규칙 없음
    
    if len(applicable_rules) > 1:
        # 비일관성(inconsistency): 여러 규칙이 적용 가능하고 다른 결과를 제안
        results = set(result for result, _ in applicable_rules)
        if len(results) > 1:
            # 해결 방법 1: 우선순위에 따라 결정
            applicable_rules.sort(key=lambda x: x[1])  # 우선순위로 정렬
            return applicable_rules[0][0]  # 가장 높은 우선순위의 규칙 적용
            
            # 해결 방법 2: 직접 모순을 확인하고 경고
            # return f"모순된 예측: {results}"
    
    # 단일 결과인 경우
    return applicable_rules[0][0]

# 기본 규칙(default rule)을 추가한 개선된 버전
def classify_email_with_default(features):
    """
    기본 규칙이 추가된 이메일 분류 함수
    
    Args:
        features: 이메일 특징
    
    Returns:
        분류 결과: 항상 'spam' 또는 'ham' 반환
    """
    result = classify_email(features)
    
    # 불완전성 해결: 기본 규칙 적용
    if result is None:
        return "ham"  # 기본 규칙: 분류할 수 없으면 ham으로 간주
    
    return result

# 특징 공간 시각화 (텍스트 기반)
def visualize_feature_space():
    """특징 공간과 분류 결과를 텍스트로 시각화"""
    print("\n=== 특징 공간 시각화 ===")
    print("  Peter=0 Peter=1")
    print("  ------- -------")
    
    # Lottery=1 행
    l1p0 = classify_email({'lottery': 1, 'peter': 0}) or "None"
    l1p1 = classify_email({'lottery': 1, 'peter': 1}) or "모순"
    print(f"Lottery=1| {l1p0:^7} | {l1p1:^7} |")
    
    # Lottery=0 행
    l0p0 = classify_email({'lottery': 0, 'peter': 0}) or "None"
    l0p1 = classify_email({'lottery': 0, 'peter': 1}) or "None"
    print(f"Lottery=0| {l0p0:^7} | {l0p1:^7} |")
    print("  ------- -------")

def main():
    """메인 함수: 테스트 케이스와 결과 출력"""
    # 테스트 예제들
    test_cases = [
        {'lottery': 1, 'peter': 0},  # lottery=1 → spam
        {'lottery': 0, 'peter': 1},  # peter=1 → ham
        {'lottery': 1, 'peter': 1},  # lottery=1, peter=1 → 모순(우선순위 적용시 spam)
        {'lottery': 0, 'peter': 0},  # 해당하는 규칙 없음 → None (기본 규칙 적용시 ham)
    ]

    print("=== 기본 분류기 (문제점 시연) ===")
    for i, features in enumerate(test_cases):
        print(f"케이스 {i+1}: {features} → {classify_email(features)}")

    print("\n=== 개선된 분류기 (기본 규칙 추가) ===")
    for i, features in enumerate(test_cases):
        print(f"케이스 {i+1}: {features} → {classify_email_with_default(features)}")

    # 특징 공간 시각화
    visualize_feature_space()

    # 규칙 집합 문제 해결하기
    print("\n=== 규칙 집합 문제 해결 방법 ===")
    print("1. 불완전성(Incompleteness) 해결: 기본 규칙 추가")
    print("   - 모든 경우에 적용될 수 있는 기본 규칙: 'ham'으로 분류")
    print("   - 결과: 모든 이메일이 분류됨")

    print("\n2. 비일관성(Inconsistency) 해결: 규칙 우선순위 부여")
    print("   - 규칙 1 (lottery=1 → spam)의 우선순위: 높음")
    print("   - 규칙 2 (peter=1 → ham)의 우선순위: 낮음") 
    print("   - 결과: lottery=1 & peter=1 인 경우 spam으로 분류")

if __name__ == "__main__":
    main() 