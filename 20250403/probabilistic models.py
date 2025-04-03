import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 결합 확률 분포 정의
# 테이블 형태로 결합 확률을 표현
# 순서: toothache, catch, cavity
joint_prob = {
    (True, True, True): 0.108,
    (True, False, True): 0.012,
    (False, True, True): 0.072,
    (False, False, True): 0.008,
    (True, True, False): 0.016,
    (True, False, False): 0.064,
    (False, True, False): 0.144,
    (False, False, False): 0.576
}

def print_joint_distribution():
    """결합 확률 분포를 표 형태로 출력"""
    print("\n결합 확률 분포 테이블:")
    print("="*50)
    print(f"{'toothache':<10} | {'catch':<10} | {'cavity':<10} | {'확률':<10}")
    print("-"*50)
    
    for (t, c, cav), prob in joint_prob.items():
        print(f"{str(t):<10} | {str(c):<10} | {str(cav):<10} | {prob:<10.3f}")

def probability_of_proposition(condition_func):
    """명제의 확률 계산 - 해당 명제가 참인 모든 원자적 사건의 확률 합계"""
    return sum(prob for event, prob in joint_prob.items() if condition_func(event))

def P_cavity():
    """P(cavity) 계산"""
    return probability_of_proposition(lambda event: event[2])

def P_toothache():
    """P(toothache) 계산"""
    return probability_of_proposition(lambda event: event[0])

def P_catch():
    """P(catch) 계산"""
    return probability_of_proposition(lambda event: event[1])

def P_cavity_or_toothache():
    """P(cavity ∨ toothache) 계산"""
    return probability_of_proposition(lambda event: event[2] or event[0])

def P_cavity_given_toothache():
    """P(cavity | toothache) 계산 - 조건부 확률"""
    p_toothache = P_toothache()
    p_cavity_and_toothache = probability_of_proposition(lambda event: event[2] and event[0])
    return p_cavity_and_toothache / p_toothache if p_toothache > 0 else 0

def P_not_cavity_given_toothache():
    """P(¬cavity | toothache) 계산 - 조건부 확률"""
    p_toothache = P_toothache()
    p_not_cavity_and_toothache = probability_of_proposition(lambda event: not event[2] and event[0])
    return p_not_cavity_and_toothache / p_toothache if p_toothache > 0 else 0

def P_not_cavity_and_toothache():
    """P(¬cavity ∧ toothache) 계산"""
    return probability_of_proposition(lambda event: not event[2] and event[0])

def P_cavity_and_toothache():
    """P(cavity ∧ toothache) 계산"""
    return probability_of_proposition(lambda event: event[2] and event[0])

def calculate_odds_ratio():
    """P(cavity, toothache) : P(¬cavity, toothache) 오즈 계산"""
    p_cavity_and_toothache = P_cavity_and_toothache()
    p_not_cavity_and_toothache = P_not_cavity_and_toothache()
    return p_cavity_and_toothache / p_not_cavity_and_toothache if p_not_cavity_and_toothache > 0 else 0

def P_cavity_given_toothache_and_catch():
    """P(cavity | toothache ∧ catch) 계산"""
    p_toothache_and_catch = probability_of_proposition(lambda event: event[0] and event[1])
    p_cavity_and_toothache_and_catch = probability_of_proposition(
        lambda event: event[2] and event[0] and event[1]
    )
    return p_cavity_and_toothache_and_catch / p_toothache_and_catch if p_toothache_and_catch > 0 else 0

def normalization_constant():
    return (P_cavity_and_toothache() + P_cavity_and_toothache_not_catch())

def P_cavity_and_toothache_not_catch():
    return probability_of_proposition(lambda event: event[2] and not event[1] and event[0])

def main():
    print("확률적 모델 - 전체 결합 분포를 사용한 추론")
    print_joint_distribution()
    
    print("\n기본 확률 계산:")
    print(f"P(cavity) = {P_cavity():.3f}")
    print(f"P(toothache) = {P_toothache():.3f}")
    print(f"P(catch) = {P_catch():.3f}")
    
    print("\n복합 명제 확률:")
    print(f"P(cavity ∨ toothache) = {P_cavity_or_toothache():.3f}")
    
    print("\n정규화 상수:")
    alpha = normalization_constant()
    print(f"정규화 상수 α = {alpha:.3f}")
    
    print("\n조건부 확률:")
    print(f"P(cavity | toothache) = {P_cavity_given_toothache():.3f}")
    print(f"P(¬cavity | toothache) = {P_not_cavity_given_toothache():.3f}")
    print(f"P(cavity | toothache ∧ catch) = {P_cavity_given_toothache_and_catch():.3f}")
    
    # 슬라이드에 있는 조건부 확률 계산
    p_not_cavity_and_toothache = P_not_cavity_and_toothache()
    p_toothache = P_toothache()
    
    print("\n슬라이드 예제 계산:")
    print(f"P(¬cavity ∧ toothache) = {p_not_cavity_and_toothache:.3f}")
    print(f"P(toothache) = {p_toothache:.3f}")
    print(f"P(¬cavity|toothache) = {p_not_cavity_and_toothache:.3f} / {p_toothache:.3f} = {p_not_cavity_and_toothache/p_toothache:.3f}")
    print(f"P(¬cavity|toothache) = (0.016 + 0.064) / (0.108 + 0.012 + 0.016 + 0.064) = 0.400")
    
    # 확장: 오즈 계산
    print("\n오즈(Odds) 계산:")
    p_cavity_and_toothache = P_cavity_and_toothache()
    p_not_cavity_and_toothache = P_not_cavity_and_toothache()
    odds = p_cavity_and_toothache / p_not_cavity_and_toothache
    print(f"P(cavity, toothache) : P(¬cavity, toothache) = {p_cavity_and_toothache:.3f} : {p_not_cavity_and_toothache:.3f} = {odds:.3f}")

if __name__ == "__main__":
    main()
