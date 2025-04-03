#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from tabulate import tabulate

def print_case(combo, marginal_likelihoods, posterior_odds):
    """특정 특징 조합에 대한 계산과 예측을 출력합니다."""
    viagra_val = combo['viagra']
    lottery_val = combo['lottery']
    
    # 우도 비율 계산
    p_viagra_spam = marginal_likelihoods['viagra']['spam'][viagra_val]
    p_viagra_ham = marginal_likelihoods['viagra']['ham'][viagra_val]
    p_lottery_spam = marginal_likelihoods['lottery']['spam'][lottery_val]
    p_lottery_ham = marginal_likelihoods['lottery']['ham'][lottery_val]
    
    likelihood_ratio = (p_viagra_spam * p_lottery_spam) / (p_viagra_ham * p_lottery_ham)
    
    # 우도 비율 계산 과정 표시
    print(f"특징 조합 (Viagra={viagra_val}, lottery={lottery_val}):")
    print(f"  Likelihood Ratio = [P(Viagra={viagra_val}|spam) × P(lottery={lottery_val}|spam)] / [P(Viagra={viagra_val}|ham) × P(lottery={lottery_val}|ham)]")
    print(f"                   = [{p_viagra_spam} × {p_lottery_spam}] / [{p_viagra_ham} × {p_lottery_ham}]")
    print(f"                   = {likelihood_ratio:.2f}")
    
    # 실제 사후 오즈와 비교
    actual_odds = posterior_odds[(int(viagra_val), int(lottery_val))]
    ml_prediction = "spam" if likelihood_ratio > 1 else "ham"
    map_prediction = "spam" if actual_odds > 1 else "ham"
    
    print(f"  ML 예측: {ml_prediction} (Likelihood Ratio {'>' if likelihood_ratio > 1 else '<'} 1)")
    print(f"  MAP 예측: {map_prediction} (Posterior Odds = {actual_odds})")
    print(f"  두 예측의 일치 여부: {'일치' if ml_prediction == map_prediction else '불일치'}")
    print()

def main():
    print("확률적 모델 - 나이브 베이즈 주변 우도 예제\n")
    
    # 주변 우도(marginal likelihoods) 정의
    marginal_likelihoods = {
        # P(Viagra=1|Y), P(Viagra=0|Y)
        'viagra': {
            'spam': {'1': 0.40, '0': 0.60},
            'ham': {'1': 0.12, '0': 0.88}
        },
        # P(lottery=1|Y), P(lottery=0|Y)
        'lottery': {
            'spam': {'1': 0.21, '0': 0.79},
            'ham': {'1': 0.13, '0': 0.87}
        }
    }
    
    # 사전 확률 (prior probabilities)
    prior_probs = {
        'spam': 0.5,  # P(Y=spam) = 0.5
        'ham': 0.5    # P(Y=ham) = 0.5
    }
    
    # 요약 테이블 생성
    case_summary = []
    headers = ["특징 조합", "우도 비율", "ML 예측", "MAP 예측", "일치 여부"]
    
    print("나이브 베이즈 모델과 주변 우도 설명:")
    print("=" * 70)
    print("나이브 베이즈 모델은 클래스가 주어졌을 때 개별 단어의 우도가 독립적이라고 가정합니다.")
    print("즉, P(Viagra, lottery|Y) = P(Viagra|Y) × P(lottery|Y)로 계산합니다.")
    print("이는 확률 테이블의 크기를 크게 줄이면서도 효과적인 분류를 가능하게 합니다.")
    
    print("\n주변 우도 테이블:")
    print("=" * 70)
    
    # Viagra 주변 우도 테이블
    viagra_data = [
        ['spam', marginal_likelihoods['viagra']['spam']['1'], marginal_likelihoods['viagra']['spam']['0']],
        ['ham', marginal_likelihoods['viagra']['ham']['1'], marginal_likelihoods['viagra']['ham']['0']]
    ]
    print("Viagra 주변 우도 테이블:")
    print(tabulate(viagra_data, headers=['Y', 'P(Viagra=1|Y)', 'P(Viagra=0|Y)'], tablefmt="grid"))
    print()
    
    # Lottery 주변 우도 테이블
    lottery_data = [
        ['spam', marginal_likelihoods['lottery']['spam']['1'], marginal_likelihoods['lottery']['spam']['0']],
        ['ham', marginal_likelihoods['lottery']['ham']['1'], marginal_likelihoods['lottery']['ham']['0']]
    ]
    print("Lottery 주변 우도 테이블:")
    print(tabulate(lottery_data, headers=['Y', 'P(lottery=1|Y)', 'P(lottery=0|Y)'], tablefmt="grid"))
    print()
    
    print("\n우도 비율(Likelihood Ratio) 계산:")
    print("=" * 70)
    print("나이브 베이즈는 우도 비율을 통해 최대 우도(ML) 의사 결정을 수행할 수 있습니다.")
    print("우도 비율 > 1 이면 'spam', 그렇지 않으면 'ham'으로 예측합니다.")
    print()
    
    # 각 조합에 대한 우도 비율 계산
    feature_combinations = [
        {'viagra': '0', 'lottery': '0'},
        {'viagra': '0', 'lottery': '1'},
        {'viagra': '1', 'lottery': '0'},
        {'viagra': '1', 'lottery': '1'}
    ]
    
    # 사후 확률 계산을 위한 정보
    posterior_odds = {
        (0, 0): 0.45,
        (0, 1): 1.9,
        (1, 0): 4.0,
        (1, 1): 0.67
    }
    
    # 각 조합에 대한 계산 및 출력
    for combo in feature_combinations:
        viagra_val = combo['viagra']
        lottery_val = combo['lottery']
        
        # 우도 비율 계산
        p_viagra_spam = marginal_likelihoods['viagra']['spam'][viagra_val]
        p_viagra_ham = marginal_likelihoods['viagra']['ham'][viagra_val]
        p_lottery_spam = marginal_likelihoods['lottery']['spam'][lottery_val]
        p_lottery_ham = marginal_likelihoods['lottery']['ham'][lottery_val]
        
        likelihood_ratio = (p_viagra_spam * p_lottery_spam) / (p_viagra_ham * p_lottery_ham)
        
        # 우도 비율 계산 과정 표시
        print(f"특징 조합 (Viagra={viagra_val}, lottery={lottery_val}):")
        print(f"  Likelihood Ratio = [P(Viagra={viagra_val}|spam) × P(lottery={lottery_val}|spam)] / [P(Viagra={viagra_val}|ham) × P(lottery={lottery_val}|ham)]")
        print(f"                   = [{p_viagra_spam} × {p_lottery_spam}] / [{p_viagra_ham} × {p_lottery_ham}]")
        print(f"                   = {likelihood_ratio:.2f}")
        
        # 실제 사후 오즈와 비교
        actual_odds = posterior_odds[(int(viagra_val), int(lottery_val))]
        ml_prediction = "spam" if likelihood_ratio > 1 else "ham"
        map_prediction = "spam" if actual_odds > 1 else "ham"
        is_match = ml_prediction == map_prediction
        
        # 결과 추가
        case_summary.append([
            f"V={viagra_val},L={lottery_val}", 
            f"{likelihood_ratio:.2f}", 
            ml_prediction, 
            f"{map_prediction} ({actual_odds})",
            "일치" if is_match else "불일치"
        ])
        
        print(f"  ML 예측: {ml_prediction} (Likelihood Ratio {'>' if likelihood_ratio > 1 else '<'} 1)")
        print(f"  MAP 예측: {map_prediction} (Posterior Odds = {actual_odds})")
        print(f"  두 예측의 일치 여부: {'일치' if is_match else '불일치'}")
        print()
    
    print("\n요약 테이블:")
    print("=" * 70)
    print(tabulate(case_summary, headers=headers, tablefmt="grid"))
    
    # 결론 내용
    print("\n결론:")
    print("=" * 70)
    print("1. ML(Maximum Likelihood) 의사결정 규칙은 앞의 세 경우에서는 MAP의 베이즈 최적 예측과 일치합니다.")
    print("2. 하지만 네 번째 경우(Viagra=1, lottery=1)에서는 일치하지 않습니다.")
    print("3. 이 불일치는 사전 분포(prior distribution)가 균일하지 않을 때 발생할 수 있습니다.")
    print("4. 이러한 예제는 ML 방법과 MAP 방법이 항상 같은 결과를 도출하지 않음을 보여줍니다.")
    print("5. 나이브 베이즈는 클래스 조건부 독립성 가정에도 불구하고 실제 응용에서 효과적인 성능을 보입니다.")

if __name__ == "__main__":
    main() 