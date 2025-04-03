#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from tabulate import tabulate

def main():
    print("확률적 모델 - 사후 오즈(Posterior Odds) 예제\n")
    
    # 사후 확률 정의
    posterior_probs = {
        # (Viagra, lottery, class): probability
        (0, 0, 'spam'): 0.31,
        (0, 0, 'ham'): 0.69,
        (0, 1, 'spam'): 0.65,
        (0, 1, 'ham'): 0.35,
        (1, 0, 'spam'): 0.80,
        (1, 0, 'ham'): 0.20,
        (1, 1, 'spam'): 0.40,
        (1, 1, 'ham'): 0.60,
    }
    
    print("이메일 스팸 분류를 위한 사후 확률 및 사후 오즈 설명:")
    print("=" * 70)
    print("이 예제에서는 이메일이 'spam'인지 'ham'인지 결정하기 위해 다음 특징을 사용합니다:")
    print("  - Viagra: 이메일에 'Viagra' 단어가 포함되어 있는지 여부 (0=없음, 1=있음)")
    print("  - lottery: 이메일에 'lottery' 단어가 포함되어 있는지 여부 (0=없음, 1=있음)")
    print("\n사후 확률 테이블 설명:")
    print("  - P(spam|Viagra=0,lottery=0) = 0.31: Viagra와 lottery가 모두 없는 경우, 스팸일 확률")
    print("  - P(ham|Viagra=0,lottery=0) = 0.69: Viagra와 lottery가 모두 없는 경우, 정상 메일일 확률")
    print("  - 오즈 0.45는 P(spam)/P(ham) = 0.31/0.69 = 0.45로 계산됩니다")
    print("  - 이는 Viagra와 lottery가 모두 없는 경우, 정상 메일일 가능성이 스팸일 가능성보다 더 높음을 나타냅니다")
    
    # 사후 오즈 계산 및 결과 표시
    print("\n사후 확률 및 사후 오즈 계산:")
    print("=" * 70)
    
    # 결과를 담을 테이블 생성
    table_data = []
    headers = ["Viagra", "lottery", "P(spam|X)", "P(ham|X)", "Posterior Odds", "예측"]
    
    for viagra in [0, 1]:
        for lottery in [0, 1]:
            p_spam = posterior_probs[(viagra, lottery, 'spam')]
            p_ham = posterior_probs[(viagra, lottery, 'ham')]
            
            # 사후 오즈 계산
            posterior_odds = p_spam / p_ham
            
            # MAP 의사결정 규칙에 따른 예측
            prediction = 'spam' if p_spam > p_ham else 'ham'
            
            # 계산 과정 표시
            print(f"특징 조합 (Viagra={viagra}, lottery={lottery}):")
            print(f"  P(spam|X) = {p_spam:.2f}")
            print(f"  P(ham|X) = {p_ham:.2f}")
            print(f"  Posterior Odds = P(spam|X)/P(ham|X) = {p_spam:.2f}/{p_ham:.2f} = {posterior_odds:.2f}")
            print(f"  예측: {'spam' if posterior_odds > 1 else 'ham'} (Posterior Odds {'>' if posterior_odds > 1 else '<'} 1)")
            print()
            
            # 테이블에 행 추가
            table_data.append([
                viagra, lottery, 
                f"{p_spam:.2f}", f"{p_ham:.2f}", 
                f"{posterior_odds:.2f}",
                prediction
            ])
    
    # 테이블 출력
    print("\n요약 테이블:")
    print("=" * 70)
    print(tabulate(table_data, headers, tablefmt="grid"))
    print()
    
    # MAP 의사결정 규칙 설명
    print("MAP 의사결정 규칙 (Maximum a Posteriori):")
    print("=" * 70)
    print("- MAP 의사결정 규칙은 사후 확률이 가장 높은 클래스를 선택합니다.")
    print("- 이진 분류의 경우, 사후 오즈 > 1 이면 'spam'을 예측하고, 그렇지 않으면 'ham'을 예측합니다.")
    print("- 위 표에서 Posterior Odds > 1인 경우를 확인하면, 다음 조합에서 'spam'으로 예측됨을 알 수 있습니다:")
    
    for row in table_data:
        viagra, lottery, p_spam, p_ham, odds, prediction = row
        if prediction == 'spam':
            print(f"  * Viagra={viagra}, lottery={lottery}: 오즈={odds}, 예측='{prediction}'")
    
    print("\n사후 오즈 해석:")
    print("=" * 70)
    print("- 사후 오즈 > 1: 'spam'일 가능성이 'ham'보다 높음")
    print("- 사후 오즈 < 1: 'ham'일 가능성이 'spam'보다 높음")
    print("- 사후 오즈 = 1: 'spam'과 'ham'의 가능성이 동일함")
    
    print("\n결론:")
    print("=" * 70)
    print("- 전체 사후 분포가 도메인에 대해 아는 모든 것이라면, 이러한 MAP 예측이 최선의 선택입니다.")
    print("- 베이즈 최적(Bayes-optimal) 예측은 주어진 확률 모델 하에서 오분류 확률을 최소화합니다.")
    print("- 각 특징 조합에 따라 사후 오즈가 1보다 크면 'spam', 그렇지 않으면 'ham'으로 예측합니다.")
    print("- 특히 Viagra=1, lottery=0인 경우 오즈가 4.0으로 가장 높아 스팸일 가능성이 가장 높습니다.")
    print("- 반면 Viagra=0, lottery=0인 경우 오즈가 0.45로 가장 낮아 정상 메일일 가능성이 가장 높습니다.")

if __name__ == "__main__":
    main() 