"""
공장 생산 라인 데이터의 나이브 베이즈 계산

문제 1-2번. 공장 생산 라인 데이터에 대해 나이브 베이즈 분류기를 사용하여
테스트 예제 [Supervisor=Pat, Operator=Jim, Machine=A, Overtime=No]의 Output을 
사후 확률 비율 계산을 통해 예측합니다.
"""

import pandas as pd
import numpy as np

# 공장 생산 라인 데이터 생성
def create_factory_data():
    """공장 생산 라인 데이터 생성"""
    # 슬라이드에 나온 데이터 생성
    data = {
        'Supervisor': ['Pat', 'Pat', 'Tom', 'Pat', 'Sally', 'Tom', 'Tom', 'Pat'],
        'Operator': ['Joe', 'Sam', 'Jim', 'Jim', 'Joe', 'Sam', 'Joe', 'Jim'],
        'Machine': ['A', 'B', 'B', 'B', 'C', 'C', 'C', 'A'],
        'Overtime': ['No', 'Yes', 'Yes', 'No', 'No', 'No', 'No', 'Yes'],
        'Output': ['High', 'Low', 'Low', 'High', 'High', 'Low', 'Low', 'Low']
    }
    
    df = pd.DataFrame(data)
    return df

# 나이브 베이즈 수작업 계산
def naive_bayes_prediction(df, test_example):
    """나이브 베이즈 수작업 계산"""
    # 데이터에서 확률 계산
    total_samples = len(df)
    
    # 클래스 사전 확률 P(Output)
    output_counts = df['Output'].value_counts()
    p_high = output_counts.get('High', 0) / total_samples
    p_low = output_counts.get('Low', 0) / total_samples
    
    print("\n1. 클래스 사전 확률:")
    print(f"P(Output=High) = {p_high:.4f}")
    print(f"P(Output=Low) = {p_low:.4f}")
    
    # 조건부 확률 계산
    # P(특성값 | Output) 계산
    print("\n2. 조건부 확률 계산:")
    
    # Supervisor=Pat 조건부 확률
    pat_high = df[(df['Supervisor'] == 'Pat') & (df['Output'] == 'High')].shape[0]
    pat_low = df[(df['Supervisor'] == 'Pat') & (df['Output'] == 'Low')].shape[0]
    p_pat_given_high = pat_high / output_counts.get('High', 1)
    p_pat_given_low = pat_low / output_counts.get('Low', 1)
    
    print(f"P(Supervisor=Pat | Output=High) = {pat_high}/{output_counts.get('High')} = {p_pat_given_high:.4f}")
    print(f"P(Supervisor=Pat | Output=Low) = {pat_low}/{output_counts.get('Low')} = {p_pat_given_low:.4f}")
    
    # Operator=Jim 조건부 확률
    jim_high = df[(df['Operator'] == 'Jim') & (df['Output'] == 'High')].shape[0]
    jim_low = df[(df['Operator'] == 'Jim') & (df['Output'] == 'Low')].shape[0]
    p_jim_given_high = jim_high / output_counts.get('High', 1)
    p_jim_given_low = jim_low / output_counts.get('Low', 1)
    
    print(f"P(Operator=Jim | Output=High) = {jim_high}/{output_counts.get('High')} = {p_jim_given_high:.4f}")
    print(f"P(Operator=Jim | Output=Low) = {jim_low}/{output_counts.get('Low')} = {p_jim_given_low:.4f}")
    
    # Machine=A 조건부 확률
    a_high = df[(df['Machine'] == 'A') & (df['Output'] == 'High')].shape[0]
    a_low = df[(df['Machine'] == 'A') & (df['Output'] == 'Low')].shape[0]
    p_a_given_high = a_high / output_counts.get('High', 1)
    p_a_given_low = a_low / output_counts.get('Low', 1)
    
    print(f"P(Machine=A | Output=High) = {a_high}/{output_counts.get('High')} = {p_a_given_high:.4f}")
    print(f"P(Machine=A | Output=Low) = {a_low}/{output_counts.get('Low')} = {p_a_given_low:.4f}")
    
    # Overtime=No 조건부 확률
    no_high = df[(df['Overtime'] == 'No') & (df['Output'] == 'High')].shape[0]
    no_low = df[(df['Overtime'] == 'No') & (df['Output'] == 'Low')].shape[0]
    p_no_given_high = no_high / output_counts.get('High', 1)
    p_no_given_low = no_low / output_counts.get('Low', 1)
    
    print(f"P(Overtime=No | Output=High) = {no_high}/{output_counts.get('High')} = {p_no_given_high:.4f}")
    print(f"P(Overtime=No | Output=Low) = {no_low}/{output_counts.get('Low')} = {p_no_given_low:.4f}")
    
    # Naive Bayes 계산: 
    # P(Output=High | 특성들) ∝ P(Output=High) * P(특성1|High) * P(특성2|High) * ...
    # P(Output=Low | 특성들) ∝ P(Output=Low) * P(특성1|Low) * P(특성2|Low) * ...
    print("\n3. 나이브 베이즈 계산:")
    
    # High에 대한 확률 계산
    p_high_given_features = p_high * p_pat_given_high * p_jim_given_high * p_a_given_high * p_no_given_high
    print(f"P(High | Pat, Jim, A, No) ∝ P(High) * P(Pat|High) * P(Jim|High) * P(A|High) * P(No|High)")
    print(f"P(High | Pat, Jim, A, No) ∝ {p_high:.4f} * {p_pat_given_high:.4f} * {p_jim_given_high:.4f} * {p_a_given_high:.4f} * {p_no_given_high:.4f}")
    print(f"P(High | Pat, Jim, A, No) ∝ {p_high_given_features:.10f}")
    
    # Low에 대한 확률 계산
    p_low_given_features = p_low * p_pat_given_low * p_jim_given_low * p_a_given_low * p_no_given_low
    print(f"P(Low | Pat, Jim, A, No) ∝ P(Low) * P(Pat|Low) * P(Jim|Low) * P(A|Low) * P(No|Low)")
    print(f"P(Low | Pat, Jim, A, No) ∝ {p_low:.4f} * {p_pat_given_low:.4f} * {p_jim_given_low:.4f} * {p_a_given_low:.4f} * {p_no_given_low:.4f}")
    print(f"P(Low | Pat, Jim, A, No) ∝ {p_low_given_features:.10f}")
    
    # 사후 확률 비율 (Posterior odds) 계산
    posterior_odds = p_high_given_features / p_low_given_features
    
    print("\n4. 사후 확률 비율 (Posterior Odds) 계산:")
    print(f"Posterior Odds = P(High | Pat, Jim, A, No) / P(Low | Pat, Jim, A, No)")
    print(f"Posterior Odds = {p_high_given_features:.10f} / {p_low_given_features:.10f}")
    print(f"Posterior Odds = {posterior_odds:.4f}")
    
    # 예측 결과
    if posterior_odds > 1:
        print("\n5. 예측 결과: Output = High (사후 확률 비율 > 1)")
    else:
        print("\n5. 예측 결과: Output = Low (사후 확률 비율 < 1)")
    
    return posterior_odds

def main():
    print("공장 생산 라인 데이터의 나이브 베이즈 계산")
    print("-" * 60)
    
    # 공장 데이터 생성
    df = create_factory_data()
    print("데이터셋:")
    print(df)
    
    # 테스트 예제 정의
    test_example = {
        'Supervisor': 'Pat',
        'Operator': 'Jim',
        'Machine': 'A',
        'Overtime': 'No'
    }
    
    print(f"\n테스트 예제: [Supervisor={test_example['Supervisor']}, "
          f"Operator={test_example['Operator']}, "
          f"Machine={test_example['Machine']}, "
          f"Overtime={test_example['Overtime']}]")
    
    # 나이브 베이즈 예측
    posterior_odds = naive_bayes_prediction(df, test_example)
    
    print("\n결론:")
    print(f"사후 확률 비율 (Posterior Odds) = {posterior_odds:.4f}")
    if posterior_odds > 1:
        print(f"테스트 예제 {test_example}에 대한 나이브 베이즈 분류기의 예측 결과는 Output = High 입니다.")
    else:
        print(f"테스트 예제 {test_example}에 대한 나이브 베이즈 분류기의 예측 결과는 Output = Low 입니다.")

if __name__ == "__main__":
    main() 