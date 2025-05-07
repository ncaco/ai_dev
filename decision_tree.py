import numpy as np
import pandas as pd
import math

# 훈련 데이터 정의
data = {
    'A': [2, 1, 1, 3, 2, 3, 1, 2, 1, 3],
    'B': ['H', 'H', 'H', 'M', 'M', 'M', 'M', 'L', 'L', 'L'],
    'C': [1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
    'D': ['F', 'F', 'T', 'F', 'T', 'T', 'F', 'T', 'F', 'T'],
    'Class': ['Y', 'N', 'N', 'Y', 'Y', 'N', 'N', 'Y', 'Y', 'N']
}

df = pd.DataFrame(data)
print("=" * 50)
print("1-1. 의사결정 트리 도출")
print("=" * 50)
print("\n[훈련 데이터]")
print(df)

# 클래스 분포 계산
class_distribution = df['Class'].value_counts()
print("\n클래스 분포:")
print(class_distribution)
total_samples = len(df)
print(f"총 샘플 수: {total_samples}")

# 엔트로피 계산 함수
def calculate_entropy(y):
    classes = y.unique()
    entropy = 0
    total = len(y)
    
    for cls in classes:
        count = len(y[y == cls])
        p = count / total
        if p > 0:
            entropy_term = -p * math.log2(p)
            entropy += entropy_term
    
    return entropy

# 정보 이득 계산 함수
def calculate_information_gain(data, feature, target, verbose=False):
    total_entropy = calculate_entropy(data[target])
    values = data[feature].unique()
    total_samples = len(data)
    weighted_entropy = 0
    
    for value in values:
        subset = data[data[feature] == value]
        subset_size = len(subset)
        weight = subset_size / total_samples
        subset_entropy = calculate_entropy(subset[target])
        weighted_entropy += weight * subset_entropy
    
    information_gain = total_entropy - weighted_entropy
    return information_gain

# 전체 데이터의 엔트로피 계산
total_entropy = calculate_entropy(df['Class'])
print(f"\n[전체 데이터 엔트로피] H(S) = {total_entropy:.4f}")

# 각 특성에 대한 정보 이득 계산
print("\n[각 특성의 정보 이득]")
features = ['A', 'B', 'C', 'D']
information_gains = {}
feature_values_count = {}

for feature in features:
    ig = calculate_information_gain(df, feature, 'Class')
    information_gains[feature] = ig
    # 특성별 값의 개수 저장
    feature_values_count[feature] = len(df[feature].unique())
    print(f"특성 {feature}의 정보 이득: {ig:.4f}")

# 정보 이득이 가장 높은 특성 선택
best_feature = max(information_gains, key=information_gains.get)
print(f"\n최적의 분할 특성: {best_feature} (정보 이득: {information_gains[best_feature]:.4f})")

# 특성 값의 개수 출력
print("\n[특성별 값의 개수]")
for feature, count in feature_values_count.items():
    print(f"특성 {feature}: {count}개 값")

# 의사결정 트리 구축
print("\n[의사결정 트리 구축]")
root_feature = best_feature
print(f"루트 노드: {root_feature}")

# 루트 노드의 각 값에 대한 하위 노드 분석
print("\n[하위 노드 분석]")
for value in sorted(df[root_feature].unique()):
    subset = df[df[root_feature] == value]
    print(f"\n{root_feature} = {value}일 때:")
    print(f"  데이터 개수: {len(subset)}")
    print(f"  클래스 분포: {subset['Class'].value_counts().to_dict()}")
    
    # 클래스가 모두 같은지 확인
    if len(subset['Class'].unique()) == 1:
        print(f"  => 모든 클래스가 {subset['Class'].iloc[0]}입니다. (리프 노드)")
    else:
        # 하위 노드에서 최적의 특성 찾기
        sub_information_gains = {}
        remaining_features = [f for f in features if f != root_feature]
        
        # 각 남은 특성의 정보 이득 계산
        for feature in remaining_features:
            ig = calculate_information_gain(subset, feature, 'Class', verbose=False)
            sub_information_gains[feature] = ig
        
        best_sub_feature = max(sub_information_gains, key=sub_information_gains.get)
        print(f"  => 최적의 하위 특성: {best_sub_feature} (정보 이득: {sub_information_gains[best_sub_feature]:.4f})")
        
        # 추가로 하위 노드 분석
        for sub_value in sorted(subset[best_sub_feature].unique()):
            sub_subset = subset[subset[best_sub_feature] == sub_value]
            print(f"    {best_sub_feature} = {sub_value}일 때:")
            class_dist = sub_subset['Class'].value_counts().to_dict()
            print(f"      클래스 분포: {class_dist}")
            
            # 최종 분류 결과 출력
            if len(sub_subset['Class'].unique()) == 1:
                print(f"      => 분류 결과: {sub_subset['Class'].iloc[0]} (리프 노드)")
            else:
                majority_class = max(class_dist, key=class_dist.get)
                print(f"      => 분류 결과: {majority_class} (다수결)")

# 최종 의사결정 트리 요약
print("\n[최종 의사결정 트리]")
print(f"루트 노드: {root_feature}")
print(f"  ├─ {root_feature} = 1")
print(f"  │   ├─ B = H → Class = N")
print(f"  │   ├─ B = M → Class = N")
print(f"  │   └─ B = L → Class = Y")
print(f"  ├─ {root_feature} = 2 → Class = Y")
print(f"  └─ {root_feature} = 3")
print(f"      ├─ D = F → Class = Y")
print(f"      └─ D = T → Class = N")

print("\n" + "=" * 50)
print("1-2. 테스트 예제 분류")
print("=" * 50)

# 테스트 예제 1: A=1, B=?, C=0, D=T
print("\n[테스트 예제 1] A=1, B=?, C=0, D=T")
print("예측 결과:")
print("  - B=H인 경우: Class = N")
print("  - B=M인 경우: Class = N")
print("  - B=L인 경우: Class = Y")
print("  - 결론: B 값이 누락되어 정확한 예측 불가능")
print("  - 판단 근거: A=1일 때 분류 결과는 B 값에 따라 달라짐")

# 테스트 예제 2: A=?, B=M, C=1, D=F
print("\n[테스트 예제 2] A=?, B=M, C=1, D=F")
subset_BM_C1_DF = df[(df['B'] == 'M') & (df['C'] == 1) & (df['D'] == 'F')]
print("데이터 분석:")
if not subset_BM_C1_DF.empty:
    class_dist = subset_BM_C1_DF['Class'].value_counts().to_dict()
    print(f"  - B=M, C=1, D=F인 데이터 클래스 분포: {class_dist}")
    print("예측 결과:")
    print("  - A=1인 경우: Class = N")
    print("  - A=2인 경우: Class = Y")
    print("  - A=3인 경우: Class = Y")
    print("  - 결론: A 값이 누락되어 정확한 예측 불가능")
    print("  - 판단 근거: 분류 결과는 A 값에 따라 달라짐")
else:
    print("  - 해당 조합의 데이터가 없습니다.") 