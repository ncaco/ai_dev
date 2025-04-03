"""
공장 생산 라인 데이터 분석

이 프로그램은 공장 생산 라인 데이터를 사용하여 의사결정 트리와 나이브 베이즈 모델을 구현합니다.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
from matplotlib.colors import ListedColormap
import pandas as pd
from sklearn import tree
from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import OrdinalEncoder
import matplotlib.font_manager as fm
import platform

# 한글 폰트 설정
def set_korean_font():
    """한글 폰트 설정 함수"""
    system_name = platform.system()
    
    if system_name == "Windows":
        # 윈도우의 경우 맑은 고딕 폰트를 사용
        font_path = r"C:\Windows\Fonts\malgun.ttf"
        font_name = fm.FontProperties(fname=font_path).get_name()
        plt.rc('font', family=font_name)
    elif system_name == "Darwin":  # macOS
        # macOS의 경우 Apple Gothic을 사용
        plt.rc('font', family='AppleGothic')
    else:
        # Linux 및 기타 시스템의 경우 나눔 고딕 또는 기본 폰트 사용 시도
        try:
            # 나눔고딕이 설치된 경우
            plt.rc('font', family='NanumGothic')
        except:
            print("적절한 한글 폰트를 찾을 수 없습니다. 시스템 기본 폰트를 사용합니다.")
    
    # 마이너스 기호가 깨지는 문제 해결
    plt.rc('axes', unicode_minus=False)

# 한글 폰트 설정 함수 호출
set_korean_font()

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

# 데이터 정보 출력
def print_data_info(df):
    """데이터 정보 출력"""
    print("\n데이터셋 정보:")
    print(df)
    
    print("\n각 특성별 고유 값:")
    for col in df.columns:
        values = df[col].unique()
        print(f"{col}: {values}")
    
    # 특성별 Output 분포 계산
    print("\n특성별 Output 분포:")
    
    for feature in ['Supervisor', 'Machine']:
        print(f"\n{feature} 기준 Output 분포:")
        output_counts = df.groupby([feature, 'Output']).size().unstack().fillna(0)
        print(output_counts)
        
        # 정보 이득(Information Gain) 계산을 위한 엔트로피 준비
        total_samples = len(df)
        feature_values = df[feature].unique()
        
        # 전체 데이터의 엔트로피 계산
        output_counts_total = df['Output'].value_counts()
        entropy_total = 0
        for count in output_counts_total:
            prob = count / total_samples
            entropy_total -= prob * np.log2(prob)
        
        # 각 특성값에 대한 조건부 엔트로피 계산
        conditional_entropy = 0
        for value in feature_values:
            subset = df[df[feature] == value]
            value_count = len(subset)
            value_prob = value_count / total_samples
            
            value_entropy = 0
            output_counts_value = subset['Output'].value_counts()
            for count in output_counts_value:
                prob = count / value_count
                value_entropy -= prob * np.log2(prob)
            
            conditional_entropy += value_prob * value_entropy
        
        # 정보 이득 계산
        information_gain = entropy_total - conditional_entropy
        print(f"{feature}의 정보 이득 (Information Gain): {information_gain:.4f}")

# 의사결정 트리 시각화 - 맞춤형 그래픽
def visualize_custom_decision_tree():
    """공장 의사결정 트리 시각화"""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis('off')  # 축 숨기기
    
    # 노드 위치 정의
    nodes = {
        'root': (8, 8),                # Machine 노드
        'A_child': (4, 6),             # Machine=A 노드
        'B_child': (8, 6),             # Machine=B 노드
        'C_child': (12, 6),            # Machine=C 노드
        'B_overtime_yes': (7, 4),      # Machine=B & Overtime=Yes 노드
        'B_overtime_no': (9, 4),       # Machine=B & Overtime=No 노드
        'A_output': (4, 4),            # Machine=A 결과 노드
        'B_yes_output': (7, 2),        # Machine=B & Overtime=Yes 결과 노드
        'B_no_output': (9, 2),         # Machine=B & Overtime=No 결과 노드
        'C_output': (12, 4)            # Machine=C 결과 노드
    }
    
    # 노드 색상 정의
    node_colors = {
        'feature': '#F5F5F5',  # 연한 회색 (특징 노드)
        'high': '#4CAF50',     # 녹색 (High)
        'low': '#F44336'       # 빨간색 (Low)
    }
    
    # 루트 노드 (Machine)
    circle = plt.Circle(nodes['root'], 0.8, color=node_colors['feature'], ec='black')
    ax.add_patch(circle)
    ax.text(nodes['root'][0], nodes['root'][1], "Machine", ha='center', va='center', fontsize=14, weight='bold')
    
    # Machine=A 노드
    # 데이터에 따르면, Machine=A에는 High와 Low가 모두 있지만, 결정 트리에서는 주요 클래스로 표시합니다.
    # High가 더 많거나 동일하면 High로 결정
    rect = plt.Rectangle((nodes['A_output'][0]-1.5, nodes['A_output'][1]-0.5), 3, 1,
                        color=node_colors['high'], ec='black')
    ax.add_patch(rect)
    ax.text(nodes['A_output'][0], nodes['A_output'][1], "Output: High (1/2)", ha='center', va='center', fontsize=12)
    
    # Machine=B 중간 노드 (Overtime 분할)
    circle = plt.Circle(nodes['B_child'], 0.8, color=node_colors['feature'], ec='black')
    ax.add_patch(circle)
    ax.text(nodes['B_child'][0], nodes['B_child'][1], "Overtime", ha='center', va='center', fontsize=14, weight='bold')
    
    # Machine=B & Overtime=Yes 노드
    rect = plt.Rectangle((nodes['B_yes_output'][0]-1.5, nodes['B_yes_output'][1]-0.5), 3, 1,
                        color=node_colors['low'], ec='black')
    ax.add_patch(rect)
    ax.text(nodes['B_yes_output'][0], nodes['B_yes_output'][1], "Output: Low (2/2)", ha='center', va='center', fontsize=12)
    
    # Machine=B & Overtime=No 노드
    rect = plt.Rectangle((nodes['B_no_output'][0]-1.5, nodes['B_no_output'][1]-0.5), 3, 1,
                        color=node_colors['high'], ec='black')
    ax.add_patch(rect)
    ax.text(nodes['B_no_output'][0], nodes['B_no_output'][1], "Output: High (1/1)", ha='center', va='center', fontsize=12)
    
    # Machine=C 노드
    # 데이터에 따르면, Machine=C에는 High와 Low가 모두 있지만, 결정 트리에서는 주요 클래스로 표시합니다.
    # Low가 더 많으면 Low로 결정
    rect = plt.Rectangle((nodes['C_output'][0]-1.5, nodes['C_output'][1]-0.5), 3, 1,
                        color=node_colors['low'], ec='black')
    ax.add_patch(rect)
    ax.text(nodes['C_output'][0], nodes['C_output'][1], "Output: Low (2/3)", ha='center', va='center', fontsize=12)
    
    # 연결선 그리기
    # Root -> Machine=A
    ax.plot([nodes['root'][0], nodes['A_child'][0]],
           [nodes['root'][1]-0.8, nodes['A_output'][1]+0.5], 'k-')
    ax.text((nodes['root'][0] + nodes['A_child'][0])/2 - 1.5,
           (nodes['root'][1] + nodes['A_output'][1])/2 + 1, "A", fontsize=12, weight='bold')
    
    # Root -> Machine=B
    ax.plot([nodes['root'][0], nodes['B_child'][0]],
           [nodes['root'][1]-0.8, nodes['B_child'][1]+0.8], 'k-')
    ax.text(nodes['root'][0],
           (nodes['root'][1] + nodes['B_child'][1])/2 + 0.3, "B", fontsize=12, weight='bold')
    
    # Root -> Machine=C
    ax.plot([nodes['root'][0], nodes['C_child'][0]],
           [nodes['root'][1]-0.8, nodes['C_output'][1]+0.5], 'k-')
    ax.text((nodes['root'][0] + nodes['C_child'][0])/2 + 1.5,
           (nodes['root'][1] + nodes['C_output'][1])/2 + 1, "C", fontsize=12, weight='bold')
    
    # Machine=B -> Overtime=Yes
    ax.plot([nodes['B_child'][0], nodes['B_overtime_yes'][0]],
           [nodes['B_child'][1]-0.8, nodes['B_yes_output'][1]+0.5], 'k-')
    ax.text((nodes['B_child'][0] + nodes['B_overtime_yes'][0])/2 - 0.5,
           (nodes['B_child'][1] + nodes['B_yes_output'][1])/2 + 0.3, "Yes", fontsize=12)
    
    # Machine=B -> Overtime=No
    ax.plot([nodes['B_child'][0], nodes['B_overtime_no'][0]],
           [nodes['B_child'][1]-0.8, nodes['B_no_output'][1]+0.5], 'k-')
    ax.text((nodes['B_child'][0] + nodes['B_overtime_no'][0])/2 + 0.5,
           (nodes['B_child'][1] + nodes['B_no_output'][1])/2 + 0.3, "No", fontsize=12)
    
    plt.title('공장 생산 데이터 의사결정 트리', fontsize=16)
    plt.tight_layout()
    plt.savefig('factory_decision_tree.png', dpi=300)
    plt.show()

# 의사결정 트리 학습 및 시각화 - scikit-learn 사용
def train_and_visualize_decision_tree(df):
    """scikit-learn을 사용하여 의사결정 트리 학습 및 시각화"""
    # 특성과 타겟 분리
    X = df.drop('Output', axis=1)
    y = df['Output']
    
    # 범주형 특성을 숫자로 변환
    encoder = OrdinalEncoder()
    X_encoded = encoder.fit_transform(X)
    
    # 의사결정 트리 모델 생성 및 학습
    clf = tree.DecisionTreeClassifier(max_depth=3)
    clf.fit(X_encoded, y)
    
    # 의사결정 트리 시각화
    plt.figure(figsize=(15, 10))
    tree.plot_tree(clf, 
                  feature_names=X.columns,
                  class_names=['High', 'Low'],
                  filled=True,
                  fontsize=12)
    plt.title('scikit-learn 의사결정 트리', fontsize=16)
    plt.tight_layout()
    plt.savefig('factory_sklearn_decision_tree.png', dpi=300)
    plt.show()
    
    return clf, encoder

# 나이브 베이즈 모델을 사용한 예측
def perform_naive_bayes_prediction(df, test_example):
    """나이브 베이즈 모델을 사용한 예측"""
    # 데이터에서 확률 계산
    total_samples = len(df)
    
    # 클래스 확률 P(Output)
    output_counts = df['Output'].value_counts()
    p_high = output_counts.get('High', 0) / total_samples
    p_low = output_counts.get('Low', 0) / total_samples
    
    print("\n나이브 베이즈 예측 계산:")
    print(f"P(Output=High) = {p_high:.4f}")
    print(f"P(Output=Low) = {p_low:.4f}")
    
    # 조건부 확률 계산
    # P(특성값 | Output) 계산
    print("\n조건부 확률:")
    
    # Supervisor=Pat 조건부 확률
    pat_high = df[(df['Supervisor'] == 'Pat') & (df['Output'] == 'High')].shape[0]
    pat_low = df[(df['Supervisor'] == 'Pat') & (df['Output'] == 'Low')].shape[0]
    p_pat_given_high = pat_high / output_counts.get('High', 1)
    p_pat_given_low = pat_low / output_counts.get('Low', 1)
    
    print(f"P(Supervisor=Pat | Output=High) = {p_pat_given_high:.4f}")
    print(f"P(Supervisor=Pat | Output=Low) = {p_pat_given_low:.4f}")
    
    # Operator=Jim 조건부 확률
    jim_high = df[(df['Operator'] == 'Jim') & (df['Output'] == 'High')].shape[0]
    jim_low = df[(df['Operator'] == 'Jim') & (df['Output'] == 'Low')].shape[0]
    p_jim_given_high = jim_high / output_counts.get('High', 1)
    p_jim_given_low = jim_low / output_counts.get('Low', 1)
    
    print(f"P(Operator=Jim | Output=High) = {p_jim_given_high:.4f}")
    print(f"P(Operator=Jim | Output=Low) = {p_jim_given_low:.4f}")
    
    # Machine=A 조건부 확률
    a_high = df[(df['Machine'] == 'A') & (df['Output'] == 'High')].shape[0]
    a_low = df[(df['Machine'] == 'A') & (df['Output'] == 'Low')].shape[0]
    p_a_given_high = a_high / output_counts.get('High', 1)
    p_a_given_low = a_low / output_counts.get('Low', 1)
    
    print(f"P(Machine=A | Output=High) = {p_a_given_high:.4f}")
    print(f"P(Machine=A | Output=Low) = {p_a_given_low:.4f}")
    
    # Overtime=No 조건부 확률
    no_high = df[(df['Overtime'] == 'No') & (df['Output'] == 'High')].shape[0]
    no_low = df[(df['Overtime'] == 'No') & (df['Output'] == 'Low')].shape[0]
    p_no_given_high = no_high / output_counts.get('High', 1)
    p_no_given_low = no_low / output_counts.get('Low', 1)
    
    print(f"P(Overtime=No | Output=High) = {p_no_given_high:.4f}")
    print(f"P(Overtime=No | Output=Low) = {p_no_given_low:.4f}")
    
    # Naive Bayes 계산: 
    # P(Output=High | 특성들) ∝ P(Output=High) * P(특성1|High) * P(특성2|High) * ...
    # P(Output=Low | 특성들) ∝ P(Output=Low) * P(특성1|Low) * P(특성2|Low) * ...
    
    # Laplace 평활화 (Laplace smoothing)를 위한 상수
    alpha = 1e-10
    
    # High에 대한 확률 계산
    p_high_given_features = p_high * p_pat_given_high * p_jim_given_high * p_a_given_high * p_no_given_high
    
    # Low에 대한 확률 계산
    p_low_given_features = p_low * p_pat_given_low * p_jim_given_low * p_a_given_low * p_no_given_low
    
    # 확률이 0이 되지 않도록 매우 작은 값을 더함
    p_high_given_features = max(p_high_given_features, alpha)
    p_low_given_features = max(p_low_given_features, alpha)
    
    print("\n나이브 베이즈 예측 결과:")
    print(f"P(Output=High | 특성들) = {p_high_given_features:.10f}")
    print(f"P(Output=Low | 특성들) = {p_low_given_features:.10f}")
    
    # 사후 확률 비율 (Posterior odds) 계산
    posterior_odds = p_high_given_features / p_low_given_features
    
    print(f"\n사후 확률 비율 (Posterior Odds) = {posterior_odds:.4f}")
    
    # 예측 결과
    if posterior_odds > 1:
        print("예측 결과: Output = High")
    else:
        print("예측 결과: Output = Low")
    
    # 시각화를 위한 데이터
    prob_data = {
        'Output': ['High', 'Low'],
        'Probability': [p_high_given_features, p_low_given_features]
    }
    
    df_prob = pd.DataFrame(prob_data)
    
    # 시각화
    plt.figure(figsize=(8, 6))
    colors = ['#4CAF50', '#F44336'] if posterior_odds > 1 else ['#F44336', '#4CAF50']
    
    bars = plt.bar(df_prob['Output'], df_prob['Probability'], color=colors)
    plt.xlabel('Output')
    plt.ylabel('Probability')
    plt.title('나이브 베이즈 예측 확률')
    
    # 확률값 표시
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.10f}',
                ha='center', va='bottom', rotation=0)
    
    plt.tight_layout()
    plt.savefig('naive_bayes_prediction.png', dpi=300)
    plt.show()
    
    return posterior_odds

# scikit-learn 나이브 베이즈 모델을 사용한 예측
def sklearn_naive_bayes_prediction(df, test_example):
    """scikit-learn 나이브 베이즈 모델을 사용한 예측"""
    # 특성과 타겟 분리
    X = df.drop('Output', axis=1)
    y = df['Output']
    
    # 범주형 특성을 숫자로 변환
    encoder = OrdinalEncoder()
    X_encoded = encoder.fit_transform(X)
    
    # 나이브 베이즈 모델 생성 및 학습
    clf = CategoricalNB()
    clf.fit(X_encoded, y)
    
    # 테스트 예제 변환
    test_df = pd.DataFrame([test_example])
    test_encoded = encoder.transform(test_df)
    
    # 예측 및 확률 계산
    prediction = clf.predict(test_encoded)
    probabilities = clf.predict_proba(test_encoded)
    
    # 결과 출력
    print("\nscikit-learn 나이브 베이즈 예측 결과:")
    print(f"예측 클래스: {prediction[0]}")
    print(f"Output=High 확률: {probabilities[0][0]:.4f}")
    print(f"Output=Low 확률: {probabilities[0][1]:.4f}")
    
    # 사후 확률 비율 계산
    if 'High' in clf.classes_ and 'Low' in clf.classes_:
        high_idx = np.where(clf.classes_ == 'High')[0][0]
        low_idx = np.where(clf.classes_ == 'Low')[0][0]
        posterior_odds = probabilities[0][high_idx] / probabilities[0][low_idx]
    else:
        # 클래스가 다르게 인코딩되었을 경우 (0과 1 등)
        posterior_odds = probabilities[0][0] / probabilities[0][1]
    
    print(f"사후 확률 비율 (Posterior Odds): {posterior_odds:.4f}")
    
    return prediction[0], posterior_odds

def main():
    print("공장 생산 라인 데이터 분석 프로그램")
    print("-" * 60)
    
    # 공장 데이터 생성
    df = create_factory_data()
    
    # 데이터 정보 출력
    print_data_info(df)
    
    # 커스텀 의사결정 트리 시각화
    print("\n1. 의사결정 트리 시각화")
    visualize_custom_decision_tree()
    
    # scikit-learn 의사결정 트리 학습 및 시각화
    print("\n2. scikit-learn 의사결정 트리 학습 및 시각화")
    clf, _ = train_and_visualize_decision_tree(df)
    
    # 나이브 베이즈 예측 - 테스트 예제: [Supervisor=Pat, Operator=Jim, Machine=A, Overtime=No]
    print("\n3. 나이브 베이즈 예측")
    test_example = {
        'Supervisor': 'Pat',
        'Operator': 'Jim',
        'Machine': 'A',
        'Overtime': 'No'
    }
    
    posterior_odds = perform_naive_bayes_prediction(df, test_example)
    
    # scikit-learn 나이브 베이즈 예측
    print("\n4. scikit-learn 나이브 베이즈 예측")
    sklearn_prediction, sklearn_odds = sklearn_naive_bayes_prediction(df, test_example)
    
    print("\n프로그램 종료")

if __name__ == "__main__":
    main() 