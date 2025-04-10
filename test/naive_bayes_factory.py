import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.naive_bayes import CategoricalNB
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import confusion_matrix, classification_report

# 공장 생산 라인 데이터 생성
def create_factory_data():
    """슬라이드에 있는 공장 생산 라인 데이터 생성"""
    data = {
        'Supervisor': ['Pat', 'Pat', 'Tom', 'Pat', 'Sally', 'Tom', 'Tom', 'Pat'],
        'Operator': ['Joe', 'Sam', 'Jim', 'Jim', 'Joe', 'Sam', 'Joe', 'Jim'],
        'Machine': ['A', 'B', 'B', 'B', 'C', 'C', 'C', 'A'],
        'Overtime': ['No', 'Yes', 'Yes', 'No', 'No', 'No', 'No', 'Yes'],
        'Output': ['High', 'Low', 'Low', 'High', 'High', 'Low', 'Low', 'Low']
    }
    
    df = pd.DataFrame(data)
    return df

# 데이터 시각화 함수
def visualize_data(df):
    """데이터 분포 시각화"""
    # 범주형 변수의 분포 시각화
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Supervisor별 Output 분포
    sns.countplot(x='Supervisor', hue='Output', data=df, ax=axes[0, 0])
    axes[0, 0].set_title('Output by Supervisor')
    
    # Operator별 Output 분포
    sns.countplot(x='Operator', hue='Output', data=df, ax=axes[0, 1])
    axes[0, 1].set_title('Output by Operator')
    
    # Machine별 Output 분포
    sns.countplot(x='Machine', hue='Output', data=df, ax=axes[1, 0])
    axes[1, 0].set_title('Output by Machine')
    
    # Overtime별 Output 분포
    sns.countplot(x='Overtime', hue='Output', data=df, ax=axes[1, 1])
    axes[1, 1].set_title('Output by Overtime')
    
    plt.tight_layout()
    plt.savefig('factory_data_distribution.png')
    plt.show()

# 나이브 베이즈 모델 학습 및 평가
def train_naive_bayes(df):
    """나이브 베이즈 분류기 학습 및 평가"""
    # 특성(X)과 타겟(y) 분리
    X = df.drop('Output', axis=1)
    y = df['Output']
    
    # 범주형 변수 인코딩
    encoder = OrdinalEncoder()
    X_encoded = encoder.fit_transform(X)
    
    # 나이브 베이즈 모델 학습
    model = CategoricalNB()
    model.fit(X_encoded, y)
    
    # 예측
    y_pred = model.predict(X_encoded)
    
    # 평가
    print("분류 보고서:")
    print(classification_report(y, y_pred))
    
    # 혼동 행렬
    cm = confusion_matrix(y, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=model.classes_, yticklabels=model.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig('naive_bayes_confusion_matrix.png')
    plt.show()
    
    return model, encoder

# 테스트 예제에 대한 예측 확률 계산
def predict_test_example(model, encoder, test_example):
    """테스트 예제에 대한 예측 확률 계산"""
    # 테스트 예제를 데이터프레임으로 변환
    test_df = pd.DataFrame([test_example])
    
    # 인코딩
    test_encoded = encoder.transform(test_df)
    
    # 클래스별 확률 예측
    class_probs = model.predict_proba(test_encoded)[0]
    classes = model.classes_
    
    # 결과 출력
    print("\n테스트 예제:", test_example)
    print("\n클래스별 확률:")
    for i, class_name in enumerate(classes):
        print(f"{class_name}: {class_probs[i]:.4f}")
    
    # 예측 클래스
    predicted_class = classes[np.argmax(class_probs)]
    print(f"\n예측 클래스: {predicted_class}")
    
    # 고확률 클래스와 저확률 클래스의 비율 계산 (사후 확률 비율)
    if len(classes) == 2:
        posterior_odds = class_probs[1] / class_probs[0] if class_probs[0] > 0 else float('inf')
        print(f"사후 확률 비율(Posterior Odds): {posterior_odds:.4f}")
    
    return predicted_class, class_probs

# 수동 계산 나이브 베이즈 (슬라이드에 있는 예제 계산)
def manual_naive_bayes(df, test_example):
    """수동으로 나이브 베이즈 확률 계산"""
    # 전체 데이터 수
    total_samples = len(df)
    
    # 클래스 사전 확률 계산: P(Output)
    class_counts = df['Output'].value_counts()
    p_high = class_counts.get('High', 0) / total_samples
    p_low = class_counts.get('Low', 0) / total_samples
    
    print("\n1. 클래스 사전 확률:")
    print(f"P(Output=High) = {p_high:.4f}")
    print(f"P(Output=Low) = {p_low:.4f}")
    
    # 조건부 확률 계산: P(특성 | Output)
    print("\n2. 조건부 확률:")
    
    # 각 특성에 대한 조건부 확률 계산
    feature_probs = {}
    for feature in df.columns:
        if feature != 'Output':
            feature_value = test_example[feature]
            
            # 각 클래스에 대한 조건부 확률
            high_count = df[(df[feature] == feature_value) & (df['Output'] == 'High')].shape[0]
            low_count = df[(df[feature] == feature_value) & (df['Output'] == 'Low')].shape[0]
            
            # 라플라스 스무딩 적용 (0이 되는 확률 방지)
            p_feature_given_high = (high_count + 1) / (class_counts.get('High', 0) + len(df[feature].unique()))
            p_feature_given_low = (low_count + 1) / (class_counts.get('Low', 0) + len(df[feature].unique()))
            
            # 결과 저장
            feature_probs[f"{feature}={feature_value}|High"] = p_feature_given_high
            feature_probs[f"{feature}={feature_value}|Low"] = p_feature_given_low
            
            print(f"P({feature}={feature_value} | High) = {high_count}/{class_counts.get('High', 0)} = {p_feature_given_high:.4f}")
            print(f"P({feature}={feature_value} | Low) = {low_count}/{class_counts.get('Low', 0)} = {p_feature_given_low:.4f}")
    
    # 나이브 베이즈 계산
    print("\n3. 나이브 베이즈 계산:")
    
    # P(High | 특성들) ∝ P(High) * P(특성1|High) * P(특성2|High) * ...
    p_high_given_features = p_high
    for feature in df.columns:
        if feature != 'Output':
            p_high_given_features *= feature_probs.get(f"{feature}={test_example[feature]}|High", 0)
    
    # P(Low | 특성들) ∝ P(Low) * P(특성1|Low) * P(특성2|Low) * ...
    p_low_given_features = p_low
    for feature in df.columns:
        if feature != 'Output':
            p_low_given_features *= feature_probs.get(f"{feature}={test_example[feature]}|Low", 0)
    
    # 출력
    print("P(High | 특성들) ∝ P(High) * P(Supervisor|High) * P(Operator|High) * P(Machine|High) * P(Overtime|High)")
    print(f"P(High | 특성들) ∝ {p_high:.4f} * ", end="")
    print(" * ".join([f"{feature_probs.get(f'{feature}={test_example[feature]}|High', 0):.4f}" for feature in df.columns if feature != 'Output']))
    print(f"P(High | 특성들) ∝ {p_high_given_features:.10f}")
    
    print("\nP(Low | 특성들) ∝ P(Low) * P(Supervisor|Low) * P(Operator|Low) * P(Machine|Low) * P(Overtime|Low)")
    print(f"P(Low | 특성들) ∝ {p_low:.4f} * ", end="")
    print(" * ".join([f"{feature_probs.get(f'{feature}={test_example[feature]}|Low', 0):.4f}" for feature in df.columns if feature != 'Output']))
    print(f"P(Low | 특성들) ∝ {p_low_given_features:.10f}")
    
    # 사후 확률 비율 (Posterior odds) 계산
    posterior_odds = p_high_given_features / p_low_given_features if p_low_given_features > 0 else float('inf')
    
    print("\n4. 사후 확률 비율 (Posterior Odds) 계산:")
    print(f"Posterior Odds = P(High | 특성들) / P(Low | 특성들)")
    print(f"Posterior Odds = {p_high_given_features:.10f} / {p_low_given_features:.10f}")
    print(f"Posterior Odds = {posterior_odds:.4f}")
    
    # 예측 결과
    predicted_class = 'High' if posterior_odds > 1 else 'Low'
    print(f"\n5. 예측 결과: Output = {predicted_class} (사후 확률 비율 {'>' if posterior_odds > 1 else '<'} 1)")
    
    return predicted_class, posterior_odds

def main():
    print("공장 생산 라인 데이터의 나이브 베이즈 분석")
    print("=" * 60)
    
    # 데이터 생성
    df = create_factory_data()
    print("\n데이터셋:")
    print(df)
    
    # 데이터 분포 시각화
    print("\n데이터 분포 시각화...")
    visualize_data(df)
    
    # 나이브 베이즈 모델 학습
    print("\n나이브 베이즈 모델 학습 및 평가...")
    model, encoder = train_naive_bayes(df)
    
    # 테스트 예제 정의 (슬라이드의 문제)
    test_example = {
        'Supervisor': 'Pat',
        'Operator': 'Jim',
        'Machine': 'A',
        'Overtime': 'No'
    }
    
    # 모델을 사용한 예측
    print("\n모델을 사용한 예측:")
    predict_test_example(model, encoder, test_example)
    
    # 수동 나이브 베이즈 계산
    print("\n수동으로 나이브 베이즈 확률 계산:")
    manual_naive_bayes(df, test_example)
    
    print("\n분석 완료!")

if __name__ == "__main__":
    main() 