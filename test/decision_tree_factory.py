import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import confusion_matrix, classification_report
import math

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

# 정보 이득(Information Gain) 계산 함수
def calculate_entropy(y):
    """엔트로피 계산"""
    # 클래스별 확률 계산
    unique_classes = np.unique(y)
    entropy = 0
    for cls in unique_classes:
        p_cls = len(y[y == cls]) / len(y)
        entropy -= p_cls * math.log2(p_cls)
    return entropy

def calculate_information_gain(X, y, feature):
    """특성에 대한 정보 이득 계산"""
    # 전체 엔트로피
    total_entropy = calculate_entropy(y)
    
    # 특성값에 따른 조건부 엔트로피
    feature_values = np.unique(X[feature])
    conditional_entropy = 0
    
    for value in feature_values:
        # 해당 특성값을 가진 샘플의 인덱스
        indices = X[feature] == value
        
        # 특성값의 비율
        p_value = sum(indices) / len(X)
        
        # 해당 특성값을 가진 샘플들의 엔트로피
        entropy_value = calculate_entropy(y[indices])
        
        # 조건부 엔트로피에 가중치를 두어 합산
        conditional_entropy += p_value * entropy_value
    
    # 정보 이득 = 전체 엔트로피 - 조건부 엔트로피
    information_gain = total_entropy - conditional_entropy
    return information_gain

# 결정 트리 학습 및 분석
def train_decision_tree(df):
    """결정 트리 분류기 학습 및 시각화"""
    # 특성(X)과 타겟(y) 분리
    X = df.drop('Output', axis=1)
    y = df['Output']
    
    # 범주형 변수 인코딩
    encoder = OrdinalEncoder()
    X_encoded = encoder.fit_transform(X)
    
    # 결정 트리 모델 학습
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_encoded, y)
    
    # 특성 중요도
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print("특성 중요도:")
    print(feature_importance)
    
    # 특성별 정보 이득 계산
    print("\n특성별 정보 이득(Information Gain):")
    for feature in X.columns:
        ig = calculate_information_gain(X, y, feature)
        print(f"{feature}: {ig:.4f}")
    
    # 결정 트리 시각화
    plt.figure(figsize=(15, 10))
    plot_tree(model, feature_names=X.columns, class_names=model.classes_, 
              filled=True, rounded=True, fontsize=10)
    plt.title('Factory Data Decision Tree')
    plt.savefig('factory_decision_tree.png')
    plt.show()
    
    # 텍스트로 트리 출력
    tree_text = export_text(model, feature_names=list(X.columns))
    print("\n결정 트리 구조:")
    print(tree_text)
    
    # 예측
    y_pred = model.predict(X_encoded)
    
    # 평가
    print("\n분류 보고서:")
    print(classification_report(y, y_pred))
    
    # 혼동 행렬
    cm = confusion_matrix(y, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=model.classes_, yticklabels=model.classes_)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.savefig('decision_tree_confusion_matrix.png')
    plt.show()
    
    return model, encoder

# 수작업 의사결정 트리 시각화 (Supervisor를 루트 노드로)
def visualize_supervisor_decision_tree():
    """Supervisor를 루트 노드로 하는 수작업 의사결정 트리 시각화"""
    # 트리 그래프 생성
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 노드 위치 계산
    nodes = {
        'root': (0.5, 0.9),
        'pat': (0.25, 0.7),
        'tom': (0.75, 0.7),
        'sally': (0.5, 0.5),
        'pat_overtime': (0.25, 0.5),
        'pat_yes': (0.1, 0.3),
        'pat_no': (0.4, 0.3)
    }
    
    # 노드 그리기
    def draw_node(position, label, decision=False):
        if decision:
            circle = plt.Circle(position, 0.05, color='lightgreen' if label == 'High' else 'salmon', 
                               alpha=0.8, ec='black')
        else:
            circle = plt.Circle(position, 0.05, color='lightblue', alpha=0.8, ec='black')
        ax.add_patch(circle)
        plt.text(position[0], position[1], label, ha='center', va='center', fontweight='bold')
    
    # 엣지 그리기
    def draw_edge(start, end, label):
        plt.plot([start[0], end[0]], [start[1], end[1]], 'k-')
        # 엣지 라벨
        mid_x = (start[0] + end[0]) / 2
        mid_y = (start[1] + end[1]) / 2
        plt.text(mid_x, mid_y, label, ha='center', va='center', 
                bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'))
    
    # 루트 노드 (Supervisor)
    draw_node(nodes['root'], 'Supervisor')
    
    # Supervisor=Pat 브랜치
    draw_node(nodes['pat'], 'Pat')
    draw_edge(nodes['root'], nodes['pat'], 'Pat')
    
    # Supervisor=Tom 브랜치
    draw_node(nodes['tom'], 'Tom', decision=True)
    draw_edge(nodes['root'], nodes['tom'], 'Tom')
    plt.text(nodes['tom'][0], nodes['tom'][1]-0.08, 'Low', ha='center', va='center')
    
    # Supervisor=Sally 브랜치
    draw_node(nodes['sally'], 'Sally', decision=True)
    draw_edge(nodes['root'], nodes['sally'], 'Sally')
    plt.text(nodes['sally'][0], nodes['sally'][1]-0.08, 'High', ha='center', va='center')
    
    # Pat -> Overtime
    draw_node(nodes['pat_overtime'], 'Overtime')
    draw_edge(nodes['pat'], nodes['pat_overtime'], '')
    
    # Overtime=Yes 브랜치
    draw_node(nodes['pat_yes'], 'Yes', decision=True)
    draw_edge(nodes['pat_overtime'], nodes['pat_yes'], 'Yes')
    plt.text(nodes['pat_yes'][0], nodes['pat_yes'][1]-0.08, 'Low', ha='center', va='center')
    
    # Overtime=No 브랜치
    draw_node(nodes['pat_no'], 'No', decision=True)
    draw_edge(nodes['pat_overtime'], nodes['pat_no'], 'No')
    plt.text(nodes['pat_no'][0], nodes['pat_no'][1]-0.08, 'High', ha='center', va='center')
    
    # 축 설정
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    
    plt.title('Decision Tree with Supervisor as Root Node', fontsize=16)
    plt.savefig('supervisor_decision_tree.png')
    plt.show()

# 데이터 분포 분석
def analyze_feature_distribution(df):
    """특성별 Output 분포 분석"""
    print("\n특성별 Output 분포:")
    
    # 각 특성에 대해 교차표 생성
    for feature in df.columns:
        if feature != 'Output':
            cross_tab = pd.crosstab(df[feature], df['Output'])
            
            # 출력
            print(f"\n{feature} 기준 Output 분포:")
            print(cross_tab)
            
            # 시각화
            plt.figure(figsize=(10, 6))
            cross_tab.plot(kind='bar', stacked=True, ax=plt.gca())
            plt.title(f'Output Distribution by {feature}')
            plt.xlabel(feature)
            plt.ylabel('Count')
            plt.legend(title='Output')
            plt.savefig(f'output_by_{feature.lower()}.png')
            plt.show()
            
            # 정보 이득 계산
            ig = calculate_information_gain(df, df['Output'], feature)
            print(f"{feature}의 정보 이득 (Information Gain): {ig:.4f}")

def main():
    print("공장 생산 라인 데이터의 의사결정 트리 분석")
    print("=" * 60)
    
    # 데이터 생성
    df = create_factory_data()
    print("\n데이터셋:")
    print(df)
    
    # 데이터 분포 분석
    print("\n데이터 분포 분석...")
    analyze_feature_distribution(df)
    
    # 특성별 고유 값 확인
    for col in df.columns:
        print(f"{col}: {df[col].unique()}")
    
    # 의사결정 트리 학습
    print("\n의사결정 트리 모델 학습 및 평가...")
    model, encoder = train_decision_tree(df)
    
    # Supervisor를 루트 노드로 하는 의사결정 트리 수작업 시각화
    print("\nSupervisor를 루트 노드로 하는 의사결정 트리 시각화...")
    visualize_supervisor_decision_tree()
    
    print("\n분석 완료!")

if __name__ == "__main__":
    main() 