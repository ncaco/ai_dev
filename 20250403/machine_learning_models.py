"""
의사결정 트리와 나이브 베이즈 모델 시각화

이 프로그램은 슬라이드에 나온 의사결정 트리와 나이브 베이즈 모델을 시각화하고,
둘을 비교 분석합니다.
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

# 의사결정 트리 시각화 - 맞춤형 그래픽
def visualize_custom_decision_tree():
    """보트 렌팅 애호가를 위한 의사결정 트리 시각화"""
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')  # 축 숨기기
    
    # 노드 위치 정의
    nodes = {
        'root': (5, 8),         # 날씨 노드
        'sunny_child': (3, 6),   # 컨디션 노드 (맑음)
        'cloudy_child': (7, 6),  # 컨디션 노드 (흐림)
        'sunny_good': (2, 4),    # 결과 노드 (좋음)
        'sunny_bad': (4, 4),    # 결과 노드 (나쁨)
        'cloudy_good': (6, 4),  # 결과 노드 (좋음)
        'cloudy_bad': (8, 4)    # 결과 노드 (나쁨)
    }
    
    # 노드 색상 정의
    node_colors = {
        'feature': '#F5F5F5',  # 연한 회색 (특징 노드)
        'good': '#4CAF50',     # 녹색 (좋음)
        'bad': '#F44336'       # 빨간색 (나쁨)
    }
    
    # 루트 노드 (날씨)
    circle = plt.Circle(nodes['root'], 0.8, color=node_colors['feature'], ec='black')
    ax.add_patch(circle)
    ax.text(nodes['root'][0], nodes['root'][1], "날씨", ha='center', va='center', fontsize=14)
    
    # 맑음 서브트리
    circle = plt.Circle(nodes['sunny_child'], 0.8, color=node_colors['feature'], ec='black')
    ax.add_patch(circle)
    ax.text(nodes['sunny_child'][0], nodes['sunny_child'][1], "컨디션", ha='center', va='center', fontsize=14)
    
    # 흐림 서브트리
    circle = plt.Circle(nodes['cloudy_child'], 0.8, color=node_colors['feature'], ec='black')
    ax.add_patch(circle)
    ax.text(nodes['cloudy_child'][0], nodes['cloudy_child'][1], "컨디션", ha='center', va='center', fontsize=14)
    
    # 맑음 & 좋음
    rect = plt.Rectangle((nodes['sunny_good'][0]-0.8, nodes['sunny_good'][1]-0.5), 1.6, 1,
                        color=node_colors['good'], ec='black')
    ax.add_patch(rect)
    ax.text(nodes['sunny_good'][0], nodes['sunny_good'][1], "보트 대여", ha='center', va='center', fontsize=14)
    
    # 맑음 & 나쁨
    rect = plt.Rectangle((nodes['sunny_bad'][0]-0.8, nodes['sunny_bad'][1]-0.5), 1.6, 1,
                        color=node_colors['bad'], ec='black')
    ax.add_patch(rect)
    ax.text(nodes['sunny_bad'][0], nodes['sunny_bad'][1], "보트 대여\n불가", ha='center', va='center', fontsize=14)
    
    # 흐림 & 좋음
    rect = plt.Rectangle((nodes['cloudy_good'][0]-0.8, nodes['cloudy_good'][1]-0.5), 1.6, 1,
                        color=node_colors['good'], ec='black')
    ax.add_patch(rect)
    ax.text(nodes['cloudy_good'][0], nodes['cloudy_good'][1], "보트 대여", ha='center', va='center', fontsize=14)
    
    # 흐림 & 나쁨
    rect = plt.Rectangle((nodes['cloudy_bad'][0]-0.8, nodes['cloudy_bad'][1]-0.5), 1.6, 1,
                        color=node_colors['bad'], ec='black')
    ax.add_patch(rect)
    ax.text(nodes['cloudy_bad'][0], nodes['cloudy_bad'][1], "보트 대여\n불가", ha='center', va='center', fontsize=14)
    
    # 연결선 그리기
    # 루트 -> 맑음
    ax.plot([nodes['root'][0], nodes['sunny_child'][0]],
           [nodes['root'][1]-0.8, nodes['sunny_child'][1]+0.8], 'k-')
    ax.text((nodes['root'][0] + nodes['sunny_child'][0])/2 - 0.5,
           (nodes['root'][1] + nodes['sunny_child'][1])/2, "맑음", fontsize=12)
    
    # 루트 -> 흐림
    ax.plot([nodes['root'][0], nodes['cloudy_child'][0]],
           [nodes['root'][1]-0.8, nodes['cloudy_child'][1]+0.8], 'k-')
    ax.text((nodes['root'][0] + nodes['cloudy_child'][0])/2 + 0.5,
           (nodes['root'][1] + nodes['cloudy_child'][1])/2, "흐림", fontsize=12)
    
    # 맑음 -> 맑음 & 좋음
    ax.plot([nodes['sunny_child'][0], nodes['sunny_good'][0]],
           [nodes['sunny_child'][1]-0.8, nodes['sunny_good'][1]+0.5], 'k-')
    ax.text((nodes['sunny_child'][0] + nodes['sunny_good'][0])/2 - 0.5,
           (nodes['sunny_child'][1] + nodes['sunny_good'][1])/2, "좋음", fontsize=12)
    
    # 맑음 -> 맑음 & 나쁨
    ax.plot([nodes['sunny_child'][0], nodes['sunny_bad'][0]],
           [nodes['sunny_child'][1]-0.8, nodes['sunny_bad'][1]+0.5], 'k-')
    ax.text((nodes['sunny_child'][0] + nodes['sunny_bad'][0])/2 + 0.5,
           (nodes['sunny_child'][1] + nodes['sunny_bad'][1])/2, "나쁨", fontsize=12)
    
    # 흐림 -> 흐림 & 좋음
    ax.plot([nodes['cloudy_child'][0], nodes['cloudy_good'][0]],
           [nodes['cloudy_child'][1]-0.8, nodes['cloudy_good'][1]+0.5], 'k-')
    ax.text((nodes['cloudy_child'][0] + nodes['cloudy_good'][0])/2 - 0.5,
           (nodes['cloudy_child'][1] + nodes['cloudy_good'][1])/2, "좋음", fontsize=12)
    
    # 흐림 -> 흐림 & 나쁨
    ax.plot([nodes['cloudy_child'][0], nodes['cloudy_bad'][0]],
           [nodes['cloudy_child'][1]-0.8, nodes['cloudy_bad'][1]+0.5], 'k-')
    ax.text((nodes['cloudy_child'][0] + nodes['cloudy_bad'][0])/2 + 0.5,
           (nodes['cloudy_child'][1] + nodes['cloudy_bad'][1])/2, "나쁨", fontsize=12)
    
    plt.title('보트 렌탈 결정을 위한 의사결정 트리', fontsize=16)
    plt.tight_layout()
    plt.savefig('custom_decision_tree.png', dpi=300)
    plt.show()

# 샘플 데이터 생성
def create_sample_data():
    """의사결정 트리와 나이브 베이즈 모델을 위한 샘플 데이터 생성"""
    # 날씨: 맑음(1), 흐림(2)
    # 컨디션: 좋음(1), 나쁨(2)
    # 부상: 있음(1), 없음(2)
    # 결과(보트 대여): 예(1), 아니오(0)
    
    # 슬라이드에 나온 데이터를 기반으로 한 데이터 생성
    data = {
        '날씨': ['맑음', '맑음', '맑음', '맑음', '흐림', '흐림', '흐림', '흐림', '흐림', '흐림'],
        '컨디션': ['좋음', '좋음', '나쁨', '나쁨', '좋음', '좋음', '좋음', '나쁨', '나쁨', '나쁨'],
        '부상': ['아니오', '예', '아니오', '예', '아니오', '예', '아니오', '아니오', '예', '예'],
        '보트_대여': ['예', '예', '아니오', '아니오', '예', '예', '예', '예', '예', '아니오']
    }
    
    df = pd.DataFrame(data)
    return df

# 의사결정 트리 학습 및 시각화 - scikit-learn 사용
def train_and_visualize_decision_tree(df):
    """scikit-learn을 사용하여 의사결정 트리 학습 및 시각화"""
    # 특성과 타겟 분리
    X = df[['날씨', '컨디션', '부상']]
    y = df['보트_대여']
    
    # 범주형 특성을 숫자로 변환
    encoder = OrdinalEncoder()
    X_encoded = encoder.fit_transform(X)
    
    # 의사결정 트리 모델 생성 및 학습
    clf = tree.DecisionTreeClassifier(max_depth=3)
    clf.fit(X_encoded, y)
    
    # 의사결정 트리 시각화
    plt.figure(figsize=(12, 8))
    tree.plot_tree(clf, 
                  feature_names=['날씨', '컨디션', '부상'],
                  class_names=['아니오', '예'],
                  filled=True,
                  fontsize=10)
    plt.title('scikit-learn 의사결정 트리', fontsize=16)
    plt.tight_layout()
    plt.savefig('sklearn_decision_tree.png', dpi=300)
    plt.show()
    
    return clf, encoder

# 나이브 베이즈 모델 학습 및 예측
def train_naive_bayes(df):
    """나이브 베이즈 모델 학습 및 예측"""
    # 특성과 타겟 분리
    X = df[['날씨', '컨디션', '부상']]
    y = df['보트_대여'].map({'예': 1, '아니오': 0})
    
    # 범주형 특성을 숫자로 변환
    encoder = OrdinalEncoder()
    X_encoded = encoder.fit_transform(X)
    
    # 나이브 베이즈 모델 생성 및 학습
    clf = CategoricalNB()
    clf.fit(X_encoded, y)
    
    return clf, encoder

# 나이브 베이즈 예측 결과 시각화
def visualize_naive_bayes_predictions(clf, encoder, test_cases):
    """나이브 베이즈 모델의 예측 결과 시각화"""
    plt.figure(figsize=(10, 6))
    
    # 테스트 케이스 인코딩
    test_cases_encoded = encoder.transform(test_cases[['날씨', '컨디션', '부상']])
    
    # 예측 및 확률 계산
    predictions = clf.predict(test_cases_encoded)
    probabilities = clf.predict_proba(test_cases_encoded)
    
    # 결과 계산
    results = []
    for i, (_, row) in enumerate(test_cases.iterrows()):
        prob_yes = probabilities[i][1]
        prob_no = probabilities[i][0]
        odds = prob_yes / prob_no if prob_no > 0 else float('inf')
        prediction = '예' if predictions[i] == 1 else '아니오'
        
        results.append({
            '날씨': row['날씨'],
            '컨디션': row['컨디션'],
            '부상': row['부상'],
            'P(예)': f'{prob_yes:.3f}',
            'P(아니오)': f'{prob_no:.3f}',
            'Odds': f'{odds:.3f}',
            '예측': prediction
        })
    
    # 결과 데이터프레임 생성 및 출력
    results_df = pd.DataFrame(results)
    print("나이브 베이즈 예측 결과:")
    print(results_df)
    
    # 표 형태로 시각화
    fig, ax = plt.subplots(figsize=(12, len(results_df) * 0.5 + 1))
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=results_df.values,
                    colLabels=results_df.columns,
                    cellLoc='center',
                    loc='center',
                    colWidths=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    # 예측값에 따라 셀 색상 변경
    for i in range(len(results_df)):
        table[(i+1, 6)].set_facecolor('#4CAF50' if results_df.iloc[i]['예측'] == '예' else '#F44336')
        table[(i+1, 6)].set_text_props(color='white')
    
    plt.title('나이브 베이즈 예측 결과', fontsize=16)
    plt.tight_layout()
    plt.savefig('naive_bayes_predictions.png', dpi=300)
    plt.show()
    
    return results_df

# 의사결정 트리와 나이브 베이즈 비교
def compare_models(tree_clf, nb_clf, encoder, test_cases):
    """의사결정 트리와 나이브 베이즈 모델의 예측 비교"""
    # 테스트 케이스 인코딩
    test_cases_encoded = encoder.transform(test_cases[['날씨', '컨디션', '부상']])
    
    # 의사결정 트리 예측
    tree_predictions = tree_clf.predict(test_cases_encoded)
    tree_pred_labels = ['예' if p else '아니오' for p in tree_predictions]
    
    # 나이브 베이즈 예측
    nb_predictions = nb_clf.predict(test_cases_encoded)
    nb_probabilities = nb_clf.predict_proba(test_cases_encoded)
    nb_pred_labels = ['예' if p == 1 else '아니오' for p in nb_predictions]
    
    # 결과 계산
    results = []
    for i, (_, row) in enumerate(test_cases.iterrows()):
        prob_yes = nb_probabilities[i][1]
        prob_no = nb_probabilities[i][0]
        odds = prob_yes / prob_no if prob_no > 0 else float('inf')
        
        results.append({
            '날씨': row['날씨'],
            '컨디션': row['컨디션'],
            '부상': row['부상'],
            '의사결정 트리': tree_pred_labels[i],
            '나이브 베이즈': nb_pred_labels[i],
            'NB Odds': f'{odds:.3f}'
        })
    
    # 결과 데이터프레임 생성 및 출력
    results_df = pd.DataFrame(results)
    print("모델 비교 결과:")
    print(results_df)
    
    # 표 형태로 시각화
    fig, ax = plt.subplots(figsize=(12, len(results_df) * 0.5 + 1))
    ax.axis('tight')
    ax.axis('off')
    table = ax.table(cellText=results_df.values,
                    colLabels=results_df.columns,
                    cellLoc='center',
                    loc='center',
                    colWidths=[0.1, 0.1, 0.1, 0.15, 0.15, 0.15])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    # 예측값에 따라 셀 색상 변경
    for i in range(len(results_df)):
        table[(i+1, 3)].set_facecolor('#4CAF50' if results_df.iloc[i]['의사결정 트리'] == '예' else '#F44336')
        table[(i+1, 3)].set_text_props(color='white')
        table[(i+1, 4)].set_facecolor('#4CAF50' if results_df.iloc[i]['나이브 베이즈'] == '예' else '#F44336')
        table[(i+1, 4)].set_text_props(color='white')
    
    plt.title('의사결정 트리와 나이브 베이즈 모델 비교', fontsize=16)
    plt.tight_layout()
    plt.savefig('model_comparison.png', dpi=300)
    plt.show()
    
    return results_df

def main():
    print("의사결정 트리와 나이브 베이즈 모델 시각화 및 비교 프로그램")
    print("-" * 60)
    
    # 커스텀 의사결정 트리 시각화
    print("\n1. 보트 렌팅 의사결정 트리 시각화")
    visualize_custom_decision_tree()
    
    # 샘플 데이터 생성
    print("\n2. 샘플 데이터 생성")
    df = create_sample_data()
    print(df)
    
    # scikit-learn 의사결정 트리 학습 및 시각화
    print("\n3. scikit-learn 의사결정 트리 학습 및 시각화")
    tree_clf, encoder = train_and_visualize_decision_tree(df)
    
    # 나이브 베이즈 모델 학습
    print("\n4. 나이브 베이즈 모델 학습")
    nb_clf, _ = train_naive_bayes(df)
    
    # 테스트 케이스 생성
    print("\n5. 테스트 케이스 생성")
    test_cases = pd.DataFrame([
        {'날씨': '맑음', '컨디션': '좋음', '부상': '아니오'},  # 슬라이드 예제 1
        {'날씨': '흐림', '컨디션': '좋음', '부상': '아니오'},  # 슬라이드 예제 2
    ])
    print(test_cases)
    
    # 나이브 베이즈 예측 결과 시각화
    print("\n6. 나이브 베이즈 예측 결과 시각화")
    nb_results = visualize_naive_bayes_predictions(nb_clf, encoder, test_cases)
    
    # 모델 비교
    print("\n7. 의사결정 트리와 나이브 베이즈 모델 비교")
    compare_results = compare_models(tree_clf, nb_clf, encoder, test_cases)
    
    print("\n프로그램 종료")

if __name__ == "__main__":
    main() 