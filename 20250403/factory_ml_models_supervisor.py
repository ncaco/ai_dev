"""
공장 생산 라인 데이터 분석 - Supervisor 루트 노드

이 프로그램은 공장 생산 라인 데이터를 사용하여 Supervisor를 루트 노드로 하는 의사결정 트리를 구현합니다.
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

# Supervisor를 루트 노드로 하는 의사결정 트리 시각화
def visualize_supervisor_decision_tree():
    """Supervisor를 루트 노드로 하는 의사결정 트리 시각화"""
    fig, ax = plt.subplots(figsize=(14, 10))
    ax.set_xlim(0, 18)
    ax.set_ylim(0, 12)
    ax.axis('off')  # 축 숨기기
    
    # 노드 위치 정의
    nodes = {
        'root': (9, 10),               # Supervisor 노드
        'pat_child': (5, 8),           # Pat 노드
        'tom_child': (11, 8),          # Tom 노드
        'sally_child': (16, 8),        # Sally 노드
        'pat_overtime': (5, 6),        # Pat & Overtime 노드
        'pat_overtime_yes': (3, 4),    # Pat & Overtime=Yes 노드
        'pat_overtime_no': (7, 4),     # Pat & Overtime=No 노드
        'tom_output': (11, 6),         # Tom 결과 노드
        'sally_output': (16, 6),       # Sally 결과 노드
        'pat_yes_output': (3, 2),      # Pat & Overtime=Yes 결과 노드
        'pat_no_output': (7, 2)        # Pat & Overtime=No 결과 노드
    }
    
    # 노드 색상 정의
    node_colors = {
        'feature': '#F5F5F5',  # 연한 회색 (특징 노드)
        'high': '#4CAF50',     # 녹색 (High)
        'low': '#F44336'       # 빨간색 (Low)
    }
    
    # 루트 노드 (Supervisor)
    circle = plt.Circle(nodes['root'], 0.8, color=node_colors['feature'], ec='black')
    ax.add_patch(circle)
    ax.text(nodes['root'][0], nodes['root'][1], "Supervisor", ha='center', va='center', fontsize=14, weight='bold')
    
    # Pat 서브트리 (중간 노드)
    circle = plt.Circle(nodes['pat_child'], 0.8, color=node_colors['feature'], ec='black')
    ax.add_patch(circle)
    ax.text(nodes['pat_child'][0], nodes['pat_child'][1], "Pat", ha='center', va='center', fontsize=14, weight='bold')
    
    # Tom 노드 (리프 노드)
    rect = plt.Rectangle((nodes['tom_output'][0]-1.5, nodes['tom_output'][1]-0.5), 3, 1,
                        color=node_colors['low'], ec='black')
    ax.add_patch(rect)
    ax.text(nodes['tom_output'][0], nodes['tom_output'][1], "Output: Low (3/3)", ha='center', va='center', fontsize=12)
    
    # Sally 노드 (리프 노드)
    rect = plt.Rectangle((nodes['sally_output'][0]-1.5, nodes['sally_output'][1]-0.5), 3, 1,
                        color=node_colors['high'], ec='black')
    ax.add_patch(rect)
    ax.text(nodes['sally_output'][0], nodes['sally_output'][1], "Output: High (1/1)", ha='center', va='center', fontsize=12)
    
    # Pat & Overtime 노드
    circle = plt.Circle(nodes['pat_overtime'], 0.8, color=node_colors['feature'], ec='black')
    ax.add_patch(circle)
    ax.text(nodes['pat_overtime'][0], nodes['pat_overtime'][1], "Overtime", ha='center', va='center', fontsize=14, weight='bold')
    
    # Pat & Overtime=Yes 노드
    rect = plt.Rectangle((nodes['pat_yes_output'][0]-1.5, nodes['pat_yes_output'][1]-0.5), 3, 1,
                        color=node_colors['low'], ec='black')
    ax.add_patch(rect)
    ax.text(nodes['pat_yes_output'][0], nodes['pat_yes_output'][1], "Output: Low (2/2)", ha='center', va='center', fontsize=12)
    
    # Pat & Overtime=No 노드
    rect = plt.Rectangle((nodes['pat_no_output'][0]-1.5, nodes['pat_no_output'][1]-0.5), 3, 1,
                        color=node_colors['high'], ec='black')
    ax.add_patch(rect)
    ax.text(nodes['pat_no_output'][0], nodes['pat_no_output'][1], "Output: High (2/2)", ha='center', va='center', fontsize=12)
    
    # 연결선 그리기
    # Root -> Pat
    ax.plot([nodes['root'][0], nodes['pat_child'][0]],
           [nodes['root'][1]-0.8, nodes['pat_child'][1]+0.8], 'k-')
    ax.text((nodes['root'][0] + nodes['pat_child'][0])/2 - 1,
           (nodes['root'][1] + nodes['pat_child'][1])/2 + 0.3, "Pat", fontsize=12, weight='bold')
    
    # Root -> Tom
    ax.plot([nodes['root'][0], nodes['tom_child'][0]],
           [nodes['root'][1]-0.8, nodes['tom_output'][1]+0.5], 'k-')
    ax.text((nodes['root'][0] + nodes['tom_child'][0])/2,
           (nodes['root'][1] + nodes['tom_output'][1])/2 + 1, "Tom", fontsize=12, weight='bold')
    
    # Root -> Sally
    ax.plot([nodes['root'][0], nodes['sally_child'][0]],
           [nodes['root'][1]-0.8, nodes['sally_output'][1]+0.5], 'k-')
    ax.text((nodes['root'][0] + nodes['sally_child'][0])/2 + 2,
           (nodes['root'][1] + nodes['sally_output'][1])/2 + 1, "Sally", fontsize=12, weight='bold')
    
    # Pat -> Overtime
    ax.plot([nodes['pat_child'][0], nodes['pat_overtime'][0]],
           [nodes['pat_child'][1]-0.8, nodes['pat_overtime'][1]+0.8], 'k-')
    
    # Overtime -> Yes
    ax.plot([nodes['pat_overtime'][0], nodes['pat_overtime_yes'][0]],
           [nodes['pat_overtime'][1]-0.8, nodes['pat_yes_output'][1]+0.5], 'k-')
    ax.text((nodes['pat_overtime'][0] + nodes['pat_overtime_yes'][0])/2 - 0.5,
           (nodes['pat_overtime'][1] + nodes['pat_yes_output'][1])/2 + 0.3, "Yes", fontsize=12)
    
    # Overtime -> No
    ax.plot([nodes['pat_overtime'][0], nodes['pat_overtime_no'][0]],
           [nodes['pat_overtime'][1]-0.8, nodes['pat_no_output'][1]+0.5], 'k-')
    ax.text((nodes['pat_overtime'][0] + nodes['pat_overtime_no'][0])/2 + 0.5,
           (nodes['pat_overtime'][1] + nodes['pat_no_output'][1])/2 + 0.3, "No", fontsize=12)
    
    plt.title('Supervisor를 루트 노드로 하는 의사결정 트리', fontsize=16)
    plt.tight_layout()
    plt.savefig('supervisor_decision_tree.png', dpi=300)
    plt.show()

def main():
    print("공장 생산 라인 데이터 분석 프로그램 - Supervisor 루트 노드")
    print("-" * 60)
    
    # 공장 데이터 생성
    df = create_factory_data()
    
    # 데이터 정보 출력
    print_data_info(df)
    
    # Supervisor 의사결정 트리 시각화
    print("\n1. Supervisor를 루트 노드로 하는 의사결정 트리 시각화")
    visualize_supervisor_decision_tree()
    
    print("\n프로그램 종료")

if __name__ == "__main__":
    main() 