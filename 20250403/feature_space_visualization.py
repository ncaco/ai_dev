"""
논리 모델(Logical Models)의 특징 공간 시각화

이 프로그램은 슬라이드에서 보여준 특징 공간(Feature Space)을 
matplotlib을 사용하여 시각적으로 표현합니다.

슬라이드 내용:
- 특징 공간의 각 영역이 다른 분류 결과를 가질 수 있음
- 비일관적(inconsistent) 영역과 분류 불가능(incomplete) 영역 표시
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import matplotlib.font_manager as fm
import platform

# 한글 폰트 설정
def set_korean_font():
    system = platform.system()
    
    if system == 'Windows':
        # 윈도우에서 많이 사용하는 한글 폰트들
        font_list = ['Malgun Gothic', '맑은 고딕', 'Gulim', '굴림', 'Dotum', '돋움', 'Batang', '바탕']
    elif system == 'Darwin':  # macOS
        font_list = ['AppleGothic', 'Apple SD Gothic Neo', 'Nanum Gothic']
    else:  # Linux 등
        font_list = ['NanumGothic', 'NanumGothicOTF', 'NanumBarunGothic', 'NanumBarunGothicOTF']
    
    font_found = False
    for font in font_list:
        if any(f.name == font for f in fm.fontManager.ttflist):
            plt.rcParams['font.family'] = font
            print(f"한글 폰트 '{font}'를 사용합니다.")
            font_found = True
            break
    
    if not font_found:
        print("사용 가능한 한글 폰트를 찾을 수 없습니다. 영문으로 표시됩니다.")
        
    # 폰트 경로를 직접 지정하려면 아래 코드를 사용할 수 있습니다
    # font_path = '/경로/NanumGothic.ttf'
    # font_prop = fm.FontProperties(fname=font_path)
    # plt.rcParams['font.family'] = font_prop.get_name()
    
    # 마이너스 기호 깨짐 방지
    plt.rcParams['axes.unicode_minus'] = False

def create_feature_space_visualization():
    """
    특징 공간을 matplotlib을 사용하여 시각화합니다.
    
    슬라이드와 유사한 형태로 특징 공간을 그리고 각 영역에 
    분류 결과와 이메일 개수를 표시합니다.
    """
    # 플롯 설정
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 축 설정
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.5, 1.5)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(['Peter = 0', 'Peter = 1'])
    ax.set_yticklabels(['Lottery = 0', 'Lottery = 1'])
    ax.set_xlabel('Peter')
    ax.set_ylabel('Lottery')
    
    # 격자 설정
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # 색상 정의
    colors = {
        'spam': '#FF6B6B',  # 빨간색 계열 (spam)
        'ham': '#4ECDC4',   # 청록색 계열 (ham)
        'contradiction': '#FFD166',  # 노란색 계열 (모순)
        'no_prediction': '#CCCCCC'   # 회색 (예측 없음)
    }
    
    # 분류 결과에 따른 영역 색칠
    # (x, y, width, height, color)
    # Lottery=1, Peter=0: spam
    ax.add_patch(plt.Rectangle((0-0.5, 1-0.5), 1, 1, color=colors['spam'], alpha=0.7))
    # Lottery=1, Peter=1: contradiction (여기선 spam 우선)
    ax.add_patch(plt.Rectangle((1-0.5, 1-0.5), 1, 1, color=colors['contradiction'], alpha=0.7))
    # Lottery=0, Peter=1: ham
    ax.add_patch(plt.Rectangle((1-0.5, 0-0.5), 1, 1, color=colors['ham'], alpha=0.7))
    # Lottery=0, Peter=0: no prediction
    ax.add_patch(plt.Rectangle((0-0.5, 0-0.5), 1, 1, color=colors['no_prediction'], alpha=0.7))
    
    # 영역별 텍스트 추가
    # (x, y, text)
    # Lottery=1, Peter=0: spam
    ax.text(0, 1, "spam: 20\nham: 5", ha='center', va='center', fontsize=12)
    # Lottery=1, Peter=1: contradiction
    ax.text(1, 1, "spam: 20\nham: 5\n(모순)", ha='center', va='center', fontsize=12)
    # Lottery=0, Peter=1: ham
    ax.text(1, 0, "spam: 10\nham: 5", ha='center', va='center', fontsize=12)
    # Lottery=0, Peter=0: no prediction
    ax.text(0, 0, "spam: 20\nham: 40\n(예측 없음)", ha='center', va='center', fontsize=12)
    
    # 제목 추가
    ax.set_title('특징 공간 시각화 (Feature Space Visualization)', fontsize=16)
    
    # 범례 추가
    legend_elements = [
        Patch(facecolor=colors['spam'], edgecolor='black', alpha=0.7, label='Spam'),
        Patch(facecolor=colors['ham'], edgecolor='black', alpha=0.7, label='Ham'),
        Patch(facecolor=colors['contradiction'], edgecolor='black', alpha=0.7, label='모순(Contradiction)'),
        Patch(facecolor=colors['no_prediction'], edgecolor='black', alpha=0.7, label='예측 없음(No Prediction)')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    # 규칙 설명 추가
    rule_text = (
        "규칙 1: if Lottery = 1 then Y = spam\n"
        "규칙 2: if Peter = 1 then Y = ham\n\n"
        "- 비일관성(Inconsistency): Lottery=1 & Peter=1\n"
        "- 불완전성(Incompleteness): Lottery=0 & Peter=0"
    )
    plt.figtext(0.15, 0.02, rule_text, fontsize=12, bbox=dict(facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig('feature_space_visualization.png', dpi=300)
    print("시각화 이미지가 'feature_space_visualization.png'로 저장되었습니다.")
    plt.show()

def visualize_decision_tree():
    """
    슬라이드에 있는 의사결정 트리를 matplotlib을 사용하여 시각화합니다.
    """
    # 플롯 설정
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')  # 축 숨기기
    
    # 노드 위치 정의
    nodes = {
        'root': (5, 8),         # Viagra 노드
        'left_child': (3, 5),   # Lottery 노드 (Viagra=0)
        'right_child': (7, 5),  # Spam 노드 (Viagra=1)
        'left_left': (2, 2),    # Ham 노드 (Viagra=0, Lottery=0)
        'left_right': (4, 2)    # Spam 노드 (Viagra=0, Lottery=1)
    }
    
    # 노드 색상 정의
    node_colors = {
        'feature': '#FFFFFF',  # 흰색 (특징 노드)
        'spam': '#FF6B6B',     # 빨간색 (스팸 노드)
        'ham': '#4ECDC4'       # 청록색 (햄 노드)
    }
    
    # 노드 그리기
    # 루트 노드 (Viagra)
    circle = plt.Circle(nodes['root'], 0.8, color=node_colors['feature'], ec='black')
    ax.add_patch(circle)
    ax.text(nodes['root'][0], nodes['root'][1], "'Viagra'", ha='center', va='center', fontsize=12)
    
    # 왼쪽 자식 노드 (Lottery, Viagra=0)
    circle = plt.Circle(nodes['left_child'], 0.8, color=node_colors['feature'], ec='black')
    ax.add_patch(circle)
    ax.text(nodes['left_child'][0], nodes['left_child'][1], "'lottery'", ha='center', va='center', fontsize=12)
    
    # 오른쪽 자식 노드 (Spam, Viagra=1)
    rectangle = plt.Rectangle((nodes['right_child'][0]-1, nodes['right_child'][1]-0.5), 2, 1, 
                             color=node_colors['spam'], ec='black')
    ax.add_patch(rectangle)
    ax.text(nodes['right_child'][0], nodes['right_child'][1], "spam: 20\nham: 5", ha='center', va='center', fontsize=12)
    
    # 왼쪽-왼쪽 노드 (Ham, Viagra=0, Lottery=0)
    rectangle = plt.Rectangle((nodes['left_left'][0]-1, nodes['left_left'][1]-0.5), 2, 1, 
                             color=node_colors['ham'], ec='black')
    ax.add_patch(rectangle)
    ax.text(nodes['left_left'][0], nodes['left_left'][1], "spam: 20\nham: 40", ha='center', va='center', fontsize=12)
    
    # 왼쪽-오른쪽 노드 (Spam, Viagra=0, Lottery=1)
    rectangle = plt.Rectangle((nodes['left_right'][0]-1, nodes['left_right'][1]-0.5), 2, 1, 
                             color=node_colors['spam'], ec='black')
    ax.add_patch(rectangle)
    ax.text(nodes['left_right'][0], nodes['left_right'][1], "spam: 10\nham: 5", ha='center', va='center', fontsize=12)
    
    # 엣지 그리기 (노드 연결)
    # Root → 왼쪽 자식 (Viagra=0)
    ax.plot([nodes['root'][0], nodes['left_child'][0]], 
            [nodes['root'][1]-0.8, nodes['left_child'][1]+0.8], 'k-')
    ax.text((nodes['root'][0] + nodes['left_child'][0])/2 - 0.5, 
            (nodes['root'][1] + nodes['left_child'][1])/2, "=0", fontsize=12)
    
    # Root → 오른쪽 자식 (Viagra=1)
    ax.plot([nodes['root'][0], nodes['right_child'][0]], 
            [nodes['root'][1]-0.8, nodes['right_child'][1]+0.5], 'k-')
    ax.text((nodes['root'][0] + nodes['right_child'][0])/2 + 0.5, 
            (nodes['root'][1] + nodes['right_child'][1])/2, "=1", fontsize=12)
    
    # 왼쪽 자식 → 왼쪽-왼쪽 (Lottery=0)
    ax.plot([nodes['left_child'][0], nodes['left_left'][0]], 
            [nodes['left_child'][1]-0.8, nodes['left_left'][1]+0.5], 'k-')
    ax.text((nodes['left_child'][0] + nodes['left_left'][0])/2 - 0.5, 
            (nodes['left_child'][1] + nodes['left_left'][1])/2, "=0", fontsize=12)
    
    # 왼쪽 자식 → 왼쪽-오른쪽 (Lottery=1)
    ax.plot([nodes['left_child'][0], nodes['left_right'][0]], 
            [nodes['left_child'][1]-0.8, nodes['left_right'][1]+0.5], 'k-')
    ax.text((nodes['left_child'][0] + nodes['left_right'][0])/2 + 0.5, 
            (nodes['left_child'][1] + nodes['left_right'][1])/2, "=1", fontsize=12)
    
    # 제목 추가
    ax.set_title('의사결정 트리 시각화 (Decision Tree Visualization)', fontsize=16)
    
    plt.tight_layout()
    plt.savefig('decision_tree_visualization.png', dpi=300)
    print("의사결정 트리 이미지가 'decision_tree_visualization.png'로 저장되었습니다.")
    plt.show()

if __name__ == "__main__":
    print("특징 공간 시각화 프로그램을 시작합니다...")
    
    try:
        import matplotlib
        
        # 한글 폰트 설정
        set_korean_font()
        
        print("특징 공간을 그래픽으로 시각화합니다.")
        create_feature_space_visualization()
        visualize_decision_tree()
    except ImportError:
        print("matplotlib 라이브러리가 설치되어 있지 않습니다.")
        print("pip install matplotlib 명령으로 설치한 후 다시 실행해주세요.")
    except Exception as e:
        print(f"시각화 과정에서 오류가 발생했습니다: {e}")
        
    print("프로그램을 종료합니다.") 