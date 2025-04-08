
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
import matplotlib.patches as patches
import matplotlib.font_manager as fm
import platform

# 폰트 설정
if platform.system() == 'Windows':
    # 윈도우의 경우 맑은 고딕 사용
    plt.rc('font', family='Malgun Gothic')
else:
    # 다른 OS의 경우 기본 폰트 사용
    plt.rc('font', family='DejaVu Sans')

# 데이터 생성
def generate_data():
    """코사인 함수 데이터 생성"""
    X = np.linspace(-1, 1, 100).reshape(-1, 1)  # -1에서 1까지의 100개 점
    y = np.cos(np.pi * X).ravel()  # y = cos(πx)
    return X, y

# 수동 예측 함수 (슬라이드에 있는 회귀 트리 모델)
def manual_predict(X):
    """수동으로 정의한 회귀 트리 모델 예측"""
    predictions = np.zeros(X.shape[0])
    
    for i, x in enumerate(X):
        if x < 0:
            # 왼쪽 리프 노드: ŷ = 2x+1
            predictions[i] = 2 * x + 1
        else:
            # 오른쪽 리프 노드: ŷ = -2x+1
            predictions[i] = -2 * x + 1
    
    return predictions

# 두 시각화 생성
def visualize_regression_tree():
    """회귀 트리 모델과 예측 결과 시각화"""
    # 데이터 생성
    X, y = generate_data()
    X_flat = X.ravel()
    
    # 수동 예측
    y_manual = manual_predict(X_flat)
    
    # scikit-learn 회귀 트리
    regressor = DecisionTreeRegressor(max_depth=1)
    regressor.fit(X, y)
    y_pred = regressor.predict(X)
    
    # 그림 1: 실제 트리 구조 시각화
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 트리 구조 그리기
    ax1.set_title('Regression Tree Model Structure')
    ax1.set_xlim(-1, 9)
    ax1.set_ylim(-1, 9)
    ax1.axis('off')
    
    # 노드 그리기
    root_node = patches.Rectangle((4, 7), 2, 1, linewidth=1, edgecolor='black', facecolor='white')
    left_node = patches.Rectangle((1, 3), 2, 1.5, linewidth=1, edgecolor='black', facecolor='lightgray')
    right_node = patches.Rectangle((7, 3), 2, 1.5, linewidth=1, edgecolor='black', facecolor='lightgray')
    
    ax1.add_patch(root_node)
    ax1.add_patch(left_node)
    ax1.add_patch(right_node)
    
    # 노드 내용 텍스트
    ax1.text(5, 7.5, 'x', horizontalalignment='center')
    ax1.text(2, 3.75, 'y = 2x+1', horizontalalignment='center')
    ax1.text(8, 3.75, 'y = -2x+1', horizontalalignment='center')
    
    # 선 연결
    ax1.plot([4.5, 2], [7, 4.5], 'k-')
    ax1.plot([5.5, 8], [7, 4.5], 'k-')
    
    # 분기 조건 텍스트
    ax1.text(3, 5.5, '<0', fontsize=12)
    ax1.text(7, 5.5, '≥0', fontsize=12)
    
    # 그림 2: 함수 근사 시각화
    ax2.set_title('Regression Tree Function Approximation')
    
    # 코사인 함수와 회귀 트리 예측
    ax2.plot(X_flat, y, 'b-', label='Actual: y = cos(πx)')
    ax2.plot(X_flat, y_manual, 'r-', label='Tree Approximation')
    
    # 축 설정
    ax2.set_xlim(-1, 1)
    ax2.set_ylim(-1, 1)
    ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax2.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlabel('x')
    ax2.set_ylabel('y')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig('regression_tree_visualization.png')
    plt.show()

if __name__ == "__main__":
    visualize_regression_tree() 