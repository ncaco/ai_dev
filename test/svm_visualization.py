import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.datasets import make_blobs
import platform

# 폰트 설정
if platform.system() == 'Windows':
    plt.rc('font', family='Malgun Gothic')
else:
    plt.rc('font', family='DejaVu Sans')

# 데이터 생성 함수
def generate_linearly_separable_data(n_samples=50, centers=None, random_state=42):
    """선형 분리 가능한 데이터 생성"""
    if centers is None:
        centers = [[-2, -2], [2, 2]]
    
    X, y = make_blobs(n_samples=n_samples, centers=centers, 
                     cluster_std=1.0, random_state=random_state)
    return X, y

# 결정 경계와 마진 시각화 함수
def plot_svm_decision_boundary(ax, X, y, svm_model, title=None):
    """SVM 결정 경계와 마진 시각화"""
    # 메시 그리드 생성
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    h = 0.02  # 그리드 간격
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # 결정 경계 계산
    Z = svm_model.decision_function(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # 결정 경계 (Z=0) 그리기
    ax.contour(xx, yy, Z, colors='k', levels=[-1, 0, 1], 
              linestyles=['--', '-', '--'], linewidths=[1, 2, 1])
    
    # 영역 색상 채우기
    ax.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.3, levels=[-100, -1, 0, 1, 100])
    
    # 서포트 벡터 강조
    ax.scatter(svm_model.support_vectors_[:, 0], svm_model.support_vectors_[:, 1],
              s=100, linewidth=1, facecolors='none', edgecolors='k')
    
    # 데이터 포인트 그리기
    ax.scatter(X[y == 0][:, 0], X[y == 0][:, 1], c='w', edgecolors='black', s=40, label='Class 0')
    ax.scatter(X[y == 1][:, 0], X[y == 1][:, 1], c='black', s=40, label='Class 1')
    
    # 축과 타이틀 설정
    ax.set_xlabel('$x_1$', fontsize=14)
    ax.set_ylabel('$x_2$', fontsize=14)
    if title:
        ax.set_title(title, fontsize=16)
    
    # 범례 추가
    ax.legend()
    
    # 플롯 영역 설정
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())

# 다양한 C 값에 따른 SVM 시각화
def visualize_svm_margins():
    """다양한 C 값에 따른 SVM 마진 시각화"""
    # 데이터 생성
    X, y = generate_linearly_separable_data(n_samples=80, random_state=42)
    
    # 다양한 C 값으로 SVM 모델 훈련
    C_values = [0.1, 1.0, 10.0]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # 각 C 값에 대해 시각화
    for i, C in enumerate(C_values):
        svm_model = SVC(kernel='linear', C=C)
        svm_model.fit(X, y)
        
        # 결정 경계 시각화
        title = f'C = {C}'
        plot_svm_decision_boundary(axes[i], X, y, svm_model, title)
        
        # 마진 정보 추가
        n_sv = len(svm_model.support_vectors_)
        w_norm = np.linalg.norm(svm_model.coef_[0])
        margin = 2 / w_norm
        axes[i].text(0.05, 0.95, f'Support Vectors: {n_sv}\nMargin Width: {margin:.2f}',
                    transform=axes[i].transAxes, fontsize=12,
                    verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 전체 타이틀
    plt.suptitle('SVM Margin Visualization with Different C Values', fontsize=18)
    plt.tight_layout()
    fig.subplots_adjust(top=0.88)
    
    # 저장 및 표시
    plt.savefig('svm_margins_visualization.png', dpi=200, bbox_inches='tight')
    plt.show()
    
    print("SVM Margin Explanation:")
    print("- The margin is the distance between the decision boundary and the closest data points (support vectors).")
    print("- Small C: Larger margin, may allow misclassifications (soft margin).")
    print("- Large C: Smaller margin, tries to classify all points correctly (hard margin).")
    print("- SVM objective: Maximize margin while controlling misclassifications.")

# 실행
if __name__ == "__main__":
    visualize_svm_margins() 