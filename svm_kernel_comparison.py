import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.datasets import make_circles, make_moons
import platform

# 폰트 설정
if platform.system() == 'Windows':
    plt.rc('font', family='Malgun Gothic')
else:
    plt.rc('font', family='DejaVu Sans')

# 비선형 데이터 생성 함수들
def generate_circle_data(n_samples=100, noise=0.1, random_state=42):
    """원형 경계를 가진 데이터 생성"""
    X, y = make_circles(n_samples=n_samples, noise=noise, factor=0.5, random_state=random_state)
    return X, y

def generate_moon_data(n_samples=100, noise=0.1, random_state=42):
    """달 모양의 비선형 경계를 가진 데이터 생성"""
    X, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
    return X, y

# 결정 경계 시각화 함수
def plot_decision_boundary(ax, X, y, svm_model, title=None):
    """SVM 결정 경계 시각화"""
    # 메시 그리드 생성
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    h = 0.02  # 그리드 간격
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    # 결정 경계 계산
    Z = svm_model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    # 영역 색상 채우기
    ax.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.3)
    
    # 결정 경계 그리기
    ax.contour(xx, yy, Z, colors='k', linewidths=1)
    
    # 데이터 포인트 그리기
    ax.scatter(X[y == 0][:, 0], X[y == 0][:, 1], c='w', edgecolors='black', s=30, label='Class 0')
    ax.scatter(X[y == 1][:, 0], X[y == 1][:, 1], c='black', s=30, label='Class 1')
    
    # 서포트 벡터 표시
    ax.scatter(svm_model.support_vectors_[:, 0], svm_model.support_vectors_[:, 1],
              s=80, facecolors='none', edgecolors='k', alpha=0.5)
    
    # 축과 타이틀 설정
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    if title:
        ax.set_title(title)
    
    # 범례 추가
    ax.legend(loc='lower right')
    
    # 플롯 영역 설정
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())

# 다양한 커널 함수 비교 시각화
def visualize_svm_kernels():
    """다양한 SVM 커널 함수 비교 시각화"""
    # 두 가지 데이터셋 생성
    X_circle, y_circle = generate_circle_data(n_samples=100, noise=0.1)
    X_moon, y_moon = generate_moon_data(n_samples=100, noise=0.1)
    
    datasets = [
        (X_circle, y_circle, 'Concentric Circles Dataset'),
        (X_moon, y_moon, 'Two Moons Dataset')
    ]
    
    # 사용할 커널들과 매개변수
    kernels = [
        ('linear', {}),
        ('poly', {'degree': 3, 'C': 1.0}),
        ('rbf', {'gamma': 0.5, 'C': 1.0}),
        ('sigmoid', {'gamma': 0.5, 'C': 1.0})
    ]
    
    # 시각화
    fig, axes = plt.subplots(len(datasets), len(kernels), figsize=(16, 8))
    
    # 각 데이터셋과 커널 조합에 대해 시각화
    for i, (X, y, dataset_name) in enumerate(datasets):
        for j, (kernel_name, params) in enumerate(kernels):
            # SVM 모델 훈련
            svm_model = SVC(kernel=kernel_name, **params)
            svm_model.fit(X, y)
            
            # 정확도 계산
            accuracy = svm_model.score(X, y)
            title = f'{kernel_name.capitalize()} Kernel\nAccuracy: {accuracy:.2f}'
            
            # 결정 경계 시각화
            ax = axes[i, j]
            plot_decision_boundary(ax, X, y, svm_model, title)
            
            # 첫 번째 열에만 데이터셋 이름 추가
            if j == 0:
                ax.text(-0.2, 0.5, dataset_name, rotation=90, 
                        transform=ax.transAxes, fontsize=14, 
                        horizontalalignment='center', verticalalignment='center')
    
    # 전체 타이틀
    plt.suptitle('Comparison of SVM Kernel Functions on Nonlinear Datasets', fontsize=16)
    plt.tight_layout()
    fig.subplots_adjust(top=0.9, left=0.08)
    
    # 저장 및 표시
    plt.savefig('svm_kernel_comparison.png', dpi=200, bbox_inches='tight')
    plt.show()
    
    print("SVM Kernel Comparison:")
    print("- Linear kernel: Performs poorly on nonlinear datasets")
    print("- Polynomial kernel: Can capture nonlinear patterns with higher degrees")
    print("- RBF kernel: Often works well for many nonlinear problems")
    print("- Sigmoid kernel: Inspired by neural networks activation function")
    print("\nNote: The kernel trick allows SVM to operate in high-dimensional spaces")
    print("without explicitly computing the transformation, making it efficient for")
    print("complex pattern recognition tasks.")

# 실행
if __name__ == "__main__":
    visualize_svm_kernels() 