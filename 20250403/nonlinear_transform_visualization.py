import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
import platform

# 폰트 설정
if platform.system() == 'Windows':
    plt.rc('font', family='Malgun Gothic')
else:
    plt.rc('font', family='DejaVu Sans')

# 원형 데이터 생성
def generate_circle_data(n_samples=300, noise=0.1, random_state=42):
    """원형 경계를 가진 데이터 생성"""
    X, y = make_circles(n_samples=n_samples, noise=noise, factor=0.5, random_state=random_state)
    return X, y

# 데이터를 새로운 특성 공간으로 변환
def transform_to_squared_features(X):
    """(x, y) -> (x², y²) 변환"""
    X_transformed = np.zeros_like(X)
    X_transformed[:, 0] = X[:, 0]**2
    X_transformed[:, 1] = X[:, 1]**2
    return X_transformed

# 원 그리기 함수
def draw_circle(ax, radius=np.sqrt(3), center=(0, 0), **kwargs):
    """지정된 반지름과 중심을 가진 원 그리기"""
    theta = np.linspace(0, 2*np.pi, 100)
    x = center[0] + radius * np.cos(theta)
    y = center[1] + radius * np.sin(theta)
    ax.plot(x, y, **kwargs)

# 선형 경계 그리기 함수
def draw_linear_boundary(ax, **kwargs):
    """x' + y' = 3 선형 경계 그리기"""
    x = np.linspace(0, 3, 100)
    y = 3 - x
    ax.plot(x, y, **kwargs)
    
# 시각화 함수
def visualize_feature_transformation():
    """특성 변환 시각화"""
    # 데이터 생성
    X, y = generate_circle_data(n_samples=300, noise=0.05)
    
    # 데이터 변환
    X_transformed = transform_to_squared_features(X)
    
    # 시각화
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 원본 데이터 시각화
    ax1.set_title("Original Data Space (x, y)")
    scatter1 = ax1.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', alpha=0.7)
    
    # 원 경계 추가
    draw_circle(ax1, radius=np.sqrt(3), center=(0, 0), color='gray', linestyle='-', alpha=0.7, label="x² + y² = 3")
    
    # 축 설정
    ax1.set_xlim(-2.5, 2.5)
    ax1.set_ylim(-2.5, 2.5)
    ax1.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax1.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.legend()
    
    # 변환된 데이터 시각화
    ax2.set_title("Transformed Data Space (x², y²)")
    scatter2 = ax2.scatter(X_transformed[:, 0], X_transformed[:, 1], c=y, cmap='coolwarm', alpha=0.7)
    
    # 선형 경계 추가
    draw_linear_boundary(ax2, color='gray', linestyle='-', alpha=0.7, label="x' + y' = 3")
    
    # 축 설정
    ax2.set_xlim(0, 5)
    ax2.set_ylim(0, 5)
    ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax2.axvline(x=0, color='k', linestyle='-', alpha=0.3)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlabel("x' = x²")
    ax2.set_ylabel("y' = y²")
    ax2.legend()
    
    # 추가 정보 표시
    plt.suptitle("비선형 데이터의 특성 변환: (x, y) → (x², y²)", fontsize=16)
    
    # 그림 저장 및 표시
    plt.tight_layout()
    plt.savefig("nonlinear_transform_visualization.png")
    plt.show()
    
    print("원본 공간에서는 비선형 경계(원)가 필요하지만, 변환 후에는 선형 경계로 데이터를 분리할 수 있습니다.")
    print("변환: (x, y) → (x², y²)")
    print("경계 방정식: x² + y² = 3 → x' + y' = 3")

if __name__ == "__main__":
    visualize_feature_transformation() 