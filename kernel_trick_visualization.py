import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import make_circles
import platform

# 폰트 설정
if platform.system() == 'Windows':
    plt.rc('font', family='Malgun Gothic')
else:
    plt.rc('font', family='DejaVu Sans')

# 데이터 생성
def generate_circle_data(n_samples=300, noise=0.1, factor=0.5, random_state=42):
    """원형 경계를 가진 데이터 생성"""
    X, y = make_circles(n_samples=n_samples, noise=noise, factor=factor, random_state=random_state)
    return X, y

# 특성 변환 함수 (2D -> 3D)
def transform_to_kernel_space(X):
    """(x1, x2) -> (x1^2, x2^2, sqrt(2)*x1*x2) 변환"""
    X_transformed = np.zeros((X.shape[0], 3))
    X_transformed[:, 0] = X[:, 0]**2  # x1^2
    X_transformed[:, 1] = X[:, 1]**2  # x2^2
    X_transformed[:, 2] = np.sqrt(2) * X[:, 0] * X[:, 1]  # sqrt(2)*x1*x2
    return X_transformed

# 원 경계 그리기
def draw_circle(ax, radius=1, center=(0, 0), **kwargs):
    """지정된 반지름과 중심을 가진 원 그리기"""
    theta = np.linspace(0, 2*np.pi, 100)
    x = center[0] + radius * np.cos(theta)
    y = center[1] + radius * np.sin(theta)
    ax.plot(x, y, **kwargs)

# 평면 그리기 (3D 공간에서의 선형 경계)
def draw_plane(ax, **kwargs):
    """3D 공간에서 x^2 + y^2 = 1을 표현하는 평면 그리기"""
    # 표시할 평면의 영역 설정
    x_min, x_max = 0, 2.5
    y_min, y_max = 0, 2.5
    
    # 평면의 모서리 점들 생성
    points = np.array([
        [x_min, y_min, 0],
        [x_max, y_min, 0],
        [x_max, y_max, 0],
        [x_min, y_max, 0]
    ])
    
    # 평면을 구성하는 삼각형 정의
    triangles = [[0, 1, 2], [0, 2, 3]]
    
    # 평면 그리기
    ax.plot_trisurf(points[:, 0], points[:, 1], points[:, 2], 
                   triangles=triangles, color='gray', alpha=0.3)
    
    # 추가로 격자선 그리기
    num_lines = 5
    for i in range(num_lines + 1):
        x_pos = x_min + (x_max - x_min) * i / num_lines
        y_pos = y_min + (y_max - y_min) * i / num_lines
        
        # x축 방향 선
        ax.plot([x_min, x_max], [y_pos, y_pos], [0, 0], 'k-', alpha=0.2)
        # y축 방향 선
        ax.plot([x_pos, x_pos], [y_min, y_max], [0, 0], 'k-', alpha=0.2)

# 커널 트릭 시각화
def visualize_kernel_trick():
    """커널 트릭 시각화"""
    # 데이터 생성
    X, y = generate_circle_data(n_samples=180, noise=0.05, factor=0.5)
    
    # 원본 데이터에서 클래스 분리
    X_class0 = X[y == 0]
    X_class1 = X[y == 1]
    
    # 데이터 변환
    X_transformed = transform_to_kernel_space(X)
    X_transformed_class0 = X_transformed[y == 0]
    X_transformed_class1 = X_transformed[y == 1]
    
    # 원본 공간 그래프만 시각화 (슬라이드와 일치하도록)
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 뷰 각도 조정
    ax.view_init(elev=20, azim=30)
    
    # 변환된 데이터 플롯
    ax.scatter(X_transformed_class0[:, 0], X_transformed_class0[:, 1], X_transformed_class0[:, 2], 
              c='w', edgecolor='black', marker='o', s=40, label='Class 0')
    ax.scatter(X_transformed_class1[:, 0], X_transformed_class1[:, 1], X_transformed_class1[:, 2], 
              c='black', marker='o', s=40, label='Class 1')
    
    # 평면 경계 추가
    draw_plane(ax)
    
    # 축 설정
    ax.set_xlim(0, 2.5)
    ax.set_ylim(0, 2.5)
    ax.set_zlim(-3, 3)
    ax.set_xlabel("$x_1^2$", fontsize=14)
    ax.set_ylabel("$x_2^2$", fontsize=14)
    ax.set_zlabel("$\\sqrt{2}x_1x_2$", fontsize=14)
    
    # 축 눈금 간격 및 위치 조정
    ax.set_xticks([0, 0.5, 1, 1.5, 2, 2.5])
    ax.set_yticks([0, 0.5, 1, 1.5, 2, 2.5])
    ax.set_zticks([-3, -2, -1, 0, 1, 2, 3])
    
    # 격자선 제거
    ax.grid(False)
    
    # 배경색 설정
    ax.set_facecolor('white')
    
    # 범례 제거 (슬라이드와 맞춤)
    ax.get_legend().remove()
    
    # 타이틀 제거 (슬라이드와 맞춤)
    ax.set_title('')
    
    # 전체 타이틀 제거
    plt.suptitle('')
    
    # 여백 조정
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    
    # 저장 및 표시
    plt.savefig("kernel_trick_visualization_improved.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Kernel Trick Visualized:")
    print("- Original nonlinear boundary x² + y² = 1")
    print("- Transformed to 3D space (x₁², x₂², √2x₁x₂)")
    print("- Linear plane separates classes in transformed space")

if __name__ == "__main__":
    visualize_kernel_trick() 