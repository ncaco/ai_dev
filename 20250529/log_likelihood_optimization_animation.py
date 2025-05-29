import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'

# 시그모이드 함수 정의
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# 로그우도함수 정의
def log_likelihood(X, y, theta):
    """
    로지스틱 회귀의 로그우도함수 계산
    X: 특성 행렬 (m x n)
    y: 레이블 벡터 (m x 1)
    theta: 파라미터 벡터 (n x 1)
    """
    m = len(y)
    h = sigmoid(X.dot(theta))
    
    # 수치적 안정성을 위한 작은 값 추가 (로그에 0이 들어가는 것 방지)
    epsilon = 1e-5
    
    # 로그우도함수 계산
    ll = np.sum(y * np.log(h + epsilon) + (1 - y) * np.log(1 - h + epsilon))
    
    # 평균 로그우도
    return ll / m

# 로그우도함수의 그래디언트 (미분값)
def log_likelihood_gradient(X, y, theta):
    """
    로그우도함수의 그래디언트 계산
    경사상승법을 위한 미분값
    """
    m = len(y)
    h = sigmoid(X.dot(theta))
    
    # 로그우도함수를 theta로 미분한 값
    return X.T.dot(y - h) / m

# 경사상승법으로 로그우도함수 최대화 (전체 최적화 과정 기록)
def gradient_ascent(X, y, theta_init, learning_rate=0.1, iterations=1000):
    """
    경사상승법을 통한 로그우도함수 최대화
    모든 단계의 파라미터와 로그우도 값을 기록하여 반환
    """
    m, n = X.shape
    theta_history = np.zeros((iterations, n))
    ll_history = np.zeros(iterations)
    
    # 초기 파라미터 설정
    theta = theta_init.copy()
    
    for i in range(iterations):
        # 그래디언트 계산
        gradient = log_likelihood_gradient(X, y, theta)
        
        # 파라미터 업데이트 (경사상승법이므로 그래디언트 방향으로 이동)
        theta = theta + learning_rate * gradient
        
        # 로그우도와 파라미터 기록
        ll_history[i] = log_likelihood(X, y, theta)
        theta_history[i] = theta.ravel()
    
    return theta, ll_history, theta_history

# 결정 경계 계산 함수
def compute_decision_boundary(theta, xx, yy):
    """
    주어진 파라미터에 대한 결정 경계 계산
    """
    # 그리드 포인트에 대한 예측 확률
    Z = sigmoid(np.c_[np.ones(xx.ravel().shape), xx.ravel(), yy.ravel()].dot(theta))
    Z = Z.reshape(xx.shape)
    return Z

# 메인 코드
if __name__ == "__main__":
    # 1. 이진 분류 데이터 생성
    np.random.seed(42)
    X, y = make_classification(n_samples=100, n_features=2, n_redundant=0,
                              n_informative=2, random_state=42,
                              n_clusters_per_class=1)
    
    # 2. 데이터 시각화
    plt.figure(figsize=(10, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral, edgecolors='k')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Binary Classification Dataset')
    plt.colorbar()
    plt.grid(True)
    plt.show()
    
    # 3. 데이터 준비
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # 4. 바이어스 항 추가
    X_train_b = np.c_[np.ones(X_train.shape[0]), X_train]
    X_test_b = np.c_[np.ones(X_test.shape[0]), X_test]
    
    # 5. 초기 파라미터 설정
    theta_init = np.zeros(X_train_b.shape[1])
    
    # 6. 로그우도함수 최대화 (경사상승법)
    iterations = 50  # 애니메이션을 위해 반복 횟수를 줄임
    learning_rate = 0.1
    
    print("초기 로그우도:", log_likelihood(X_train_b, y_train, theta_init))
    
    # 경사상승법 실행
    theta_opt, ll_history, theta_history = gradient_ascent(X_train_b, y_train, 
                                                        theta_init, learning_rate, iterations)
    
    print("최종 로그우도:", log_likelihood(X_train_b, y_train, theta_opt))
    print("최적화된 파라미터:", theta_opt)
    
    # 7. 로그우도 변화 시각화
    plt.figure(figsize=(10, 6))
    plt.plot(range(iterations), ll_history)
    plt.xlabel('Iterations')
    plt.ylabel('Log-Likelihood')
    plt.title('Log-Likelihood Over Iterations')
    plt.grid(True)
    plt.show()
    
    # 8. 파라미터 공간에서 로그우도함수 시각화 (2D 등고선)
    # 바이어스(theta[0])를 고정하고 나머지 두 파라미터에 대한 로그우도 시각화
    theta0_fixed = theta_opt[0]
    theta1_range = np.linspace(theta_opt[1] - 3, theta_opt[1] + 3, 100)
    theta2_range = np.linspace(theta_opt[2] - 3, theta_opt[2] + 3, 100)
    
    # 로그우도 그리드 계산
    ll_grid = np.zeros((len(theta1_range), len(theta2_range)))
    
    for i, t1 in enumerate(theta1_range):
        for j, t2 in enumerate(theta2_range):
            theta_test = np.array([theta0_fixed, t1, t2])
            ll_grid[i, j] = log_likelihood(X_train_b, y_train, theta_test)
    
    # 9. 결정 경계를 위한 메시 그리드 생성
    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    
    # 10. 최적화 과정의 스냅샷 시각화 (애니메이션 대신 주요 단계별 시각화)
    num_snapshots = 5
    snapshot_indices = np.linspace(0, iterations-1, num_snapshots, dtype=int)
    
    # 로그우도함수 등고선과 결정 경계 시각화
    fig, axes = plt.subplots(num_snapshots, 2, figsize=(12, 3*num_snapshots))
    
    for i, idx in enumerate(snapshot_indices):
        # 현재 파라미터
        current_theta = theta_history[idx]
        
        # 첫 번째 열: 파라미터 공간에서 로그우도함수 등고선과 최적화 경로
        contour = axes[i, 0].contourf(theta1_range, theta2_range, ll_grid, 50, cmap='viridis')
        axes[i, 0].plot(theta_history[:idx+1, 1], theta_history[:idx+1, 2], 'r-', linewidth=2)
        axes[i, 0].plot(theta_history[idx, 1], theta_history[idx, 2], 'ro', markersize=6)
        
        axes[i, 0].set_xlabel('Theta 1')
        axes[i, 0].set_ylabel('Theta 2')
        axes[i, 0].set_title(f'Iteration {idx}: Log-Likelihood = {ll_history[idx]:.4f}')
        axes[i, 0].grid(True)
        
        # 두 번째 열: 현재 파라미터의 결정 경계
        current_Z = compute_decision_boundary(current_theta, xx, yy)
        axes[i, 1].contourf(xx, yy, current_Z, alpha=0.3, cmap=plt.cm.Spectral, levels=np.linspace(0, 1, 11))
        axes[i, 1].scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.Spectral, edgecolors='k')
        
        axes[i, 1].set_xlabel('Feature 1')
        axes[i, 1].set_ylabel('Feature 2')
        axes[i, 1].set_title(f'Decision Boundary at Iteration {idx}')
        axes[i, 1].grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # 11. 파라미터 공간에서 최적화 경로 시각화 (2D 등고선)
    plt.figure(figsize=(10, 8))
    
    # 등고선 그리기
    contour = plt.contourf(theta1_range, theta2_range, ll_grid, 50, cmap='viridis')
    plt.colorbar(label='Log-Likelihood')
    
    # 최적화 경로 표시
    plt.plot(theta_history[:, 1], theta_history[:, 2], 'r-', linewidth=2, label='Optimization Path')
    plt.plot(theta_history[0, 1], theta_history[0, 2], 'go', markersize=8, label='Initial Point')
    plt.plot(theta_history[-1, 1], theta_history[-1, 2], 'bo', markersize=8, label='Final Point')
    
    # 경로를 따라 반복 횟수 표시 (10개 지점)
    path_markers = np.linspace(0, iterations-1, 10, dtype=int)
    for idx in path_markers:
        plt.text(theta_history[idx, 1]+0.1, theta_history[idx, 2], f'{idx}', 
                fontsize=9, verticalalignment='center')
    
    plt.xlabel('Theta 1')
    plt.ylabel('Theta 2')
    plt.title('Log-Likelihood Contours and Gradient Ascent Path')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # 12. 최종 결정 경계 시각화
    plt.figure(figsize=(10, 6))
    Z = compute_decision_boundary(theta_opt, xx, yy)
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.Spectral, levels=np.linspace(0, 1, 11))
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.Spectral, edgecolors='k')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Final Decision Boundary (Maximum Likelihood Solution)')
    plt.colorbar()
    plt.grid(True)
    plt.show()
    
    # 13. 3D 로그우도 표면 시각화
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    # 메시 그리드 생성
    theta1_grid, theta2_grid = np.meshgrid(theta1_range, theta2_range)
    
    # 3D 표면 플롯
    surf = ax.plot_surface(theta1_grid, theta2_grid, ll_grid.T, cmap='viridis', alpha=0.8)
    
    # 최적화 경로 추가
    ax.plot(theta_history[:, 1], theta_history[:, 2], ll_history, 'r-', linewidth=2, label='Optimization Path')
    ax.plot([theta_history[0, 1]], [theta_history[0, 2]], [ll_history[0]], 'go', markersize=8, label='Initial Point')
    ax.plot([theta_history[-1, 1]], [theta_history[-1, 2]], [ll_history[-1]], 'bo', markersize=8, label='Final Point')
    
    ax.set_xlabel('Theta 1')
    ax.set_ylabel('Theta 2')
    ax.set_zlabel('Log-Likelihood')
    ax.set_title('Log-Likelihood Surface and Optimization Path')
    
    plt.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    ax.legend()
    plt.tight_layout()
    plt.show()
    
    # 14. 모델 평가
    def predict(X, theta, threshold=0.5):
        return sigmoid(X.dot(theta)) >= threshold
        
    y_pred_train = predict(X_train_b, theta_opt)
    y_pred_test = predict(X_test_b, theta_opt)
    
    # 정확도 계산
    train_accuracy = np.mean(y_pred_train == y_train)
    test_accuracy = np.mean(y_pred_test == y_test)
    
    print(f"훈련 데이터 정확도: {train_accuracy:.4f}")
    print(f"테스트 데이터 정확도: {test_accuracy:.4f}") 