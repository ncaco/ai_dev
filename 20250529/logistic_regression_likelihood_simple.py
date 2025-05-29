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
    
    # 로그우도함수 계산 (슬라이드의 수식과 동일)
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

# 경사상승법으로 로그우도함수 최대화
def gradient_ascent(X, y, theta, learning_rate=0.1, iterations=1000):
    """
    경사상승법을 통한 로그우도함수 최대화
    """
    m, n = X.shape
    theta_history = np.zeros((iterations, n))
    ll_history = np.zeros(iterations)
    
    # theta를 복사하여 사용
    current_theta = theta.copy()
    
    for i in range(iterations):
        # 그래디언트 계산
        gradient = log_likelihood_gradient(X, y, current_theta)
        
        # 파라미터 업데이트 (경사상승법이므로 그래디언트 방향으로 이동)
        current_theta = current_theta + learning_rate * gradient
        
        # 로그우도와 파라미터 기록
        ll_history[i] = log_likelihood(X, y, current_theta)
        theta_history[i] = current_theta.ravel()
    
    return current_theta, ll_history, theta_history

# 예측 함수
def predict(X, theta, threshold=0.5):
    """
    로지스틱 회귀 모델의 예측
    """
    return sigmoid(X.dot(theta)) >= threshold

# 결정 경계 시각화 함수
def plot_decision_boundary(X, y, theta):
    """
    학습된 모델의 결정 경계 시각화
    """
    # 메시 그리드 생성
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    
    # 그리드 포인트에 대한 예측
    Z = sigmoid(np.c_[np.ones(xx.ravel().shape), xx.ravel(), yy.ravel()].dot(theta))
    Z = Z.reshape(xx.shape)
    
    # 결정 경계 시각화
    plt.figure(figsize=(10, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.Spectral)
    plt.scatter(X[:, 1], X[:, 2], c=y, cmap=plt.cm.Spectral, edgecolors='k')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Decision Boundary (Maximum Likelihood Estimation)')
    plt.colorbar()
    plt.grid(True)
    plt.show()

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
    
    # 3. 데이터 분할
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # 4. 바이어스 항 추가
    X_train_b = np.c_[np.ones(X_train.shape[0]), X_train]
    X_test_b = np.c_[np.ones(X_test.shape[0]), X_test]
    
    # 5. 초기 파라미터 설정
    theta = np.zeros(X_train_b.shape[1])
    
    # 6. 로그우도함수 최대화 (경사상승법)
    iterations = 1000
    learning_rate = 0.1
    
    print("초기 로그우도:", log_likelihood(X_train_b, y_train, theta))
    
    # 경사상승법 실행
    theta_opt, ll_history, theta_history = gradient_ascent(X_train_b, y_train, 
                                                        theta, learning_rate, iterations)
    
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
    
    # 8. 결정 경계 시각화
    plot_decision_boundary(X_train_b, y_train, theta_opt)
    
    # 9. 모델 성능 평가
    y_pred_train = predict(X_train_b, theta_opt)
    y_pred_test = predict(X_test_b, theta_opt)
    
    # 정확도 계산
    train_accuracy = np.mean(y_pred_train == y_train)
    test_accuracy = np.mean(y_pred_test == y_test)
    
    print(f"훈련 데이터 정확도: {train_accuracy:.4f}")
    print(f"테스트 데이터 정확도: {test_accuracy:.4f}")
    
    # 10. 파라미터 공간에서 로그우도함수 시각화 (2D 등고선)
    # 바이어스(theta[0])를 고정하고 나머지 두 파라미터에 대한 로그우도 시각화
    theta0_fixed = theta_opt[0]
    theta1_range = np.linspace(theta_opt[1] - 2, theta_opt[1] + 2, 100)
    theta2_range = np.linspace(theta_opt[2] - 2, theta_opt[2] + 2, 100)
    
    # 로그우도 그리드 계산
    ll_grid = np.zeros((len(theta1_range), len(theta2_range)))
    
    for i, t1 in enumerate(theta1_range):
        for j, t2 in enumerate(theta2_range):
            theta_test = np.array([theta0_fixed, t1, t2])
            ll_grid[i, j] = log_likelihood(X_train_b, y_train, theta_test)
    
    # 등고선으로 시각화
    plt.figure(figsize=(10, 8))
    plt.contourf(theta1_range, theta2_range, ll_grid, 50, cmap='viridis')
    
    # 경사상승법 경로 표시
    plt.plot(theta_history[:, 1], theta_history[:, 2], 'r-', linewidth=2, label='Optimization Path')
    plt.plot(theta_history[0, 1], theta_history[0, 2], 'go', markersize=8, label='Initial Point')
    plt.plot(theta_history[-1, 1], theta_history[-1, 2], 'bo', markersize=8, label='Final Point')
    
    plt.xlabel('Theta 1')
    plt.ylabel('Theta 2')
    plt.title('Log-Likelihood Contours and Gradient Ascent Path')
    plt.colorbar(label='Log-Likelihood')
    plt.legend()
    plt.grid(True)
    plt.show() 