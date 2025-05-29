import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'

# 시그모이드 함수 정의
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# 로그우도함수 정의 (최대화해야 하므로 음수 취함 -> 최소화 문제로 변환)
def negative_log_likelihood(X, y, theta):
    m = len(y)
    h = sigmoid(X.dot(theta))
    
    # 수치적 안정성을 위한 작은 값 추가 (로그에 0이 들어가는 것 방지)
    epsilon = 1e-5
    
    # 로그우도함수 계산 (위 슬라이드의 최종 수식과 동일)
    log_likelihood = np.sum(y * np.log(h + epsilon) + (1 - y) * np.log(1 - h + epsilon))
    
    # 음수 로그우도 (최소화 문제로 변환)
    return -log_likelihood / m

# 로그우도함수의 그래디언트 (미분값)
def log_likelihood_gradient(X, y, theta):
    m = len(y)
    h = sigmoid(X.dot(theta))
    
    # 로그우도함수를 theta로 미분한 값 (경사상승법을 위한 그래디언트)
    gradient = X.T.dot(h - y) / m
    
    # 경사하강법을 위해 음수 부호 (우리는 최소화하므로)
    return gradient

# 경사하강법으로 로그우도함수 최대화 (음수 로그우도 최소화)
def gradient_descent(X, y, theta, learning_rate=0.1, iterations=1000, plot_cost=True):
    m = len(y)
    n = theta.shape[0]  # 파라미터 개수
    cost_history = np.zeros(iterations)
    theta_history = np.zeros((iterations, n))
    
    for i in range(iterations):
        gradient = log_likelihood_gradient(X, y, theta)
        theta = theta - learning_rate * gradient
        
        cost_history[i] = negative_log_likelihood(X, y, theta)
        theta_history[i] = theta.flatten()
        
    if plot_cost:
        plt.figure(figsize=(10, 6))
        plt.plot(range(iterations), cost_history)
        plt.title('Negative Log-Likelihood Over Iterations')
        plt.xlabel('Iterations')
        plt.ylabel('Negative Log-Likelihood')
        plt.grid(True)
        plt.show()
        
    return theta, cost_history, theta_history

# 메인 코드 실행
if __name__ == "__main__":
    # 1. 데이터 생성
    np.random.seed(42)
    X, y = make_classification(n_samples=300, n_features=2, n_redundant=0, 
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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 4. 특성 정규화
    X_mean = np.mean(X_train, axis=0)
    X_std = np.std(X_train, axis=0)
    
    X_train_norm = (X_train - X_mean) / X_std
    X_test_norm = (X_test - X_mean) / X_std
    
    # 5. 바이어스 항 추가
    X_train_b = np.c_[np.ones((X_train_norm.shape[0])), X_train_norm]
    X_test_b = np.c_[np.ones((X_test_norm.shape[0])), X_test_norm]
    
    # 6. 초기 파라미터 설정 및 로그우도함수 시각화
    if X_train_b.shape[1] == 3:  # 특성이 2개 + 바이어스일 때만 시각화
        # 파라미터 공간의 그리드 포인트 생성
        w0_vals = np.linspace(-5, 5, 100)
        w1_vals = np.linspace(-5, 5, 100)
        
        # 각 그리드 포인트에서 로그우도 계산
        log_likelihood_vals = np.zeros((len(w0_vals), len(w1_vals)))
        
        for i, w0 in enumerate(w0_vals):
            for j, w1 in enumerate(w1_vals):
                theta = np.array([0, w0, w1]).reshape(-1, 1)  # 바이어스(w2)는 0으로 고정
                log_likelihood_vals[i, j] = negative_log_likelihood(X_train_b, y_train, theta)
        
        # 3D 표면으로 로그우도함수 시각화
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        w0_grid, w1_grid = np.meshgrid(w0_vals, w1_vals)
        surf = ax.plot_surface(w0_grid, w1_grid, log_likelihood_vals.T, 
                              cmap='viridis', alpha=0.8, antialiased=True)
        
        ax.set_xlabel('Weight 0')
        ax.set_ylabel('Weight 1')
        ax.set_zlabel('Negative Log-Likelihood')
        ax.set_title('Log-Likelihood Function Surface')
        plt.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
        plt.show()
        
        # 등고선으로도 시각화
        plt.figure(figsize=(10, 8))
        contour = plt.contour(w0_vals, w1_vals, log_likelihood_vals.T, 50, cmap='viridis')
        plt.colorbar(contour)
        plt.xlabel('Weight 0')
        plt.ylabel('Weight 1')
        plt.title('Negative Log-Likelihood Contour Plot')
        plt.grid(True)
        plt.show()
    
    # 7. 로그우도함수 최대화(경사하강법)
    theta = np.zeros((X_train_b.shape[1], 1))  # 초기 파라미터는 0으로 설정
    iterations = 3000
    learning_rate = 0.5
    
    print("Initial Negative Log-Likelihood:", negative_log_likelihood(X_train_b, y_train, theta))
    
    theta, cost_history, theta_history = gradient_descent(X_train_b, y_train, theta, 
                                                       learning_rate, iterations)
    
    print("Final Negative Log-Likelihood:", negative_log_likelihood(X_train_b, y_train, theta))
    print("Optimized Parameters (theta):", theta.flatten())
    
    # 8. 최적화 경로 시각화 (파라미터가 3개인 경우)
    if X_train_b.shape[1] == 3:
        plt.figure(figsize=(10, 8))
        contour = plt.contour(w0_vals, w1_vals, log_likelihood_vals.T, 50, cmap='viridis')
        plt.colorbar(contour)
        
        # 경사하강법의 경로 표시 (처음 500개 반복만)
        path_length = min(500, iterations)
        plt.plot(theta_history[:path_length, 1], theta_history[:path_length, 2], 
                'r-', linewidth=2, label='Optimization Path')
        plt.plot(theta_history[0, 1], theta_history[0, 2], 
                'go', markersize=10, label='Initial Point')
        plt.plot(theta_history[path_length-1, 1], theta_history[path_length-1, 2], 
                'bo', markersize=10, label='Final Point')
        
        plt.xlabel('Weight 0')
        plt.ylabel('Weight 1')
        plt.title('Optimization Path on Log-Likelihood Contour')
        plt.legend()
        plt.grid(True)
        plt.show()
    
    # 9. 결정 경계 시각화
    def plot_decision_boundary(X, y, theta):
        # 결정 경계를 시각화하기 위한 메시 그리드 생성
        x_min, x_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        y_min, y_max = X[:, 2].min() - 1, X[:, 2].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                             np.arange(y_min, y_max, 0.1))
        
        # 모든 메시 그리드 포인트의 예측 계산
        Z = sigmoid(np.c_[np.ones((xx.ravel().shape[0])), xx.ravel(), yy.ravel()].dot(theta))
        Z = Z.reshape(xx.shape)
        
        # 데이터 포인트와 결정 경계 시각화
        plt.figure(figsize=(10, 8))
        plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.Spectral)
        plt.scatter(X[:, 1], X[:, 2], c=y, cmap=plt.cm.Spectral, edgecolors='k')
        plt.xlabel('Feature 1')
        plt.ylabel('Feature 2')
        plt.title('Decision Boundary (Maximum Likelihood Solution)')
        plt.colorbar()
        plt.grid(True)
        plt.show()
    
    plot_decision_boundary(X_train_b, y_train, theta)
    
    # 10. 모델 성능 평가
    def predict(X, theta, threshold=0.5):
        return sigmoid(X.dot(theta)) >= threshold
    
    y_pred_train = predict(X_train_b, theta)
    y_pred_test = predict(X_test_b, theta)
    
    # 정확도 계산
    train_accuracy = np.mean(y_pred_train == y_train.reshape(-1, 1))
    test_accuracy = np.mean(y_pred_test == y_test.reshape(-1, 1))
    
    print(f"Training Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    # 11. 학습 데이터와 테스트 데이터의 로그우도함수 값 비교
    train_ll = -negative_log_likelihood(X_train_b, y_train, theta)
    test_ll = -negative_log_likelihood(X_test_b, y_test, theta)
    
    print(f"Training Log-Likelihood: {train_ll:.4f}")
    print(f"Test Log-Likelihood: {test_ll:.4f}") 