import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Sigmoid 함수 정의
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# 로지스틱 회귀 비용 함수
def compute_cost(X, y, theta):
    m = len(y)
    h = sigmoid(X.dot(theta))
    epsilon = 1e-5  # 로그에 0이 들어가는 것을 방지
    cost = (-1/m) * np.sum(y * np.log(h + epsilon) + (1 - y) * np.log(1 - h + epsilon))
    return cost

# 로지스틱 회귀의 경사하강법
def gradient_descent(X, y, theta, learning_rate=0.01, iterations=1000):
    m = len(y)
    cost_history = np.zeros(iterations)
    theta_history = np.zeros((iterations, theta.shape[0]))
    
    for i in range(iterations):
        z = X.dot(theta)
        h = sigmoid(z)
        gradient = (1/m) * X.T.dot(h - y)
        theta = theta - learning_rate * gradient
        
        cost_history[i] = compute_cost(X, y, theta)
        theta_history[i] = theta.T
        
    return theta, cost_history, theta_history

# 예측 함수
def predict(X, theta, threshold=0.5):
    return sigmoid(X.dot(theta)) >= threshold

# 결정 경계 시각화 함수
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
    plt.title('Decision Boundary for Logistic Regression')
    plt.colorbar()
    plt.grid(True)
    plt.show()

# 메인 코드

# 1. 데이터 생성
np.random.seed(42)
X, y = make_classification(n_samples=500, n_features=2, n_redundant=0, 
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

# 6. 모델 학습
theta = np.random.randn(3, 1)  # 바이어스 항 포함한 파라미터
iterations = 10000
learning_rate = 0.1

theta, cost_history, theta_history = gradient_descent(X_train_b, y_train.reshape(-1, 1), 
                                                     theta, learning_rate, iterations)

# 7. 비용 함수 변화 시각화
plt.figure(figsize=(10, 6))
plt.plot(range(iterations), cost_history)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Cost Function Over Iterations')
plt.grid(True)
plt.show()

# 8. 결정 경계 시각화
plot_decision_boundary(X_train_b, y_train, theta)

# 9. 모델 평가
y_pred_train = predict(X_train_b, theta)
y_pred_test = predict(X_test_b, theta)

# 학습 데이터에 대한 성능
train_accuracy = accuracy_score(y_train, y_pred_train)
print(f"Training Accuracy: {train_accuracy:.4f}")

# 테스트 데이터에 대한 성능
test_accuracy = accuracy_score(y_test, y_pred_test)
print(f"Test Accuracy: {test_accuracy:.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_test))

print("\nClassification Report:")
print(classification_report(y_test, y_pred_test))

# 10. 시그모이드 함수 시각화
plt.figure(figsize=(10, 6))
z = np.linspace(-10, 10, 100)
sigmoid_values = sigmoid(z)

plt.plot(z, sigmoid_values)
plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.3)
plt.axvline(x=0, color='r', linestyle='--', alpha=0.3)
plt.xlabel('z')
plt.ylabel('sigmoid(z)')
plt.title('Sigmoid Function')
plt.grid(True)
plt.show()

# 11. 분류 확률 시각화
probs = sigmoid(X_test_b.dot(theta)).flatten()
plt.figure(figsize=(10, 6))
plt.hist(probs[y_test == 0], bins=20, alpha=0.5, label='Class 0')
plt.hist(probs[y_test == 1], bins=20, alpha=0.5, label='Class 1')
plt.axvline(x=0.5, color='r', linestyle='--')
plt.xlabel('Probability of Class 1')
plt.ylabel('Frequency')
plt.title('Classification Probabilities')
plt.legend()
plt.grid(True)
plt.show()

# 12. 경사하강법 경로 시각화 (처음 500회 반복만)
iterations_to_plot = min(500, iterations)

# 비용 함수의 등고선을 그리기 위한 파라미터 공간 설정
theta0_vals = np.linspace(-2, 2, 100)
theta1_vals = np.linspace(-2, 2, 100)
J_vals = np.zeros((len(theta0_vals), len(theta1_vals)))

# 각 파라미터 조합에 대한 비용 계산
for i in range(len(theta0_vals)):
    for j in range(len(theta1_vals)):
        t = np.array([1.0, theta0_vals[i], theta1_vals[j]]).reshape(3, 1)
        J_vals[i, j] = compute_cost(X_train_b, y_train.reshape(-1, 1), t)

# 등고선 그래프로 시각화
plt.figure(figsize=(10, 8))
plt.contour(theta0_vals, theta1_vals, J_vals.T, 50, cmap='viridis')

# 경사하강법의 파라미터 경로 추가
plt.plot(theta_history[:iterations_to_plot, 1], theta_history[:iterations_to_plot, 2], 
         'r-', linewidth=2, label='Gradient Descent Path')
plt.plot(theta_history[0, 1], theta_history[0, 2], 'go', markersize=10, label='Initial Point')
plt.plot(theta_history[iterations_to_plot-1, 1], theta_history[iterations_to_plot-1, 2], 
         'bo', markersize=10, label='Final Point')

plt.xlabel('Theta 1')
plt.ylabel('Theta 2')
plt.title(f'Gradient Descent Path (First {iterations_to_plot} Iterations)')
plt.legend()
plt.grid(True)
plt.show() 