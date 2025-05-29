import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'  # 폰트 변경

# 데이터 생성
np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# 데이터 시각화
plt.figure(figsize=(10, 6))
plt.scatter(X, y)
plt.title('Generated Data')
plt.xlabel('X')
plt.ylabel('y')
plt.show()

# 경사하강법 구현
def compute_cost(X, y, theta):
    """비용 함수 (MSE) 계산"""
    m = len(y)
    predictions = X.dot(theta)
    cost = (1/(2*m)) * np.sum(np.square(predictions - y))
    return cost

def gradient_descent(X, y, theta, learning_rate=0.01, iterations=1000):
    """경사하강법 구현"""
    m = len(y)
    cost_history = np.zeros(iterations)
    theta_history = np.zeros((iterations, 2))
    
    for i in range(iterations):
        prediction = np.dot(X, theta)
        
        # 기울기(그래디언트) 계산
        gradient = (1/m) * X.T.dot(prediction - y)
        
        # 파라미터 업데이트
        theta = theta - learning_rate * gradient
        
        # 비용과 파라미터 값 저장
        cost_history[i] = compute_cost(X, y, theta)
        theta_history[i] = theta.T
        
    return theta, cost_history, theta_history

# 데이터 준비 (절편항 추가)
X_b = np.c_[np.ones((len(X), 1)), X]
theta = np.random.randn(2, 1)  # 초기 파라미터 (random)

# 경사하강법 실행
iterations = 100
theta, cost_history, theta_history = gradient_descent(X_b, y, theta, learning_rate=0.1, iterations=iterations)

print(f"Final Parameters: θ₀ = {theta[0][0]:.4f}, θ₁ = {theta[1][0]:.4f}")
print(f"Actual Parameters: θ₀ = 4, θ₁ = 3")

# 비용 함수 감소 시각화
plt.figure(figsize=(10, 6))
plt.plot(range(iterations), cost_history)
plt.title('Cost Function Decrease Over Iterations')
plt.xlabel('Iterations')
plt.ylabel('Cost (MSE)')
plt.grid(True)
plt.show()

# 최종 회귀선 시각화
plt.figure(figsize=(10, 6))
plt.scatter(X, y)
plt.plot(X, X_b.dot(theta), color='red', linewidth=3)
plt.title('Final Linear Regression Model')
plt.xlabel('X')
plt.ylabel('y')
plt.show()

# 경사하강법 애니메이션
fig, ax = plt.subplots(figsize=(10, 6))

# 산점도 표시
scatter = ax.scatter(X, y)

# 초기 회귀선
line, = ax.plot([], [], 'r-', linewidth=2)

# 텍스트 요소
cost_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
theta_text = ax.text(0.02, 0.90, '', transform=ax.transAxes)

# 축 범위 설정
ax.set_xlim(0, 2)
ax.set_ylim(0, 14)
ax.set_xlabel('X')
ax.set_ylabel('y')
ax.set_title('Gradient Descent Simulation')
ax.grid(True)

# 애니메이션 업데이트 함수
def update(frame):
    # 현재 파라미터로 회귀선 업데이트
    current_theta = theta_history[frame].reshape(2, 1)
    x_line = np.array([0, 2])
    y_line = current_theta[0] + current_theta[1] * x_line
    
    line.set_data(x_line, y_line)
    cost_text.set_text(f'Cost: {cost_history[frame]:.4f}')
    theta_text.set_text(f'θ₀: {current_theta[0][0]:.4f}, θ₁: {current_theta[1][0]:.4f}')
    
    return line, cost_text, theta_text

# 애니메이션 생성
ani = FuncAnimation(fig, update, frames=iterations, 
                    blit=True, interval=100, repeat=False)

plt.tight_layout()
plt.show()

# 3D 공간에서 비용함수 시각화
from mpl_toolkits.mplot3d import Axes3D

# 비용함수 그리드 생성
theta0_vals = np.linspace(0, 10, 100)
theta1_vals = np.linspace(0, 5, 100)

# 그리드 좌표
theta0_grid, theta1_grid = np.meshgrid(theta0_vals, theta1_vals)

# 비용 계산
cost_grid = np.zeros((len(theta0_vals), len(theta1_vals)))
for i in range(len(theta0_vals)):
    for j in range(len(theta1_vals)):
        t = np.array([[theta0_vals[i]], [theta1_vals[j]]])
        cost_grid[j, i] = compute_cost(X_b, y, t)

# 3D 표면 플롯
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(theta0_grid, theta1_grid, cost_grid, cmap='viridis', alpha=0.6)
ax.set_xlabel('θ₀')
ax.set_ylabel('θ₁')
ax.set_zlabel('Cost')
ax.set_title('3D Visualization of Cost Function')

# 경사하강법 경로 시각화
ax.plot(theta_history[:, 0], theta_history[:, 1], cost_history, 'r-', linewidth=2, label='Gradient Descent Path')
ax.plot(theta_history[:, 0], theta_history[:, 1], cost_history, 'mo', markersize=3)

plt.legend()
plt.tight_layout()
plt.show()

# 등고선에서 경사하강법 시각화
plt.figure(figsize=(10, 8))
contour = plt.contour(theta0_grid, theta1_grid, cost_grid, 50, cmap='viridis')
plt.plot(theta_history[:, 0], theta_history[:, 1], 'r-', linewidth=2, label='Gradient Descent Path')
plt.plot(theta_history[:, 0], theta_history[:, 1], 'mo', markersize=4)
plt.plot(theta_history[0, 0], theta_history[0, 1], 'go', markersize=8, label='Initial Point')
plt.plot(theta_history[-1, 0], theta_history[-1, 1], 'bo', markersize=8, label='Final Point')
plt.xlabel('θ₀')
plt.ylabel('θ₁')
plt.title('Gradient Descent Path on Contour Plot')
plt.colorbar(contour)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show() 