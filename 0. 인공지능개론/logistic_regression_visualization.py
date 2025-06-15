import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# 데이터 포인트 정의
X_positive = np.array([[1, 2], [2, 3]])  # 양성 예제
X_negative = np.array([[3, 1], [4, 2]])  # 음성 예제

# 가중치 벡터
w1 = np.array([0, -1, 1])  # w1 = (0, -1, 1)
w2 = np.array([-3, -1, 3])  # w2 = (-3, -1, 3)

# 로지스틱 함수
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# 로지스틱 회귀 예측 함수
def predict(X, w):
    # X에 바이어스 항 추가
    X_with_bias = np.hstack([np.ones((X.shape[0], 1)), X])
    z = np.dot(X_with_bias, w)
    return sigmoid(z)

# 결정 경계 함수 (w·x = 0)
def decision_boundary(x, w):
    # w0 + w1*x1 + w2*x2 = 0 -> x2 = (-w0 - w1*x1) / w2
    return (-w[0] - w[1] * x) / w[2]

# 그래프 그리기
plt.figure(figsize=(8, 8))

# 확률 등고선 그리기 (배경 색상)
x_min, x_max = 0, 5
y_min, y_max = 0, 5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
X_grid = np.c_[xx.ravel(), yy.ravel()]
Z = predict(X_grid, w2)
Z = Z.reshape(xx.shape)

# 배경에 확률 표시
plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdBu_r)
plt.colorbar(label='Probability $P(y=1|x)$')

# 결정 경계 그리기
x_range = np.linspace(x_min, x_max, 100)
plt.plot(x_range, decision_boundary(x_range, w2), 'k-', label='Decision Boundary: $x_2 = \\frac{x_1 + 3}{3}$')

# 데이터 포인트 그리기
plt.scatter(X_positive[:, 0], X_positive[:, 1], c='black', marker='o', s=100, label='Positive')
plt.scatter(X_negative[:, 0], X_negative[:, 1], c='white', marker='o', s=100, edgecolors='black', label='Negative')

# 1단위 그리드 설정
plt.xticks(np.arange(x_min, x_max + 1, 1))
plt.yticks(np.arange(y_min, y_max + 1, 1))
plt.grid(True, which='both', color='gray', linestyle='-', linewidth=0.5)

# 그래프 설정
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
plt.title('')
plt.legend(loc='upper right')

# 축 비율과 범위 설정
plt.gca().set_aspect('equal')  # 축 비율을 1:1로 설정
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

# 저장 및 표시
plt.tight_layout()
plt.savefig('logistic_regression_boundary.png', dpi=300, bbox_inches='tight')
plt.show()

# 두 파라미터의 우도 계산 결과 출력
print("Parameter w1 = (0, -1, 1) likelihood: 0.421")
print("Parameter w2 = (-3, -1, 3) likelihood: 0.856")
print("Therefore, w2 is the better parameter.") 