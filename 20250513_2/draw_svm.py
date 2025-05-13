import numpy as np
import matplotlib.pyplot as plt

# 데이터 포인트 정의
x1 = np.array([1, 0])
x2 = np.array([-1, 2])
points = np.array([x1, x2])
labels = np.array([1, -1])

# 계산된 SVM 파라미터
w = np.array([0.5, -0.5])  # 계산된 가중치
w0 = 0.5  # 계산된 바이어스

# 결정 경계 그리기 위한 함수
def decision_boundary(x):
    # w^T x + w0 = 0 에서 x2(y축) 값 계산
    # w[0]*x + w[1]*y + w0 = 0
    # y = -(w[0]*x + w0) / w[1]
    return -(w[0] * x + w0) / w[1]

# 마진 경계 그리기 위한 함수
def margin_plus(x):
    # w^T x + w0 = 1
    return -(w[0] * x + w0 - 1) / w[1]

def margin_minus(x):
    # w^T x + w0 = -1
    return -(w[0] * x + w0 + 1) / w[1]

# 그래프 설정
plt.figure(figsize=(10, 8))
plt.grid(True)

# 데이터 포인트 그리기
plt.scatter(x1[0], x1[1], c='red', s=100, marker='o', label='클래스 +1')
plt.scatter(x2[0], x2[1], c='blue', s=100, marker='x', label='클래스 -1')

# 결정 경계와 마진 그리기
x_range = np.linspace(-2, 2, 100)
plt.plot(x_range, decision_boundary(x_range), 'g-', label='결정 경계 (w^T x + w0 = 0)')
plt.plot(x_range, margin_plus(x_range), 'g--', label='마진 경계 (w^T x + w0 = 1)')
plt.plot(x_range, margin_minus(x_range), 'g--', label='마진 경계 (w^T x + w0 = -1)')

# "a - b > -1" 영역 표시 (양성 클래스 영역)
x_fill = np.linspace(-2, 2, 100)
y_fill_upper = np.full_like(x_fill, 3)
y_fill_lower = x_fill + 1  # a - b > -1 => b < a + 1
plt.fill_between(x_fill, y_fill_lower, y_fill_upper, alpha=0.2, color='red', label='양성 분류 영역 (a - b > -1)')

# "a - b < -1" 영역 표시 (음성 클래스 영역)
y_fill_upper = x_fill + 1  # a - b = -1 => b = a + 1
y_fill_lower = np.full_like(x_fill, -1)
plt.fill_between(x_fill, y_fill_lower, y_fill_upper, alpha=0.2, color='blue', label='음성 분류 영역 (a - b < -1)')

# 그래프 레이블 및 제목 설정
plt.xlabel('x축 (a)')
plt.ylabel('y축 (b)')
plt.title('SVM 결정 경계와 마진')
plt.legend(loc='upper right')
plt.xlim(-2, 2)
plt.ylim(-1, 3)

# w 벡터 표시
plt.arrow(0, 0, w[0], w[1], head_width=0.1, head_length=0.1, fc='purple', ec='purple', label='가중치 벡터 w')
plt.text(w[0]/2, w[1]/2, 'w = (0.5, -0.5)', color='purple')

# 저장 및 표시
plt.savefig('20250513_2/svm_visualization.png', dpi=300, bbox_inches='tight')
plt.show() 