# Decision Boundary Visualization for a Basic Linear Classifier
import numpy as np
import matplotlib.pyplot as plt

# Updated sample data points
positive_points = np.array([[0, 3], [1, 2], [1, 4], [2, 3]])
negative_points = np.array([[0, 2], [1, 0], [3, 0], [4, 2]])

# Mean calculation - p와 n 계산
mu_positive = np.mean(positive_points, axis=0)  # p
mu_negative = np.mean(negative_points, axis=0)  # n

# Decision boundary calculation
w = mu_positive - mu_negative  # w = p - n
midpoint = (mu_positive + mu_negative) / 2  # (p + n)/2

# 결정 방정식 계산 과정 출력
print("========= 결정 방정식 계산 과정 =========")
print(f"1. 양성 클래스 중심점 p = {mu_positive}")
print(f"2. 음성 클래스 중심점 n = {mu_negative}")
print(f"3. 방향 벡터 w = p - n = {mu_positive} - {mu_negative} = {w}")
print(f"4. 중점 M = (p + n)/2 = ({mu_positive} + {mu_negative})/2 = {midpoint}")

# 결정 경계 오프셋 계산: t = (p-n)⋅(p+n)/2 = (||p||² - ||n||²)/2 = w⋅midpoint
p_norm_squared = np.dot(mu_positive, mu_positive)
n_norm_squared = np.dot(mu_negative, mu_negative)
t_from_norms = (p_norm_squared - n_norm_squared) / 2
t = np.dot(w, midpoint)

print(f"5. 오프셋 t 계산 방법 1: w⋅midpoint = ({w[0]}*{midpoint[0]} + {w[1]}*{midpoint[1]}) = {t}")
print(f"6. 오프셋 t 계산 방법 2: (||p||² - ||n||²)/2 = ({p_norm_squared} - {n_norm_squared})/2 = {t_from_norms}")
print(f"   (두 방법의 결과가 같은 것을 확인할 수 있습니다: {t} ≈ {t_from_norms})")

# 결정 방정식 구성: w⋅x = t, 즉 w[0]*x + w[1]*y = t
decision_equation = f"{w[0]:.2f}x + {w[1]:.2f}y = {t:.2f}"
print(f"7. 결정 방정식: {decision_equation}")
print("======================================")

# 두 중점 사이의 벡터와 중점 출력
print(f"벡터 w (두 중점 사이의 방향 벡터): {w}")
print(f"중점 M: {midpoint}")
print(f"결정 경계 오프셋 t: {t}")

# 오류 개수 계산
error_count = 0
total_points = len(positive_points) + len(negative_points)

# 각 점에 대한 결정 값 계산 및 오류 여부 출력
print("\n각 점에 대한 결정 값 및 오류 여부:")

# Positive 포인트 분류 오류 확인
for point in positive_points:
    x, y = point
    # 결정 값 계산: w⋅x > t
    decision_value = np.dot(w, point) - t
    
    # 사칙연산 과정 출력
    calculation = f"Positive Point ({x}, {y}): w⋅x - t = ({w[0]}*{x} + {w[1]}*{y}) - {t} = {decision_value}"
    print(calculation)
    
    if decision_value <= 0:  # Positive 포인트가 negative로 분류되는 경우
        error_count += 1
        print(f"  -> 오류 발생")
    else:
        print(f"  -> 정상")

# Negative 포인트 분류 오류 확인
for point in negative_points:
    x, y = point
    # 결정 값 계산: w⋅x > t
    decision_value = np.dot(w, point) - t
    
    # 사칙연산 과정 출력
    calculation = f"Negative Point ({x}, {y}): w⋅x - t = ({w[0]}*{x} + {w[1]}*{y}) - {t} = {decision_value}"
    print(calculation)
    
    if decision_value > 0:  # Negative 포인트가 positive로 분류되는 경우
        error_count += 1
        print(f"  -> 오류 발생")
    else:
        print(f"  -> 정상")

# 오류율 계산
error_rate = error_count / total_points
print(f"\n총 오류 개수: {error_count}")
print(f"오류율: {error_rate:.2%}")

# Plotting
plt.scatter(positive_points[:, 0], positive_points[:, 1], color='blue', label='Positive')
plt.scatter(negative_points[:, 0], negative_points[:, 1], color='red', label='Negative')
plt.scatter(*mu_positive, color='cyan', marker='x', s=100, label='Positive Mean')
plt.scatter(*mu_negative, color='orange', marker='x', s=100, label='Negative Mean')

# 중점 표시
plt.scatter(*midpoint, color='magenta', marker='o', s=100, label='Midpoint M')

# 두 중점을 이어주는 벡터 시각화
plt.arrow(mu_negative[0], mu_negative[1], w[0], w[1], 
          head_width=0.2, head_length=0.2, fc='purple', ec='purple', 
          length_includes_head=True, label='Vector w')

# Decision boundary line: w[0]*x + w[1]*y = t
x_vals = np.linspace(-1, 5, 100)
if w[1] != 0:
    y_vals = (t - w[0] * x_vals) / w[1]
else:
    y_vals = np.linspace(-1, 5, 100)
    x_vals = np.ones_like(y_vals) * (t / w[0])

plt.plot(x_vals, y_vals, color='green', linestyle='--', label='Decision Boundary')

plt.axhline(0, color='black',linewidth=0.8)
plt.axvline(0, color='black',linewidth=0.8)

plt.legend(loc='upper right')
plt.xlabel('x1')
plt.ylabel('x2')
plt.title(f'Linear Classifier Decision Boundary\n{decision_equation}\nError Rate: {error_rate:.2%}')
# plt.grid(True)
plt.show()