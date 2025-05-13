import numpy as np
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import platform

# 한글 폰트 설정
system_name = platform.system()
if system_name == 'Windows':
    plt.rc('font', family='Malgun Gothic')  # 윈도우의 경우 맑은 고딕
elif system_name == 'Darwin':  # Mac
    plt.rc('font', family='AppleGothic')
else:  # Linux
    plt.rc('font', family='NanumGothic')
    
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지

# 데이터 정의
# 양성 예제들 (검은 점)
positive_points = np.array([
    [1, 2],  # 첫 번째 점 (1, 2)
    [2, 3]   # 두 번째 점 (2, 3)
])

# 음성 예제들 (흰 점)
negative_points = np.array([
    [3, 1],  # 첫 번째 점 (3, 1)
    [4, 2]   # 두 번째 점 (4, 2)
])

# 원하는 경계점 (이 점들을 지나야 함)
boundary_points = np.array([
    [2, 1.5],  # 첫 번째 경계점
    [3, 2.5]   # 두 번째 경계점
])

# 경계선을 계산하기 위한 방정식 설정
# 로지스틱 회귀에서 클래스 경계는 w₀ + w₁x₁ + w₂x₂ = 0을 만족하는 점들
# 주어진 두 점 (2, 1.5)와 (3, 2.5)를 지나는 경계선을 구하기 위해
# w₀ + w₁·2 + w₂·1.5 = 0
# w₀ + w₁·3 + w₂·2.5 = 0
# 를 만족하는 w = [w₀, w₁, w₂]를 찾아야 함

# w₂ = 1로 정규화 (임의로 설정)
w2 = 1

# 두 방정식으로부터 w₀와 w₁을 구함
# 두 번째 점의 방정식에서 첫 번째 점의 방정식을 빼면:
# w₁(3-2) + w₂(2.5-1.5) = 0
# w₁ + w₂ = 0
# w₂ = 1이므로, w₁ = -1

w1 = -1

# 이제 w₀를 구함:
# w₀ + w₁·2 + w₂·1.5 = 0
# w₀ + (-1)·2 + 1·1.5 = 0
# w₀ = 2 - 1.5 = 0.5

w0 = 0.5

# 새로운 파라미터
w_new = np.array([w0, w1, w2])
print(f"새로운 파라미터: w = [{w0}, {w1}, {w2}]")

# 새로운 경계식: w₀ + w₁x₁ + w₂x₂ = 0
# 0.5 + (-1)x₁ + 1·x₂ = 0
# x₂ = x₁ - 0.5
print(f"새로운 경계식: x₂ = x₁ - 0.5")

# 경계선 확인 - 주어진 점들이 경계선에 있는지 검증
point1 = [1, boundary_points[0, 0], boundary_points[0, 1]]  # [1, 2, 1.5]
point2 = [1, boundary_points[1, 0], boundary_points[1, 1]]  # [1, 3, 2.5]

eqn1 = np.dot(w_new, point1)
eqn2 = np.dot(w_new, point2)

print(f"첫 번째 경계점 검증: {w0} + {w1}*{boundary_points[0, 0]} + {w2}*{boundary_points[0, 1]} = {eqn1}")
print(f"두 번째 경계점 검증: {w0} + {w1}*{boundary_points[1, 0]} + {w2}*{boundary_points[1, 1]} = {eqn2}")

# 그래프 범위 설정
x1_range = np.linspace(0, 5, 100)
x2_boundary = x1_range - 0.5  # 새로운 경계식 x₂ = x₁ - 0.5

plt.figure(figsize=(8, 6))

# 경계선 그리기
plt.plot(x1_range, x2_boundary, 'r-', label='새 클래스 경계 (x₂ = x₁ - 0.5)')

# 양성 예제 그리기 (검은 점)
plt.scatter(positive_points[:, 0], positive_points[:, 1], color='black', marker='o', 
            s=100, label='양성 예제')

# 음성 예제 그리기 (흰 점)
plt.scatter(negative_points[:, 0], negative_points[:, 1], color='white', marker='o', 
            s=100, edgecolors='black', label='음성 예제')

# 경계점 그리기 (파란색 X)
plt.scatter(boundary_points[:, 0], boundary_points[:, 1], color='blue', marker='x', 
            s=100, label='경계점')

# 격자 그리기
plt.grid(True)

# 축 범위 설정
plt.xlim(0, 5)
plt.ylim(0, 4)

# 축 레이블
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')

# 제목 및 범례
plt.title(f'로지스틱 회귀 모델의 새 클래스 경계 (w = [{w0}, {w1}, {w2}])')
plt.legend()

# 양쪽 영역에 색상 채우기
x1_fill = np.linspace(0, 5, 100)
x2_fill_upper = np.ones_like(x1_fill) * 4  # 상단 경계
x2_fill_lower = np.zeros_like(x1_fill) * 0  # 하단 경계

# 양성 영역(경계선 위쪽) 색상 채우기
plt.fill_between(x1_fill, x2_boundary, x2_fill_upper, color='lightblue', alpha=0.3)

# 음성 영역(경계선 아래쪽) 색상 채우기
plt.fill_between(x1_fill, x2_fill_lower, x2_boundary, color='lightcoral', alpha=0.3)

# 그래프 저장 및 표시
plt.savefig('correct_boundary.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\nw = [{w0}, {w1}, {w2}]일 때의 클래스 경계식: x₂ = x₁ - 0.5")
print("이 직선은 (2,1.5)와 (3,2.5)를 지나게 됩니다.") 