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

# 클래스 경계를 그리기 위한 코드
# w_2 = (-3, -1, 3)에 대한 모델의 클래스 경계

# 파라미터 w_2
w = np.array([-3, -1, 3])

# 로지스틱 함수
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

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

# 클래스 경계 그리기
# 클래스 경계는 h_w(x) = 0.5일 때, 즉 w·x = 0일 때 발생합니다.
# w = [-3, -1, 3]이고 x = [1, x1, x2]이므로 
# -3 + (-1)*x1 + 3*x2 = 0
# 따라서 x2 = (3 + x1)/3

# 그래프 범위 설정
x1_range = np.linspace(0, 5, 100)
x2_boundary = (3 + x1_range) / 3

plt.figure(figsize=(8, 6))

# 경계선 그리기
plt.plot(x1_range, x2_boundary, 'r-', label='클래스 경계 (w·x = 0)')

# 양성 예제 그리기 (검은 점)
plt.scatter(positive_points[:, 0], positive_points[:, 1], color='black', marker='o', 
            s=100, label='양성 예제')

# 음성 예제 그리기 (흰 점)
plt.scatter(negative_points[:, 0], negative_points[:, 1], color='white', marker='o', 
            s=100, edgecolors='black', label='음성 예제')

# 격자 그리기
plt.grid(True)

# 축 범위 설정
plt.xlim(0, 5)
plt.ylim(0, 4)

# 축 레이블
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')

# 제목 및 범례
plt.title('로지스틱 회귀 모델의 클래스 경계 (w = [-3, -1, 3])')
plt.legend()

# 양쪽 영역에 색상 채우기
x1_fill = np.linspace(0, 5, 100)
x2_fill_upper = np.ones_like(x1_fill) * 4  # 상단 경계
x2_fill_lower = np.zeros_like(x1_fill) * 0  # 하단 경계

# 양성 영역(경계선 위쪽) 색상 채우기
plt.fill_between(x1_fill, x2_boundary, x2_fill_upper, color='lightblue', alpha=0.3, label='양성 영역')

# 음성 영역(경계선 아래쪽) 색상 채우기
plt.fill_between(x1_fill, x2_fill_lower, x2_boundary, color='lightcoral', alpha=0.3, label='음성 영역')

# 그래프 저장 및 표시
plt.savefig('class_boundary.png', dpi=300, bbox_inches='tight')
plt.show()

print("w = [-3, -1, 3]일 때의 클래스 경계식: x2 = (3 + x1)/3")
print("이 직선 위의 점들은 모델의 예측값이 0.5가 되는 지점입니다.")
print("직선 위쪽은 양성으로 예측되고, 아래쪽은 음성으로 예측됩니다.") 