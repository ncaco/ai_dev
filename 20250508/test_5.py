import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import matplotlib

# 한글 폰트 설정 (Windows 환경)
matplotlib.rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 깨짐 방지


# 평균 0, 분산 1인 정규분포 (표준정규분포)
mu = 0.75
sigma = np.sqrt(0.75/1000)

# 90% 신뢰구간에 해당하는 z값
z_left = -1.28
z_right = z_left * -1

# x축 범위 설정
x = np.linspace(mu - 4*sigma, mu + 4*sigma, 1000)
y = norm.pdf(x, mu, sigma)

plt.figure(figsize=(8, 5))
plt.plot(x, y, label='표준정규분포')

# 90% 신뢰구간 영역 색칠
plt.fill_between(x, y, where=(x >= z_left) & (x <= z_right), color='skyblue', alpha=0.5, label='90% 신뢰구간')

# z=-1.65, z=1.65 선 그리기
plt.axvline(z_left, color='purple', linestyle='--', label=f'z={z_left}')
plt.axvline(z_right, color='purple', linestyle='--', label=f'z={z_right}')

# z 값 텍스트 추가
plt.text(z_left-0.2, 0.05, f'z={z_left}', color='purple', fontsize=10)
plt.text(z_right+0.1, 0.05, f'z={z_right}', color='purple', fontsize=10)

plt.title('표준정규분포에서 90% 신뢰구간')
plt.xlabel('x')
plt.ylabel('확률 밀도')
plt.legend()
plt.grid(True)
plt.show()

# 90% 신뢰구간 확률 계산 (확인용)
prob = norm.cdf(z_right) - norm.cdf(z_left)
print(f"(-1.65 <= X <= 1.65) 구간의 확률: {prob*100:.1f}%")




p = (0.75 + (1.28**2) / 1000 +- np.sqrt(0.75/1000 - 0.75**2/1000 + (1.28**2) / (4*1000**2))) / (1+(1.28**2) / 1000)
print(p)











