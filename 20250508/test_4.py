import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# 한글 폰트 설정 (Windows 환경)
matplotlib.rc('font', family='Malgun Gothic')
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 깨짐 방지

# 베르누이 시행의 성공 확률 p와 시행 횟수 n
p = 0.73  # 예시: 관찰된 성공률 (예: 750/1000)
n = 1000

# 정규 근사: 평균과 표준편차
mu = p
sigma = np.sqrt(p * (1 - p) / n)

# 신뢰구간 (예: 80% 신뢰구간)
confidence = 0.80
z = norm.ppf(0.5 + confidence / 2)  # 양쪽 합쳐서 80%가 되는 z값

ci_lower = mu - z * sigma
ci_upper = mu + z * sigma

# x축: 확률의 범위
x = np.linspace(mu - 4*sigma, mu + 4*sigma, 1000)
y = norm.pdf(x, mu, sigma)

plt.figure(figsize=(8, 5))
plt.plot(x, y, label='정규 근사 (중심극한정리)')
plt.fill_between(x, y, where=(x >= ci_lower) & (x <= ci_upper), color='orange', alpha=0.3, label=f'{int(confidence*100)}% 신뢰구간')
plt.axvline(mu, color='red', linestyle='--', label='관찰된 성공률')
plt.title('베르누이 시행의 성공률 정규분포 근사 및 신뢰구간')
plt.xlabel('성공 확률 p')
plt.ylabel('확률 밀도')
plt.legend()
plt.grid(True)
plt.show()
