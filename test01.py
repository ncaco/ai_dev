import numpy as np
import matplotlib.pyplot as plt

# 데이터 설정
positives = 2  # TP
negatives = 6  # TN

# 커버리지 플롯 생성
coverage_matrix = np.zeros((positives + 1, negatives + 1))

# 올바르게 분류된 경우
coverage_matrix[1:, 1:] = 1  # 모든 Positive-Negative 쌍을 올바르게 분류된 것으로 설정

# 잘못 분류된 경우
coverage_matrix[0, 1:] = 0  # 첫 번째 Positive가 잘못 분류된 경우
coverage_matrix[1:, 0] = 0  # 첫 번째 Negative가 잘못 분류된 경우

# 커버리지 플롯
plt.figure(figsize=(10, 5))

# 커버리지 플롯
plt.imshow(coverage_matrix, cmap='RdYlGn', interpolation='nearest')

# 색상 구분을 위한 레이블 설정
plt.xticks(np.arange(negatives + 1), [''] + [f'Neg {i}' for i in range(1, negatives + 1)])
plt.yticks(np.arange(positives + 1), [''] + [f'Pos {i}' for i in range(1, positives + 1)])

# 색상 바 추가
plt.colorbar(label='Classification Result')

# 제목 및 레이블 설정
plt.title('Coverage Plot')
plt.xlabel('Negatives sorted on decreasing score')
plt.ylabel('Positives sorted on decreasing score')

# 경계선 추가
for i in range(positives + 1):
    plt.axhline(i - 0.5, color='black', linewidth=1)
for j in range(negatives + 1):
    plt.axvline(j - 0.5, color='black', linewidth=1)

# 텍스트 추가
plt.text(0.5, 0.5, 'Correctly ranked', ha='center', va='center', color='green', fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
plt.text(0.5, 0.1, 'Ranking errors', ha='center', va='center', color='red', fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
plt.text(0.5, 0.3, 'Ties', ha='center', va='center', color='orange', fontsize=12, bbox=dict(facecolor='white', alpha=0.5))

# 플롯 표시
plt.tight_layout()
plt.show()