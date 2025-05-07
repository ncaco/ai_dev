import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# 예시 데이터
TP = 2
TN = 6
FP = 2
FN = 0

# 인스턴스 정렬 (예시)
# 실제 데이터와 결정 경계에서의 거리로 대체해야 함
distances = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])  # 거리
labels = np.array([1, 1, 0, 0, 0, 0, 1, 1])  # 실제 레이블

# 거리 기준으로 정렬
sorted_indices = np.argsort(distances)
sorted_labels = labels[sorted_indices]

# 커버리지 곡선
TPR = np.cumsum(sorted_labels)  # True Positive Count
FPR = np.cumsum(1 - sorted_labels)  # False Positive Count

# 커버리지 곡선 그리기
plt.figure(figsize=(10, 6))

# 커버리지 곡선
plt.plot(FPR, TPR, marker='o', color='red', label='Coverage Curve')
plt.title('Coverage Curve')
plt.xlabel('False Positives')
plt.ylabel('True Positives')
plt.xlim(0, FP + 1)  # x축 범위 설정
plt.ylim(0, TP + 1)  # y축 범위 설정
plt.grid()

# TP1, TP2, FP1, FP2 포인트 정의
TP1 = 1
TP2 = 2
FP1 = 1
FP2 = 2

# 포인트 표시
plt.scatter([FP1, FP2], [TP1, TP2], color='blue')
plt.text(FP1, TP1, 'TP1', fontsize=12, verticalalignment='bottom', horizontalalignment='right')
plt.text(FP2, TP2, 'TP2', fontsize=12, verticalalignment='bottom', horizontalalignment='right')

# 최대 TP 및 FP 표시
plt.axhline(y=TP, color='green', linestyle='--', label='TP Max')
plt.axvline(x=FP, color='blue', linestyle='--', label='FP Max')
plt.legend()

plt.tight_layout()
plt.show()