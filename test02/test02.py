import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# 예시 데이터 (임의의 점수와 실제 레이블)
y_true = np.array([1, 1, 0, 0, 1, 0, 1, 0])  # 실제 레이블
y_scores = np.array([0.9, 0.8, 0.7, 0.4, 0.6, 0.3, 0.2, 0.1])  # 예측 점수

# ROC 곡선 계산
fpr, tpr, thresholds = roc_curve(y_true, y_scores)
roc_auc = auc(fpr, tpr)

# ROC 곡선 그리기
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label='ROC curve (area = {:.2f})'.format(roc_auc))
plt.plot([0, 1], [0, 1], color='red', linestyle='--')  # 대각선
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.grid()
plt.show()