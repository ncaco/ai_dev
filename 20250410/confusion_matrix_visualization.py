import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import matplotlib.font_manager as fm
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

# 폰트 설정
# Windows의 경우 맑은 고딕 폰트 사용
if os.name == 'nt':  # Windows
    font_path = 'C:/Windows/Fonts/malgun.ttf'  # 맑은 고딕
    font_name = fm.FontProperties(fname=font_path).get_name()
    plt.rc('font', family=font_name)
else:  # macOS나 Linux의 경우
    plt.rc('font', family='AppleGothic')  # macOS
    
# 마이너스 기호 깨짐 방지
plt.rc('axes', unicode_minus=False)

# 현재 스크립트 위치 확인
script_dir = os.path.dirname(os.path.abspath(__file__))

# 예시 데이터 (슬라이드의 예시를 사용)
y_true = np.array([1] * 50 + [0] * 50)  # 실제 값 (1: 양성, 0: 음성)
y_pred = np.array([1] * 30 + [0] * 20 + [1] * 10 + [0] * 40)  # 예측 값

# 혼동 행렬 계산
cm = confusion_matrix(y_true, y_pred)
tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

# 성능 지표 계산
accuracy = accuracy_score(y_true, y_pred)
tpr = tp / (tp + fn)  # 민감도(Sensitivity) or 재현율(Recall) or 진양성률(TPR)
tnr = tn / (tn + fp)  # 특이도(Specificity) or 진음성률(TNR)
fpr = fp / (fp + tn)  # 위양성률(FPR) or 거짓 경보율(False Alarm Rate)
fnr = fn / (fn + tp)  # 위음성률(FNR)
precision = tp / (tp + fp)  # 정밀도(Precision)
f1 = 2 * (precision * tpr) / (precision + tpr)  # F1 점수

# 결과 출력
print(f"혼동 행렬:\n{cm}")
print(f"정확도(Accuracy): {accuracy:.2f}")
print(f"민감도/재현율(TPR): {tpr:.2f}")
print(f"특이도(TNR): {tnr:.2f}")
print(f"위양성률(FPR): {fpr:.2f}")
print(f"위음성률(FNR): {fnr:.2f}")
print(f"정밀도(Precision): {precision:.2f}")
print(f"F1 점수: {f1:.2f}")

# 저장 경로 설정
confusion_matrix_path = os.path.join(script_dir, 'confusion_matrix.png')
roc_curve_path = os.path.join(script_dir, 'roc_curve.png')
performance_metrics_path = os.path.join(script_dir, 'performance_metrics.png')

# 혼동 행렬 시각화
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['예측: 음성', '예측: 양성'],
            yticklabels=['실제: 음성', '실제: 양성'])
plt.ylabel('실제 레이블')
plt.xlabel('예측 레이블')
plt.title('혼동 행렬')
plt.tight_layout()
plt.savefig(confusion_matrix_path)
plt.close()

# ROC 곡선 (Threshold 변화에 따른 TPR vs FPR)을 위한 임의의 확률 생성
# 실제로는 모델이 출력하는 확률값을 사용해야 함
np.random.seed(42)
y_scores = np.random.random(len(y_true))

# 다양한 임계값에서의 TPR, FPR 계산
thresholds = np.linspace(0, 1, 100)
tprs = []
fprs = []

for threshold in thresholds:
    y_pred_threshold = (y_scores >= threshold).astype(int)
    cm_t = confusion_matrix(y_true, y_pred_threshold)
    tn_t, fp_t, fn_t, tp_t = cm_t.ravel()
    
    tpr_t = tp_t / (tp_t + fn_t) if (tp_t + fn_t) > 0 else 0
    fpr_t = fp_t / (fp_t + tn_t) if (fp_t + tn_t) > 0 else 0
    
    tprs.append(tpr_t)
    fprs.append(fpr_t)

# ROC 곡선 그리기
plt.figure(figsize=(8, 6))
plt.plot(fprs, tprs, 'b-', linewidth=2, label='ROC 곡선')
plt.plot([0, 1], [0, 1], 'r--', linewidth=2, label='무작위 분류')
plt.xlabel('위양성률(FPR)')
plt.ylabel('진양성률(TPR)')
plt.title('ROC 곡선')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(roc_curve_path)
plt.close()

# 성능 지표 시각화
metrics = ['정확도', '민감도/재현율', '특이도', '정밀도', 'F1 점수']
values = [accuracy, tpr, tnr, precision, f1]

plt.figure(figsize=(10, 6))
plt.bar(metrics, values, color='teal')
plt.ylim(0, 1)
plt.ylabel('점수')
plt.title('분류 성능 지표')
for i, v in enumerate(values):
    plt.text(i, v + 0.02, f'{v:.2f}', ha='center')
plt.tight_layout()
plt.savefig(performance_metrics_path)
plt.close()

print("모든 그래프가 생성되었습니다.")
print(f"- {confusion_matrix_path}: 혼동 행렬")
print(f"- {roc_curve_path}: ROC 곡선")
print(f"- {performance_metrics_path}: 성능 지표 그래프") 