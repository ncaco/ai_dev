import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.font_manager as fm
from sklearn.metrics import confusion_matrix, roc_curve, auc

# 한글 폰트 설정
if os.name == 'nt':  # Windows
    font_path = 'C:/Windows/Fonts/malgun.ttf'  # 맑은 고딕
    font_name = fm.FontProperties(fname=font_path).get_name()
    plt.rc('font', family=font_name)
else:  # macOS나 Linux의 경우
    plt.rc('font', family='AppleGothic')

# 마이너스 기호 깨짐 방지
plt.rc('axes', unicode_minus=False)

# 현재 스크립트 위치 확인
script_dir = os.path.dirname(os.path.abspath(__file__))

# 주어진 값
# 첫 번째 행: 21 10 (TP와 FN)
# 두 번째 행: 1 55 (FP와 TN)
TP = 21
FN = 10
FP = 1
TN = 55

# 혼동 행렬 생성
conf_matrix = np.array([[TN, FP], [FN, TP]])

# 혼동 행렬에서 계산할 수 있는 지표들
total = TP + FN + FP + TN
accuracy = (TP + TN) / total
sensitivity = TP / (TP + FN)  # TPR (True Positive Rate)
specificity = TN / (TN + FP)  # TNR (True Negative Rate)
precision = TP / (TP + FP)
npv = TN / (TN + FN)  # Negative Predictive Value
f1_score = 2 * (precision * sensitivity) / (precision + sensitivity)
fpr = FP / (FP + TN)  # False Positive Rate
fnr = FN / (TP + FN)  # False Negative Rate

print("혼동 행렬:")
print(f"[[TN={TN}, FP={FP}],")
print(f" [FN={FN}, TP={TP}]]")
print("\n측정 지표:")
print(f"정확도(Accuracy): {accuracy:.4f}")
print(f"민감도(Sensitivity/TPR): {sensitivity:.4f}")
print(f"특이도(Specificity/TNR): {specificity:.4f}")
print(f"정밀도(Precision): {precision:.4f}")
print(f"음성예측도(NPV): {npv:.4f}")
print(f"F1 점수: {f1_score:.4f}")
print(f"위양성률(FPR): {fpr:.4f}")
print(f"위음성률(FNR): {fnr:.4f}")

# 그래프 생성을 위한 설정
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# 1. 혼동 행렬 시각화
ax1.set_title('혼동 행렬 (Confusion Matrix)', fontsize=14)
im = ax1.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
ax1.set_xticks(np.arange(2))
ax1.set_yticks(np.arange(2))
ax1.set_xticklabels(['음성 예측', '양성 예측'], fontsize=12)
ax1.set_yticklabels(['실제 음성', '실제 양성'], fontsize=12)

# 혼동 행렬에 값 표시
thresh = conf_matrix.max() / 2.
for i in range(2):
    for j in range(2):
        text = ax1.text(j, i, conf_matrix[i, j],
                       ha="center", va="center", 
                       color="white" if conf_matrix[i, j] > thresh else "black",
                       fontsize=14)

# 정확도, 민감도, 특이도 등의 정보 추가
metrics_text = f"정확도: {accuracy:.4f}\n" \
               f"민감도: {sensitivity:.4f}\n" \
               f"특이도: {specificity:.4f}\n" \
               f"정밀도: {precision:.4f}\n" \
               f"F1 점수: {f1_score:.4f}"
               
ax1.text(1.1, 1.0, metrics_text, transform=ax1.transAxes, fontsize=12,
        bbox=dict(facecolor='white', alpha=0.8))

# 2. ROC 곡선
# 실제 ROC 곡선을 그리기 위해 임의의 모델 점수를 생성
# 실제 클래스 레이블 생성: 1은 양성, 0은 음성
y_true = np.concatenate([np.ones(TP + FN), np.zeros(FP + TN)])

# 임의의 모델 점수를 생성
# 실제 양성 클래스에는 높은 점수, 음성 클래스에는 낮은 점수 부여
# 주어진 혼동 행렬과 일치하도록 점수 분포 조정

# 주어진 점에서의 TPR과 FPR을 사용하여 점수 분포를 조정
np.random.seed(42)  # 재현성을 위한 시드 설정

# 양성 클래스의 점수: TP는 높은 점수, FN은 낮은 점수
positive_scores = np.concatenate([
    np.random.uniform(0.6, 1.0, TP),    # TP: 높은 점수 (임계값 위)
    np.random.uniform(0.0, 0.6, FN)     # FN: 낮은 점수 (임계값 아래)
])

# 음성 클래스의 점수: TN은 낮은 점수, FP는 높은 점수
negative_scores = np.concatenate([
    np.random.uniform(0.6, 1.0, FP),    # FP: 높은 점수 (임계값 위)
    np.random.uniform(0.0, 0.6, TN)     # TN: 낮은 점수 (임계값 아래)
])

# 모든 점수를 결합
y_scores = np.concatenate([positive_scores, negative_scores])

# ROC 곡선 계산
fpr_curve, tpr_curve, thresholds = roc_curve(y_true, y_scores)
roc_auc = auc(fpr_curve, tpr_curve)

# ROC 곡선 그리기
ax2.set_title('ROC 곡선 (ROC Curve)', fontsize=14)
ax2.plot(fpr_curve, tpr_curve, 'b-', lw=2, label=f'ROC 곡선 (AUC = {roc_auc:.4f})')
ax2.plot([0, 1], [0, 1], 'k--', label='무작위 분류 (AUC = 0.5)')
ax2.set_xlim([0.0, 1.0])
ax2.set_ylim([0.0, 1.05])
ax2.set_xlabel('위양성률 (False Positive Rate)', fontsize=12)
ax2.set_ylabel('진양성률 (True Positive Rate)', fontsize=12)
ax2.grid(True, linestyle='--', alpha=0.7)

# 주어진 혼동 행렬에서의 ROC 포인트 표시
ax2.plot(fpr, sensitivity, 'ro', markersize=10, label=f'모델 (TPR={sensitivity:.4f}, FPR={fpr:.4f})')

# 임계값 점들 표시
# 몇 개의 임계값 포인트만 표시
threshold_points = np.linspace(0, len(thresholds)-1, 5, dtype=int)
for i in threshold_points:
    if i < len(thresholds):  # 인덱스 에러 방지
        ax2.plot(fpr_curve[i], tpr_curve[i], 'go', markersize=6)
        ax2.text(fpr_curve[i]+0.02, tpr_curve[i]-0.02, f'T={thresholds[i]:.2f}', 
                fontsize=8, bbox=dict(facecolor='white', alpha=0.7))

ax2.legend(loc='lower right', fontsize=12)

# ROC 곡선 관련 정보 추가
roc_text = f"TPR: {sensitivity:.4f}\n" \
           f"FPR: {fpr:.4f}\n" \
           f"AUC: {roc_auc:.4f}"
           
ax2.text(0.05, 0.95, roc_text, transform=ax2.transAxes, fontsize=12,
        bbox=dict(facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig(os.path.join(script_dir, 'custom_roc_confusion_matrix.png'), dpi=150, bbox_inches='tight')
plt.close()

print(f"\n그래프가 성공적으로 생성되었습니다: {os.path.join(script_dir, 'custom_roc_confusion_matrix.png')}")

# 추가: 전체 ROC 곡선 상세 그래프
plt.figure(figsize=(10, 8))
plt.plot(fpr_curve, tpr_curve, 'b-', lw=2, label=f'ROC 곡선 (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], 'k--', label='무작위 분류 (AUC = 0.5)')
plt.plot(fpr, sensitivity, 'ro', markersize=10, label=f'주어진 모델 (TPR={sensitivity:.4f}, FPR={fpr:.4f})')

# 더 많은 임계값 포인트 표시
threshold_points = np.linspace(0, len(thresholds)-1, 10, dtype=int)
for i in threshold_points:
    if i < len(thresholds):  # 인덱스 에러 방지
        plt.plot(fpr_curve[i], tpr_curve[i], 'go', markersize=6)
        plt.text(fpr_curve[i]+0.02, tpr_curve[i]-0.02, f'T={thresholds[i]:.2f}', 
                fontsize=8, bbox=dict(facecolor='white', alpha=0.7))

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('위양성률 (False Positive Rate)', fontsize=14)
plt.ylabel('진양성률 (True Positive Rate)', fontsize=14)
plt.title('ROC 곡선 상세 (Detailed ROC Curve)', fontsize=16)
plt.legend(loc='lower right', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)

# ROC 곡선의 특정 영역 확대 (주어진 점 주변)
plt.axes([0.55, 0.3, 0.3, 0.3])  # 왼쪽, 아래, 너비, 높이
plt.plot(fpr_curve, tpr_curve, 'b-', lw=2)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, sensitivity, 'ro', markersize=10)

# 확대 영역 설정
zoom_margin = 0.1
plt.xlim(max(0, fpr - zoom_margin), min(1, fpr + zoom_margin))
plt.ylim(max(0, sensitivity - zoom_margin), min(1, sensitivity + zoom_margin))
plt.grid(True, linestyle='--', alpha=0.7)
plt.title('확대 보기', fontsize=10)

plt.tight_layout()
plt.savefig(os.path.join(script_dir, 'roc_curve.png'), dpi=150, bbox_inches='tight')
plt.close()

print(f"상세 ROC 곡선 그래프가 생성되었습니다: {os.path.join(script_dir, 'roc_curve.png')}") 