import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.font_manager as fm
from sklearn.metrics import roc_curve, auc

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

# 생성한 분류기 지점들 (ROC 플롯상의 지점들)
# 슬라이드 예시의 C1, C2, C3 지점
points = {
    'C1': {'fpr': 0.2, 'tpr': 0.6},
    'C2': {'fpr': 0.1, 'tpr': 0.8},
    'C3': {'fpr': 0.3, 'tpr': 0.9}
}

# 테스트 세트 크기와 레이블 분포 (슬라이드에서 주어진 값)
Te = 20
Pos = 5
Neg = 15

# 그림 1: Coverage 플롯과 ROC 플롯 함께 시각화
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

# 1. Coverage 플롯 (왼쪽)
ax1.set_title('Coverage 플롯', fontsize=14)
ax1.set_xlim(0, Neg)
ax1.set_ylim(0, Pos)
ax1.set_xlabel('Negatives', fontsize=12)
ax1.set_ylabel('Positives', fontsize=12)
ax1.grid(True, linestyle='--', alpha=0.7)

# 평균 재현율 등곡선 (Isometric) - 기울기 Pos/Neg의 선
# 슬라이드에서 보이는 등곡선 그리기
# 기울기 = Pos/Neg = 5/15 = 1/3
x_recall = np.linspace(0, Neg, 100)
for i, offset in enumerate([0.2, 0.4, 0.6, 0.8]):
    y_recall = (Pos/Neg) * x_recall + offset * Pos
    mask = (y_recall >= 0) & (y_recall <= Pos)
    if i == 1:  # 두 번째 선에만 레이블 추가
        ax1.plot(x_recall[mask], y_recall[mask], '--', color='blue', alpha=0.7, 
                 label='평균 재현율 등곡선\n(기울기 = Pos/Neg)')
    else:
        ax1.plot(x_recall[mask], y_recall[mask], '--', color='blue', alpha=0.7)

# 정확도 등곡선 (Isometric) - 기울기 1인 선
x_acc = np.linspace(0, Neg, 100)
for i, offset in enumerate([0.1, 0.3, 0.5, 0.7]):
    y_acc = x_acc + offset * Pos
    mask = (y_acc >= 0) & (y_acc <= Pos)
    if i == 1:  # 두 번째 선에만 레이블 추가
        ax1.plot(x_acc[mask], y_acc[mask], '-.', color='purple', alpha=0.7, 
                 label='정확도 등곡선\n(기울기 = 1)')
    else:
        ax1.plot(x_acc[mask], y_acc[mask], '-.', color='purple', alpha=0.7)

# C1, C2, C3 점 표시
for point_name, point_data in points.items():
    # Coverage 플롯에서의 좌표 계산
    fp = point_data['fpr'] * Neg
    tp = point_data['tpr'] * Pos
    ax1.scatter(fp, tp, s=100, label=point_name)
    ax1.text(fp + 0.2, tp + 0.2, point_name, fontsize=12)

ax1.legend(loc='upper left', fontsize=10)

# 2. ROC 플롯 (오른쪽)
ax2.set_title('ROC 플롯', fontsize=14)
ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)
ax2.set_xlabel('False Positive Rate (FPR)', fontsize=12)
ax2.set_ylabel('True Positive Rate (TPR)', fontsize=12)
ax2.grid(True, linestyle='--', alpha=0.7)

# 대각선 그리기 (무작위 분류기)
ax2.plot([0, 1], [0, 1], color='gray', linestyle='--', alpha=0.8, label='무작위 분류')

# 정확도 등곡선 (Isometric) - 기울기 1/clr인 선
# clr = Pos/Neg, 따라서 기울기 = 1/(Pos/Neg) = Neg/Pos = 15/5 = 3
clr = Pos / Neg
x_acc_iso = np.linspace(0, 1, 100)
for i, offset in enumerate([0.1, 0.3, 0.5, 0.7]):
    y_acc_iso = (Neg/Pos) * x_acc_iso + offset
    mask = (y_acc_iso >= 0) & (y_acc_iso <= 1)
    if i == 1:
        ax2.plot(x_acc_iso[mask], y_acc_iso[mask], '-.', color='purple', alpha=0.7, 
                 label=f'정확도 등곡선\n(기울기 = 1/clr = {Neg/Pos:.1f})')
    else:
        ax2.plot(x_acc_iso[mask], y_acc_iso[mask], '-.', color='purple', alpha=0.7)

# 평균 재현율 등곡선 (Isometric) - 기울기 1인 선
x_recall_iso = np.linspace(0, 1, 100)
for i, offset in enumerate([0.1, 0.3, 0.5, 0.7]):
    y_recall_iso = x_recall_iso + offset
    mask = (y_recall_iso >= 0) & (y_recall_iso <= 1)
    if i == 1:
        ax2.plot(x_recall_iso[mask], y_recall_iso[mask], '--', color='blue', alpha=0.7, 
                 label='평균 재현율 등곡선\n(기울기 = 1)')
    else:
        ax2.plot(x_recall_iso[mask], y_recall_iso[mask], '--', color='blue', alpha=0.7)

# C1, C2, C3 점 표시
for point_name, point_data in points.items():
    ax2.scatter(point_data['fpr'], point_data['tpr'], s=100, label=point_name)
    ax2.text(point_data['fpr'] + 0.02, point_data['tpr'] + 0.02, point_name, fontsize=12)

# 각 점의 정확도 계산
for point_name, point_data in points.items():
    fpr = point_data['fpr']
    tpr = point_data['tpr']
    
    # 이 지점에서의 TP, FP, TN, FN 계산
    tp = tpr * Pos
    fp = fpr * Neg
    fn = Pos - tp
    tn = Neg - fp
    
    # 정확도 계산
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    avg_recall = (tpr + (1 - fpr)) / 2
    
    print(f"{point_name}:")
    print(f"  FPR: {fpr:.2f}, TPR: {tpr:.2f}")
    print(f"  TP: {tp:.1f}, FP: {fp:.1f}, FN: {fn:.1f}, TN: {tn:.1f}")
    print(f"  정확도: {accuracy:.2f}")
    print(f"  평균 재현율: {avg_recall:.2f}")

# C2와 C3의 성능 비교
c2_acc = (points['C2']['tpr'] * Pos + (1 - points['C2']['fpr']) * Neg) / Te
c2_avg_recall = (points['C2']['tpr'] + (1 - points['C2']['fpr'])) / 2

c3_acc = (points['C3']['tpr'] * Pos + (1 - points['C3']['fpr']) * Neg) / Te
c3_avg_recall = (points['C3']['tpr'] + (1 - points['C3']['fpr'])) / 2

comparison_text = f"C2 vs C3:\nC2 정확도: {c2_acc:.2f}, 평균 재현율: {c2_avg_recall:.2f}\nC3 정확도: {c3_acc:.2f}, 평균 재현율: {c3_avg_recall:.2f}"
if c2_acc > c3_acc and c2_avg_recall > c3_avg_recall:
    comparison_text += "\nC2가 C3보다 정확도와 평균 재현율 모두 높음"
else:
    comparison_text += "\n성능 비교가 명확하지 않음"

ax2.text(0.05, 0.05, comparison_text, transform=ax2.transAxes, fontsize=10,
        bbox=dict(facecolor='white', alpha=0.8))

ax2.legend(loc='lower right', fontsize=10)

# 전체 그림 제목 설정
plt.suptitle('ROC 플롯과 등곡선(Isometric Curves)', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])

# 그래프 저장
output_path = os.path.join(script_dir, 'roc_isometric_curves.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight')
plt.close()

print(f"ROC 플롯과 등곡선이 성공적으로 생성되었습니다: {output_path}")
print("\n분류기별 성능 비교:")
print(comparison_text) 