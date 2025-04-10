import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.font_manager as fm
from sklearn.metrics import roc_curve, auc
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

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

# 임의의 분류 데이터 생성
X, y = make_classification(n_samples=1000, n_features=20, n_informative=10, n_redundant=5, 
                          random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 세 가지 다른 분류기 학습
# Classifier 1: Logistic Regression
clf1 = LogisticRegression(random_state=42)
clf1.fit(X_train, y_train)
y_score1 = clf1.predict_proba(X_test)[:, 1]

# Classifier 2: Random Forest
clf2 = RandomForestClassifier(n_estimators=100, random_state=42)
clf2.fit(X_train, y_train)
y_score2 = clf2.predict_proba(X_test)[:, 1]

# Classifier 3: SVM
clf3 = SVC(probability=True, random_state=42)
clf3.fit(X_train, y_train)
y_score3 = clf3.predict_proba(X_test)[:, 1]

# ROC 곡선 계산
fpr1, tpr1, thresholds1 = roc_curve(y_test, y_score1)
fpr2, tpr2, thresholds2 = roc_curve(y_test, y_score2)
fpr3, tpr3, thresholds3 = roc_curve(y_test, y_score3)

# AUC 계산
auc1 = auc(fpr1, tpr1)
auc2 = auc(fpr2, tpr2)
auc3 = auc(fpr3, tpr3)

# ROC 곡선 그리기
plt.figure(figsize=(12, 10))

# 대각선 그리기 (무작위 분류기)
plt.plot([0, 1], [0, 1], color='gray', linestyle='--', alpha=0.8, label='무작위 분류')

# 각 분류기의 ROC 곡선 그리기
plt.plot(fpr1, tpr1, color='blue', lw=2, label=f'Logistic Regression (AUC = {auc1:.2f})')
plt.plot(fpr2, tpr2, color='green', lw=2, label=f'Random Forest (AUC = {auc2:.2f})')
plt.plot(fpr3, tpr3, color='red', lw=2, label=f'SVM (AUC = {auc3:.2f})')

# C1, C2, C3 점 표시
# 각 분류기에서 특정 임계값에서의 점 선택
idx1 = np.abs(thresholds1 - 0.3).argmin()  # 임계값 0.3에 가장 가까운 지점
idx2 = np.abs(thresholds2 - 0.5).argmin()  # 임계값 0.5에 가장 가까운 지점
idx3 = np.abs(thresholds3 - 0.7).argmin()  # 임계값 0.7에 가장 가까운 지점

plt.scatter(fpr1[idx1], tpr1[idx1], s=100, color='blue', label='C1')
plt.scatter(fpr2[idx2], tpr2[idx2], s=100, color='green', label='C2')
plt.scatter(fpr3[idx3], tpr3[idx3], s=100, color='red', label='C3')

# 기울기 1인 선을 그리기 위한 함수
def create_slope1_line(fpr, tpr, index):
    y0 = tpr[index] - fpr[index]  # y0 계산
    avg_recall = (1 - y0) / 2  # 평균 재현율 계산
    x = np.linspace(0, 1, 100)
    y = x + y0
    return x, y, y0, avg_recall

# 각 C1, C2, C3를 지나는 기울기 1인 선 그리기
x1, y1, y01, avg_recall1 = create_slope1_line(fpr1, tpr1, idx1)
valid_mask1 = (y1 >= 0) & (y1 <= 1)
plt.plot(x1[valid_mask1], y1[valid_mask1], '--', color='blue', alpha=0.5, 
         label=f'Slope 1 Line C1 (avg recall={avg_recall1:.2f})')

x2, y2, y02, avg_recall2 = create_slope1_line(fpr2, tpr2, idx2)
valid_mask2 = (y2 >= 0) & (y2 <= 1)
plt.plot(x2[valid_mask2], y2[valid_mask2], '--', color='green', alpha=0.5, 
         label=f'Slope 1 Line C2 (avg recall={avg_recall2:.2f})')

x3, y3, y03, avg_recall3 = create_slope1_line(fpr3, tpr3, idx3)
valid_mask3 = (y3 >= 0) & (y3 <= 1)
plt.plot(x3[valid_mask3], y3[valid_mask3], '--', color='red', alpha=0.5, 
         label=f'Slope 1 Line C3 (avg recall={avg_recall3:.2f})')

# 그래프 꾸미기
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('위양성률 (False Positive Rate)', fontsize=14)
plt.ylabel('진양성률 (True Positive Rate)', fontsize=14)
plt.title('ROC 곡선 (Receiver Operating Characteristic Curve)', fontsize=16)
plt.grid(True, linestyle='--', alpha=0.7)

# 수식 추가
plt.figtext(0.15, 0.2, r'$tpr = fpr + y_0$', fontsize=12)
plt.figtext(0.15, 0.17, r'$rec_{avg} = (tpr + tnr)/2$', fontsize=12)
plt.figtext(0.15, 0.14, r'$= (tpr + 1 - fpr)/2$', fontsize=12)
plt.figtext(0.15, 0.11, r'$= (1 - y_0)/2$', fontsize=12)
plt.figtext(0.15, 0.07, r'기울기 1인 선은 같은 평균 재현율을 가짐', fontsize=12)

# 범례 위치 조정 및 크기 조정
plt.legend(loc='lower right', fontsize=10)

# 그래프 저장
output_path = os.path.join(script_dir, 'roc_curve_visualization.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight')
plt.close()

print(f"ROC 곡선이 성공적으로 생성되었습니다: {output_path}")

# 각 분류기의 혼동 행렬과 성능 지표 계산 및 출력
def calculate_metrics(y_true, y_score, threshold=0.5):
    y_pred = (y_score >= threshold).astype(int)
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = 2 * (precision * tpr) / (precision + tpr) if (precision + tpr) > 0 else 0
    avg_recall = (tpr + tnr) / 2
    
    return {
        'TP': tp, 'TN': tn, 'FP': fp, 'FN': fn,
        'Accuracy': accuracy, 'TPR': tpr, 'TNR': tnr,
        'FPR': fpr, 'FNR': fnr, 'Precision': precision,
        'F1': f1, 'AvgRecall': avg_recall
    }

# 임계값 0.3, 0.5, 0.7에서의 지표 계산
thresholds = [0.3, 0.5, 0.7]
classifiers = [
    ("Logistic Regression", y_score1),
    ("Random Forest", y_score2),
    ("SVM", y_score3)
]

# 각 분류기의 각 임계값에서의 성능 지표 출력
print("\n분류기별 성능 지표:")
for clf_name, y_score in classifiers:
    print(f"\n{clf_name}:")
    for threshold in thresholds:
        metrics = calculate_metrics(y_test, y_score, threshold)
        print(f"  임계값 {threshold}:")
        print(f"    혼동 행렬: TP={metrics['TP']}, FN={metrics['FN']}, FP={metrics['FP']}, TN={metrics['TN']}")
        print(f"    정확도: {metrics['Accuracy']:.2f}")
        print(f"    TPR(민감도/재현율): {metrics['TPR']:.2f}")
        print(f"    TNR(특이도): {metrics['TNR']:.2f}")
        print(f"    FPR(위양성률): {metrics['FPR']:.2f}")
        print(f"    Precision(정밀도): {metrics['Precision']:.2f}")
        print(f"    F1 점수: {metrics['F1']:.2f}")
        print(f"    평균 재현율: {metrics['AvgRecall']:.2f}") 