# SVM(Support Vector Machine) Classifier Example
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.preprocessing import StandardScaler
import matplotlib.font_manager as fm
import matplotlib as mpl

# 한글 폰트 설정 - 윈도우의 경우
import platform
if platform.system() == 'Windows':
    # 나눔고딕 또는 맑은 고딕 폰트 사용
    font_path = r'C:\Windows\Fonts\malgun.ttf'  # 맑은 고딕 폰트 경로
    font_name = fm.FontProperties(fname=font_path).get_name()
    mpl.rc('font', family=font_name)
else:
    # macOS나 Linux의 경우 다른 폰트 설정이 필요할 수 있음
    plt.rcParams['font.family'] = 'AppleGothic' if platform.system() == 'Darwin' else 'NanumGothic'

# 그래프에서 마이너스 기호 깨짐 방지
mpl.rcParams['axes.unicode_minus'] = False

# 데이터 포인트 설정 (1-3.py와 동일)
positive_points = np.array([[0, 3], [1, 2], [1, 4], [2, 3]])
negative_points = np.array([[0, 2], [1, 0], [3, 0], [4, 2]])

# 데이터 준비
X = np.vstack([positive_points, negative_points])
y = np.array([1] * len(positive_points) + [0] * len(negative_points))

# 테스트 포인트
test_point = np.array([[-1, 1]])

# 특성 스케일링 (SVM 성능 향상을 위해)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
test_point_scaled = scaler.transform(test_point)

# SVM 모델 학습 (선형 커널과 RBF 커널 비교)
svm_linear = svm.SVC(kernel='linear', C=1.0)
svm_rbf = svm.SVC(kernel='rbf', gamma='auto', C=1.0)

# 모델 학습
svm_linear.fit(X_scaled, y)
svm_rbf.fit(X_scaled, y)

# 테스트 포인트 예측
linear_prediction = svm_linear.predict(test_point_scaled)
rbf_prediction = svm_rbf.predict(test_point_scaled)

# 학습 결과 출력
print("========= SVM 분류 결과 =========")
print(f"테스트 포인트: ({test_point[0][0]}, {test_point[0][1]})")
print(f"선형 SVM 예측: {'양성(Positive)' if linear_prediction[0] == 1 else '음성(Negative)'}")
print(f"RBF 커널 SVM 예측: {'양성(Positive)' if rbf_prediction[0] == 1 else '음성(Negative)'}")

# 결정 경계 시각화를 위한 그리드 생성
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                     np.arange(y_min, y_max, 0.1))

# 그리드 포인트 예측
grid = np.c_[xx.ravel(), yy.ravel()]
grid_scaled = scaler.transform(grid)

# 선형 SVM 결정 함수 정보
print("\n========= 선형 SVM 세부 정보 =========")
coef = svm_linear.coef_[0]
intercept = svm_linear.intercept_[0]
# 원래 스케일로 변환
w_original = coef / scaler.scale_
b_original = intercept - np.sum(coef * scaler.mean_ / scaler.scale_)
print(f"결정 경계 방정식: {w_original[0]:.4f}x + {w_original[1]:.4f}y + {b_original:.4f} = 0")
print(f"가중치 벡터 w: [{w_original[0]:.4f}, {w_original[1]:.4f}]")
print(f"절편 b: {b_original:.4f}")

# 서포트 벡터 확인
support_vector_indices = svm_linear.support_
print(f"\n선형 SVM 서포트 벡터 수: {len(support_vector_indices)}")
print("서포트 벡터:")
for i, idx in enumerate(support_vector_indices):
    sv_class = "양성(Positive)" if y[idx] == 1 else "음성(Negative)"
    print(f"  {i+1}. 포인트 {X[idx]}, 클래스: {sv_class}")

# 테스트 포인트에 대한 결정 함수 값 계산
linear_decision_value = svm_linear.decision_function(test_point_scaled)[0]
original_decision_value = np.dot(test_point[0], w_original) + b_original
print(f"\n테스트 포인트 ({test_point[0][0]}, {test_point[0][1]})에 대한 결정 함수 값:")
print(f"  스케일된 값: {linear_decision_value:.4f}")
print(f"  원래 스케일: {original_decision_value:.4f}")
print(f"  해석: 결정 경계와의 거리, 부호는 클래스(양수: 양성, 음수: 음성)를 나타냅니다.")

# 모델 마진 계산
margin = 1 / np.sqrt(np.sum(coef**2))
print(f"\n선형 SVM 마진: {margin:.4f}")
print("  - 마진은 결정 경계와 가장 가까운 서포트 벡터 사이의 거리를 의미합니다.")
print("  - 마진이 클수록 일반화 능력이 좋은 모델입니다.")
print("======================================")

# 시각화
plt.figure(figsize=(15, 6))

# 첫 번째 그래프: 선형 SVM
plt.subplot(1, 2, 1)
# 결정 경계 시각화 - 배경 색상 구분
Z_linear = svm_linear.predict(grid_scaled).reshape(xx.shape)
plt.contourf(xx, yy, Z_linear, alpha=0.2, cmap=plt.cm.coolwarm)
# 결정 경계선 굵기 얇게 설정
plt.contour(xx, yy, Z_linear, colors='k', linestyles='-', linewidths=0.5)

# 데이터 포인트 시각화
plt.scatter(positive_points[:, 0], positive_points[:, 1], color='blue', edgecolors='k', label='양성(Positive)')
plt.scatter(negative_points[:, 0], negative_points[:, 1], color='red', edgecolors='k', label='음성(Negative)')

# 서포트 벡터 강조 표시
plt.scatter(X[support_vector_indices, 0], X[support_vector_indices, 1], 
            s=100, facecolors='none', edgecolors='green', label='서포트 벡터')

# 테스트 포인트 표시
plt.scatter(test_point[0][0], test_point[0][1], color='green', marker='*', s=200, 
            label=f'테스트 포인트 ({test_point[0][0]}, {test_point[0][1]})')

plt.title(f'선형 SVM\n결정 방정식: {w_original[0]:.2f}x + {w_original[1]:.2f}y + {b_original:.2f} = 0')
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.3)

# 두 번째 그래프: RBF 커널 SVM
plt.subplot(1, 2, 2)
# 결정 경계 시각화 - 배경 색상 구분
Z_rbf = svm_rbf.predict(grid_scaled).reshape(xx.shape)
plt.contourf(xx, yy, Z_rbf, alpha=0.2, cmap=plt.cm.coolwarm)
# 결정 경계선 굵기 얇게 설정
plt.contour(xx, yy, Z_rbf, colors='k', linestyles='-', linewidths=0.5)

# 데이터 포인트 시각화
plt.scatter(positive_points[:, 0], positive_points[:, 1], color='blue', edgecolors='k', label='양성(Positive)')
plt.scatter(negative_points[:, 0], negative_points[:, 1], color='red', edgecolors='k', label='음성(Negative)')

# RBF 커널의 서포트 벡터 표시
rbf_support_vector_indices = svm_rbf.support_
plt.scatter(X[rbf_support_vector_indices, 0], X[rbf_support_vector_indices, 1], 
            s=100, facecolors='none', edgecolors='green', label='서포트 벡터')

# 테스트 포인트 표시
plt.scatter(test_point[0][0], test_point[0][1], color='green', marker='*', s=200, 
            label=f'테스트 포인트 ({test_point[0][0]}, {test_point[0][1]})')

plt.title('RBF 커널 SVM\n(비선형 결정 경계)')
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.3)

plt.tight_layout()
plt.show() 