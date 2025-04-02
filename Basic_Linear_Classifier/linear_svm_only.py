# 선형 SVM(Support Vector Machine) 분류기 예제
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.preprocessing import StandardScaler
import matplotlib.font_manager as fm
import matplotlib as mpl
import sys

# 한글 폰트 설정 - 윈도우의 경우
import platform
if platform.system() == 'Windows':
    font_path = r'C:\Windows\Fonts\malgun.ttf'  # 맑은 고딕 폰트 경로
    font_name = fm.FontProperties(fname=font_path).get_name()
    mpl.rc('font', family=font_name)
else:
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

# 선형 SVM 모델 학습
svm_linear = svm.SVC(kernel='linear', C=1.0)
svm_linear.fit(X_scaled, y)

# 테스트 포인트 예측
linear_prediction = svm_linear.predict(test_point_scaled)
test_class = "양성(Positive)" if linear_prediction[0] == 1 else "음성(Negative)"

# 학습 결과 출력
print("========= 선형 SVM 분류 결과 =========")
print(f"테스트 포인트: ({test_point[0][0]}, {test_point[0][1]})")
print(f"선형 SVM 예측: {test_class}")

# 결정 경계 시각화를 위한 그리드 생성
x_min, x_max = min(X[:, 0].min(), test_point[0][0]) - 1, X[:, 0].max() + 1
y_min, y_max = min(X[:, 1].min(), test_point[0][1]) - 1, X[:, 1].max() + 1
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
    
    # 서포트 벡터와 결정 경계 사이의 거리 계산
    sv_scaled = X_scaled[idx]
    sv_distance = abs(np.dot(coef, sv_scaled) + intercept) / np.sqrt(np.sum(coef**2))
    print(f"     결정 경계까지의 거리: {sv_distance:.4f}")

# 테스트 포인트 결정 함수 값 계산
linear_decision_value = svm_linear.decision_function(test_point_scaled)[0]
original_decision_value = np.dot(test_point[0], w_original) + b_original

print(f"\n테스트 포인트 ({test_point[0][0]}, {test_point[0][1]})에 대한 결정 함수 값:")
print(f"  스케일된 값: {linear_decision_value:.4f}")
print(f"  원래 스케일: {original_decision_value:.4f}")

if linear_decision_value > 0:
    print("  해석: 테스트 포인트는 양성 클래스 영역에 위치합니다(결정 함수 값 > 0)")
else:
    print("  해석: 테스트 포인트는 음성 클래스 영역에 위치합니다(결정 함수 값 < 0)")

# 모델 마진 계산
margin = 1 / np.sqrt(np.sum(coef**2))
print(f"\n선형 SVM 마진: {margin:.4f}")
print("  - 마진은 결정 경계와 가장 가까운 서포트 벡터 사이의 거리를 의미합니다.")
print("  - 마진이 클수록 일반화 능력이 좋은 모델입니다.")

# 테스트 포인트와 결정 경계 사이의 거리 계산
test_distance = abs(linear_decision_value) / np.sqrt(np.sum(coef**2))
print(f"테스트 포인트와 결정 경계 사이의 거리: {test_distance:.4f}")
print("======================================")

# 시각화
plt.figure(figsize=(10, 8))

# 결정 경계 시각화
Z_linear = svm_linear.predict(grid_scaled).reshape(xx.shape)

# 등고선 그리기 - 배경 색상으로 클래스 표시 (투명도 낮춤)
plt.contourf(xx, yy, Z_linear, alpha=0.2, cmap=plt.cm.coolwarm)

# 결정 경계선 그리기 (얇게 설정)
plt.contour(xx, yy, Z_linear, colors='k', linestyles='-', linewidths=0.5)

# 데이터 포인트 시각화
plt.scatter(positive_points[:, 0], positive_points[:, 1], color='blue', edgecolors='k', 
            marker='o', s=80, label='양성(Positive)')
plt.scatter(negative_points[:, 0], negative_points[:, 1], color='red', edgecolors='k', 
            marker='o', s=80, label='음성(Negative)')

# 서포트 벡터 강조 표시
plt.scatter(X[support_vector_indices, 0], X[support_vector_indices, 1], 
            s=120, facecolors='none', edgecolors='green', linewidth=2, 
            label='서포트 벡터')

# 테스트 포인트 표시
plt.scatter(test_point[0][0], test_point[0][1], color='green', marker='*', s=200, 
            label=f'테스트 포인트 ({test_point[0][0]}, {test_point[0][1]}): {test_class}')

# 가중치 벡터 시각화 (결정 경계에 수직)
# 중앙점 계산
midpoint_x = np.mean([x_min, x_max])
midpoint_y = (-w_original[0] * midpoint_x - b_original) / w_original[1]

# 가중치 벡터 스케일링 및 표시
scale = 2.0  # 벡터 크기 조정
plt.arrow(midpoint_x, midpoint_y, scale * w_original[0], scale * w_original[1], 
          head_width=0.3, head_length=0.3, fc='orange', ec='orange', 
          length_includes_head=True, label='가중치 벡터 w')

# 마진 경계 그리기 (점선으로 표시)
def get_hyperplane_value(x, w, b, offset):
    # w[0]*x + w[1]*y + b = offset => y = (-w[0]*x - b + offset) / w[1]
    return (-w_original[0] * x - b_original + offset) / w_original[1]

# 마진 경계선 그리기
x_plot = np.linspace(x_min, x_max, 1000)
y_pos_margin = get_hyperplane_value(x_plot, w_original, b_original, 1)
y_neg_margin = get_hyperplane_value(x_plot, w_original, b_original, -1)
plt.plot(x_plot, y_pos_margin, 'r--', linewidth=1.0)
plt.plot(x_plot, y_neg_margin, 'r--', linewidth=1.0)
plt.fill_between(x_plot, y_pos_margin, y_neg_margin, color='gray', alpha=0.1)

# 테스트 포인트에서 결정 경계까지의 거리 표시
test_decision_y = (-w_original[0] * test_point[0][0] - b_original) / w_original[1]
plt.plot([test_point[0][0], test_point[0][0]], [test_point[0][1], test_decision_y], 'g:', linewidth=1.5)

# 그래프 제목 및 레이블 설정
plt.title(f'선형 SVM 분류\n결정 경계: {w_original[0]:.2f}x + {w_original[1]:.2f}y + {b_original:.2f} = 0\n마진: {margin:.4f}')
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend(loc='best')
plt.grid(True, linestyle='--', alpha=0.3)

# 축 범위 설정
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

plt.tight_layout()
plt.show() 