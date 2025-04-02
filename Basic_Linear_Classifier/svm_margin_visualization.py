# SVM 마진 최대화와 서포트 벡터 시각화
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.preprocessing import StandardScaler
import matplotlib.font_manager as fm
import matplotlib as mpl
import platform

# 한글 폰트 설정
if platform.system() == 'Windows':
    font_path = r'C:\Windows\Fonts\malgun.ttf'  # 맑은 고딕 폰트 경로
    font_name = fm.FontProperties(fname=font_path).get_name()
    mpl.rc('font', family=font_name)
else:
    plt.rcParams['font.family'] = 'AppleGothic' if platform.system() == 'Darwin' else 'NanumGothic'

# 마이너스 기호 깨짐 방지
mpl.rcParams['axes.unicode_minus'] = False

# 원본 데이터셋 사용 (1-3.py와 동일한 데이터)
positive_points = np.array([[0, 3], [1, 2], [1, 4], [2, 3]])
negative_points = np.array([[0, 2], [1, 0], [3, 0], [4, 2]])

# 테스트 포인트
test_point = np.array([[-1, 1]])

# 데이터 결합
X = np.vstack([positive_points, negative_points])
y = np.array([1] * len(positive_points) + [0] * len(negative_points))

# 특성 스케일링
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
test_point_scaled = scaler.transform(test_point)

# 선형 SVM 모델 (C는 마진과 오류 사이의 균형을 조절)
# C 값이 작을수록 더 넓은 마진을 선호, C 값이 클수록 훈련 데이터의 올바른 분류를 더 중요시
svm_model = svm.SVC(kernel='linear', C=1.0)
svm_model.fit(X_scaled, y)

# 결정 경계 시각화를 위한 그리드 생성
x_min, x_max = min(X[:, 0].min(), test_point[0][0]) - 1, X[:, 0].max() + 1
y_min, y_max = min(X[:, 1].min(), test_point[0][1]) - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))

# 그리드 포인트 예측
grid = np.c_[xx.ravel(), yy.ravel()]
grid_scaled = scaler.transform(grid)
Z = svm_model.predict(grid_scaled).reshape(xx.shape)

# 서포트 벡터 인덱스
support_vector_indices = svm_model.support_

# 모델 가중치와 절편 (원래 스케일로 변환)
w = svm_model.coef_[0]
b = svm_model.intercept_[0]
w_original = w / scaler.scale_
b_original = b - np.sum(w * scaler.mean_ / scaler.scale_)

# 결정 경계 방정식: w[0]*x + w[1]*y + b = 0
# 마진 경계 방정식: w[0]*x + w[1]*y + b = 1 (양성 마진)
#                 w[0]*x + w[1]*y + b = -1 (음성 마진)

# 원래 스케일에서의 선 그리기를 위한 직선 함수
def get_hyperplane_value(x, w, b, offset):
    # w[0]*x + w[1]*y + b = offset
    # => y = (-w[0]*x - b + offset) / w[1]
    return (-w[0] * x - b + offset) / w[1]

# 마진 계산
margin = 1 / np.sqrt(np.sum(w**2))
print(f"SVM 마진: {margin:.4f}")

# 시각화
plt.figure(figsize=(10, 8))

# 데이터 포인트 시각화
plt.scatter(positive_points[:, 0], positive_points[:, 1], s=80, facecolors='none', 
            edgecolors='blue', linewidth=1.5, label='양성 클래스')
plt.scatter(negative_points[:, 0], negative_points[:, 1], s=80, facecolors='none', 
            edgecolors='red', linewidth=1.5, label='음성 클래스')

# 테스트 포인트 표시
plt.scatter(test_point[0][0], test_point[0][1], color='green', marker='*', s=200, 
            label=f'테스트 포인트 ({test_point[0][0]}, {test_point[0][1]})')

# 서포트 벡터 강조 표시
plt.scatter(X[support_vector_indices, 0], X[support_vector_indices, 1], 
            s=120, facecolors='none', edgecolors='green', linewidth=2.5, 
            label='서포트 벡터')

# 결정 경계 및 마진 경계 그리기
x_plot = np.linspace(x_min, x_max, 1000)

# 결정 경계
y_decision = get_hyperplane_value(x_plot, w, b, 0)
plt.plot(x_plot, y_decision, 'k-', label='결정 경계', linewidth=1.5)

# 마진 경계
y_pos_margin = get_hyperplane_value(x_plot, w, b, 1)
y_neg_margin = get_hyperplane_value(x_plot, w, b, -1)
plt.plot(x_plot, y_pos_margin, 'r--', label='마진 경계', linewidth=1.2)
plt.plot(x_plot, y_neg_margin, 'r--', linewidth=1.2)

# 마진 영역 표시 (옅은 색으로 표시)
plt.fill_between(x_plot, y_pos_margin, y_neg_margin, color='gray', alpha=0.1)

# 가중치 벡터 시각화 (결정 경계에 수직)
# 가중치 벡터의 방향이 잘 보이도록 중앙 위치에서 시작하게 설정
midpoint_x = np.mean([X[:, 0].min(), X[:, 0].max()])
midpoint_y = get_hyperplane_value(midpoint_x, w, b, 0)

scale = 1.0  # 벡터 크기 조정
plt.arrow(midpoint_x, midpoint_y, scale * w[0], scale * w[1], 
          head_width=0.2, head_length=0.2, fc='orange', ec='orange', 
          length_includes_head=True, label='가중치 벡터 w')

# 테스트 포인트 분류 결과
test_prediction = svm_model.predict(test_point_scaled)[0]
test_class = "양성(Positive)" if test_prediction == 1 else "음성(Negative)"
test_decision_value = svm_model.decision_function(test_point_scaled)[0]
print(f"\n테스트 포인트 ({test_point[0][0]}, {test_point[0][1]}) 분류 결과: {test_class}")
print(f"결정 함수 값: {test_decision_value:.4f}")

# 테스트 포인트에서 결정 경계까지의 거리 계산
test_distance = abs(test_decision_value) / np.sqrt(np.sum(w**2))
print(f"결정 경계까지의 거리: {test_distance:.4f}")

# 상세 정보 표시
decision_equation = f"{w[0]:.2f}x + {w[1]:.2f}y + {b:.2f} = 0"

# 그래프 제목 및 레이블 설정
plt.title(f'서포트 벡터 머신(SVM) 마진 최대화\n결정 경계: {decision_equation}\n마진: {margin:.4f}')
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend(loc='upper left')
plt.grid(True, linestyle='--', alpha=0.3)

# 축 범위 조정
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

# 가중치 벡터와 서포트 벡터 정보 출력
print(f"\n결정 경계 방정식: {w_original[0]:.4f}x + {w_original[1]:.4f}y + {b_original:.4f} = 0")
print(f"가중치 벡터 w: [{w_original[0]:.4f}, {w_original[1]:.4f}]")

print("\n서포트 벡터:")
for i, idx in enumerate(support_vector_indices):
    sv_class = "양성" if y[idx] == 1 else "음성"
    sv_point = X[idx]
    print(f"  {i+1}. 포인트 [{sv_point[0]:.2f}, {sv_point[1]:.2f}], 클래스: {sv_class}")
    
    # 서포트 벡터에서 결정 경계까지의 거리 계산
    sv_scaled = X_scaled[idx]
    distance = abs(np.dot(w, sv_scaled) + b) / np.sqrt(np.sum(w**2))
    print(f"     결정 경계까지의 거리: {distance:.4f}")

# 1-NN과 비교하여 테스트 포인트 (-1, 1)에 대한 예측 비교
# 모든 훈련 데이터 포인트와의 거리 계산
distances = []
for i, point in enumerate(X):
    dist = np.sqrt(np.sum((point - test_point[0])**2))
    distances.append((dist, 'positive' if y[i] == 1 else 'negative', point))

# 거리에 따라 정렬
distances.sort(key=lambda x: x[0])

# 가장 가까운 이웃 찾기 (k=1)
nearest_neighbor = distances[0]
nearest_distance, nearest_class, nearest_point = nearest_neighbor

print("\n1-NN 예측 결과:")
print(f"가장 가까운 이웃: {nearest_point}, 클래스: {nearest_class}")
print(f"거리: {nearest_distance:.4f}")
print(f"1-NN 분류 결과: {nearest_class}")

print("\nSVM과 1-NN 비교:")
print(f"SVM 예측: {test_class}")
print(f"1-NN 예측: {nearest_class}")

# 1-NN 가장 가까운 이웃으로의 선 표시
plt.plot([test_point[0][0], nearest_point[0]], [test_point[0][1], nearest_point[1]], 
         'g--', linewidth=1.2, label='1-NN 최근접 이웃')

plt.legend(loc='upper left')
plt.tight_layout()
plt.show() 