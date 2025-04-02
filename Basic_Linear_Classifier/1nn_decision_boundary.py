# 1-NN (1-Nearest Neighbor) 결정 경계 시각화
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
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

# 그래프에서 마이너스 기호 깨짐 방지
mpl.rcParams['axes.unicode_minus'] = False

# 데이터 포인트 설정 (1-3.py와 동일)
positive_points = np.array([[0, 3], [1, 2], [1, 4], [2, 3]])
negative_points = np.array([[0, 2], [1, 0], [3, 0], [4, 2]])

# 데이터 준비
X = np.vstack([positive_points, negative_points])
y = np.array([1] * len(positive_points) + [0] * len(negative_points))

# 테스트 포인트
test_point = np.array([-1, 1])

# 각 데이터 포인트와의 거리 계산
distances = []
for i, point in enumerate(X):
    dist = np.sqrt(np.sum((point - test_point)**2))
    distances.append((dist, 'positive' if y[i] == 1 else 'negative', point, y[i]))

# 거리에 따라 정렬
distances.sort(key=lambda x: x[0])

# 가장 가까운 이웃 찾기 (k=1)
nearest_neighbor = distances[0]
nearest_distance, nearest_class, nearest_point, nearest_label = nearest_neighbor

# 결과 출력
print("========= 1-NN 분류 결과 =========")
print(f"테스트 포인트: ({test_point[0]}, {test_point[1]})")
print(f"가장 가까운 이웃: {nearest_point}, 클래스: {nearest_class}")
print(f"거리: {nearest_distance:.4f}")
print(f"1-NN 분류 결과: {nearest_class}")
print("======================================")

# 1-NN 메시 그리드 생성
x_min, x_max = min(X[:, 0].min(), test_point[0]) - 1, X[:, 0].max() + 1
y_min, y_max = min(X[:, 1].min(), test_point[1]) - 1, X[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05),
                     np.arange(y_min, y_max, 0.05))

# 각 메시 포인트에 대해 1-NN 분류 수행
mesh_points = np.c_[xx.ravel(), yy.ravel()]
Z = np.zeros(mesh_points.shape[0], dtype=int)

# 각 메시 포인트에 대해 가장 가까운 훈련 데이터 포인트 찾기
for i, point in enumerate(mesh_points):
    # 모든 훈련 데이터와의 거리 계산
    dists = np.sqrt(np.sum((X - point)**2, axis=1))
    # 가장 가까운 포인트의 인덱스 찾기
    nearest_idx = np.argmin(dists)
    # 해당 포인트의 클래스로 분류
    Z[i] = y[nearest_idx]

# 메시 그리드 형태로 재구성
Z = Z.reshape(xx.shape)

# 시각화
plt.figure(figsize=(10, 8))

# 결정 경계 시각화 (배경 색상으로 표시)
cmap_light = ListedColormap(['#FFAAAA', '#AAAAFF'])
cmap_bold = ListedColormap(['#FF0000', '#0000FF'])

# 배경 색상으로 결정 경계 표시 (투명도 조절)
plt.contourf(xx, yy, Z, alpha=0.3, cmap=cmap_light)
# 결정 경계선 표시 (얇게 설정)
plt.contour(xx, yy, Z, colors='k', linestyles='-', linewidths=0.5)

# 데이터 포인트 시각화
plt.scatter(positive_points[:, 0], positive_points[:, 1], color='blue', edgecolors='k',
            marker='o', s=80, label='양성(Positive)')
plt.scatter(negative_points[:, 0], negative_points[:, 1], color='red', edgecolors='k',
            marker='o', s=80, label='음성(Negative)')

# 테스트 포인트 표시
plt.scatter(test_point[0], test_point[1], color='green', marker='*', s=200,
            label=f'테스트 포인트 ({test_point[0]}, {test_point[1]}): {nearest_class}')

# 가장 가까운 이웃으로의 선 표시
plt.plot([test_point[0], nearest_point[0]], [test_point[1], nearest_point[1]],
         'g--', linewidth=1.5, label=f'거리: {nearest_distance:.4f}')

# 그래프 제목 및 레이블 설정
plt.title('1-NN (1-최근접 이웃) 결정 경계\n테스트 포인트 (-1, 1)에 대한 분류 결과: ' + nearest_class)
plt.xlabel('x1')
plt.ylabel('x2')
plt.legend(loc='best')
plt.grid(True, linestyle='--', alpha=0.3)

# 축 범위 설정
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

# 추가: 보로노이 다이어그램 효과 표시
for i, (cls, color) in enumerate(zip([0, 1], ['red', 'blue'])):
    idx = np.where(y == cls)[0]
    plt.scatter(X[idx, 0], X[idx, 1], c=color, edgecolors='k',
                marker='o', s=80, alpha=0.8)

plt.tight_layout()
plt.show() 