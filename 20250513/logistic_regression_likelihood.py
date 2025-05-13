import numpy as np
import matplotlib.pyplot as plt

# 데이터 정의
# 양성 예제들 (검은 점)
positive_examples = np.array([
    [1, 1, 2],  # 첫 번째 점 (1, 2)
    [1, 2, 3]   # 두 번째 점 (2, 3)
])

# 음성 예제들 (흰 점)
negative_examples = np.array([
    [1, 3, 1],  # 첫 번째 점 (3, 1)
    [1, 4, 2]   # 두 번째 점 (4, 2)
])

# 모든 데이터 포인트
X = np.vstack((positive_examples, negative_examples))
# 레이블: 양성 = 1, 음성 = 0
y = np.array([1, 1, 0, 0])

# 두 파라미터 후보
w1 = np.array([0, -1, 1])
w2 = np.array([-3, -1, 3])

# 로지스틱 함수
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# 로그 가능도 계산 함수
def log_likelihood(X, y, w):
    z = np.dot(X, w)
    h = sigmoid(z)
    
    # 로그 가능도 계산: sum(y*log(h) + (1-y)*log(1-h))
    log_like = np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
    return log_like

# 각 파라미터 후보에 대한 로그 가능도 계산
log_like_w1 = log_likelihood(X, y, w1)
log_like_w2 = log_likelihood(X, y, w2)

print(f"w1 = {w1}의 로그 가능도: {log_like_w1:.4f}")
print(f"w2 = {w2}의 로그 가능도: {log_like_w2:.4f}")

if log_like_w1 > log_like_w2:
    print("w1이 더 나은 파라미터입니다.")
else:
    print("w2가 더 나은 파라미터입니다.")

# 각 데이터 포인트에 대한 예측값 계산
pred_w1 = sigmoid(np.dot(X, w1))
pred_w2 = sigmoid(np.dot(X, w2))

print("\n각 데이터 포인트에 대한 예측값:")
print("w1 예측값:", pred_w1)
print("w2 예측값:", pred_w2) 