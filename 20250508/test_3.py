# 제공 오차(제곱오차)가 예측 확률 p^가 경험적 확률 p와 같을 때 최소화됨을 설명하는 코드

# n(+) : 양성 샘플 개수, n(-) : 음성 샘플 개수
n_pos = 7   # 예시: 양성 샘플 개수
n_neg = 3   # 예시: 음성 샘플 개수

# 경험적 확률 p
p = n_pos / (n_pos + n_neg)

# 예측 확률 p^ (여기서는 여러 값을 실험)
p_hat_list = [0.0, 0.2, 0.4, 0.7, 0.9, 1.0, p]

print(f"n(+)={n_pos}, n(-)={n_neg}, 경험적 확률 p={p:.3f}")
print("p^ 값에 따른 제공 오차(제곱오차) 계산:")

for p_hat in p_hat_list:
    # 제공 오차(제곱오차) 공식:
    # n(+) * (p^ - 1)^2 + n(-) * (p^ - 0)^2
    squared_error = n_pos * (p_hat - 1) ** 2 + n_neg * (p_hat - 0) ** 2
    print(f"  p^={p_hat:.3f} -> 제공 오차(제곱오차): {squared_error:.4f}")

print("\n※ p^ = p 일 때 제공 오차가 최소가 됨을 확인할 수 있습니다.")
