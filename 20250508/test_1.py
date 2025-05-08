
p_hat = [
    0.99,
    0,
    0.01
]

y = [
    0,
    0,
    1
]
for idx, (ph, yt) in enumerate(zip(p_hat, y), 1):
    error = ph - yt
    squared = ph ** 2
    squared_error = squared - yt
    print(f"[{idx}] 예측값: {ph:.2f}, 실제값: {yt}, 오차: {error:.2f}, 제곱: {squared:.4f}, (제곱-실제값): {squared_error:.4f}")



    # 모든 (제곱 - 실제값) 값을 합산하여 출력
    # squared_error 값을 리스트에 저장
    if idx == 1:
        squared_error_list = []
    squared_error_list.append(squared_error)

    if idx == len(p_hat):
        total = sum(squared_error_list)
        print(f"제곱오차를 뺀 값의 합: {total:.4f}")
