import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'

# 시그모이드 함수 정의
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# 로그우도함수 정의
def log_likelihood(X, y, theta):
    """
    로지스틱 회귀의 로그우도함수 계산
    X: 특성 행렬 (m x n)
    y: 레이블 벡터 (m x 1)
    theta: 파라미터 벡터 (n x 1)
    """
    m = len(y)
    h = sigmoid(X.dot(theta))
    
    # 수치적 안정성을 위한 작은 값 추가 (로그에 0이 들어가는 것 방지)
    epsilon = 1e-5
    
    # 로그우도함수 계산
    ll = np.sum(y * np.log(h + epsilon) + (1 - y) * np.log(1 - h + epsilon))
    
    # 평균 로그우도
    return ll / m

# 로그우도함수의 그래디언트 (미분값)
def log_likelihood_gradient(X, y, theta):
    """
    로그우도함수의 그래디언트 계산
    경사상승법을 위한 미분값
    """
    m = len(y)
    h = sigmoid(X.dot(theta))
    
    # 로그우도함수를 theta로 미분한 값
    return X.T.dot(y - h) / m

# 경사상승법으로 로그우도함수 최대화
def gradient_ascent_step(X, y, theta, learning_rate=0.1):
    """
    경사상승법의 한 스텝을 수행
    """
    gradient = log_likelihood_gradient(X, y, theta)
    return theta + learning_rate * gradient

# 수렴 조건을 적용한 경사상승법
def gradient_ascent_with_convergence(X, y, theta_init, learning_rate=0.1, max_iterations=1000, tol=1e-6):
    """
    수렴 조건을 적용한 경사상승법
    tol: 로그우도 변화량의 임계값 (이 값보다 작아지면 수렴한 것으로 간주)
    """
    theta = theta_init.copy()
    theta_history = [theta.copy()]
    
    ll_prev = log_likelihood(X, y, theta)
    ll_history = [ll_prev]
    
    for i in range(1, max_iterations):
        # 파라미터 업데이트
        theta = gradient_ascent_step(X, y, theta, learning_rate)
        
        # 로그우도 계산
        ll = log_likelihood(X, y, theta)
        
        # 결과 저장
        theta_history.append(theta.copy())
        ll_history.append(ll)
        
        # 수렴 여부 확인
        if abs(ll - ll_prev) < tol:
            print(f'수렴 완료! 반복 횟수: {i}, 로그우도 변화량: {abs(ll - ll_prev):.8f}')
            break
            
        ll_prev = ll
    else:
        print(f'최대 반복 횟수({max_iterations})에 도달했습니다. 수렴하지 않았을 수 있습니다.')
    
    return np.array(theta_history), np.array(ll_history), i+1

# 결정 경계 계산 함수
def compute_decision_boundary(theta, xx, yy):
    """
    주어진 파라미터에 대한 결정 경계 계산
    """
    # 그리드 포인트에 대한 예측 확률
    Z = sigmoid(np.c_[np.ones(xx.ravel().shape), xx.ravel(), yy.ravel()].dot(theta))
    Z = Z.reshape(xx.shape)
    return Z

# 메인 코드
if __name__ == "__main__":
    # 1. 이진 분류 데이터 생성
    np.random.seed(42)
    X, y = make_classification(n_samples=100, n_features=2, n_redundant=0,
                              n_informative=2, random_state=42,
                              n_clusters_per_class=1)
    
    # 2. 데이터 시각화
    plt.figure(figsize=(10, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral, edgecolors='k')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Binary Classification Dataset')
    plt.colorbar()
    plt.grid(True)
    plt.show()
    
    # 3. 데이터 준비
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # 4. 바이어스 항 추가
    X_train_b = np.c_[np.ones(X_train.shape[0]), X_train]
    X_test_b = np.c_[np.ones(X_test.shape[0]), X_test]
    
    # 5. 초기 파라미터 설정
    theta_init = np.zeros(X_train_b.shape[1])
    
    # 6. 수렴 조건을 적용한 경사상승법 실행
    # 수렴 조건 설정
    max_iterations = 1000  # 최대 반복 횟수
    learning_rate = 0.1    # 학습률
    tol = 1e-6             # 수렴 임계값
    
    # 경사상승법 실행
    theta_history, ll_history, iterations = gradient_ascent_with_convergence(
        X_train_b, y_train, theta_init, learning_rate, max_iterations, tol
    )
    
    # 7. 파라미터 공간에서 로그우도함수 시각화 (2D 등고선)
    # 로그우도함수의 그리드 범위 설정
    theta1_range = np.linspace(-2, 2, 100)
    theta2_range = np.linspace(-2, 2, 100)
    
    # 로그우도 그리드 계산
    ll_grid = np.zeros((len(theta1_range), len(theta2_range)))
    
    for i, t1 in enumerate(theta1_range):
        for j, t2 in enumerate(theta2_range):
            # 바이어스는 0으로 고정하고 계산
            theta_test = np.array([0, t1, t2])
            ll_grid[i, j] = log_likelihood(X_train_b, y_train, theta_test)
    
    # 8. 결정 경계를 위한 메시 그리드 생성
    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1),
                         np.arange(y_min, y_max, 0.1))
    
    # 9. 애니메이션 설정
    fig = plt.figure(figsize=(16, 8))
    
    # 화면 분할: 4개의 서브플롯 (2x2 그리드)
    ax1 = plt.subplot2grid((2, 2), (0, 0))  # 파라미터 공간 등고선
    ax2 = plt.subplot2grid((2, 2), (0, 1))  # 결정 경계
    ax3 = plt.subplot2grid((2, 2), (1, 0))  # 로그우도 변화 그래프
    ax4 = plt.subplot2grid((2, 2), (1, 1), projection='3d')  # 3D 로그우도 표면
    
    # 등고선 그리기 (파라미터 공간)
    contour = ax1.contourf(theta1_range, theta2_range, ll_grid, 50, cmap='viridis')
    path_line, = ax1.plot([], [], 'r-', linewidth=2, label='Optimization Path')
    current_point, = ax1.plot([], [], 'ro', markersize=8)
    
    ax1.set_xlabel('Theta 1')
    ax1.set_ylabel('Theta 2')
    ax1.set_title('Log-Likelihood Contours')
    plt.colorbar(contour, ax=ax1, label='Log-Likelihood')
    ax1.grid(True)
    ax1.legend()
    
    # 결정 경계 초기화
    scatter = ax2.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.Spectral, edgecolors='k')
    decision_boundary = ax2.contourf(xx, yy, np.zeros_like(xx), alpha=0.3, cmap=plt.cm.Spectral, levels=np.linspace(0, 1, 11))
    
    ax2.set_xlabel('Feature 1')
    ax2.set_ylabel('Feature 2')
    ax2.set_title('Decision Boundary')
    ax2.grid(True)
    
    # 로그우도 변화 그래프 초기화
    log_line, = ax3.plot([], [], 'b-', linewidth=2)
    ax3.set_xlim(0, iterations)  # 실제 반복 횟수 범위로 설정
    ax3.set_ylim(min(ll_history) - 0.1, max(ll_history) + 0.1)  # 로그우도 범위
    ax3.set_xlabel('Iterations')
    ax3.set_ylabel('Log-Likelihood')
    ax3.set_title('Log-Likelihood Convergence')
    ax3.grid(True)
    
    # 3D 로그우도 표면 그리기
    theta1_grid, theta2_grid = np.meshgrid(theta1_range, theta2_range)
    surf = ax4.plot_surface(theta1_grid, theta2_grid, ll_grid.T, cmap='viridis', alpha=0.7, antialiased=True)
    path_3d, = ax4.plot([], [], [], 'r-', linewidth=2, label='Optimization Path')
    current_point_3d, = ax4.plot([], [], [], 'ro', markersize=6)
    
    ax4.set_xlabel('Theta 1')
    ax4.set_ylabel('Theta 2')
    ax4.set_zlabel('Log-Likelihood')
    ax4.set_title('Log-Likelihood Surface')
    
    # 텍스트 요소 추가
    info_text = ax2.text(0.02, 0.95, '', transform=ax2.transAxes, va='top', fontsize=10,
                         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 수렴 여부를 보여주는 텍스트 추가
    convergence_text = ax3.text(0.5, 0.9, '', transform=ax3.transAxes, ha='center',
                              bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))
    
    # 애니메이션 초기화 함수
    def init():
        path_line.set_data([], [])
        current_point.set_data([], [])
        log_line.set_data([], [])
        path_3d.set_data([], [])
        path_3d.set_3d_properties([])
        current_point_3d.set_data([], [])
        current_point_3d.set_3d_properties([])
        info_text.set_text('')
        convergence_text.set_text('')
        return path_line, current_point, log_line, path_3d, current_point_3d, info_text, convergence_text
    
    # 애니메이션 업데이트 함수
    def update(frame):
        # 파라미터 공간에서의 경로 업데이트
        path_line.set_data(theta_history[:frame+1, 1], theta_history[:frame+1, 2])
        current_point.set_data([theta_history[frame, 1]], [theta_history[frame, 2]])
        
        # 로그우도 변화 그래프 업데이트
        log_line.set_data(range(frame+1), ll_history[:frame+1])
        
        # 3D 경로 업데이트
        path_3d.set_data(theta_history[:frame+1, 1], theta_history[:frame+1, 2])
        path_3d.set_3d_properties(ll_history[:frame+1])
        current_point_3d.set_data([theta_history[frame, 1]], [theta_history[frame, 2]])
        current_point_3d.set_3d_properties([ll_history[frame]])
        
        # 결정 경계 업데이트
        for c in ax2.collections:
            if c != scatter:  # 산점도는 유지
                c.remove()
        
        current_Z = compute_decision_boundary(theta_history[frame], xx, yy)
        ax2.contourf(xx, yy, current_Z, alpha=0.3, cmap=plt.cm.Spectral, levels=np.linspace(0, 1, 11))
        
        # 텍스트 정보 업데이트
        info_text.set_text(f'Iteration: {frame}\nLog-Likelihood: {ll_history[frame]:.4f}\n' +
                          f'θ₀: {theta_history[frame, 0]:.4f}, θ₁: {theta_history[frame, 1]:.4f}, θ₂: {theta_history[frame, 2]:.4f}')
        
        # 수렴 여부 텍스트 업데이트
        if frame == iterations - 1:
            convergence_text.set_text(f'수렴 완료! (총 {iterations}회 반복)')
        elif frame > 0:
            diff = abs(ll_history[frame] - ll_history[frame-1])
            convergence_text.set_text(f'로그우도 변화량: {diff:.8f}')
        
        return path_line, current_point, log_line, path_3d, current_point_3d, info_text, convergence_text
    
    # 애니메이션 생성
    ani = FuncAnimation(fig, update, frames=iterations, init_func=init, 
                        interval=100, blit=True)
    
    plt.tight_layout()
    plt.show()
    
    # 10. 최종 결과 출력
    final_theta = theta_history[-1]
    final_ll = ll_history[-1]
    
    print("\n===== 최종 결과 =====")
    print("초기 로그우도:", ll_history[0])
    print("최종 로그우도:", final_ll)
    print("로그우도 증가량:", final_ll - ll_history[0])
    print("최적화된 파라미터:", final_theta)
    print(f"총 반복 횟수: {iterations}")
    
    # 11. 모델 평가
    def predict(X, theta, threshold=0.5):
        return sigmoid(X.dot(theta)) >= threshold
        
    y_pred_train = predict(X_train_b, final_theta)
    y_pred_test = predict(X_test_b, final_theta)
    
    # 정확도 계산
    train_accuracy = np.mean(y_pred_train == y_train)
    test_accuracy = np.mean(y_pred_test == y_test)
    
    print(f"훈련 데이터 정확도: {train_accuracy:.4f}")
    print(f"테스트 데이터 정확도: {test_accuracy:.4f}")
    
    # 12. 최종 결정 경계 시각화
    plt.figure(figsize=(10, 6))
    Z = compute_decision_boundary(final_theta, xx, yy)
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.Spectral, levels=np.linspace(0, 1, 11))
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=plt.cm.Spectral, edgecolors='k')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title(f'Final Decision Boundary (After {iterations} Iterations)')
    plt.colorbar()
    plt.grid(True)
    plt.show()
    
    # 13. 로그우도 변화 그래프 시각화 (전체)
    plt.figure(figsize=(10, 6))
    plt.plot(range(iterations), ll_history, 'b-', linewidth=2)
    plt.xlabel('Iterations')
    plt.ylabel('Log-Likelihood')
    plt.title('Log-Likelihood Convergence')
    plt.grid(True)
    
    # 수렴 부분 강조
    plt.axhline(y=final_ll, color='r', linestyle='--', alpha=0.5)
    plt.text(iterations * 0.7, final_ll, f'최종값: {final_ll:.4f}', va='bottom')
    
    plt.show() 