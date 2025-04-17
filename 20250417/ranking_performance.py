import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import make_classification

# 랭킹 에러 계산 함수
def calculate_ranking_error(y_true, scores):
    """
    y_true: 실제 레이블 (1: positive, 0: negative)
    scores: 분류기의 예측 점수
    """
    pos_indices = np.where(y_true == 1)[0]
    neg_indices = np.where(y_true == 0)[0]
    
    ranking_errors = 0
    total_pairs = len(pos_indices) * len(neg_indices)
    
    for pos_idx in pos_indices:
        pos_score = scores[pos_idx]
        for neg_idx in neg_indices:
            neg_score = scores[neg_idx]
            # 랭킹 에러: positive 샘플의 점수가 negative 샘플의 점수보다 낮을 때
            if pos_score < neg_score:
                ranking_errors += 1
            # 동점인 경우 0.5 에러로 계산
            elif pos_score == neg_score:
                ranking_errors += 0.5
    
    # 랭킹 에러율 계산
    ranking_error_rate = ranking_errors / total_pairs
    
    # 랭킹 정확도 계산
    ranking_accuracy = 1 - ranking_error_rate
    
    return ranking_error_rate, ranking_accuracy

# Coverage Plot 그리기 함수
def plot_coverage_curve(y_true, scores):
    """
    y_true: 실제 레이블 (1: positive, 0: negative)
    scores: 분류기의 예측 점수
    """
    pos_indices = np.where(y_true == 1)[0]
    neg_indices = np.where(y_true == 0)[0]
    
    # 점수에 따라 내림차순 정렬
    sorted_pos_indices = pos_indices[np.argsort(-scores[pos_indices])]
    sorted_neg_indices = neg_indices[np.argsort(-scores[neg_indices])]
    
    # Coverage Plot을 위한 그리드 생성
    num_pos = len(sorted_pos_indices)
    num_neg = len(sorted_neg_indices)
    
    grid = np.zeros((num_pos, num_neg))
    
    # 그리드 채우기: 랭킹 에러(빨간색), 올바른 랭킹(녹색)
    for i, pos_idx in enumerate(sorted_pos_indices):
        pos_score = scores[pos_idx]
        for j, neg_idx in enumerate(sorted_neg_indices):
            neg_score = scores[neg_idx]
            if pos_score > neg_score:  # 올바른 랭킹
                grid[i, j] = 1
            elif pos_score == neg_score:  # 동점
                grid[i, j] = 0.5
            else:  # 랭킹 에러
                grid[i, j] = 0
    
    # 시각화
    plt.figure(figsize=(10, 8))
    plt.imshow(grid, cmap='RdYlGn', interpolation='none', aspect='auto')
    plt.colorbar(label='Ranking Correctness')
    plt.xlabel('Negatives sorted on decreasing score')
    plt.ylabel('Positives sorted on decreasing score')
    plt.title('Coverage Plot')
    plt.tight_layout()
    
    return plt

# 예제 실행
def main():
    # 인공 데이터 생성
    X, y = make_classification(n_samples=100, n_features=2, n_informative=2, 
                               n_redundant=0, n_clusters_per_class=1, random_state=42)
    
    # 분류기 학습
    clf = DecisionTreeClassifier(max_depth=3, random_state=42)
    clf.fit(X, y)
    
    # 예측 점수 계산
    scores = clf.predict_proba(X)[:, 1]  # 클래스 1의 확률
    
    # 랭킹 에러와 정확도 계산
    rank_err, rank_acc = calculate_ranking_error(y, scores)
    print(f"Ranking Error Rate: {rank_err:.4f}")
    print(f"Ranking Accuracy: {rank_acc:.4f}")
    
    # sklearn의 AUC 계산 (랭킹 정확도와 동일)
    auc = roc_auc_score(y, scores)
    print(f"AUC (using sklearn): {auc:.4f}")
    
    # Coverage Plot 생성
    plt = plot_coverage_curve(y, scores)
    plt.savefig('coverage_plot.png')
    plt.show()
    
    # 데이터 포인트 시각화
    plt.figure(figsize=(10, 8))
    plt.scatter(X[y==0, 0], X[y==0, 1], c='red', marker='o', label='Negative')
    plt.scatter(X[y==1, 0], X[y==1, 1], c='blue', marker='^', label='Positive')
    
    # 결정 경계 시각화
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                         np.arange(y_min, y_max, 0.02))
    Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    Z = Z.reshape(xx.shape)
    
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='RdBu')
    plt.colorbar(label='Probability of class 1')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Decision Boundary and Data Points')
    plt.legend()
    plt.tight_layout()
    plt.savefig('decision_boundary.png')
    plt.show()

if __name__ == "__main__":
    main() 