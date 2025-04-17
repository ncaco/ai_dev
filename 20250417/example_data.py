import numpy as np
import matplotlib.pyplot as plt

# 슬라이드에서 본 예제 데이터 생성
def create_example_data():
    """
    슬라이드에서 본 예제와 유사한 데이터 생성:
    - 'Viagra'와 'lottery' 키워드가 있는 이메일 분류
    - spam/ham 레이블이 있는 데이터
    """
    # 샘플 데이터 생성
    # 형식: [키워드 존재 여부, spam 확률, 실제 레이블(1: spam, 0: ham)]
    data = [
        # 'Viagra' 키워드 포함
        {'id': 'v1', 'features': {'viagra': 1, 'lottery': 0}, 'score': 2.0, 'label': 1},  # spam
        {'id': 'v2', 'features': {'viagra': 1, 'lottery': 0}, 'score': 2.0, 'label': 1},  # spam
        {'id': 'v3', 'features': {'viagra': 1, 'lottery': 0}, 'score': 2.0, 'label': 0},  # ham
        {'id': 'v4', 'features': {'viagra': 1, 'lottery': 0}, 'score': 2.0, 'label': 0},  # ham
        {'id': 'v5', 'features': {'viagra': 1, 'lottery': 0}, 'score': 2.0, 'label': 0},  # ham
        
        # 'lottery' 키워드 포함
        {'id': 'l1', 'features': {'viagra': 0, 'lottery': 1}, 'score': -1.0, 'label': 1},  # spam
        {'id': 'l2', 'features': {'viagra': 0, 'lottery': 1}, 'score': -1.0, 'label': 1},  # spam
        {'id': 'l3', 'features': {'viagra': 0, 'lottery': 1}, 'score': -1.0, 'label': 1},  # spam
        {'id': 'l4', 'features': {'viagra': 0, 'lottery': 1}, 'score': -1.0, 'label': 0},  # ham
        {'id': 'l5', 'features': {'viagra': 0, 'lottery': 1}, 'score': -1.0, 'label': 0},  # ham
        
        # 'Viagra'와 'lottery' 둘 다 포함 (leaf 1)
        {'id': 'vl1', 'features': {'viagra': 1, 'lottery': 1}, 'score': 1.0, 'label': 1},  # spam
        {'id': 'vl2', 'features': {'viagra': 1, 'lottery': 1}, 'score': 1.0, 'label': 1},  # spam
        {'id': 'vl3', 'features': {'viagra': 1, 'lottery': 1}, 'score': 1.0, 'label': 0},  # ham
        {'id': 'vl4', 'features': {'viagra': 1, 'lottery': 1}, 'score': 1.0, 'label': 0},  # ham
        {'id': 'vl5', 'features': {'viagra': 1, 'lottery': 1}, 'score': 1.0, 'label': 0},  # ham
    ]
    
    return data

def visualize_feature_tree():
    """
    슬라이드에서 본 Feature Tree 시각화
    """
    plt.figure(figsize=(10, 8))
    
    # 노드 위치 정의
    nodes = {
        'root': (5, 10),
        'viagra_yes': (3, 8),
        'viagra_no': (7, 8),
        'lottery_yes_1': (2, 6),
        'lottery_no_1': (4, 6),
        'lottery_yes_2': (6, 6),
        'lottery_no_2': (8, 6),
        'leaf1': (2, 4),
        'leaf2': (4, 4),
        'leaf3': (6, 4),
        'leaf4': (8, 4)
    }
    
    # 노드 그리기
    for node, pos in nodes.items():
        if node == 'root':
            plt.plot(pos[0], pos[1], 'o', markersize=30, color='lightblue', alpha=0.7)
            plt.text(pos[0], pos[1], "'Viagra'", ha='center', va='center', fontsize=12)
        elif node in ['viagra_yes', 'viagra_no']:
            plt.plot(pos[0], pos[1], 'o', markersize=30, color='lightblue', alpha=0.7)
            plt.text(pos[0], pos[1], "'lottery'", ha='center', va='center', fontsize=12)
        else:
            plt.plot(pos[0], pos[1], 's', markersize=30, color='lightgray', alpha=0.7)
            
            if node == 'leaf1':
                plt.text(pos[0], pos[1], "spam: 20\nham: 5", ha='center', va='center', fontsize=10)
            elif node == 'leaf2':
                plt.text(pos[0], pos[1], "spam: 10\nham: 5", ha='center', va='center', fontsize=10)
            elif node == 'leaf3':
                plt.text(pos[0], pos[1], "spam: 20\nham: 40", ha='center', va='center', fontsize=10)
            elif node == 'leaf4':
                plt.text(pos[0], pos[1], "spam: 0\nham: 30", ha='center', va='center', fontsize=10)
    
    # 간선 그리기
    plt.plot([nodes['root'][0], nodes['viagra_yes'][0]], [nodes['root'][1], nodes['viagra_yes'][1]], 'k-')
    plt.plot([nodes['root'][0], nodes['viagra_no'][0]], [nodes['root'][1], nodes['viagra_no'][1]], 'k-')
    
    plt.plot([nodes['viagra_yes'][0], nodes['lottery_yes_1'][0]], [nodes['viagra_yes'][1], nodes['lottery_yes_1'][1]], 'k-')
    plt.plot([nodes['viagra_yes'][0], nodes['lottery_no_1'][0]], [nodes['viagra_yes'][1], nodes['lottery_no_1'][1]], 'k-')
    
    plt.plot([nodes['viagra_no'][0], nodes['lottery_yes_2'][0]], [nodes['viagra_no'][1], nodes['lottery_yes_2'][1]], 'k-')
    plt.plot([nodes['viagra_no'][0], nodes['lottery_no_2'][0]], [nodes['viagra_no'][1], nodes['lottery_no_2'][1]], 'k-')
    
    plt.plot([nodes['lottery_yes_1'][0], nodes['leaf1'][0]], [nodes['lottery_yes_1'][1], nodes['leaf1'][1]], 'k-')
    plt.plot([nodes['lottery_no_1'][0], nodes['leaf2'][0]], [nodes['lottery_no_1'][1], nodes['leaf2'][1]], 'k-')
    plt.plot([nodes['lottery_yes_2'][0], nodes['leaf3'][0]], [nodes['lottery_yes_2'][1], nodes['leaf3'][1]], 'k-')
    plt.plot([nodes['lottery_no_2'][0], nodes['leaf4'][0]], [nodes['lottery_no_2'][1], nodes['leaf4'][1]], 'k-')
    
    # 간선에 레이블 추가
    plt.text(nodes['root'][0]-0.3, nodes['root'][1]-0.5, "=0", ha='right', va='bottom', fontsize=10)
    plt.text(nodes['root'][0]+0.3, nodes['root'][1]-0.5, "=1", ha='left', va='bottom', fontsize=10)
    
    plt.text(nodes['viagra_yes'][0]-0.3, nodes['viagra_yes'][1]-0.5, "=0", ha='right', va='bottom', fontsize=10)
    plt.text(nodes['viagra_yes'][0]+0.3, nodes['viagra_yes'][1]-0.5, "=1", ha='left', va='bottom', fontsize=10)
    
    plt.text(nodes['viagra_no'][0]-0.3, nodes['viagra_no'][1]-0.5, "=0", ha='right', va='bottom', fontsize=10)
    plt.text(nodes['viagra_no'][0]+0.3, nodes['viagra_no'][1]-0.5, "=1", ha='left', va='bottom', fontsize=10)
    
    plt.title("Feature Tree Example")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('feature_tree.png')
    plt.show()

def visualize_scoring_tree():
    """
    슬라이드에서 본 Scoring Tree 시각화
    """
    plt.figure(figsize=(10, 8))
    
    # 노드 위치 정의
    nodes = {
        'root': (5, 10),
        'viagra_yes': (3, 8),
        'viagra_no': (7, 8),
        'lottery_yes_1': (2, 6),
        'lottery_no_1': (4, 6),
        'lottery_yes_2': (6, 6),
        'lottery_no_2': (8, 6),
        'leaf1': (2, 4),
        'leaf2': (4, 4),
        'leaf3': (6, 4),
        'leaf4': (8, 4)
    }
    
    # 노드 그리기
    for node, pos in nodes.items():
        if node == 'root':
            plt.plot(pos[0], pos[1], 'o', markersize=30, color='lightblue', alpha=0.7)
            plt.text(pos[0], pos[1], "'Viagra'", ha='center', va='center', fontsize=12)
        elif node in ['viagra_yes', 'viagra_no']:
            plt.plot(pos[0], pos[1], 'o', markersize=30, color='lightblue', alpha=0.7)
            plt.text(pos[0], pos[1], "'lottery'", ha='center', va='center', fontsize=12)
        else:
            plt.plot(pos[0], pos[1], 's', markersize=30, color='lightgray', alpha=0.7)
            
            if node == 'leaf1':
                plt.text(pos[0], pos[1], "s(x) = +2", ha='center', va='center', fontsize=10)
            elif node == 'leaf2':
                plt.text(pos[0], pos[1], "s(x) = +1", ha='center', va='center', fontsize=10)
            elif node == 'leaf3':
                plt.text(pos[0], pos[1], "s(x) = -1", ha='center', va='center', fontsize=10)
            elif node == 'leaf4':
                plt.text(pos[0], pos[1], "s(x) = -2", ha='center', va='center', fontsize=10)
    
    # 간선 그리기
    plt.plot([nodes['root'][0], nodes['viagra_yes'][0]], [nodes['root'][1], nodes['viagra_yes'][1]], 'k-')
    plt.plot([nodes['root'][0], nodes['viagra_no'][0]], [nodes['root'][1], nodes['viagra_no'][1]], 'k-')
    
    plt.plot([nodes['viagra_yes'][0], nodes['lottery_yes_1'][0]], [nodes['viagra_yes'][1], nodes['lottery_yes_1'][1]], 'k-')
    plt.plot([nodes['viagra_yes'][0], nodes['lottery_no_1'][0]], [nodes['viagra_yes'][1], nodes['lottery_no_1'][1]], 'k-')
    
    plt.plot([nodes['viagra_no'][0], nodes['lottery_yes_2'][0]], [nodes['viagra_no'][1], nodes['lottery_yes_2'][1]], 'k-')
    plt.plot([nodes['viagra_no'][0], nodes['lottery_no_2'][0]], [nodes['viagra_no'][1], nodes['lottery_no_2'][1]], 'k-')
    
    plt.plot([nodes['lottery_yes_1'][0], nodes['leaf1'][0]], [nodes['lottery_yes_1'][1], nodes['leaf1'][1]], 'k-')
    plt.plot([nodes['lottery_no_1'][0], nodes['leaf2'][0]], [nodes['lottery_no_1'][1], nodes['leaf2'][1]], 'k-')
    plt.plot([nodes['lottery_yes_2'][0], nodes['leaf3'][0]], [nodes['lottery_yes_2'][1], nodes['leaf3'][1]], 'k-')
    plt.plot([nodes['lottery_no_2'][0], nodes['leaf4'][0]], [nodes['lottery_no_2'][1], nodes['leaf4'][1]], 'k-')
    
    # 간선에 레이블 추가
    plt.text(nodes['root'][0]-0.3, nodes['root'][1]-0.5, "=0", ha='right', va='bottom', fontsize=10)
    plt.text(nodes['root'][0]+0.3, nodes['root'][1]-0.5, "=1", ha='left', va='bottom', fontsize=10)
    
    plt.text(nodes['viagra_yes'][0]-0.3, nodes['viagra_yes'][1]-0.5, "=0", ha='right', va='bottom', fontsize=10)
    plt.text(nodes['viagra_yes'][0]+0.3, nodes['viagra_yes'][1]-0.5, "=1", ha='left', va='bottom', fontsize=10)
    
    plt.text(nodes['viagra_no'][0]-0.3, nodes['viagra_no'][1]-0.5, "=0", ha='right', va='bottom', fontsize=10)
    plt.text(nodes['viagra_no'][0]+0.3, nodes['viagra_no'][1]-0.5, "=1", ha='left', va='bottom', fontsize=10)
    
    plt.title("Scoring Tree Example")
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('scoring_tree.png')
    plt.show()

def main():
    # 예제 데이터 생성
    data = create_example_data()
    
    # Feature Tree와 Scoring Tree 시각화
    visualize_feature_tree()
    visualize_scoring_tree()
    
    # 데이터 정보 출력
    positives = [item for item in data if item['label'] == 1]
    negatives = [item for item in data if item['label'] == 0]
    
    print(f"총 데이터: {len(data)}")
    print(f"Positive 샘플 수: {len(positives)}")
    print(f"Negative 샘플 수: {len(negatives)}")
    
    # 평균 점수 출력
    pos_scores = [item['score'] for item in positives]
    neg_scores = [item['score'] for item in negatives]
    
    print(f"Positive 샘플 평균 점수: {sum(pos_scores)/len(pos_scores):.2f}")
    print(f"Negative 샘플 평균 점수: {sum(neg_scores)/len(neg_scores):.2f}")

if __name__ == "__main__":
    main() 