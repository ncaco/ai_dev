import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import math
from collections import Counter

# 주어진 훈련 데이터
data = [
    {'A': 2, 'B': 'H', 'C': 1, 'D': 'F', 'Class': 'Y'},
    {'A': 1, 'B': 'H', 'C': 1, 'D': 'F', 'Class': 'N'},
    {'A': 1, 'B': 'H', 'C': 1, 'D': 'T', 'Class': 'N'},
    {'A': 3, 'B': 'M', 'C': 1, 'D': 'F', 'Class': 'Y'},
    {'A': 2, 'B': 'M', 'C': 1, 'D': 'T', 'Class': 'Y'},
    {'A': 3, 'B': 'M', 'C': 1, 'D': 'T', 'Class': 'N'},
    {'A': 1, 'B': 'M', 'C': 1, 'D': 'F', 'Class': 'N'},
    {'A': 2, 'B': 'L', 'C': 0, 'D': 'T', 'Class': 'Y'},
    {'A': 1, 'B': 'L', 'C': 0, 'D': 'F', 'Class': 'Y'},
    {'A': 3, 'B': 'L', 'C': 0, 'D': 'T', 'Class': 'N'},
]

df = pd.DataFrame(data)
print("훈련 데이터:")
print(df)

class Node:
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature      # 특성(속성) 인덱스
        self.threshold = threshold  # 분할 임계값
        self.left = left            # 왼쪽 서브트리
        self.right = right          # 오른쪽 서브트리
        self.value = value          # 리프 노드의 클래스

class FixedDecisionTree:
    def __init__(self):
        self.root = None
    
    def fit(self, X, y, feature_values):
        """
        순서대로 A, B, C, D 특성을 사용해 결정 트리를 생성합니다.
        """
        # 데이터프레임으로 변환하여 작업
        data = pd.DataFrame(X, columns=feature_values)
        data['Class'] = y
        
        # 루트 노드 생성 (A 특성으로 분할)
        self.root = self._build_tree_fixed_order(data, 0, feature_values)
    
    def _build_tree_fixed_order(self, data, depth, feature_values):
        """
        A, B, C, D 순서로 강제 분기하는 트리를 구축합니다.
        """
        # 데이터가 비어있으면 'Y' 클래스 (다수결)로 설정
        if len(data) == 0:
            return Node(value='Y')  # 기본값으로 'Y' 설정
        
        # 모든 클래스가 같으면 리프 노드 반환
        if len(data['Class'].unique()) == 1:
            return Node(value=data['Class'].iloc[0])
        
        # 최대 깊이에 도달하면 다수결로 클래스 결정
        if depth >= len(feature_values):
            most_common = Counter(data['Class']).most_common(1)[0][0]
            return Node(value=most_common)
        
        # 현재 특성 선택
        current_feature = feature_values[depth]
        
        # 현재 특성의 고유 값
        unique_values = data[current_feature].unique()
        
        # 고유 값이 없는 경우 다수결로 결정
        if len(unique_values) == 0:
            most_common = Counter(data['Class']).most_common(1)[0][0]
            return Node(value=most_common)
        
        # 숫자형 특성인 경우 (A, C)
        if current_feature in ['A', 'C']:
            # 중간값 계산
            threshold = float(unique_values.mean())
            
            # 데이터 분할
            left_data = data[data[current_feature] <= threshold]
            right_data = data[data[current_feature] > threshold]
            
            # 자식 노드 생성
            left_child = self._build_tree_fixed_order(left_data, depth + 1, feature_values)
            right_child = self._build_tree_fixed_order(right_data, depth + 1, feature_values)
            
            return Node(feature=current_feature, threshold=threshold, left=left_child, right=right_child)
        
        # 범주형 특성인 경우 (B, D)
        else:
            # 가장 빈도가 높은 값을 기준으로 분할
            value_counts = data[current_feature].value_counts()
            
            if len(value_counts) == 0:
                most_common = Counter(data['Class']).most_common(1)[0][0]
                return Node(value=most_common)
            
            threshold = value_counts.index[0]
            
            # 데이터 분할
            left_data = data[data[current_feature] == threshold]
            right_data = data[data[current_feature] != threshold]
            
            # 자식 노드 생성
            left_child = self._build_tree_fixed_order(left_data, depth + 1, feature_values)
            right_child = self._build_tree_fixed_order(right_data, depth + 1, feature_values)
            
            return Node(feature=current_feature, threshold=threshold, left=left_child, right=right_child)
    
    def predict_one(self, x, node, feature_values):
        """
        단일 샘플에 대한 예측을 수행합니다.
        """
        if node.value is not None:
            return node.value
        
        feature_idx = feature_values.index(node.feature)
        feature_val = x[feature_idx]
        
        # 누락된 값('?')인 경우 양쪽 경로 모두 탐색하고 다수결 결정
        if feature_val == '?':
            # 자식 노드가 모두 리프 노드인 경우
            if node.left.value is not None and node.right.value is not None:
                # 왼쪽 노드를 우선 (임의 선택)
                return node.left.value
            # 왼쪽 자식 노드가 리프 노드인 경우
            elif node.left.value is not None:
                return node.left.value
            # 오른쪽 자식 노드가 리프 노드인 경우
            elif node.right.value is not None:
                return node.right.value
            # 둘 다 내부 노드인 경우
            else:
                # 양쪽 경로를 모두 탐색하고 결과를 종합 (간단히 왼쪽 경로 선택)
                return self.predict_one(x, node.left, feature_values)
        
        # 숫자형 특성인 경우
        if node.feature in ['A', 'C']:
            if feature_val <= node.threshold:
                return self.predict_one(x, node.left, feature_values)
            else:
                return self.predict_one(x, node.right, feature_values)
        # 범주형 특성인 경우
        else:
            if feature_val == node.threshold:
                return self.predict_one(x, node.left, feature_values)
            else:
                return self.predict_one(x, node.right, feature_values)
    
    def predict(self, X, feature_values):
        """
        여러 샘플에 대한 예측을 수행합니다.
        """
        return [self.predict_one(x, self.root, feature_values) for x in X]
    
    def print_tree(self, node=None, indent=""):
        """
        훈련된 의사 결정 트리를 출력합니다.
        """
        if node is None:
            node = self.root
        
        if node.value is not None:
            print(f"{indent}Class: {node.value}")
            return
        
        if node.feature in ['A', 'C']:
            print(f"{indent}{node.feature} <= {node.threshold}")
        else:
            print(f"{indent}{node.feature} == '{node.threshold}'")
        
        print(f"{indent}왼쪽:")
        self.print_tree(node.left, indent + "  ")
        print(f"{indent}오른쪽:")
        self.print_tree(node.right, indent + "  ")
    
    def visualize_tree(self, node=None, depth=0, pos=None, parent_pos=None, is_left=None):
        """
        의사 결정 트리를 시각화합니다.
        """
        if node is None:
            node = self.root
            self.fig, self.ax = plt.subplots(figsize=(12, 8))
            self.ax.set_xlim(-1, 1)
            self.ax.set_ylim(0, 1)
            self.ax.axis('off')
            pos = (0, 0.9)
            
        width = 0.5 ** depth
        
        # 노드 그리기
        if node.value is not None:
            color = 'lightgreen' if node.value == 'Y' else 'lightcoral'
            self.ax.text(pos[0], pos[1], f"Class: {node.value}", 
                         ha='center', va='center', bbox=dict(boxstyle='round,pad=0.5', facecolor=color))
        else:
            text = f"{node.feature} <= {node.threshold:.2f}" if node.feature in ['A', 'C'] else f"{node.feature} == '{node.threshold}'"
            self.ax.text(pos[0], pos[1], text, 
                         ha='center', va='center', bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue'))
        
        # 가지 연결
        if parent_pos:
            self.ax.plot([parent_pos[0], pos[0]], [parent_pos[1], pos[1]], 'k-')
            if is_left:
                label = "Yes" if node.feature in ['A', 'C'] else "Equal"
                self.ax.text((parent_pos[0] + pos[0])/2, (parent_pos[1] + pos[1])/2, label, 
                             ha='center', va='center', bbox=dict(boxstyle='round,pad=0.1', facecolor='white'))
            else:
                label = "No" if node.feature in ['A', 'C'] else "Not Equal"
                self.ax.text((parent_pos[0] + pos[0])/2, (parent_pos[1] + pos[1])/2, label, 
                             ha='center', va='center', bbox=dict(boxstyle='round,pad=0.1', facecolor='white'))
        
        # 자식 노드 재귀적으로 그리기
        if node.left:
            left_pos = (pos[0] - width/2, pos[1] - 0.1)
            self.visualize_tree(node.left, depth + 1, left_pos, pos, True)
        
        if node.right:
            right_pos = (pos[0] + width/2, pos[1] - 0.1)
            self.visualize_tree(node.right, depth + 1, right_pos, pos, False)
        
        if depth == 0:
            plt.tight_layout()
            plt.savefig('fixed_decision_tree.png', dpi=300, bbox_inches='tight')
            plt.show()

# 데이터 준비
data_np = df.values
X = data_np[:, :-1]  # 모든 특성
y = data_np[:, -1]   # 클래스 레이블

# 특성 이름
feature_values = ['A', 'B', 'C', 'D']

# 각 특성을 원래 타입으로 가져오기
X_original = []
for i in range(len(data)):
    X_original.append([
        data[i]['A'],
        data[i]['B'],
        data[i]['C'],
        data[i]['D']
    ])
X_original = np.array(X_original, dtype=object)

# 고정 순서 의사결정 트리 생성 및 훈련
fixed_tree = FixedDecisionTree()
fixed_tree.fit(X_original, y, feature_values)

# 의사 결정 트리 출력
print("\n고정 순서 결정 트리 구조 (A, B, C, D 순):")
fixed_tree.print_tree()

# 의사 결정 트리 시각화
print("\n결정 트리 시각화:")
fixed_tree.visualize_tree()

# 테스트 예제 분류
test_examples = [
    {'A': 1, 'B': '?', 'C': 0, 'D': 'T'},  # 첫 번째 테스트 예제
    {'A': '?', 'B': 'M', 'C': 1, 'D': 'F'}   # 두 번째 테스트 예제
]

print("\n테스트 예제:")
for i, example in enumerate(test_examples, 1):
    print(f"예제 {i}: {example}")

# 누락된 값이 있는 테스트 예제를 위한 처리
print("\n수동으로 테스트 예제 분류:")
print("예제 1: A=1 -> B=? (모든 B 값에 대해 경로 확인) -> C=0 -> D=T")
print("예제 2: A=? (모든 A 값에 대해 경로 확인) -> B=M -> C=1 -> D=F")

# 예제 1 수동 분류
print("\n예제 1 분류 결과:")
for b_val in ['H', 'M', 'L']:
    test_sample = np.array([1, b_val, 0, 'T'], dtype=object)
    result = fixed_tree.predict_one(test_sample, fixed_tree.root, feature_values)
    print(f"  B={b_val}: {result}")

# 예제 2 수동 분류
print("\n예제 2 분류 결과:")
for a_val in [1, 2, 3]:
    test_sample = np.array([a_val, 'M', 1, 'F'], dtype=object)
    result = fixed_tree.predict_one(test_sample, fixed_tree.root, feature_values)
    print(f"  A={a_val}: {result}") 