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

class MultiNode:
    def __init__(self, feature=None, children=None, value=None):
        self.feature = feature      # 특성(속성) 이름
        self.children = children or {}  # 자식 노드 (값 -> 노드 매핑)
        self.value = value          # 리프 노드의 클래스

class MultiWayDecisionTree:
    def __init__(self):
        self.root = None
    
    def fit(self, X, y, feature_values):
        """
        A, B, C, D 순서대로 다중 분기 결정 트리를 생성합니다.
        """
        # 데이터프레임으로 변환하여 작업
        data = pd.DataFrame(X, columns=feature_values)
        data['Class'] = y
        
        # 루트 노드 생성 (A 특성으로 분할)
        self.root = self._build_tree(data, 0, feature_values)
    
    def _build_tree(self, data, depth, feature_values):
        """
        다중 분기 결정 트리를 구축합니다.
        """
        # 데이터가 비어있으면 'Y' 클래스 (다수결)로 설정
        if len(data) == 0:
            return MultiNode(value='Y')  # 기본값으로 'Y' 설정
        
        # 모든 클래스가 같으면 리프 노드 반환
        if len(data['Class'].unique()) == 1:
            return MultiNode(value=data['Class'].iloc[0])
        
        # 최대 깊이에 도달하면 다수결로 클래스 결정
        if depth >= len(feature_values):
            most_common = Counter(data['Class']).most_common(1)[0][0]
            return MultiNode(value=most_common)
        
        # 현재 특성 선택
        current_feature = feature_values[depth]
        
        # 노드 생성
        node = MultiNode(feature=current_feature)
        
        # 현재 특성의 모든 고유 값에 대해 분기
        unique_values = sorted(data[current_feature].unique())
        
        # 고유 값이 없는 경우 다수결로 결정
        if len(unique_values) == 0:
            most_common = Counter(data['Class']).most_common(1)[0][0]
            return MultiNode(value=most_common)
        
        # 각 고유 값에 대해 자식 노드 생성
        children = {}
        for value in unique_values:
            # 현재 값에 해당하는 데이터만 선택
            subset = data[data[current_feature] == value]
            
            # 자식 노드 재귀적으로 구축
            child = self._build_tree(subset, depth + 1, feature_values)
            
            # 자식 노드 추가
            children[value] = child
        
        # 노드에 자식 노드 설정
        node.children = children
        
        return node
    
    def predict_one(self, x, node, feature_values):
        """
        단일 샘플에 대한 예측을 수행합니다.
        """
        # 리프 노드이면 클래스 반환
        if node.value is not None:
            return node.value
        
        # 현재 특성의 값 가져오기
        feature_idx = feature_values.index(node.feature)
        feature_val = x[feature_idx]
        
        # 누락된 값('?')인 경우 가장 많은 클래스 예측
        if feature_val == '?':
            # 모든 가능한 값에 대한 예측 결과 수집
            results = []
            for value, child_node in node.children.items():
                new_x = x.copy()
                new_x[feature_idx] = value
                results.append(self.predict_one(new_x, child_node, feature_values))
            
            # 가장 많은 클래스 반환
            return Counter(results).most_common(1)[0][0]
        
        # 특성 값이 훈련 데이터에 없는 경우
        if feature_val not in node.children:
            # 가장 일반적인 자식 노드의 클래스 찾기
            child_classes = []
            for child in node.children.values():
                if child.value is not None:
                    child_classes.append(child.value)
                else:
                    # 내부 노드인 경우, 첫 번째 자식의 결과를 사용
                    first_child = next(iter(child.children.values()))
                    if first_child.value is not None:
                        child_classes.append(first_child.value)
            
            # 자식 클래스가 있으면 가장 많은 클래스 반환
            if child_classes:
                return Counter(child_classes).most_common(1)[0][0]
            return 'Y'  # 기본값
        
        # 해당 값의 자식 노드로 이동
        return self.predict_one(x, node.children[feature_val], feature_values)
    
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
        
        print(f"{indent}{node.feature} 분기:")
        
        for value, child in sorted(node.children.items()):
            print(f"{indent}  {node.feature} = {value}:")
            self.print_tree(child, indent + "    ")
    
    def visualize_tree(self, node=None, depth=0, pos=None, parent_pos=None, branch_label=None):
        """
        다중 분기 의사 결정 트리를 시각화합니다.
        """
        if node is None:
            node = self.root
            self.fig, self.ax = plt.subplots(figsize=(14, 10))
            self.ax.set_xlim(-1, 1)
            self.ax.set_ylim(0, 1)
            self.ax.axis('off')
            pos = (0, 0.9)
            
        width = 0.8 ** (depth + 1)
        
        # 노드 그리기
        if node.value is not None:
            color = 'lightgreen' if node.value == 'Y' else 'lightcoral'
            self.ax.text(pos[0], pos[1], f"Class: {node.value}", 
                         ha='center', va='center', bbox=dict(boxstyle='round,pad=0.5', facecolor=color))
        else:
            self.ax.text(pos[0], pos[1], f"{node.feature}", 
                         ha='center', va='center', bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue'))
        
        # 가지 연결
        if parent_pos:
            self.ax.plot([parent_pos[0], pos[0]], [parent_pos[1], pos[1]], 'k-')
            self.ax.text((parent_pos[0] + pos[0])/2, (parent_pos[1] + pos[1])/2, branch_label, 
                         ha='center', va='center', bbox=dict(boxstyle='round,pad=0.1', facecolor='white'))
        
        # 자식 노드가 있는 경우
        if node.children:
            n_children = len(node.children)
            
            # 자식 노드가 많을 때 간격 조정
            if n_children > 2:
                width *= 1.5
            
            # 자식 노드 위치 계산
            positions = np.linspace(-width/2, width/2, n_children)
            
            # 자식 노드 정렬 (A는 숫자순, 나머지는 알파벳순)
            sorted_children = sorted(node.children.items()) 
            
            # 각 자식 노드 그리기
            for i, (value, child_node) in enumerate(sorted_children):
                child_pos = (pos[0] + positions[i], pos[1] - 0.15)
                self.visualize_tree(child_node, depth + 1, child_pos, pos, str(value))
        
        if depth == 0:
            plt.tight_layout()
            plt.savefig('multi_way_tree.png', dpi=300, bbox_inches='tight')
            plt.show()

# 데이터 준비
X_original = []
y = []
for item in data:
    X_original.append([
        item['A'],
        item['B'],
        item['C'],
        item['D']
    ])
    y.append(item['Class'])

X_original = np.array(X_original, dtype=object)
y = np.array(y)

# 특성 이름
feature_values = ['A', 'B', 'C', 'D']

# 다중 분기 의사결정 트리 생성 및 훈련
tree = MultiWayDecisionTree()
tree.fit(X_original, y, feature_values)

# 의사 결정 트리 출력
print("\n다중 분기 결정 트리 구조 (A, B, C, D 순):")
tree.print_tree()

# 의사 결정 트리 시각화
print("\n결정 트리 시각화:")
tree.visualize_tree()

# 테스트 예제 분류
test_examples = [
    {'A': 1, 'B': '?', 'C': 0, 'D': 'T'},  # 첫 번째 테스트 예제
    {'A': '?', 'B': 'M', 'C': 1, 'D': 'F'}   # 두 번째 테스트 예제
]

print("\n테스트 예제:")
for i, example in enumerate(test_examples, 1):
    print(f"예제 {i}: {example}")

# 예제 1 수동 분류
print("\n예제 1 분류 결과:")
test_array1 = np.array([1, '?', 0, 'T'], dtype=object)
for b_val in ['H', 'M', 'L']:
    test_sample = test_array1.copy()
    test_sample[1] = b_val
    result = tree.predict_one(test_sample, tree.root, feature_values)
    print(f"  A=1, B={b_val}, C=0, D=T: {result}")

# 예제 2 수동 분류
print("\n예제 2 분류 결과:")
test_array2 = np.array(['?', 'M', 1, 'F'], dtype=object)
for a_val in [1, 2, 3]:
    test_sample = test_array2.copy()
    test_sample[0] = a_val
    result = tree.predict_one(test_sample, tree.root, feature_values)
    print(f"  A={a_val}, B=M, C=1, D=F: {result}")

# 자동 분류 (누락 값 처리 포함)
print("\n자동 분류 결과:")
test_samples = []
for example in test_examples:
    sample = [example.get('A', '?'), 
              example.get('B', '?'), 
              example.get('C', '?'), 
              example.get('D', '?')]
    test_samples.append(sample)

test_samples = np.array(test_samples, dtype=object)
results = tree.predict(test_samples, feature_values)

for i, (example, result) in enumerate(zip(test_examples, results), 1):
    print(f"예제 {i}: {example} -> {result}") 