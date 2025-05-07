import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches
import matplotlib.font_manager as fm

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'  # 윈도우 기본 한글 폰트
plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지

# 훈련 데이터 정의
data = {
    'A': [2, 1, 1, 3, 2, 3, 1, 2, 1, 3],
    'B': ['H', 'H', 'H', 'M', 'M', 'M', 'M', 'L', 'L', 'L'],
    'C': [1, 1, 1, 1, 1, 1, 1, 0, 0, 0],
    'D': ['F', 'F', 'T', 'F', 'T', 'T', 'F', 'T', 'F', 'T'],
    'Class': ['Y', 'N', 'N', 'Y', 'Y', 'N', 'N', 'Y', 'Y', 'N']
}

df = pd.DataFrame(data)

# 사용 가능한 폰트 확인 및 출력
print("사용 가능한 폰트:")
fonts = [f.name for f in fm.fontManager.ttflist]
for font in fonts:
    if any(keyword in font.lower() for keyword in ['gothic', 'gulim', 'batang', 'dotum', 'malgun']):
        print(f"- {font}")

# 의사결정 트리 시각화
plt.figure(figsize=(12, 8))
plt.axis('off')

# 노드 좌표 및 크기 정의
node_width = 0.08
node_height = 0.08
text_offset = 0.02

# 루트 노드 (A)
root_x, root_y = 0.5, 0.9
plt.gca().add_patch(Rectangle((root_x-node_width/2, root_y-node_height/2), node_width, node_height, 
                             facecolor='lightblue', edgecolor='black'))
plt.text(root_x, root_y, 'A', ha='center', va='center')

# A=1 노드
a1_x, a1_y = 0.25, 0.7
plt.gca().add_patch(Rectangle((a1_x-node_width/2, a1_y-node_height/2), node_width, node_height, 
                             facecolor='lightgreen', edgecolor='black'))
plt.text(a1_x, a1_y, 'A=1', ha='center', va='center')

# A=2 노드
a2_x, a2_y = 0.5, 0.7
plt.gca().add_patch(Rectangle((a2_x-node_width/2, a2_y-node_height/2), node_width, node_height, 
                             facecolor='lightgreen', edgecolor='black'))
plt.text(a2_x, a2_y, 'A=2', ha='center', va='center')

# A=3 노드
a3_x, a3_y = 0.75, 0.7
plt.gca().add_patch(Rectangle((a3_x-node_width/2, a3_y-node_height/2), node_width, node_height, 
                             facecolor='lightgreen', edgecolor='black'))
plt.text(a3_x, a3_y, 'A=3', ha='center', va='center')

# B 노드
b_x, b_y = 0.25, 0.5
plt.gca().add_patch(Rectangle((b_x-node_width/2, b_y-node_height/2), node_width, node_height, 
                             facecolor='lightblue', edgecolor='black'))
plt.text(b_x, b_y, 'B', ha='center', va='center')

# A=2 -> Class Y
plt.gca().add_patch(Rectangle((a2_x-node_width, a2_y-node_height*2), node_width*2, node_height, 
                             facecolor='salmon', edgecolor='black'))
plt.text(a2_x, a2_y-node_height*1.5, 'Class Y', ha='center', va='center')

# D 노드
d_x, d_y = 0.75, 0.5
plt.gca().add_patch(Rectangle((d_x-node_width/2, d_y-node_height/2), node_width, node_height, 
                             facecolor='lightblue', edgecolor='black'))
plt.text(d_x, d_y, 'D', ha='center', va='center')

# B=H 노드
bh_x, bh_y = 0.15, 0.3
plt.gca().add_patch(Rectangle((bh_x-node_width/2, bh_y-node_height/2), node_width, node_height, 
                             facecolor='lightgreen', edgecolor='black'))
plt.text(bh_x, bh_y, 'B=H', ha='center', va='center')

# B=M 노드
bm_x, bm_y = 0.25, 0.3
plt.gca().add_patch(Rectangle((bm_x-node_width/2, bm_y-node_height/2), node_width, node_height, 
                             facecolor='lightgreen', edgecolor='black'))
plt.text(bm_x, bm_y, 'B=M', ha='center', va='center')

# B=L 노드
bl_x, bl_y = 0.35, 0.3
plt.gca().add_patch(Rectangle((bl_x-node_width/2, bl_y-node_height/2), node_width, node_height, 
                             facecolor='lightgreen', edgecolor='black'))
plt.text(bl_x, bl_y, 'B=L', ha='center', va='center')

# D=F 노드
df_x, df_y = 0.65, 0.3
plt.gca().add_patch(Rectangle((df_x-node_width/2, df_y-node_height/2), node_width, node_height, 
                             facecolor='lightgreen', edgecolor='black'))
plt.text(df_x, df_y, 'D=F', ha='center', va='center')

# D=T 노드
dt_x, dt_y = 0.85, 0.3
plt.gca().add_patch(Rectangle((dt_x-node_width/2, dt_y-node_height/2), node_width, node_height, 
                             facecolor='lightgreen', edgecolor='black'))
plt.text(dt_x, dt_y, 'D=T', ha='center', va='center')

# B=H -> Class N
plt.gca().add_patch(Rectangle((bh_x-node_width/2, bh_y-node_height*2), node_width, node_height, 
                             facecolor='salmon', edgecolor='black'))
plt.text(bh_x, bh_y-node_height*1.5, 'Class N', ha='center', va='center')

# B=M -> Class N
plt.gca().add_patch(Rectangle((bm_x-node_width/2, bm_y-node_height*2), node_width, node_height, 
                             facecolor='salmon', edgecolor='black'))
plt.text(bm_x, bm_y-node_height*1.5, 'Class N', ha='center', va='center')

# B=L -> Class Y
plt.gca().add_patch(Rectangle((bl_x-node_width/2, bl_y-node_height*2), node_width, node_height, 
                             facecolor='salmon', edgecolor='black'))
plt.text(bl_x, bl_y-node_height*1.5, 'Class Y', ha='center', va='center')

# D=F -> Class Y
plt.gca().add_patch(Rectangle((df_x-node_width/2, df_y-node_height*2), node_width, node_height, 
                             facecolor='salmon', edgecolor='black'))
plt.text(df_x, df_y-node_height*1.5, 'Class Y', ha='center', va='center')

# D=T -> Class N
plt.gca().add_patch(Rectangle((dt_x-node_width/2, dt_y-node_height*2), node_width, node_height, 
                             facecolor='salmon', edgecolor='black'))
plt.text(dt_x, dt_y-node_height*1.5, 'Class N', ha='center', va='center')

# 연결선 그리기
plt.plot([root_x, a1_x], [root_y-node_height/2, a1_y+node_height/2], 'k-')
plt.plot([root_x, a2_x], [root_y-node_height/2, a2_y+node_height/2], 'k-')
plt.plot([root_x, a3_x], [root_y-node_height/2, a3_y+node_height/2], 'k-')
plt.plot([a1_x, b_x], [a1_y-node_height/2, b_y+node_height/2], 'k-')
plt.plot([a2_x, a2_x], [a2_y-node_height/2, a2_y-node_height], 'k-')
plt.plot([a3_x, d_x], [a3_y-node_height/2, d_y+node_height/2], 'k-')
plt.plot([b_x, bh_x], [b_y-node_height/2, bh_y+node_height/2], 'k-')
plt.plot([b_x, bm_x], [b_y-node_height/2, bm_y+node_height/2], 'k-')
plt.plot([b_x, bl_x], [b_y-node_height/2, bl_y+node_height/2], 'k-')
plt.plot([d_x, df_x], [d_y-node_height/2, df_y+node_height/2], 'k-')
plt.plot([d_x, dt_x], [d_y-node_height/2, dt_y+node_height/2], 'k-')
plt.plot([bh_x, bh_x], [bh_y-node_height/2, bh_y-node_height], 'k-')
plt.plot([bm_x, bm_x], [bm_y-node_height/2, bm_y-node_height], 'k-')
plt.plot([bl_x, bl_x], [bl_y-node_height/2, bl_y-node_height], 'k-')
plt.plot([df_x, df_x], [df_y-node_height/2, df_y-node_height], 'k-')
plt.plot([dt_x, dt_x], [dt_y-node_height/2, dt_y-node_height], 'k-')

# 범례 추가
decision_node = mpatches.Patch(color='lightblue', label='결정 노드')
value_node = mpatches.Patch(color='lightgreen', label='값 노드')
class_node = mpatches.Patch(color='salmon', label='클래스 노드')
plt.legend(handles=[decision_node, value_node, class_node], loc='upper right')

# 제목 추가
plt.title('훈련 데이터로부터 도출된 의사결정 트리', fontsize=16)

# 그래프 저장 및 표시
plt.savefig('decision_tree_visualization.png', dpi=300, bbox_inches='tight')
plt.tight_layout()
plt.show()

print("의사결정 트리 시각화 완료! 'decision_tree_visualization.png' 파일로 저장되었습니다.")

# scikit-learn 라이브러리를 사용한 결정 트리 시각화
try:
    from sklearn import tree
    from sklearn.tree import DecisionTreeClassifier
    import graphviz
    
    # 데이터 준비
    # 범주형 변수를 숫자로 변환
    df_encoded = df.copy()
    df_encoded['B'] = df_encoded['B'].map({'H': 0, 'M': 1, 'L': 2})
    df_encoded['D'] = df_encoded['D'].map({'F': 0, 'T': 1})
    df_encoded['Class'] = df_encoded['Class'].map({'N': 0, 'Y': 1})
    
    # 특성과 타겟 분리
    X = df_encoded[['A', 'B', 'C', 'D']]
    y = df_encoded['Class']
    
    # 결정 트리 모델 생성 및 학습
    clf = DecisionTreeClassifier(random_state=42)
    clf = clf.fit(X, y)
    
    # 결정 트리 시각화
    dot_data = tree.export_graphviz(clf, out_file=None, 
                                    feature_names=['A', 'B', 'C', 'D'],
                                    class_names=['N', 'Y'],
                                    filled=True, rounded=True,
                                    special_characters=True)
    
    graph = graphviz.Source(dot_data)
    graph.render("decision_tree_sklearn")
    print("scikit-learn 결정 트리 시각화 완료! 'decision_tree_sklearn.pdf' 파일로 저장되었습니다.")
except ImportError:
    print("scikit-learn 또는 graphviz 라이브러리가 설치되어 있지 않습니다.")
    print("설치하려면 다음 명령어를 실행하세요:")
    print("pip install scikit-learn graphviz python-graphviz") 