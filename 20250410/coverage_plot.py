import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.font_manager as fm

# 한글 폰트 설정
if os.name == 'nt':  # Windows
    font_path = 'C:/Windows/Fonts/malgun.ttf'  # 맑은 고딕
    font_name = fm.FontProperties(fname=font_path).get_name()
    plt.rc('font', family=font_name)
else:  # macOS나 Linux의 경우
    plt.rc('font', family='AppleGothic')

# 마이너스 기호 깨짐 방지
plt.rc('axes', unicode_minus=False)

# 현재 스크립트 위치 확인
script_dir = os.path.dirname(os.path.abspath(__file__))

# 예시 데이터 1 (슬라이드의 왼쪽 혼동 행렬)
conf_matrix1 = {
    'TP': 30, 'FN': 20, 'FP': 10, 'TN': 40,
    'Pos': 50, 'Neg': 50, 'Total': 100
}

# 예시 데이터 2 (슬라이드의 오른쪽 혼동 행렬)
conf_matrix2 = {
    'TP': 60, 'FN': 15, 'FP': 10, 'TN': 15,
    'Pos': 75, 'Neg': 25, 'Total': 100
}

# 첫 번째 혼동 행렬의 Coverage Plot (왼쪽)
def plot_coverage1(conf_matrix, ax):
    # 축 범위 설정
    ax.set_xlim(0, conf_matrix['Neg'])
    ax.set_ylim(0, conf_matrix['Pos'])
    
    # 격자 설정
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # FP1 지점에 C1 표시
    fp1 = conf_matrix['FP'] / 2
    tp1 = conf_matrix['Pos']
    ax.scatter(fp1, tp1, color='green', s=50, zorder=5)
    ax.annotate('C1', (fp1, tp1), xytext=(0, 5), 
                textcoords='offset points', ha='center', fontsize=10)
    
    # FP2 지점에 C2 표시
    fp2 = conf_matrix['FP']
    tp2 = conf_matrix['TP']
    ax.scatter(fp2, tp2, color='green', s=50, zorder=5)
    ax.annotate('C2', (fp2, tp2), xytext=(0, 5), 
                textcoords='offset points', ha='center', fontsize=10)
    
    # 축 레이블 설정
    ax.set_xlabel('Negatives')
    ax.set_ylabel('Positives')
    
    # FP1, FP2, TP1, TP2 지점 표시
    ax.axvline(x=fp1, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(x=fp2, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=tp1, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=tp2, color='gray', linestyle='--', alpha=0.5)
    
    # 축 아래 레이블 추가
    ax.text(fp1, -5, 'FP1', ha='center', va='top', fontsize=10)
    ax.text(fp2, -5, 'FP2', ha='center', va='top', fontsize=10)
    
    # 왼쪽 축 레이블 추가
    ax.text(-5, tp1, 'TP1', ha='right', va='center', fontsize=10)
    ax.text(-5, tp2, 'TP2', ha='right', va='center', fontsize=10)

    # 혼동 행렬을 텍스트로 표시
    conf_text = f"혼동 행렬:\nTP: {conf_matrix['TP']}, FN: {conf_matrix['FN']}\nFP: {conf_matrix['FP']}, TN: {conf_matrix['TN']}"
    ax.text(0.5, 0.2, conf_text, transform=ax.transAxes, ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    # 두 번째 작은 혼동 행렬 표시
    small_conf_text = f"작은 혼동 행렬:\nTP: 20, FN: 30\nFP: 20, TN: 30"
    ax.text(0.5, 0.1, small_conf_text, transform=ax.transAxes, ha='center', fontsize=8, bbox=dict(boxstyle='round', facecolor='lavender', alpha=0.9))

# 두 번째 혼동 행렬의 Coverage Plot (오른쪽)
def plot_coverage2(conf_matrix, ax):
    # 축 범위 설정
    ax.set_xlim(0, conf_matrix['Neg'])
    ax.set_ylim(0, conf_matrix['Pos'])
    
    # 격자 설정
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # FP3 지점에 C3 표시
    fp3 = conf_matrix['FP']
    tp3 = conf_matrix['TP']
    ax.scatter(fp3, tp3, color='green', s=50, zorder=5)
    ax.annotate('C3', (fp3, tp3), xytext=(0, 5), 
                textcoords='offset points', ha='center', fontsize=10)
    
    # 축 레이블 설정
    ax.set_xlabel('Negatives')
    ax.set_ylabel('Positives')
    
    # FP3 지점 표시
    ax.axvline(x=fp3, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=tp3, color='gray', linestyle='--', alpha=0.5)
    
    # 축 아래 레이블 추가
    ax.text(fp3, -3, 'FP3', ha='center', va='top', fontsize=10)
    
    # 왼쪽 축 레이블 추가
    ax.text(-3, tp3, 'TP3', ha='right', va='center', fontsize=10)

    # 혼동 행렬을 텍스트로 표시
    conf_text = f"혼동 행렬:\nTP: {conf_matrix['TP']}, FN: {conf_matrix['FN']}\nFP: {conf_matrix['FP']}, TN: {conf_matrix['TN']}"
    ax.text(0.5, 0.2, conf_text, transform=ax.transAxes, ha='center', fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

# 그래프 그리기
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

# 각 서브플롯 제목 설정
ax1.set_title('Left Coverage Plot', fontsize=14)
ax2.set_title('Right Coverage Plot', fontsize=14)

# 첫 번째, 두 번째 Coverage Plot 그리기
plot_coverage1(conf_matrix1, ax1)
plot_coverage2(conf_matrix2, ax2)

plt.suptitle('Coverage Plot', fontsize=16)
plt.tight_layout(rect=[0, 0, 1, 0.95])

# 그래프 저장
output_path = os.path.join(script_dir, 'coverage_plot.png')
plt.savefig(output_path, dpi=150, bbox_inches='tight')
plt.close()

print(f"Coverage Plot이 성공적으로 생성되었습니다: {output_path}") 