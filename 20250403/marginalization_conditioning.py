import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 결합 확률 분포 정의 (이전 예제와 동일)
# 순서: toothache, catch, cavity
joint_prob = {
    (True, True, True): 0.108,
    (True, False, True): 0.012,
    (False, True, True): 0.072,
    (False, False, True): 0.008,
    (True, True, False): 0.016,
    (True, False, False): 0.064,
    (False, True, False): 0.144,
    (False, False, False): 0.576
}

def print_joint_distribution():
    """결합 확률 분포를 표 형태로 출력"""
    print("\n결합 확률 분포 테이블:")
    print("="*50)
    print(f"{'toothache':<10} | {'catch':<10} | {'cavity':<10} | {'확률':<10}")
    print("-"*50)
    
    for (t, c, cav), prob in joint_prob.items():
        print(f"{str(t):<10} | {str(c):<10} | {str(cav):<10} | {prob:<10.3f}")

def probability_of_proposition(condition_func):
    """명제의 확률 계산 - 해당 명제가 참인 모든 원자적 사건의 확률 합계"""
    return sum(prob for event, prob in joint_prob.items() if condition_func(event))

# ---------- 주변화(Marginalization) 구현 ----------

def marginalize_toothache():
    """
    주변화: P(toothache) 계산
    P(toothache) = ∑_catch,cavity P(toothache, catch, cavity)
    """
    # toothache=True인 모든 확률을 합산
    p_toothache_true = sum(prob for (t, c, cav), prob in joint_prob.items() if t)
    
    # toothache=False인 모든 확률을 합산
    p_toothache_false = sum(prob for (t, c, cav), prob in joint_prob.items() if not t)
    
    return {"True": p_toothache_true, "False": p_toothache_false}

def marginalize_cavity():
    """
    주변화: P(cavity) 계산
    P(cavity) = ∑_toothache,catch P(toothache, catch, cavity)
    """
    # cavity=True인 모든 확률을 합산
    p_cavity_true = sum(prob for (t, c, cav), prob in joint_prob.items() if cav)
    
    # cavity=False인 모든 확률을 합산
    p_cavity_false = sum(prob for (t, c, cav), prob in joint_prob.items() if not cav)
    
    return {"True": p_cavity_true, "False": p_cavity_false}

def marginalize_catch():
    """
    주변화: P(catch) 계산
    P(catch) = ∑_toothache,cavity P(toothache, catch, cavity)
    """
    # catch=True인 모든 확률을 합산
    p_catch_true = sum(prob for (t, c, cav), prob in joint_prob.items() if c)
    
    # catch=False인 모든 확률을 합산
    p_catch_false = sum(prob for (t, c, cav), prob in joint_prob.items() if not c)
    
    return {"True": p_catch_true, "False": p_catch_false}

def marginalize_toothache_and_cavity():
    """
    주변화: P(toothache, cavity) 계산
    P(toothache, cavity) = ∑_catch P(toothache, catch, cavity)
    """
    # 가능한 모든 (toothache, cavity) 조합에 대한 확률 계산
    result = {}
    for t in [True, False]:
        for cav in [True, False]:
            # catch 변수를 주변화(sum out)
            prob = sum(p for (t1, c, cav1), p in joint_prob.items() 
                      if t1 == t and cav1 == cav)
            result[(t, cav)] = prob
    
    return result

# ---------- 조건화(Conditioning) 구현 ----------

def condition_cavity_given_toothache(toothache_value=True):
    """
    조건화: P(cavity | toothache) 계산
    P(cavity | toothache) = P(cavity, toothache) / P(toothache)
    """
    # cavity(쿼리 변수)가 True이고 toothache(증거 변수)가 주어진 값일 때의 결합 확률
    p_cavity_true_and_toothache = sum(prob for (t, c, cav), prob in joint_prob.items() 
                                    if cav and t == toothache_value)
    
    # toothache(증거 변수)가 주어진 값일 때의 확률
    p_toothache = sum(prob for (t, c, cav), prob in joint_prob.items() 
                     if t == toothache_value)
    
    # 조건부 확률 계산 (P(cavity|toothache) = P(cavity,toothache)/P(toothache))
    p_cavity_true_given_toothache = p_cavity_true_and_toothache / p_toothache
    
    # P(cavity, toothache)
    p_cavity_false_and_toothache = sum(prob for (t, c, cav), prob in joint_prob.items() 
                                     if not cav and t == toothache_value)
    
    # 조건부 확률 계산
    if p_toothache > 0:
        p_cavity_false_given_toothache = p_cavity_false_and_toothache / p_toothache
        return {"True": p_cavity_true_given_toothache, "False": p_cavity_false_given_toothache}
    else:
        return {"True": 0, "False": 0}

def condition_cavity_given_catch(catch_value=True):
    """
    조건화: P(cavity | catch) 계산
    P(cavity | catch) = P(cavity, catch) / P(catch)
    """
    # P(cavity, catch)
    p_cavity_true_and_catch = sum(prob for (t, c, cav), prob in joint_prob.items() 
                                if cav and c == catch_value)
    p_cavity_false_and_catch = sum(prob for (t, c, cav), prob in joint_prob.items() 
                                 if not cav and c == catch_value)
    
    # P(catch)
    p_catch = sum(prob for (t, c, cav), prob in joint_prob.items() 
                 if c == catch_value)
    
    # 조건부 확률 계산
    if p_catch > 0:
        p_cavity_true_given_catch = p_cavity_true_and_catch / p_catch
        p_cavity_false_given_catch = p_cavity_false_and_catch / p_catch
        return {"True": p_cavity_true_given_catch, "False": p_cavity_false_given_catch}
    else:
        return {"True": 0, "False": 0}

def condition_rule_example():
    """
    조건화 규칙 예제: P(Y) = ∑_z P(Y|z)P(z)
    
    구체적으로 P(cavity) = P(cavity|toothache=True)P(toothache=True) + P(cavity|toothache=False)P(toothache=False)
    를 계산하여 직접 cavity의 확률을 계산한 것과 비교
    """
    # 직접 계산한 P(cavity)
    p_cavity_direct = marginalize_cavity()
    
    # 조건화 규칙을 사용하여 계산
    p_cavity_true_given_toothache_true = condition_cavity_given_toothache(True)["True"]
    p_cavity_true_given_toothache_false = condition_cavity_given_toothache(False)["True"]
    
    p_toothache_true = marginalize_toothache()["True"]
    p_toothache_false = marginalize_toothache()["False"]
    
    p_cavity_using_conditioning = p_cavity_true_given_toothache_true * p_toothache_true + \
                                p_cavity_true_given_toothache_false * p_toothache_false
    
    return {
        "P(cavity) 직접 계산": p_cavity_direct["True"],
        "P(cavity) 조건화 규칙 사용": p_cavity_using_conditioning
    }

def main():
    print("확률적 모델 - 주변화(Marginalization)와 조건화(Conditioning)")
    print_joint_distribution()
    
    print("\n--- 주변화(Marginalization) 예제 ---")
    print("P(toothache):")
    p_toothache = marginalize_toothache()
    print(f"  P(toothache=True) = {p_toothache['True']:.3f}")
    print(f"  P(toothache=False) = {p_toothache['False']:.3f}")
    
    print("\nP(cavity):")
    p_cavity = marginalize_cavity()
    print(f"  P(cavity=True) = {p_cavity['True']:.3f}")
    print(f"  P(cavity=False) = {p_cavity['False']:.3f}")
    
    print("\nP(catch):")
    p_catch = marginalize_catch()
    print(f"  P(catch=True) = {p_catch['True']:.3f}")
    print(f"  P(catch=False) = {p_catch['False']:.3f}")
    
    print("\nP(toothache, cavity):")
    p_toothache_cavity = marginalize_toothache_and_cavity()
    for (t, cav), prob in p_toothache_cavity.items():
        print(f"  P(toothache={t}, cavity={cav}) = {prob:.3f}")
    
    print("\n--- 조건화(Conditioning) 예제 ---")
    print("P(cavity | toothache=True):")
    p_cavity_given_toothache = condition_cavity_given_toothache(True)
    print(f"  P(cavity=True | toothache=True) = {p_cavity_given_toothache['True']:.3f}")
    print(f"  P(cavity=False | toothache=True) = {p_cavity_given_toothache['False']:.3f}")
    
    print("\nP(cavity | catch=True):")
    p_cavity_given_catch = condition_cavity_given_catch(True)
    print(f"  P(cavity=True | catch=True) = {p_cavity_given_catch['True']:.3f}")
    print(f"  P(cavity=False | catch=True) = {p_cavity_given_catch['False']:.3f}")
    
    print("\n--- 조건화 규칙(Conditioning Rule) 검증 ---")
    conditioning_rule_result = condition_rule_example()
    print(f"  직접 계산한 P(cavity=True): {conditioning_rule_result['P(cavity) 직접 계산']:.3f}")
    print(f"  조건화 규칙으로 계산한 P(cavity=True): {conditioning_rule_result['P(cavity) 조건화 규칙 사용']:.3f}")
    
if __name__ == "__main__":
    main() 