# 랭킹 성능 평가 (Ranking Performance)

이 프로젝트는 머신러닝 모델의 랭킹 성능을 평가하는 방법을 구현한 코드입니다. 슬라이드에서 배운 개념들을 Python으로 구현했습니다.

## 구현된 기능

1. **랭킹 에러 및 정확도 계산**
   - Positive-Negative 쌍에 대해 랭킹 에러를 계산
   - 동점(Ties) 처리 포함
   - 랭킹 정확도 계산

2. **Coverage Plot 시각화**
   - Positive와 Negative 인스턴스의 점수에 따른 그리드 시각화
   - 올바른 랭킹과 랭킹 에러 구분

3. **모델 시각화**
   - 결정 경계 시각화
   - 데이터 포인트 분포 확인

## 사용 방법

필요한 라이브러리를 설치합니다:

```
pip install numpy matplotlib scikit-learn
```

코드 실행:

```
python ranking_performance.py
```

## 예제 출력

실행 시 다음과 같은 출력을 얻을 수 있습니다:
- Ranking Error Rate
- Ranking Accuracy
- AUC (Area Under Curve)
- Coverage Plot 이미지
- 결정 경계 시각화 이미지

## 개념 설명

### 랭킹 에러 (Ranking Error)
- x가 실제 positive이고 x'가 실제 negative일 때, 예측된 점수 s(x)가 s(x')보다 낮을 때 발생
- 총 랭킹 에러 수는 sum(I(s(x)<s(x')))로 계산 (x∈Pos, x'∈Neg)

### 랭킹 정확도 (Ranking Accuracy)
- 1 - 랭킹 에러율
- AUC(Area Under Curve)와 동일

### Coverage Plot
- 양성 및 음성 인스턴스의 점수에 따른 랭킹 시각화
- 올바르게 랭킹된 쌍과 랭킹 에러를 한 눈에 확인 가능

## 참고 자료
- 동의대학교 Machine Learning Programming 강의자료