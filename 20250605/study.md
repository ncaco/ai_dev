# Distance-based Model

## 인스턴스 기반 학습(Instance-based Learning) 개요

### 파라메트릭 모델(Parametric model)
- 고정된 크기의 파라미터 집합으로 데이터를 요약함
  - 데이터가 알려진 형태의 모델에서 생성된다고 가정
- 예시: 선형 회귀(linear regression)

### 논파라메트릭 모델(Nonparametric model)
- 데이터를 유한한 파라미터 집합으로 특징지을 수 없을 때 사용 (예: 데이터가 매우 비선형적일 때)
- 데이터 자체에 더 많은 의존을 두고, 데이터가 말하게 함
  - 특히 데이터가 대량으로 있을 때 효과적
- 예시: 인스턴스 기반(메모리 기반) 학습(instance-based (memory-based) learning)

> 요약:  
> 파라메트릭 모델은 소수의 파라미터로 전체 데이터를 요약하는 반면, 논파라메트릭 모델은 데이터 자체를 많이 활용하여 복잡한 패턴을 포착할 수 있습니다. 인스턴스 기반 학습은 논파라메트릭 모델의 대표적인 예시입니다.


## k-최근접 이웃(k-NN, k-Nearest Neighbors) 알고리즘

### 개념 및 동작 방식
- 쿼리 데이터 $x_q$가 주어지면, 데이터셋에서 가장 가까운 k개의 이웃 $NN(k, x_q)$를 찾습니다.
  - **분류(Classification)**: 이웃들의 다수결(plurality vote)로 쿼리의 클래스를 결정합니다.
  - **회귀(Regression)**: 이웃들의 평균(mean) 또는 중앙값(median)으로 예측값을 산출하거나, 이웃들에 대해 선형 회귀(linear regression)을 수행할 수 있습니다.
- 최적의 k값을 선택하기 위해 교차 검증(cross-validation)을 사용할 수 있습니다.

### k 값에 따른 모델의 특성
- k=1: 매우 작은 k값은 모델이 데이터에 과적합(overfitting)될 수 있습니다. (왼쪽 그림 참고)
- k=5: 적절한 k값은 모델의 일반화 성능을 높여줍니다. (오른쪽 그림 참고)

> **요약:**  
> k-NN은 새로운 데이터가 주어졌을 때, 학습 데이터 중 가장 가까운 k개의 이웃을 찾아 예측을 수행하는 인스턴스 기반(논파라메트릭) 학습 방법입니다. k값이 작으면 과적합, 크면 과소적합이 발생할 수 있으므로, 교차 검증을 통해 최적의 k를 선택하는 것이 중요합니다.

## 차원의 저주(Curse of Dimensionality)

- **차원의 저주란?**
  - 고차원 공간에서는 최근접 이웃(Nearest Neighbor)들이 실제로는 멀리 떨어져 있는 현상이 발생합니다.
  - 즉, 차원이 증가할수록 데이터 포인트 간의 거리가 멀어지고, 이웃의 개념이 약해집니다.

- **수식적 설명**
  - $l$: 이웃 영역의 평균 한 변의 길이
  - $l^n$: n차원에서 이웃 영역(하이퍼큐브)의 부피
  - 데이터가 부피 1의 n차원 큐브에 균일하게 분포한다고 가정할 때, 전체 데이터 수가 $N$, 최근접 이웃의 수가 $k$라면,
    - $$\frac{k}{N} = l^n$$
    - $$l = \left(\frac{k}{N}\right)^{1/n}$$
  - 즉, 차원이 커질수록 $l$이 1에 가까워져, 이웃의 범위가 전체 공간에 퍼지게 됩니다.

- **예시**
  - $k = 10$, $N = 1,000,000$일 때,
    - $n = 3$ → $l \approx 0.02$
    - $n = 17$ → $l \approx 0.5$
    - $n = 200$ → $l \approx 0.94$
  - 차원이 높아질수록 이웃을 포함하는 영역이 전체 공간의 대부분을 차지하게 됨을 알 수 있습니다.

> **요약:**  
> k-NN과 같은 인스턴스 기반 학습에서는 차원이 높아질수록 "가까운" 이웃을 찾기가 어려워집니다. 이는 모델의 성능 저하로 이어질 수 있으므로, 차원 축소(dimensionality reduction) 등의 전처리 기법이 중요합니다.

### 인스턴스 기반 학습(Instance-based Learning) 추가 설명

#### 시간 복잡도(Time complexity)
- **순차 테이블(sequential table)**: $O(N)$
- **이진 트리(binary tree)**: $O(\log N)$  
  - 예시: kD 트리(kD tree), 볼 트리(ball tree)
- **해시 테이블(hash table)**: $O(1)$  
  - 예시: locality-sensitive hashing(LSH, 지역 민감 해싱)

#### 느린 학습(Lazy learning)
- **Lazy learning(게으른 학습)**:  
  - 새로운 인스턴스를 분류할 때 실제 연산이 이루어집니다.  
    (즉, 데이터를 저장할 때가 아니라 예측 시점에 계산이 집중됨)
  - 이 방식 덕분에 **맞춤형 예측(tailor-made predictions)**이 가능합니다.

> **정리:**  
> 인스턴스 기반 학습은 저장된 데이터를 그대로 보관하고, 예측 시점에만 계산을 수행하는 "게으른 학습" 방식입니다.  
> 효율적인 최근접 이웃 탐색을 위해 kD 트리, 볼 트리, 해시 테이블 등 다양한 자료구조가 활용되며, 각각의 시간 복잡도 특성이 다릅니다.

### 거리 함수(Distance Function)와 정규화(Normalization)

- **거리 측정(metric):**  
  - 대표적으로 **Minkowski 거리** 또는 $L^p$ 노름(norm)이 사용됩니다.
  - 일반식:  
    $$L^p(x_j, x_q) = \left( \sum_i |x_{j,i} - x_{q,i}|^p \right)^{1/p}$$
    - $p = 2$일 때: **유클리드 거리(Euclidean distance)**
    - $p = 1$일 때: **맨해튼 거리(Manhattan distance)**
    - **불리언 속성(Boolean attributes)**에 대해 $p = 2$를 적용하면 **해밍 거리(Hamming distance)**와 유사하게 동작

- **정규화(Normalization):**
  - 각 차원의 값이 서로 다른 스케일을 가질 경우, 거리 계산에 영향을 줄 수 있으므로 정규화가 필요합니다.
  - 정규화 방법:  
    $$x_{j,i} \rightarrow \frac{x_{j,i} - \mu_i}{\sigma_i}$$
    - $\mu_i$: i번째 차원의 평균(mean)
    - $\sigma_i$: i번째 차원의 표준편차(standard deviation)

> **정리:**  
> 인스턴스 기반 학습에서 거리 함수는 데이터 간 유사도를 측정하는 핵심 요소입니다.  
> 데이터의 스케일 차이로 인한 왜곡을 방지하기 위해 정규화가 필수적으로 적용됩니다.

#### Locality-Sensitive Hashing(LSH, 지역 민감 해싱)

- **근사 최근접 이웃(Approximate near-neighbors) 문제:**
  - 예시 데이터 집합과 쿼리 포인트 $x_q$가 주어졌을 때, 높은 확률로 $x_q$와 가까운(near) 예시 포인트(들)를 찾는 문제입니다.

- **지역 민감 해시(Locality-sensitive hash):**
  - 데이터를 한 축(선)으로 사영(projection)하여 해시 테이블을 만들고, 이 선을 따라 해시 구간(hash bin)으로 나눕니다.
  - $l$개의 임의 투영(random projection)을 선택하여 $g_1(x), g_2(x), ..., g_l(x)$와 같은 해시 함수를 생성합니다.
  - 쿼리 포인트 $x_q$에 대해, 각 $g_k(x_q)$에 해당하는 해시 테이블에서 후보 집합 $C$를 모읍니다.
  - 후보 집합 $C$에서 실제 거리 계산을 통해 $x_q$와 가장 가까운 $k$개의 포인트를 최종적으로 선택합니다.

> **정리:**  
> LSH는 고차원 공간에서 효율적으로 근사 최근접 이웃을 찾기 위한 해시 기반 방법입니다.  
> 여러 개의 무작위 투영을 활용해 후보군을 빠르게 좁히고, 최종적으로 실제 거리를 계산하여 최근접 이웃을 결정합니다.

#### 비모수 회귀(Nonparametric Regression)와 인스턴스 기반 학습

- **k-최근접 이웃 회귀(k-nearest-neighbors regression):**
  - **k-NN 평균:**  
    - 예측값 $h(x) = \sum y_j / k$ (k개의 최근접 이웃의 타깃값 평균)
    - 한쪽 방향(outlier)에서만 근거가 모일 경우, 경향(trend)을 무시하고 이상치에 취약  
      (예: 경계점에서 평균이 왜곡될 수 있음)
  - **k-NN 선형 회귀(linear regression):**  
    - k개의 최근접 이웃을 대상으로 최적의 직선을 찾음
    - 이상치(outlier)에서도 데이터의 경향(trend)을 포착할 수 있음

- **국소 가중 선형 회귀(Locally weighted linear regression):**
  - 예측 함수 $h(x)$의 불연속(discontinuity) 문제를 완화
  - 각 예시가 **커널 함수(kernel function)**에 의해 가중치(weight)를 부여받음
  - 쿼리 포인트와의 거리가 멀수록 가중치가 점차 감소  
    (즉, 가까운 이웃일수록 예측에 더 큰 영향)

> **정리:**  
> 인스턴스 기반 회귀에서는 k-NN 평균, k-NN 선형 회귀, 국소 가중 선형 회귀 등 다양한 비모수적 방법이 활용됩니다.  
> 국소 가중 회귀는 커널 함수를 통해 거리 기반 가중치를 적용하여, 예측의 부드러움과 이상치에 대한 강건성을 높입니다.

#### 쿼리 포인트에 대한 국소 가중 선형 회귀의 수식 및 k-NN의 장단점

- 쿼리 포인트 $x_q$에 대해, **국소 가중 선형 회귀**는 다음과 같은 최적화 문제를 푼다:
  $$
  w^* = \arg\min_w \sum_j K\left(\text{Distance}(x_q, x_j)\right) \left(y_j - w \cdot x_j\right)^2
  $$
  - 여기서 $K(\cdot)$는 커널 함수(예: 가우시안, 삼각형 등)로, $x_q$와 $x_j$의 거리에 따라 가중치를 부여합니다.
  - 최적의 $w^*$를 구한 뒤, 쿼리 포인트 $x_q$에 대한 예측값은 $h(x_q) = w^* \cdot x_q$로 계산합니다.
  - **특징:** 쿼리마다 새로운 회귀 문제를 풀지만, 전체 데이터가 아니라 k개의 최근접 이웃(또는 커널로 가중치가 충분히 큰 이웃)만 사용하므로 계산 효율을 높일 수 있습니다.

- **k-NN의 장단점:**
  - **장점:**
    - 비선형 함수 근사 가능 (Can approximate nonlinear function)
    - 점진적 학습(incremental learning)에 적합
  - **단점:**
    - 예측 시 계산이 느림 (Slow at deriving a prediction)
    - 데이터의 전체 구조(global structure)에 대한 정보 제공이 부족

> **요약:**  
> 국소 가중 선형 회귀는 쿼리마다 커널 가중치를 적용해 선형 회귀를 수행하며, k-NN은 단순 평균부터 선형 회귀까지 다양한 방식으로 비모수적 예측이 가능하지만, 예측 속도와 데이터 구조 파악 측면에서 한계가 있습니다.
