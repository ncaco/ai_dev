# 이진 분류 문제 시각화

이 프로젝트는 이진 분류 문제를 좌표상에 시각화하고, 결정 경계, 커버리지 곡선 및 ROC 곡선을 그리는 파이썬 코드를 제공합니다.

## 파일 설명

- `binary_classification_visualization.py`: 기본적인 이진 분류 문제 시각화
- `binary_classification_curves.py`: 이진 분류 문제, 커버리지 곡선 및 ROC 곡선 시각화
- `requirements.txt`: 필요한 패키지 목록

## 설치 방법

필요한 패키지를 설치하려면 다음 명령어를 실행하세요:

```bash
pip install -r requirements.txt
```

## 실행 방법

기본 시각화를 실행하려면:

```bash
python binary_classification_visualization.py
```

커버리지 곡선과 ROC 곡선을 포함한 전체 시각화를 실행하려면:

```bash
python binary_classification_curves.py
```

## 코드 설명

이 코드는 다음과 같은 기능을 제공합니다:

1. 2차원 평면에 긍정적인 예(●)와 부정적인 예(○)를 시각화
2. 선형 결정 경계를, 그리고 각 포인트에서 경계까지의 거리 표시
3. 결정 경계에서의 거리를 기준으로 인스턴스 정렬
4. 정렬된 인스턴스에 대한 커버리지 곡선과 ROC 곡선 그리기

## 참고사항

현재 코드는 예시 데이터를 사용합니다. 실제 문제의 데이터에 맞게 `positive_examples`와 `negative_examples` 배열, 그리고 결정 경계 함수 매개변수를 조정해야 합니다.