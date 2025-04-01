# N-gram 파이썬 예제

이 저장소는 텍스트 처리 및 분석을 위한 n-gram의 다양한 사용법을 보여주는 파이썬 예제를 포함하고 있습니다.

## N-gram이란?

N-gram은 주어진 텍스트나 음성 샘플에서 연속된 n개 항목의 시퀀스입니다. 응용 분야에 따라 항목은 문자, 단어, 음절 또는 다른 단위가 될 수 있습니다.

- **유니그램 (n=1)**: 단일 단위 (예: 개별 단어 또는 문자)
- **바이그램 (n=2)**: 두 단위의 시퀀스
- **트라이그램 (n=3)**: 세 단위의 시퀀스
- 등등...

## 포함된 예제

1. **ngram_example.py**: 단어 수준 및 문자 수준 n-gram으로 기본 n-gram 처리
2. **ngram_text_classification.py**: 언어 식별을 위한 간단한 n-gram 기반 분류기
3. **ngram_sklearn_example.py**: n-gram 기반 감정 분석을 위한 scikit-learn 사용

## 의존성

이 예제를 실행하려면 Python 3.6+ 및 다음 패키지가 필요합니다:

```
numpy
scikit-learn
```

pip를 사용하여 이러한 의존성을 설치할 수 있습니다:

```
pip install numpy scikit-learn
```

## 예제 실행하기

아래 명령어로 예제를 실행할 수 있습니다:

```
python <파일명>.py
```

예를 들어:

```
python ngram_example.py
```

## 예제 설명

### 1. 기본 N-gram 처리 (ngram_example.py)

이 예제는 다음을 보여줍니다:
- 단어 수준 n-gram(유니그램, 바이그램, 트라이그램) 생성
- 문자 수준 n-gram 생성
- 서로 다른 텍스트 간의 공통 n-gram 찾기

### 2. N-gram을 이용한 텍스트 분류 (ngram_text_classification.py)

이 예제는 다음을 보여줍니다:
- 간단한 n-gram 기반 분류기를 처음부터 구축하는 방법
- 언어 식별(영어, 스페인어, 한국어)에 문자 수준 n-gram 사용
- 확률적 n-gram 모델 구현

### 3. Scikit-learn N-gram 분석 (ngram_sklearn_example.py)

이 예제는 다음을 보여줍니다:
- n-gram 추출을 위한 CountVectorizer 사용
- 감정 분석을 위한 단어 유니그램, 바이그램 및 문자 n-gram 비교
- n-gram과 함께 TF-IDF 사용
- 분류에 가장 유용한 n-gram 식별

## N-gram의 응용

N-gram은 다양한 자연어 처리 작업에서 널리 사용됩니다:

- 언어 식별
- 텍스트 분류
- 감정 분석
- 맞춤법 검사
- 기계 번역
- 음성 인식
- 텍스트 생성
- 문서 유사성
- 표절 감지 