# 음성 및 텍스트 처리 모델 예제

이 프로젝트는 음성 처리(wav2vec)와 텍스트 처리(word2vec) 모델을 사용하는 예제 코드를 제공합니다.

## 주요 기능

### 1. Wav2Vec 음성 인식
- **음성 인식 (Speech-to-Text)**: 사전 학습된 wav2vec 모델을 사용하여 오디오 파일을 텍스트로 변환
- **모델 파인튜닝**: 한국어 음성 데이터셋을 사용하여 wav2vec 모델 파인튜닝

### 2. Word2Vec 단어 임베딩
- **단어 임베딩 생성**: 텍스트 데이터로부터 단어 임베딩을 생성
- **CBOW 및 Skip-gram 구현**: 두 가지 Word2Vec 아키텍처 비교
- **직접 구현 및 Gensim 라이브러리 사용**: 기본 원리 이해와 실용적 구현 모두 제공

## 설치 방법

필요한 패키지를 설치하려면 다음 명령어를 실행하세요:

```bash
pip install -r requirements.txt
```

## 사용 방법

### 1. 음성 인식 예제

기본적인 음성 인식 기능을 사용하려면 다음과 같이 실행하세요:

```bash
python wav2vec_example.py
```

오디오 파일을 변경하려면 코드의 `audio_file` 변수를 수정하세요.

### 2. Wav2Vec 모델 파인튜닝

wav2vec 모델을 한국어 데이터셋으로 파인튜닝하려면 다음과 같이 실행하세요:

```bash
python wav2vec_fine_tuning.py
```

### 3. Word2Vec 라이브러리 사용 예제

Gensim 라이브러리를 사용한 Word2Vec 예제를 실행하려면 다음과 같이 실행하세요:

```bash
python word2vec_example.py
```

### 4. Word2Vec 직접 구현 예제

PyTorch로 직접 구현한 Word2Vec 모델을 실행하려면 다음과 같이 실행하세요:

```bash
python word2vec_implementation.py
```

## 데이터 준비

- **음성 인식**: Common Voice 한국어 데이터셋 또는 직접 준비한 오디오 데이터셋 사용
- **단어 임베딩**: 자체 한국어 텍스트 데이터를 준비하거나 예제에 포함된 샘플 데이터 사용

## 주의사항

- GPU 메모리가 최소 8GB 이상 필요합니다 (특히 wav2vec 파인튜닝).
- 한국어 형태소 분석을 위해 KoNLPy가 필요하며, 필요한 경우 JDK 설치가 필요할 수 있습니다.
- 학습 과정은 상당한 시간이 소요될 수 있습니다.

## 참고 자료

### Wav2Vec 관련
- [Wav2Vec 2.0 논문](https://arxiv.org/abs/2006.11477)
- [Hugging Face Transformers 문서](https://huggingface.co/docs/transformers/model_doc/wav2vec2)
- [한국어 Wav2Vec 모델](https://huggingface.co/kresnik/wav2vec2-large-xlsr-korean)

### Word2Vec 관련
- [Word2Vec 원본 논문](https://arxiv.org/abs/1301.3781)
- [Gensim Word2Vec 문서](https://radimrehurek.com/gensim/models/word2vec.html)
- [KoNLPy 문서](https://konlpy.org/en/latest/) 