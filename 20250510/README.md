# Whisper STT 도구

이 도구는 OpenAI의 Whisper 모델을 사용하여 음성을 텍스트로 변환하고, 추가적으로 화자 분리(Speaker Diarization) 기능을 제공합니다.

## 기능

- 다양한 오디오 파일 형식 지원 (wav, mp3, flac 등)
- 여러 언어 자동 감지 및 전사 지원
- 다양한 크기의 Whisper 모델 선택 가능 (tiny부터 large까지)
- 화자 분리 기능으로 여러 사람이 말하는 오디오에서 각 화자 구분
- 세부적인 세그먼트 정보 제공
- 화자별 대화 요약
- 결과 저장 (텍스트 및 JSON 형식)

## 설치 방법

### Windows 사용자

1. 먼저 [Python 3.8 이상](https://www.python.org/downloads/)을 설치하세요.
2. `install_dependencies.bat` 파일을 실행하여 필요한 패키지를 모두 설치합니다.

### 다른 운영체제 사용자

다음 명령어로 필요한 패키지를 설치하세요:

```bash
pip install openai-whisper soundfile matplotlib pandas pyannote.audio
```

## 사용 방법

### Windows 사용자

1. `run_whisper_stt.bat` 파일을 실행합니다.
2. 대화형 프롬프트에 따라 오디오 파일 경로, 모델 크기, 언어 코드 등을 입력합니다.
3. 화자 분리 기능을 사용하려면 해당 옵션을 선택하세요.

### 명령줄 사용자

다음 명령어로 스크립트를 직접 실행할 수 있습니다:

```bash
python whisper_stt.py --file 오디오파일경로 --model base --language ko --diarize
```

또는 대화형 모드:

```bash
python whisper_stt.py --interactive
```

## 화자 분리 사용 시 주의사항

화자 분리 기능을 사용하기 위해서는:

1. [Hugging Face](https://huggingface.co/) 계정이 필요합니다.
2. [pyannote/speaker-diarization](https://huggingface.co/pyannote/speaker-diarization) 모델 사용 동의가 필요합니다.
3. [Access Tokens](https://huggingface.co/settings/tokens) 페이지에서 토큰을 생성해야 합니다.

## 문제 해결

- 화자 분리가 실행되지 않을 경우, Hugging Face 토큰이 유효한지 확인하세요.
- GPU가 있는 시스템에서는 화자 분리 성능이 더 좋습니다.
- 오디오 길이가 너무 짧으면 화자 분리가 정확하지 않을 수 있습니다. 