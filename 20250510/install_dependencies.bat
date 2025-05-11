@echo off
echo Whisper STT 의존성 설치 스크립트
echo ------------------------------

REM Python 설치 확인
where python >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo Python이 설치되어 있지 않거나 PATH에 설정되어 있지 않습니다.
    echo Python을 설치한 후 다시 시도해주세요.
    pause
    exit /b
)

echo 필요한 패키지를 설치합니다...
echo 1/6 pip 업그레이드
python -m pip install --upgrade pip

echo 2/6 whisper 설치 중...
python -m pip install openai-whisper

echo 3/6 soundfile 설치 중...
python -m pip install soundfile

echo 4/6 matplotlib 설치 중...
python -m pip install matplotlib

echo 5/6 pandas 설치 중...
python -m pip install pandas

echo 6/6 화자 분리 라이브러리 설치 중 (시간이 좀 걸릴 수 있습니다)...
python -m pip install pyannote.audio

echo.
echo 모든 패키지가 설치되었습니다.
echo 이제 run_whisper_stt.bat을 실행하여 Whisper STT를 사용할 수 있습니다.
pause 