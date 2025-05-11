@echo off
echo Whisper STT 실행 스크립트
echo ------------------------------

REM Python 설치 확인
where python >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo Python이 설치되어 있지 않거나 PATH에 설정되어 있지 않습니다.
    echo Python을 설치한 후 다시 시도해주세요.
    pause
    exit /b
)

REM 대화형 모드로 스크립트 실행
echo Python 스크립트 실행 중...
python whisper_stt.py --interactive

echo.
echo 처리가 완료되었습니다.
pause 