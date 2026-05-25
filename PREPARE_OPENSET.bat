@echo off
title Openset Dataset Preparation
setlocal EnableDelayedExpansion
cd /d "%~dp0"

chcp 65001 >nul
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"

echo ============================================================
echo   오픈소스 원형 객체 데이터셋 준비 (pill_cap 백업용)
echo ============================================================
echo.
echo  [Roboflow]
echo    1. bottle_cap      — RF  병뚜껑 ~639장
echo    2. coin            — RF  동전  ~2,815장
echo    3. pills           — RF  알약   ~822장
echo    4. all (RF)        — RF  1+2+3 전부 병합
echo.
echo  [Google Open Images V7]
echo    5. oiv7_bottle_cap — OIV7 병뚜껑 최대2,500장 (Google 직접 어노테이션)
echo    6. oiv7_coin       — OIV7 동전   최대2,500장
echo.
echo  [비교 실험 - RF vs OIV7 별도 폴더 준비]
echo    7. 비교: bottle_cap (RF vs OIV7)
echo    8. 비교: coin       (RF vs OIV7)
echo.
echo    9. ZIP 파일 사용
echo.
set /p "PRESET_CHOICE=선택 [1-9]: "
set "PRESET_CHOICE=%PRESET_CHOICE: =%"

:: ── 선택에 따른 프리셋 인수 결정 ─────────────────────────────────
set "PRESET_ARGS="
set "NEED_RF_KEY=1"
set "NEED_OIV7=0"
set "IS_COMPARE=0"

if "%PRESET_CHOICE%"=="1" (
    set "PRESET_ARGS=--preset bottle_cap"
) else if "%PRESET_CHOICE%"=="2" (
    set "PRESET_ARGS=--preset coin"
) else if "%PRESET_CHOICE%"=="3" (
    set "PRESET_ARGS=--preset pills"
) else if "%PRESET_CHOICE%"=="4" (
    set "PRESET_ARGS=--preset bottle_cap --preset coin --preset pills"
) else if "%PRESET_CHOICE%"=="5" (
    set "PRESET_ARGS=--preset oiv7_bottle_cap"
    set "NEED_RF_KEY=0"
    set "NEED_OIV7=1"
) else if "%PRESET_CHOICE%"=="6" (
    set "PRESET_ARGS=--preset oiv7_coin"
    set "NEED_RF_KEY=0"
    set "NEED_OIV7=1"
) else if "%PRESET_CHOICE%"=="7" (
    set "PRESET_ARGS=--preset bottle_cap --preset oiv7_bottle_cap --compare"
    set "NEED_OIV7=1"
    set "IS_COMPARE=1"
) else if "%PRESET_CHOICE%"=="8" (
    set "PRESET_ARGS=--preset coin --preset oiv7_coin --compare"
    set "NEED_OIV7=1"
    set "IS_COMPARE=1"
) else if "%PRESET_CHOICE%"=="9" (
    call :ask_zip
    if errorlevel 1 goto :failed
    set "NEED_RF_KEY=0"
) else (
    echo [ERROR] 잘못된 선택입니다.
    pause
    exit /b 1
)

:: ── Roboflow API 키 (RF 소스 사용 시) ────────────────────────────
set "API_KEY_ARG="
if "!NEED_RF_KEY!"=="1" (
    echo.
    echo  Roboflow API 키를 입력하세요.
    echo  발급: https://app.roboflow.com ^> Settings ^> API Keys
    echo  (이미 데이터가 있으면 Enter 건너뛰기 가능)
    echo.
    set /p "RF_API_KEY=API Key: "
    set "RF_API_KEY=!RF_API_KEY: =!"
    if not "!RF_API_KEY!"=="" (
        set "API_KEY_ARG=--api-key !RF_API_KEY!"
    )
)

:: ── 학습 여부 선택 ────────────────────────────────────────────────
echo.
if "!IS_COMPARE!"=="1" (
    echo  비교 모드: RF 모델과 OIV7 모델을 각각 학습합니다.
    set /p "DO_TRAIN=두 모델 모두 학습도 실행할까요? [y/N]: "
) else (
    set /p "DO_TRAIN=데이터 준비 완료 후 바로 학습도 실행할까요? [y/N]: "
)
if "!DO_TRAIN!"=="" set "DO_TRAIN=N"
set "TRAIN_ARG="
if /I "!DO_TRAIN!"=="Y" set "TRAIN_ARG=--train"

:: ── Python 환경 확인 ──────────────────────────────────────────────
set "PY=python"
python --version >nul 2>&1
if errorlevel 1 set "PY=py -3"

call :ensure_deps
if errorlevel 1 goto :failed

if "!NEED_OIV7!"=="1" (
    call :ensure_fiftyone
    if errorlevel 1 goto :failed
)

:: ── 학습 에포크 (학습 선택 시) ───────────────────────────────────
set "EPOCH_ARG="
if /I "!DO_TRAIN!"=="Y" (
    echo.
    set /p "EPOCHS=학습 에포크 수 [기본 80]: "
    if not "!EPOCHS!"=="" (
        set "EPOCH_ARG=--epochs !EPOCHS!"
    )
)

echo.
echo ============================================================
if "!IS_COMPARE!"=="1" (
    echo   비교 실험 데이터셋 준비 ^(RF vs OIV7^)
) else (
    echo   데이터셋 준비 시작
)
echo ============================================================
echo.

!PY! prepare_openset.py !API_KEY_ARG! !PRESET_ARGS! !TRAIN_ARG! !EPOCH_ARG!
if errorlevel 1 goto :failed

if "!IS_COMPARE!"=="1" (
    echo.
    echo ============================================================
    echo   비교 실험 준비 완료
    echo ============================================================
    echo.
    echo   데이터셋 위치: %CD%\openset_compare\
    echo.
    echo   모델 학습이 완료됐다면 각 best.pt 로 a4_plane_research.py 실행:
    echo     python a4_detect\a4_plane_research.py --model openset_runs\..._rf_compare\...best.pt
    echo     python a4_detect\a4_plane_research.py --model openset_runs\..._oiv7_compare\...best.pt
    echo.
    echo   탐지율 및 중심점 오차를 비교해 더 적합한 소스 판단.
)

echo.
pause
exit /b 0

:: ---------------------------------------------------------------
:ask_zip
echo.
echo  ZIP 파일 경로를 입력하세요 (여러 개는 세미콜론으로 구분).
echo  예: C:\Downloads\bottle-cap.zip
echo  예: C:\Downloads\cap.zip;C:\Downloads\coin.zip
echo.
set /p "ZIP_PATHS=ZIP 경로: "
if "!ZIP_PATHS!"=="" (
    echo [ERROR] 경로가 없습니다.
    exit /b 1
)
set "PRESET_ARGS="
for %%Z in ("!ZIP_PATHS:;=" "!") do (
    set "PRESET_ARGS=!PRESET_ARGS! --zip %%~Z"
)
exit /b 0

:: ---------------------------------------------------------------
:ensure_deps
echo.
echo [DEPS] Python 패키지 확인 중...
%PY% -c "import cv2, PIL, pillow_heif, ultralytics" >nul 2>&1
if not errorlevel 1 (
    echo [DEPS] 필수 패키지 확인 완료.
    exit /b 0
)
echo [DEPS] 패키지 없음. 가상환경 준비 중...
set "VENV_DIR=%CD%\.venv_research"
if not exist "!VENV_DIR!\Scripts\python.exe" (
    %PY% -m venv "!VENV_DIR!"
    if errorlevel 1 exit /b 1
)
set "PY=!VENV_DIR!\Scripts\python.exe"
"!PY!" -m pip install --upgrade pip
if errorlevel 1 exit /b 1
"!PY!" -m pip install -r requirements.txt
if errorlevel 1 exit /b 1
exit /b 0

:: ---------------------------------------------------------------
:ensure_fiftyone
echo.
echo [DEPS] fiftyone 확인 중 ^(OIV7 필수^)...
!PY! -c "import fiftyone" >nul 2>&1
if not errorlevel 1 (
    echo [DEPS] fiftyone 확인 완료.
    exit /b 0
)
echo [DEPS] fiftyone 설치 중...
echo        ^(대용량 패키지 — 수 분 소요될 수 있습니다^)
"!PY!" -m pip install fiftyone
if errorlevel 1 (
    echo [ERROR] fiftyone 설치 실패.
    echo         수동 설치: pip install fiftyone
    exit /b 1
)
exit /b 0

:: ---------------------------------------------------------------
:failed
echo.
echo [ERROR] 작업 중 오류가 발생했습니다.
echo 위의 출력 내용을 확인하세요.
echo.
pause
exit /b 1
