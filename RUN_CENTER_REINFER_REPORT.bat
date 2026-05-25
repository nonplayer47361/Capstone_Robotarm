@echo off
title YOLO Center Re-inference Report
setlocal EnableDelayedExpansion
cd /d "%~dp0"

chcp 65001 >nul
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"

echo ============================================================
echo   YOLO Center Re-inference Report
echo ============================================================
echo.
echo  Target:
echo    1. blueberry
echo    2. strawberry
echo    3. both
echo.
set /p "TARGET_CHOICE=Select target [1/2/3]: "
set "TARGET_CHOICE=%TARGET_CHOICE: =%"
if "%TARGET_CHOICE%"=="1" (
    set "TARGET=blueberry"
) else if "%TARGET_CHOICE%"=="2" (
    set "TARGET=strawberry"
) else if "%TARGET_CHOICE%"=="3" (
    set "TARGET=both"
) else (
    echo [ERROR] Invalid target.
    pause
    exit /b 1
)

echo.
echo  Model set:
echo    1. reviewed final model
echo    2. fullauto final model
echo    3. reviewed + fullauto
echo.
set /p "MODEL_CHOICE=Select model set [1/2/3]: "
set "MODEL_CHOICE=%MODEL_CHOICE: =%"
if "%MODEL_CHOICE%"=="1" (
    set "MODEL_SET=reviewed"
) else if "%MODEL_CHOICE%"=="2" (
    set "MODEL_SET=fullauto"
) else if "%MODEL_CHOICE%"=="3" (
    set "MODEL_SET=both"
) else (
    echo [ERROR] Invalid model set.
    pause
    exit /b 1
)

echo.
set /p "CONF=Confidence threshold [default 0.25]: "
if "%CONF%"=="" set "CONF=0.25"

set "PY=python"
python --version >nul 2>&1
if errorlevel 1 set "PY=py -3"

%PY% center_reinfer_report.py --target "%TARGET%" --model-set "%MODEL_SET%" --conf "%CONF%"
if errorlevel 1 (
    echo [ERROR] Center re-inference report failed.
    pause
    exit /b 1
)

echo.
pause
exit /b 0
