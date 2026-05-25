@echo off
title YOLO Full Auto Expansion
setlocal EnableDelayedExpansion
cd /d "%~dp0"

chcp 65001 >nul
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"

echo ============================================================
echo   YOLO Full Auto Expansion
echo ============================================================
echo.
echo  Target:
echo    1. blueberry
echo    2. strawberry
echo    3. both sequential
echo.
set /p "TARGET_CHOICE=Select target [1/2/3]: "
set "TARGET_CHOICE=%TARGET_CHOICE: =%"
if "%TARGET_CHOICE%"=="1" (
    set "CONFIG_LIST=configs\blueberry.json"
) else if "%TARGET_CHOICE%"=="2" (
    set "CONFIG_LIST=configs\strawberry.json"
) else if "%TARGET_CHOICE%"=="3" (
    set "CONFIG_LIST=configs\blueberry.json configs\strawberry.json"
) else (
    echo [ERROR] Invalid target.
    pause
    exit /b 1
)

echo.
echo  Action:
echo    1. run full-auto expansion
echo    2. status only
echo.
set /p "ACTION_CHOICE=Select action [1/2]: "
set "ACTION_CHOICE=%ACTION_CHOICE: =%"

set "PY=python"
python --version >nul 2>&1
if errorlevel 1 set "PY=py -3"

for %%C in (!CONFIG_LIST!) do (
    echo.
    echo ============================================================
    echo   Config: %%C
    echo ============================================================
    if "%ACTION_CHOICE%"=="1" (
        %PY% full_auto_pipeline.py --config "%%C"
    ) else if "%ACTION_CHOICE%"=="2" (
        %PY% full_auto_pipeline.py --config "%%C" --status
    ) else (
        echo [ERROR] Invalid action.
        pause
        exit /b 1
    )
    if errorlevel 1 (
        echo [ERROR] Full-auto pipeline failed.
        pause
        exit /b 1
    )
)

echo.
pause
exit /b 0
