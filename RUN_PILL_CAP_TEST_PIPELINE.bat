@echo off
title Pill Cap Test YOLO Pipeline
setlocal EnableDelayedExpansion
cd /d "%~dp0"

chcp 65001 >nul
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "TARGET_NAME=pill_cap"
set "CONFIG=research_configs\pill_cap.json"
set "STATE_FILE=%CD%\research_runs\pill_cap_last_session.cmd"

echo ============================================================
echo   Pill Cap Test YOLO Pipeline
echo ============================================================
echo.
echo  Source images:
echo    %CD%\test
echo.
echo  Action:
echo    1. fresh start: reset pill_cap outputs, then label 50 seed images
echo    2. label/complete 50 seed images only
echo    3. run reviewed active-learning pipeline from seed labels
echo    4. run Stage 1 only, then stop for review/resume
echo    5. resume from Stage 2
echo    6. status
echo    7. evaluate latest model on full source image pool
echo    8. reset pill_cap outputs only
echo    9. continue last saved session
echo   10. screen source photo quality
echo   11. add confirmed no-object images as empty labels
echo.
set /p "ACTION_CHOICE=Select action [1/2/3/4/5/6/7/8/9/10/11]: "
set "ACTION_CHOICE=%ACTION_CHOICE: =%"

set "PY=python"
python --version >nul 2>&1
if errorlevel 1 set "PY=py -3"

call :ensure_deps
if errorlevel 1 goto :failed

if "%ACTION_CHOICE%"=="1" (
    call :reset_one
    if errorlevel 1 goto :failed
    call :run_pipeline --init --status
    if errorlevel 1 goto :failed
    call :save_state stage1
    call :ask_continue
) else if "%ACTION_CHOICE%"=="2" (
    call :run_pipeline --init --status
    if errorlevel 1 goto :failed
    call :save_state stage1
) else if "%ACTION_CHOICE%"=="3" (
    call :run_pipeline --run
    if errorlevel 1 goto :failed
    call :save_state complete
) else if "%ACTION_CHOICE%"=="4" (
    call :run_pipeline --run --stage1-only
    if errorlevel 1 goto :failed
    call :save_state stage2
) else if "%ACTION_CHOICE%"=="5" (
    call :run_pipeline --run --start-stage 2
    if errorlevel 1 goto :failed
    call :save_state complete
) else if "%ACTION_CHOICE%"=="6" (
    call :run_pipeline --status
    if errorlevel 1 goto :failed
) else if "%ACTION_CHOICE%"=="7" (
    call :run_pipeline --eval-source
    if errorlevel 1 goto :failed
) else if "%ACTION_CHOICE%"=="8" (
    call :reset_one
    if errorlevel 1 goto :failed
) else if "%ACTION_CHOICE%"=="9" (
    call :continue_saved
    if errorlevel 1 goto :failed
) else if "%ACTION_CHOICE%"=="10" (
    call :screen_quality
    if errorlevel 1 goto :failed
) else if "%ACTION_CHOICE%"=="11" (
    call :add_confirmed_negatives
    if errorlevel 1 goto :failed
) else (
    echo [ERROR] Invalid action.
    pause
    exit /b 1
)

echo.
pause
exit /b 0

:: ---------------------------------------------------------------
:ask_continue
echo.
set /p "CONTINUE_RUN=Seed step is ready. Continue full reviewed pipeline now? [Y/n]: "
if "!CONTINUE_RUN!"=="" set "CONTINUE_RUN=Y"
if /I "!CONTINUE_RUN!"=="Y" (
    call :run_pipeline --run
    if errorlevel 1 exit /b 1
    call :save_state complete
)
exit /b %errorlevel%

:: ---------------------------------------------------------------
:ensure_deps
echo.
echo [DEPS] Checking Python packages...
%PY% -c "import cv2, PIL, pillow_heif, ultralytics" >nul 2>&1
if not errorlevel 1 (
    echo [DEPS] Required packages already available.
    exit /b 0
)
echo [DEPS] Missing packages detected. Preparing local virtual environment...
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
"!PY!" -c "import cv2, PIL, pillow_heif, ultralytics"
exit /b %errorlevel%

:: ---------------------------------------------------------------
:run_pipeline
if not exist "%CONFIG%" (
    echo [ERROR] Config not found: %CONFIG%
    exit /b 1
)
set "LOG_DIR=%CD%\research_runs\%TARGET_NAME%\logs"
if not exist "!LOG_DIR!" mkdir "!LOG_DIR!"
for /f %%I in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set "TS=%%I"
set "LOG_FILE=!LOG_DIR!\pipeline_!TS!.log"
echo [LOG] !LOG_FILE!
if "!PY!"=="python" (
    powershell -NoProfile -ExecutionPolicy Bypass -Command "$enc=[System.Text.UTF8Encoding]::new($false); [Console]::OutputEncoding=$enc; $OutputEncoding=$enc; $env:PYTHONUTF8='1'; $env:PYTHONIOENCODING='utf-8'; & python research_pipeline.py --config '%CONFIG%' %* 2>&1 | Tee-Object -FilePath '!LOG_FILE!'; exit $LASTEXITCODE"
) else if "!PY!"=="py -3" (
    powershell -NoProfile -ExecutionPolicy Bypass -Command "$enc=[System.Text.UTF8Encoding]::new($false); [Console]::OutputEncoding=$enc; $OutputEncoding=$enc; $env:PYTHONUTF8='1'; $env:PYTHONIOENCODING='utf-8'; & py -3 research_pipeline.py --config '%CONFIG%' %* 2>&1 | Tee-Object -FilePath '!LOG_FILE!'; exit $LASTEXITCODE"
) else (
    powershell -NoProfile -ExecutionPolicy Bypass -Command "$enc=[System.Text.UTF8Encoding]::new($false); [Console]::OutputEncoding=$enc; $OutputEncoding=$enc; $env:PYTHONUTF8='1'; $env:PYTHONIOENCODING='utf-8'; & '!PY!' research_pipeline.py --config '%CONFIG%' %* 2>&1 | Tee-Object -FilePath '!LOG_FILE!'; exit $LASTEXITCODE"
)
exit /b %errorlevel%

:: ---------------------------------------------------------------
:reset_one
echo.
echo [WARN] This will remove pill_cap research outputs only:
echo        %CD%\research_runs\pill_cap
echo.
set /p "RESET_CONFIRM=Type RESET to continue: "
if not "%RESET_CONFIRM%"=="RESET" (
    echo Reset cancelled.
    exit /b 1
)
set "RESET_DIR=%CD%\research_runs\pill_cap"
if exist "%RESET_DIR%" (
    powershell -NoProfile -ExecutionPolicy Bypass -Command "$root=(Resolve-Path -LiteralPath '%CD%').Path; $target='%RESET_DIR%'; if (Test-Path -LiteralPath $target) { $resolved=(Resolve-Path -LiteralPath $target).Path; if (-not $resolved.StartsWith($root, [System.StringComparison]::OrdinalIgnoreCase)) { throw 'Refusing to remove outside labeling_tools folder' }; Remove-Item -LiteralPath $resolved -Recurse -Force }"
    if errorlevel 1 exit /b 1
) else (
    echo [RESET] pill_cap already clean.
)
exit /b 0

:: ---------------------------------------------------------------
:screen_quality
set "QUALITY_DIR=%CD%\research_runs\pill_cap\00_quality_check"
if not exist "%CD%\research_runs\pill_cap" mkdir "%CD%\research_runs\pill_cap"
echo.
echo [QUALITY] Screening source images.
echo           Output: !QUALITY_DIR!
echo.
"!PY!" screen_image_quality.py --images "%CD%\test" --output-dir "!QUALITY_DIR!"
exit /b %errorlevel%

:: ---------------------------------------------------------------
:add_confirmed_negatives
set "NEG_DIR=%CD%\research_runs\pill_cap\00_quality_check\confirmed_negative"
set "DATASET_DIR=%CD%\research_runs\pill_cap\01_manual_seed_dataset"
echo.
echo [NEGATIVE] This adds confirmed no-object photos as empty YOLO labels.
echo            Source : !NEG_DIR!
echo            Dataset: !DATASET_DIR!
echo.
if not exist "!NEG_DIR!" (
    echo [ERROR] Folder not found. Run action 10 first, then copy confirmed no-object photos into:
    echo         !NEG_DIR!
    exit /b 1
)
"!PY!" add_empty_labels.py --images "!NEG_DIR!" --dataset-dir "!DATASET_DIR!" --class-names pill_cap
exit /b %errorlevel%

:: ---------------------------------------------------------------
:save_state
set "NEXT_PHASE=%~1"
if not exist "%CD%\research_runs" mkdir "%CD%\research_runs"
for /f %%I in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set "SAVED_AT=%%I"
> "%STATE_FILE%" echo set "TARGET_NAME=pill_cap"
>> "%STATE_FILE%" echo set "NEXT_PHASE=!NEXT_PHASE!"
>> "%STATE_FILE%" echo set "SAVED_AT=!SAVED_AT!"
echo [STATE] saved next phase: !NEXT_PHASE!
echo         %STATE_FILE%
exit /b 0

:: ---------------------------------------------------------------
:continue_saved
if not exist "%STATE_FILE%" (
    echo [ERROR] No saved pill_cap session found.
    echo         Expected: %STATE_FILE%
    exit /b 1
)
call "%STATE_FILE%"
if not defined NEXT_PHASE (
    echo [ERROR] Saved session is invalid: NEXT_PHASE is missing.
    exit /b 1
)
echo.
echo [RESUME] next phase: !NEXT_PHASE!
if defined SAVED_AT echo [RESUME] saved at  : !SAVED_AT!
echo.
set /p "RESUME_CONFIRM=Continue this session now? [Y/n]: "
if "!RESUME_CONFIRM!"=="" set "RESUME_CONFIRM=Y"
if /I not "!RESUME_CONFIRM!"=="Y" (
    echo Resume cancelled.
    exit /b 0
)
if /I "!NEXT_PHASE!"=="stage1" (
    call :run_pipeline --run
    if errorlevel 1 exit /b 1
    call :save_state complete
    exit /b %errorlevel%
) else if /I "!NEXT_PHASE!"=="stage2" (
    call :run_pipeline --run --start-stage 2
    if errorlevel 1 exit /b 1
    call :save_state complete
    exit /b %errorlevel%
) else if /I "!NEXT_PHASE!"=="complete" (
    echo Last saved session is already complete. Showing status instead.
    call :run_pipeline --status
    exit /b %errorlevel%
) else (
    echo [ERROR] Unknown resume phase: !NEXT_PHASE!
    exit /b 1
)

:: ---------------------------------------------------------------
:failed
echo.
echo [ERROR] Pill cap pipeline failed.
echo Check the console output above and the latest log in:
echo   %CD%\research_runs\pill_cap\logs
echo.
pause
exit /b 1
