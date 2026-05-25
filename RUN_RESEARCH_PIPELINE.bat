@echo off
title YOLO Research Pipeline
setlocal EnableDelayedExpansion
cd /d "%~dp0"

chcp 65001 >nul
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "STATE_FILE=%CD%\research_runs\last_session.cmd"
set "CONTINUE_MODE=0"

echo ============================================================
echo   YOLO Research Pipeline
echo ============================================================
echo.
echo  Research scope:
echo    1. blueberry only
echo    2. strawberry only
echo    3. both sequential  (blueberry -^> strawberry)
echo    4. continue last saved session
echo.
set /p "SCOPE_CHOICE=Select scope [1/2/3/4]: "
set "SCOPE_CHOICE=%SCOPE_CHOICE: =%"
if "%SCOPE_CHOICE%"=="1" (
    set "TARGET_LIST=blueberry"
) else if "%SCOPE_CHOICE%"=="2" (
    set "TARGET_LIST=strawberry"
) else if "%SCOPE_CHOICE%"=="3" (
    set "TARGET_LIST=blueberry strawberry"
) else if "%SCOPE_CHOICE%"=="4" (
    set "CONTINUE_MODE=1"
    call :load_state
    if errorlevel 1 (
        pause
        exit /b 1
    )
) else (
    echo [ERROR] Invalid scope.
    pause
    exit /b 1
)

if "%CONTINUE_MODE%"=="0" (
    echo.
    echo  Main action:
    echo    1. fresh start: reset, label holdout, then seed/manual labeling
    echo    2. seed/manual labeling, then optionally continue phased reviewed pipeline
    echo    3. run phased reviewed pipeline from seed/manual labels
    echo    4. label excluded 50 holdout images
    echo    5. status
    echo    6. evaluate latest model on holdout images
    echo    7. evaluate latest model on full source image pool
    echo.
    echo  Advanced restarts are run directly with research_pipeline.py.
    echo  Examples are documented in README.md.
    echo.
    set /p "ACTION_CHOICE=Select action [1/2/3/4/5/6/7]: "
    set "ACTION_CHOICE=!ACTION_CHOICE: =!"
)

set "PY=python"
python --version >nul 2>&1
if errorlevel 1 set "PY=py -3"

call :ensure_deps
if errorlevel 1 goto :failed_global

if "%CONTINUE_MODE%"=="1" (
    call :continue_saved
    if errorlevel 1 goto :failed_global
) else if "%ACTION_CHOICE%"=="1" (
    call :reset_targets
    if errorlevel 1 goto :failed_global
    call :label_holdout_phase
    if errorlevel 1 goto :failed_global
    call :seed_targets
    if errorlevel 1 goto :failed_global
    call :ask_continue_reviewed
    if errorlevel 1 goto :failed_global
) else if "%ACTION_CHOICE%"=="2" (
    call :seed_targets
    if errorlevel 1 goto :failed_global
    call :ask_continue_reviewed
    if errorlevel 1 goto :failed_global
) else if "%ACTION_CHOICE%"=="3" (
    call :run_reviewed_phases
    if errorlevel 1 goto :failed_global
) else if "%ACTION_CHOICE%"=="4" (
    call :run_for_targets --label-holdout
    if errorlevel 1 goto :failed_global
) else if "%ACTION_CHOICE%"=="5" (
    call :run_for_targets --status
    if errorlevel 1 goto :failed_global
) else if "%ACTION_CHOICE%"=="6" (
    call :run_for_targets --eval-holdout
    if errorlevel 1 goto :failed_global
) else if "%ACTION_CHOICE%"=="7" (
    call :run_for_targets --eval-source
    if errorlevel 1 goto :failed_global
) else (
    echo [ERROR] Invalid action.
    pause
    exit /b 1
)

:end
echo.
pause
exit /b 0

:: ---------------------------------------------------------------
:label_holdout_phase
echo.
echo [PHASE] Holdout labeling for final evaluation
echo        Selected target order: !TARGET_LIST!
echo.
call :run_for_targets --label-holdout
exit /b %errorlevel%

:: ---------------------------------------------------------------
:seed_targets
echo.
echo [PHASE] Manual seed preparation
echo        Selected target order: !TARGET_LIST!
echo.
call :run_for_targets --init --status
if errorlevel 1 exit /b 1
call :save_state stage1
exit /b %errorlevel%

:: ---------------------------------------------------------------
:ask_continue_reviewed
echo.
set /p "CONTINUE_RUN=Seed step is ready. Continue phased reviewed pipeline now? [Y/n]: "
if "!CONTINUE_RUN!"=="" set "CONTINUE_RUN=Y"
if /I "!CONTINUE_RUN!"=="Y" (
    call :run_reviewed_phases
    exit /b %errorlevel%
)
echo Pipeline stopped after seed/manual labeling.
echo Saved resume point: stage1
exit /b 0

:: ---------------------------------------------------------------
:run_reviewed_phases
echo.
echo [PHASE] Stage 1 seed model, auto-labeling, and required review
echo        Target order: !TARGET_LIST!
echo.
call :run_for_targets --run --stage1-only
if errorlevel 1 exit /b 1
call :save_state stage2
echo.
set /p "CONTINUE_STAGE2=Stage 1 review is complete. Continue stages 2-3-final now? [Y/n]: "
if "!CONTINUE_STAGE2!"=="" set "CONTINUE_STAGE2=Y"
if /I "!CONTINUE_STAGE2!"=="Y" (
    echo.
    echo [PHASE] Stages 2-3-final
    call :run_for_targets --run --start-stage 2
    if errorlevel 1 exit /b 1
    call :save_state complete
    exit /b %errorlevel%
)
echo Pipeline stopped after Stage 1. Resume later with:
for %%T in (!TARGET_LIST!) do echo   python research_pipeline.py --config research_configs\%%T.json --run --start-stage 2
echo Saved resume point: stage2
exit /b 0

:: ---------------------------------------------------------------
:continue_saved
echo.
echo [RESUME] Loaded saved session
echo          targets    : !TARGET_LIST!
echo          next phase : !NEXT_PHASE!
if defined SAVED_AT echo          saved at   : !SAVED_AT!
echo.
set /p "RESUME_CONFIRM=Continue this session now? [Y/n]: "
if "!RESUME_CONFIRM!"=="" set "RESUME_CONFIRM=Y"
if /I not "!RESUME_CONFIRM!"=="Y" (
    echo Resume cancelled.
    exit /b 0
)
if /I "!NEXT_PHASE!"=="stage1" (
    call :run_reviewed_phases
    exit /b %errorlevel%
) else if /I "!NEXT_PHASE!"=="stage2" (
    echo.
    echo [PHASE] Stages 2-3-final
    call :run_for_targets --run --start-stage 2
    if errorlevel 1 exit /b 1
    call :save_state complete
    exit /b %errorlevel%
) else if /I "!NEXT_PHASE!"=="complete" (
    echo Last saved session is already complete. Showing status instead.
    call :run_for_targets --status
    exit /b %errorlevel%
) else (
    echo [ERROR] Unknown resume phase: !NEXT_PHASE!
    exit /b 1
)

:: ---------------------------------------------------------------
:run_for_targets
for %%T in (!TARGET_LIST!) do (
    set "CONFIG=research_configs\%%T.json"
    set "TARGET_NAME=%%T"
    echo.
    echo ============================================================
    echo   Target: %%T
    echo ============================================================
    if not exist "!CONFIG!" (
        echo [ERROR] Config not found: !CONFIG!
        exit /b 1
    )
    call :run_pipeline %*
    if errorlevel 1 exit /b 1
)
exit /b 0

:: ---------------------------------------------------------------
:failed_global
echo.
echo [ERROR] Research pipeline failed.
if defined TARGET_NAME (
    echo Check the console output above and the latest log in:
    echo   %CD%\research_runs\!TARGET_NAME!\logs
) else (
    echo Check the console output above.
)
echo.
pause
exit /b 1

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
set "LOG_DIR=%CD%\research_runs\!TARGET_NAME!\logs"
if not exist "!LOG_DIR!" mkdir "!LOG_DIR!"
for /f %%I in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set "TS=%%I"
set "LOG_FILE=!LOG_DIR!\pipeline_!TS!.log"
echo [LOG] !LOG_FILE!
if "!PY!"=="python" (
    powershell -NoProfile -ExecutionPolicy Bypass -Command "$enc=[System.Text.UTF8Encoding]::new($false); [Console]::OutputEncoding=$enc; $OutputEncoding=$enc; $env:PYTHONUTF8='1'; $env:PYTHONIOENCODING='utf-8'; & python research_pipeline.py --config '!CONFIG!' %* 2>&1 | Tee-Object -FilePath '!LOG_FILE!'; exit $LASTEXITCODE"
) else if "!PY!"=="py -3" (
    powershell -NoProfile -ExecutionPolicy Bypass -Command "$enc=[System.Text.UTF8Encoding]::new($false); [Console]::OutputEncoding=$enc; $OutputEncoding=$enc; $env:PYTHONUTF8='1'; $env:PYTHONIOENCODING='utf-8'; & py -3 research_pipeline.py --config '!CONFIG!' %* 2>&1 | Tee-Object -FilePath '!LOG_FILE!'; exit $LASTEXITCODE"
) else (
    powershell -NoProfile -ExecutionPolicy Bypass -Command "$enc=[System.Text.UTF8Encoding]::new($false); [Console]::OutputEncoding=$enc; $OutputEncoding=$enc; $env:PYTHONUTF8='1'; $env:PYTHONIOENCODING='utf-8'; & '!PY!' research_pipeline.py --config '!CONFIG!' %* 2>&1 | Tee-Object -FilePath '!LOG_FILE!'; exit $LASTEXITCODE"
)
exit /b %errorlevel%

:: ---------------------------------------------------------------
:reset_targets
echo.
echo [WARN] This will remove all research outputs for:
echo        !TARGET_LIST!
echo        Base folder: %CD%\research_runs
echo.
set /p "RESET_CONFIRM=Type RESET to continue: "
if not "!RESET_CONFIRM!"=="RESET" (
    echo Reset cancelled.
    exit /b 1
)
for %%T in (!TARGET_LIST!) do (
    call :reset_one %%T
    if errorlevel 1 exit /b 1
)
echo Reset complete.
exit /b 0

:: ---------------------------------------------------------------
:save_state
set "NEXT_PHASE=%~1"
if not exist "%CD%\research_runs" mkdir "%CD%\research_runs"
for /f %%I in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set "SAVED_AT=%%I"
> "!STATE_FILE!" echo set "TARGET_LIST=!TARGET_LIST!"
>> "!STATE_FILE!" echo set "NEXT_PHASE=!NEXT_PHASE!"
>> "!STATE_FILE!" echo set "SAVED_AT=!SAVED_AT!"
echo [STATE] saved next phase: !NEXT_PHASE!
echo         !STATE_FILE!
exit /b 0

:: ---------------------------------------------------------------
:load_state
if not exist "!STATE_FILE!" (
    echo [ERROR] No saved session found.
    echo         Expected: !STATE_FILE!
    exit /b 1
)
call "!STATE_FILE!"
if not defined TARGET_LIST (
    echo [ERROR] Saved session is invalid: TARGET_LIST is missing.
    exit /b 1
)
if not defined NEXT_PHASE (
    echo [ERROR] Saved session is invalid: NEXT_PHASE is missing.
    exit /b 1
)
exit /b 0

:: ---------------------------------------------------------------
:reset_one
set "RESET_NAME=%~1"
set "RESET_DIR=%CD%\research_runs\!RESET_NAME!"
if exist "!RESET_DIR!" (
    echo [RESET] !RESET_NAME!
    powershell -NoProfile -ExecutionPolicy Bypass -Command "$root=(Resolve-Path -LiteralPath '%CD%').Path; $target='!RESET_DIR!'; if (Test-Path -LiteralPath $target) { $resolved=(Resolve-Path -LiteralPath $target).Path; if (-not $resolved.StartsWith($root, [System.StringComparison]::OrdinalIgnoreCase)) { throw 'Refusing to remove outside labeling_tools folder' }; Remove-Item -LiteralPath $resolved -Recurse -Force }"
    if errorlevel 1 exit /b 1
) else (
    echo [RESET] !RESET_NAME! already clean.
)
exit /b 0
