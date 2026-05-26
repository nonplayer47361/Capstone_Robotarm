@echo off
title A4 Pill-Cap Experiment Runner
setlocal EnableDelayedExpansion
cd /d "%~dp0"

chcp 65001 >nul
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"

set "PY_CMD=python"
python --version >nul 2>&1
if errorlevel 1 set "PY_CMD=py -3"

set "OBJECT_TYPE=pill_cap"
set "EXPECTED_CLASS=pill_cap"
set "DEFAULT_MODEL=..\research_runs\pill_cap\runs\04_final_model\weights\best.pt"

call :ensure_deps
if errorlevel 1 goto :failed

:menu
echo.
echo ============================================================
echo   A4 + YOLO Pill-Cap Experiment Runner
echo ============================================================
echo  1. STEP 1  Camera calibration capture
echo  2. STEP 1b Camera calibration preview
echo  3. STEP 2  A4 methods precheck  (edge / aruco / grid)
echo  4. STEP 3  YOLO pill-cap object-only precheck
echo  5. STEP 4  A4 + YOLO integration precheck
echo  6. STEP 5  Coordinate eval: one method
echo  7. STEP 5b Coordinate eval: edge - aruco - grid sequence
echo  8. STEP 6  Rebuild report from CSV
echo  9. Show pill-cap checklist
echo  0. Custom a4_plane_research.py args
echo.
set /p "MENU=Select [0-9]: "
set "MENU=%MENU: =%"

if "%MENU%"=="1" goto :calib_capture
if "%MENU%"=="2" goto :calib_preview
if "%MENU%"=="3" goto :precheck_a4
if "%MENU%"=="4" goto :precheck_object
if "%MENU%"=="5" goto :precheck_both
if "%MENU%"=="6" goto :eval_one
if "%MENU%"=="7" goto :eval_three_methods
if "%MENU%"=="8" goto :report
if "%MENU%"=="9" goto :checklist
if "%MENU%"=="0" goto :custom
echo [ERROR] Invalid menu.
goto :end

:calib_capture
call :ask_camera
%PY_CMD% calibrate_camera.py --capture --camera !CAMERA!
goto :end

:calib_preview
call :ask_camera
call :ask_calib_required
if errorlevel 1 goto :end
%PY_CMD% calibrate_camera.py --preview !CALIB_ARG! --camera !CAMERA!
goto :end

:precheck_a4
call :ask_camera
call :ask_calib
call :ask_condition
%PY_CMD% a4_plane_research.py --precheck --precheck-target a4 --all-methods --condition "!CONDITION!" !CALIB_ARG! --camera !CAMERA!
goto :end

:precheck_object
call :ask_camera
call :ask_model
if "!MODEL!"=="" goto :model_missing
call :ask_calib
call :ask_condition
%PY_CMD% a4_plane_research.py --precheck --precheck-target object --model "!MODEL!" --object-type %OBJECT_TYPE% --expected-class %EXPECTED_CLASS% --condition "!CONDITION!" !CALIB_ARG! --camera !CAMERA!
goto :end

:precheck_both
call :ask_camera
call :ask_model
if "!MODEL!"=="" goto :model_missing
call :ask_calib
call :ask_condition
call :ask_method
%PY_CMD% a4_plane_research.py --precheck --precheck-target both --method "!METHOD!" --model "!MODEL!" --object-type %OBJECT_TYPE% --expected-class %EXPECTED_CLASS% --condition "!CONDITION!" !CALIB_ARG! --camera !CAMERA!
goto :end

:eval_one
call :ask_camera
call :ask_model
if "!MODEL!"=="" goto :model_missing
call :ask_calib
call :ask_condition
call :ask_method
call :ask_repeats
%PY_CMD% a4_plane_research.py --eval --method "!METHOD!" --model "!MODEL!" --object-type %OBJECT_TYPE% --expected-class %EXPECTED_CLASS% --one-point --manual --repeats !REPEATS! --condition "!CONDITION!" !CALIB_ARG! --camera !CAMERA!
goto :end

:eval_three_methods
call :ask_camera
call :ask_model
if "!MODEL!"=="" goto :model_missing
call :ask_calib
call :ask_condition
call :ask_repeats

echo.
echo [1/3] Place EDGE sheet, then press any key.
pause >nul
%PY_CMD% a4_plane_research.py --eval --method edge --model "!MODEL!" --object-type %OBJECT_TYPE% --expected-class %EXPECTED_CLASS% --one-point --manual --repeats !REPEATS! --condition "!CONDITION!" !CALIB_ARG! --camera !CAMERA!
if errorlevel 1 goto :failed

echo.
echo [2/3] Place ARUCO sheet, then press any key.
pause >nul
%PY_CMD% a4_plane_research.py --eval --method aruco --model "!MODEL!" --object-type %OBJECT_TYPE% --expected-class %EXPECTED_CLASS% --one-point --manual --repeats !REPEATS! --condition "!CONDITION!" !CALIB_ARG! --camera !CAMERA!
if errorlevel 1 goto :failed

echo.
echo [3/3] Place GRID sheet, then press any key.
pause >nul
%PY_CMD% a4_plane_research.py --eval --method grid --model "!MODEL!" --object-type %OBJECT_TYPE% --expected-class %EXPECTED_CLASS% --one-point --manual --repeats !REPEATS! --condition "!CONDITION!" !CALIB_ARG! --camera !CAMERA!
goto :end

:report
echo.
set /p "CSV_PATH=CSV path: "
if "%CSV_PATH%"=="" (
    echo [ERROR] CSV path is required.
    goto :end
)
%PY_CMD% a4_plane_research.py --report --csv "%CSV_PATH%"
goto :end

:custom
echo.
echo Example:
echo   --eval --method aruco --model "..\research_runs\pill_cap\runs\04_final_model\weights\best.pt" --object-type pill_cap --expected-class pill_cap --one-point --manual --condition level --calib calib_camera0.json --camera 1
echo.
set /p "ARGS=Args: "
%PY_CMD% a4_plane_research.py %ARGS%
goto :end

:checklist
echo.
echo Recommended pill-cap flow:
echo   1) Calibration capture with checkerboard
echo   2) Calibration preview and RMS check
echo   3) A4 precheck: level, then tilt_low / tilt_mid / tilt_high
echo   4) YOLO object-only precheck: pill_cap
echo   5) Integration precheck: edge / aruco / grid
echo   6) Coordinate eval: level for edge / aruco / grid
echo   7) Coordinate eval: tilt condition for edge / aruco / grid
echo   8) Compare CSV reports and choose the best method
echo.
echo Printed sheets:
echo   checkerboard calibration sheet
echo   001 edge center guide
echo   006 aruco center guide
echo   016 grid center guide
echo.
goto :end

:: ---------------------------------------------------------------------------
:ask_camera
echo.
set /p "CAMERA=Camera ID (Enter=1): "
if "!CAMERA!"=="" set "CAMERA=1"
exit /b 0

:ask_model
echo.
echo Default model: %DEFAULT_MODEL%
set /p "MODEL=YOLO model path (.pt, Enter=default): "
if "!MODEL!"=="" (
    if exist "%DEFAULT_MODEL%" (
        set "MODEL=%DEFAULT_MODEL%"
    ) else (
        echo [WARN] Default model file not found: %DEFAULT_MODEL%
        echo        Enter a model path manually.
    )
)
exit /b 0

:ask_calib
set "CALIB_ARG="
echo.
set /p "CALIB=Calibration JSON (Enter=calib_camera0.json if exists, '-'=none): "
if "!CALIB!"=="" (
    if exist "calib_camera0.json" set "CALIB=calib_camera0.json"
)
if not "!CALIB!"=="" if not "!CALIB!"=="-" set CALIB_ARG=--calib "!CALIB!"
exit /b 0

:ask_calib_required
call :ask_calib
if "!CALIB_ARG!"=="" (
    echo [ERROR] Calibration file is required for preview.
    exit /b 1
)
exit /b 0

:ask_condition
echo.
set /p "CONDITION=Condition label (Enter=level; e.g. level/tilt_low/tilt_mid/tilt_high): "
if "!CONDITION!"=="" set "CONDITION=level"
exit /b 0

:ask_method
echo.
set /p "METHOD=A4 method (Enter=aruco; edge/aruco/grid): "
if "!METHOD!"=="" set "METHOD=aruco"
exit /b 0

:ask_repeats
echo.
set /p "REPEATS=Repeats per point (Enter=5): "
if "!REPEATS!"=="" set "REPEATS=5"
exit /b 0

:model_missing
echo [ERROR] model path is required.
goto :end

:ensure_deps
echo [DEPS] Checking packages...
%PY_CMD% -c "import cv2, numpy, ultralytics, reportlab; assert hasattr(cv2, 'aruco')" >nul 2>&1
if not errorlevel 1 (
    echo [DEPS] OK.
    exit /b 0
)
echo [DEPS] Missing packages detected. Preparing local virtual environment...
set "VENV_DIR=%CD%\.venv_a4"
if not exist "!VENV_DIR!\Scripts\python.exe" (
    %PY_CMD% -m venv "!VENV_DIR!"
    if errorlevel 1 exit /b 1
)
set "PY_CMD="!VENV_DIR!\Scripts\python.exe""
%PY_CMD% -m pip install --upgrade pip
if errorlevel 1 exit /b 1
%PY_CMD% -m pip install opencv-contrib-python numpy ultralytics reportlab
if errorlevel 1 exit /b 1
%PY_CMD% -c "import cv2, numpy, ultralytics, reportlab; assert hasattr(cv2, 'aruco')"
exit /b %errorlevel%

:failed
echo.
echo [ERROR] Experiment runner failed. Check the message above.
echo.
pause
exit /b 1

:end
echo.
pause
exit /b 0
