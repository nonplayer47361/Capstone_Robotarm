@echo off
title Berry YOLO Labeling Tool - GUI
setlocal
cd /d "%~dp0"

echo ================================================
echo   Berry YOLO Labeling Tool  v2.1  [GUI mode]
echo   Select image folder in the GUI window.
echo   Do NOT close this console window.
echo ================================================
echo.

:: ── Check Python ─────────────────────────────────
set "PYTHON_CMD=python"
python --version >nul 2>&1
if errorlevel 1 (
    set "PYTHON_CMD=py -3"
    py -3 --version >nul 2>&1
    if errorlevel 1 (
        echo [ERROR] Python 3 is not installed or not in PATH.
        echo Please install Python 3.10 or newer:
        echo   https://www.python.org/downloads/
        pause
        exit /b 1
    )
)

echo [1/3] Python version:
%PYTHON_CMD% --version

:: ── Check pip ────────────────────────────────────
%PYTHON_CMD% -m pip --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] pip is not available. Reinstall Python with pip enabled.
    pause
    exit /b 1
)

:: ── Install packages ─────────────────────────────
echo [2/3] Checking required packages...
%PYTHON_CMD% -c "import cv2, PIL, pillow_heif" >nul 2>&1
if errorlevel 1 (
    echo [DEPS] Missing packages detected. Installing from requirements.txt...
    %PYTHON_CMD% -m pip install -q -r requirements.txt
    if errorlevel 1 (
        echo.
        echo [ERROR] Package installation failed.
        echo Check your internet connection, or run manually:
        echo   pip install -r requirements.txt
        pause
        exit /b 1
    )
) else (
    echo [DEPS] Required packages already available.
)

:: ── Launch GUI ───────────────────────────────────
echo [3/3] Launching GUI...
echo (Keep this console window open until you are done)
echo.
%PYTHON_CMD% labeling_gui.py %*
if errorlevel 1 (
    echo.
    echo [ERROR] GUI exited with an error.
)
pause
