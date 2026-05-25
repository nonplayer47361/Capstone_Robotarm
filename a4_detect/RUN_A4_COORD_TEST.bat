@echo off
title A4 Coordinate Test
setlocal EnableDelayedExpansion
cd /d "%~dp0"

chcp 65001 >nul
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"

echo ============================================================
echo   A4 평면 좌표계 실험 도구
echo ============================================================
echo.
echo  [1] Step 0  — 실험 시트 생성
echo  [2] Step 1  — 단일 방법 실시간 테스트 (--live)
echo  [3] Step 2  — 모든 방법 동시 비교    (--compare)
echo  [4] Step 3  — 이미지 파일 벤치마크  (--benchmark)
echo  [5] Step 4  — A4 + YOLO 실시간 검증 (--validate)
echo  [6] Step 5a — 선행 테스트: A4 전체  (--precheck --all-methods)
echo  [7] Step 5b — 선행 테스트: YOLO     (--precheck --precheck-target object)
echo  [8] Step 6  — 좌표 오차 측정 실험  (--eval)
echo  [9] Step 7  — CSV → 오차 리포트    (--report)
echo  [0] 직접 인자 입력
echo.
set /p "MENU=메뉴 선택 [0-9]: "
set "MENU=%MENU: =%"

:: Python 경로 확인
set "PY=python"
python --version >nul 2>&1
if errorlevel 1 set "PY=py -3"

call :ensure_deps
if errorlevel 1 goto :failed

if "%MENU%"=="1" goto :gen_sheets
if "%MENU%"=="2" goto :live
if "%MENU%"=="3" goto :compare
if "%MENU%"=="4" goto :benchmark
if "%MENU%"=="5" goto :validate
if "%MENU%"=="6" goto :precheck_a4
if "%MENU%"=="7" goto :precheck_obj
if "%MENU%"=="8" goto :eval
if "%MENU%"=="9" goto :report
if "%MENU%"=="0" goto :custom
echo [ERROR] 잘못된 선택: %MENU%
goto :end

:: ── 메뉴 분기 ────────────────────────────────────────────────────────────────

:gen_sheets
echo.
echo  생성 옵션:
echo    [1] 전체 시트
echo    [2] eval 30점 시트만
echo    [3] ArUco 캘리브레이션 변형 시트
echo.
set /p "GS=선택 [1/2/3]: "
if "%GS%"=="1" (
    %PY% a4_plane_research.py --gen-sheets --one-point --calib-sheet
) else if "%GS%"=="2" (
    %PY% a4_plane_research.py --gen-sheets --only eval
) else if "%GS%"=="3" (
    %PY% a4_plane_research.py --gen-sheets --calib-variants
) else (
    echo [ERROR] 잘못된 선택
    goto :end
)
goto :end

:live
echo.
set /p "METHOD=탐지 방법 [aruco/checkerboard/color_dot/edge/grid/composite] (기본: aruco): "
if "%METHOD%"=="" set "METHOD=aruco"
%PY% a4_plane_research.py --live --method %METHOD%
goto :end

:compare
%PY% a4_plane_research.py --compare
goto :end

:benchmark
echo.
set /p "IMG_DIR=이미지 디렉터리 경로 (기본: ./test_images): "
if "%IMG_DIR%"=="" set "IMG_DIR=./test_images"
%PY% a4_plane_research.py --benchmark --images "%IMG_DIR%"
goto :end

:validate
echo.
set /p "MODEL=YOLO 모델 경로 (.pt): "
if "%MODEL%"=="" (
    echo [ERROR] --validate 에는 모델 경로가 필요합니다.
    goto :end
)
%PY% a4_plane_research.py --validate --method aruco --model "%MODEL%"
goto :end

:precheck_a4
%PY% a4_plane_research.py --precheck --precheck-target a4 --all-methods
goto :end

:precheck_obj
echo.
set /p "MODEL=YOLO 모델 경로 (.pt): "
if "%MODEL%"=="" (
    echo [ERROR] --precheck-target object 에는 모델 경로가 필요합니다.
    goto :end
)
%PY% a4_plane_research.py --precheck --precheck-target object --model "%MODEL%"
goto :end

:eval
echo.
set /p "MODEL=YOLO 모델 경로 (.pt): "
if "%MODEL%"=="" (
    echo [ERROR] --eval 에는 모델 경로가 필요합니다.
    goto :end
)
set /p "OBJECT_TYPE=객체 종류 (기본: cap): "
if "%OBJECT_TYPE%"=="" set "OBJECT_TYPE=cap"
%PY% a4_plane_research.py --eval --method aruco --model "%MODEL%" --object-type %OBJECT_TYPE%
goto :end

:report
echo.
set /p "CSV_PATH=CSV 파일 경로: "
if "%CSV_PATH%"=="" (
    echo [ERROR] --report 에는 CSV 경로가 필요합니다.
    goto :end
)
%PY% a4_plane_research.py --report --csv "%CSV_PATH%"
goto :end

:custom
echo.
echo  예: --live --method aruco
echo       --eval --model best.pt --object-type cap
echo.
set /p "ARGS=인자 입력: "
%PY% a4_plane_research.py %ARGS%
goto :end

:: ── 공통 ─────────────────────────────────────────────────────────────────────

:failed
echo.
echo [ERROR] 실행 실패. 위 오류 메시지를 확인하세요.
echo.
pause
exit /b 1

:end
echo.
pause
exit /b 0


:: ── 의존성 확인 ──────────────────────────────────────────────────────────────
:ensure_deps
echo [DEPS] Python 패키지 확인 중...
%PY% -c "import cv2, numpy, ultralytics, reportlab" >nul 2>&1
if not errorlevel 1 (
    echo [DEPS] 필수 패키지 확인 완료.
    exit /b 0
)
echo [DEPS] 누락된 패키지 감지 — 가상 환경 준비 중...
set "VENV_DIR=%CD%\.venv_a4"
if not exist "!VENV_DIR!\Scripts\python.exe" (
    %PY% -m venv "!VENV_DIR!"
    if errorlevel 1 exit /b 1
)
set "PY=!VENV_DIR!\Scripts\python.exe"
"!PY!" -m pip install --upgrade pip
if errorlevel 1 exit /b 1

:: a4_detect 용 최소 패키지 목록
"!PY!" -m pip install opencv-python numpy ultralytics reportlab
if errorlevel 1 exit /b 1

"!PY!" -c "import cv2, numpy, ultralytics, reportlab"
exit /b %errorlevel%
