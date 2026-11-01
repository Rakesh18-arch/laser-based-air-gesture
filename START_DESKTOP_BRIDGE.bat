@echo off
setlocal EnableExtensions
cd /d "%~dp0"
title AIRWRITE - Local Desktop Bridge
set "PYTHON_COMMAND="
set "SETUP_VERSION=AIRWRITE_2026_10_31_SPLIT_VIEW_NO_HAND_CAL_V8"

echo.
echo ==============================================================
echo  AIRWRITE  ^|  LOCAL DESKTOP BRIDGE
echo  Allows the hosted website to launch this PC's Windows app
echo ==============================================================
echo.

where py >nul 2>&1
if not errorlevel 1 (
  py -3.12 -c "import sys,struct; raise SystemExit(0 if sys.version_info[:2] == (3,12) and struct.calcsize('P') == 8 else 1)" >nul 2>&1
  if not errorlevel 1 set "PYTHON_COMMAND=py -3.12"
)
if defined PYTHON_COMMAND goto PYTHON_READY
where python >nul 2>&1
if not errorlevel 1 (
  python -c "import sys,struct; raise SystemExit(0 if sys.version_info[:2] == (3,12) and struct.calcsize('P') == 8 else 1)" >nul 2>&1
  if not errorlevel 1 set "PYTHON_COMMAND=python"
)
if defined PYTHON_COMMAND goto PYTHON_READY

echo [ERROR] Python 3.12 64-bit is required. Run START_AIRWRITE.bat once to install the environment.
pause
exit /b 1

:PYTHON_READY
if not exist ".venv\Scripts\python.exe" (
  %PYTHON_COMMAND% -m venv ".venv"
  if errorlevel 1 goto FAILED
)
set "INSTALLED_VERSION="
if exist ".venv\.airwrite_setup_complete" set /p INSTALLED_VERSION=<".venv\.airwrite_setup_complete"
if not "%INSTALLED_VERSION%"=="%SETUP_VERSION%" (
  ".venv\Scripts\python.exe" -m pip install --disable-pip-version-check --upgrade pip
  if errorlevel 1 goto FAILED
  ".venv\Scripts\python.exe" -m pip install --disable-pip-version-check -r "requirements.txt"
  if errorlevel 1 goto FAILED
  >".venv\.airwrite_setup_complete" echo %SETUP_VERSION%
)
set "AIRWRITE_NO_BROWSER=1"
set "AIRWRITE_ALLOWED_ORIGINS=https://laserhandgesture.netlify.app"
echo [OK] Starting local bridge on http://127.0.0.1:8765
echo Keep this window open while using the hosted AIRWRITE website.
echo If the browser asks, allow local-network access for laserhandgesture.netlify.app.
echo Press Ctrl+C to stop the bridge.
echo.
".venv\Scripts\python.exe" "server.py"
if errorlevel 1 goto FAILED
endlocal
exit /b 0

:FAILED
echo.
echo [ERROR] The AIRWRITE desktop bridge could not start. Check the error above.
pause
endlocal
exit /b 1
