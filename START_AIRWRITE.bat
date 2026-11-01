@echo off
setlocal EnableExtensions
cd /d "%~dp0"
title AIRWRITE - Local Website and Desktop Launcher
set "PYTHON_COMMAND="
set "SETUP_VERSION=AIRWRITE_2026_10_31_SPLIT_VIEW_NO_HAND_CAL_V8"

echo.
echo ==============================================================
echo  AIRWRITE  ^|  ONE-CLICK STARTUP
echo  Website + private registration database + b.py Smart Board
echo ==============================================================
echo.

REM Prefer a real 64-bit Python 3.12 runtime already installed on Windows.
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

REM If the Python Install Manager is available, ask it to install 3.12.
where py >nul 2>&1
if not errorlevel 1 (
  echo Python 3.12 was not found. Asking Python Install Manager to install it...
  py install 3.12
  py -3.12 -c "import sys,struct; raise SystemExit(0 if sys.version_info[:2] == (3,12) and struct.calcsize('P') == 8 else 1)" >nul 2>&1
  if not errorlevel 1 set "PYTHON_COMMAND=py -3.12"
)
if defined PYTHON_COMMAND goto PYTHON_READY

goto PYTHON_MISSING

:PYTHON_READY
echo [OK] A 64-bit Python 3.12 runtime is available.
if not exist ".venv\Scripts\python.exe" goto CREATE_VENV
".venv\Scripts\python.exe" -c "import sys; raise SystemExit(0 if sys.version_info[:2] == (3,12) else 1)" >nul 2>&1
if not errorlevel 1 goto VENV_READY
echo Recreating an incompatible AIRWRITE environment...
rmdir /s /q ".venv"

:CREATE_VENV
echo Creating AIRWRITE's private Python environment...
%PYTHON_COMMAND% -m venv ".venv"
if errorlevel 1 goto VENV_FAILED

:VENV_READY
set "INSTALLED_VERSION="
if exist ".venv\.airwrite_setup_complete" set /p INSTALLED_VERSION=<".venv\.airwrite_setup_complete"
if "%INSTALLED_VERSION%"=="%SETUP_VERSION%" goto START_PORTAL

echo.
echo Installing / updating AIRWRITE dependencies. This runs on first launch or after a package update.
".venv\Scripts\python.exe" -m pip install --disable-pip-version-check --upgrade pip
if errorlevel 1 goto DEPENDENCY_FAILED
".venv\Scripts\python.exe" -m pip install --disable-pip-version-check -r "requirements.txt"
if errorlevel 1 goto DEPENDENCY_FAILED
>".venv\.airwrite_setup_complete" echo %SETUP_VERSION%

goto START_PORTAL

:PYTHON_MISSING
echo.
echo [ERROR] Python 3.12 64-bit was not found or could not be installed automatically.
echo Install Python 3.12 for Windows (64-bit) from https://www.python.org/downloads/.
echo During setup, enable the Python Launcher and Tcl/Tk options.
echo Then double-click START_AIRWRITE.bat again.
start "" "https://www.python.org/downloads/"
pause
exit /b 1

:VENV_FAILED
echo [ERROR] Could not create AIRWRITE's Python environment.
echo Check the Python 3.12 64-bit installation and available disk space.
pause
exit /b 1

:DEPENDENCY_FAILED
echo.
echo [ERROR] Dependency installation did not finish.
echo Check internet access, then run REPAIR_AIRWRITE.bat.
pause
exit /b 1

:START_PORTAL
echo.
echo [OK] Starting AIRWRITE local portal...
echo Your browser will open automatically.
echo Keep this black window open while using the website and desktop app.
echo Press Ctrl+C here when you want to stop the website service.
echo.
".venv\Scripts\python.exe" "server.py"
if errorlevel 1 (
  echo.
  echo [ERROR] AIRWRITE portal stopped. Read the message above.
  pause
)
endlocal
