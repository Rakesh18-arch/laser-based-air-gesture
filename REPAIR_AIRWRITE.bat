@echo off
setlocal
cd /d "%~dp0"
echo Repairing AIRWRITE Python dependencies...
if not exist ".venv\Scripts\python.exe" (
  echo No AIRWRITE environment found. Run START_AIRWRITE.bat first.
  pause
  exit /b 1
)
".venv\Scripts\python.exe" -m pip install --disable-pip-version-check --upgrade -r "requirements.txt"
if errorlevel 1 (
  echo Repair failed. Check the internet connection and the error above.
  pause
  exit /b 1
)
>".venv\.airwrite_setup_complete" echo AIRWRITE_2026_10_31_HAND_STUDIO_RELEASE_V6
echo AIRWRITE dependencies are repaired.
pause
endlocal
