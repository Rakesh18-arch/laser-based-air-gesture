@echo off
cd /d "%~dp0"
echo Installing Air Writer dependencies...
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
echo.
echo Install finished. Launching Air Writer Smart Board...
python air_writer_app.py
pause
