@echo off
cd /d "%~dp0"
python air_writer_app.py
if errorlevel 1 pause
