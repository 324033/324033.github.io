@echo off
title MediaPipe Object Recognition Web App
cd /d "%~dp0"
echo ========================================================
echo   Starting MediaPipe Object Recognition Web App
echo   URL: http://127.0.0.1:5000
echo ========================================================
echo.
if exist ".venv\Scripts\python.exe" (
    ".venv\Scripts\python.exe" app.py
) else (
    python app.py
)
pause
