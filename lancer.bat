@echo off
setlocal
cd /d "%~dp0"
title Tap Software

if not exist venv\Scripts\python.exe goto notinstalled

call venv\Scripts\activate.bat

REM open the browser a few seconds after the app starts
start "" cmd /c "timeout /t 6 /nobreak >nul & start http://localhost:5000"

echo Tap Software is running. Close this window to stop it.
echo Participant page : http://localhost:5000
echo Examiner page    : http://localhost:5000/examiner
echo.
python app_v3.py
pause
exit /b 0

:notinstalled
echo The app is not installed yet. Run "installer.bat" first.
pause
exit /b 1
