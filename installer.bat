@echo off
setlocal
cd /d "%~dp0"
title Tap Software - Installation

REM ---------------------------------------------------------------
REM  Set this to the Python version the app was built with
REM  (the same one written in README.txt).  Example: 3.11
REM ---------------------------------------------------------------
set PYVER=3.11

echo ============================================
echo   Tap Software - Installation
echo ============================================
echo.

REM --- 1. Check that Python is installed -------------------------
py -%PYVER% --version >nul 2>&1
if errorlevel 1 goto nopython

REM --- 2. Create the private environment -------------------------
if exist venv\Scripts\python.exe goto venvok
echo Creating the environment...
py -%PYVER% -m venv venv
if errorlevel 1 goto fail
:venvok

REM --- 3. Install the exact library versions ---------------------
call venv\Scripts\activate.bat
python -m pip install --upgrade pip
if errorlevel 1 goto fail

if exist wheels\ goto offline
echo Installing libraries (internet required, this can take several minutes)...
pip install -r requirements.txt
if errorlevel 1 goto fail
goto done

:offline
echo Installing libraries from the "wheels" folder (no internet needed)...
pip install --no-index --find-links wheels -r requirements.txt
if errorlevel 1 goto fail

:done
echo.
echo ============================================
echo   Installation finished.
echo   Start the app with "lancer.bat".
echo ============================================
pause
exit /b 0

:nopython
echo Python %PYVER% was not found.
echo.
echo 1. Download Python %PYVER% from https://www.python.org/downloads/windows/
echo 2. During installation, tick "Add Python to PATH".
echo 3. Run this file again.
echo.
pause
exit /b 1

:fail
echo.
echo The installation failed. Read the messages above,
echo then see the "Troubleshooting" part of README.txt.
echo.
pause
exit /b 1
