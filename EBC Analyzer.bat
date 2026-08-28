@echo off
title EBC Analyzer
cd /d "%~dp0"

rem Find a Python. Prefer the launcher, fall back to python on PATH.
set "PY="
where py >nul 2>&1 && set "PY=py -3"
if not defined PY ( where python >nul 2>&1 && set "PY=python" )

if not defined PY (
  echo.
  echo   Python was not found on this computer.
  echo   Install it from https://www.python.org/downloads/  ^(tick "Add python.exe to PATH"^),
  echo   then double-click this file again.
  echo.
  pause
  exit /b 1
)

echo.
echo   Starting EBC Analyzer...
echo   A browser tab will open. Leave THIS window open while it works.
echo   Close this window to stop the app.
echo.

%PY% ebc_app.py
if errorlevel 1 (
  echo.
  echo   The app stopped with an error. The message above says why.
  echo   A missing package is the usual cause - install them with:
  echo       %PY% -m pip install opencv-python mediapipe numpy scipy matplotlib openpyxl pillow
  echo   ffmpeg must also be on PATH.
  echo.
  pause
)
