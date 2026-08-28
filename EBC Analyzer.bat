@echo off
title EBC Analyzer
setlocal EnableDelayedExpansion

rem ---------------------------------------------------------------------------
rem Find ebc_app.py.
rem
rem It normally sits next to this file, but this launcher is meant to be copied
rem somewhere convenient - the desktop, an analysis folder - so if it is not
rem here we look for an "ebc_run_all" folder beside or above wherever this file
rem landed, up to six levels up.
rem ---------------------------------------------------------------------------
set "APP="
if exist "%~dp0ebc_app.py" set "APP=%~dp0ebc_app.py"

set "P=%~dp0"
for /L %%i in (1,1,6) do (
  if not defined APP (
    if exist "!P!ebc_run_all\ebc_app.py" set "APP=!P!ebc_run_all\ebc_app.py"
  )
  if not defined APP (
    for %%d in ("!P!..") do set "P=%%~fd\"
  )
)

if not defined APP (
  echo.
  echo   Could not find ebc_app.py.
  echo.
  echo   This launcher looked next to itself, and for an "ebc_run_all" folder
  echo   beside or above:
  echo.
  echo       %~dp0
  echo.
  echo   Move this file into the ebc_run_all folder - or anywhere alongside it -
  echo   and run it again.
  echo.
  pause
  exit /b 1
)

rem Run from the app's own folder so it finds the rest of the pipeline.
for %%f in ("!APP!") do set "APPDIR=%%~dpf"

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
echo     using !APP!
echo.
echo   A browser tab will open. Leave THIS window open while it works.
echo   Close this window to stop the app.
echo.

cd /d "!APPDIR!"
%PY% "!APP!"
if errorlevel 1 (
  echo.
  echo   The app stopped with an error. The message above says why.
  echo   A missing package is the usual cause - install them with:
  echo       %PY% -m pip install opencv-python mediapipe numpy scipy matplotlib openpyxl pillow
  echo   ffmpeg must also be on PATH.
  echo.
  pause
)
