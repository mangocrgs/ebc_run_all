@echo off
rem ===========================================================================
rem Build EBC Analyzer into one installer you can send to anybody.
rem
rem   packaging\build.bat
rem
rem What comes out is
rem
rem     packaging\Setup EBC Analyzer 1.1.exe        (~240 MB)
rem
rem which installs an app carrying Python, OpenCV, MediaPipe, SciPy, matplotlib and
rem ffmpeg inside it.  The machine it lands on needs none of those, and no
rem administrator rights: it installs into the user's own AppData.
rem
rem The app is staged in
rem
rem     <project>\build\dist\EBC Analyzer\      (~885 MB, git-ignored)
rem
rem which is deliberately not the folder the installer installs into, so a build never
rem overwrites the copy someone already has installed.
rem
rem Needs, on this machine and once:
rem     py -3 -m pip install pyinstaller opencv-python mediapipe numpy scipy matplotlib openpyxl pillow
rem     winget install JRSoftware.InnoSetup
rem plus an ffmpeg on PATH - that is the one that gets packaged.
rem ===========================================================================
setlocal
set "HERE=%~dp0"
set "ROOT=%HERE%.."
set "STAGE=%ROOT%\build"
set "APP=%STAGE%\dist\EBC Analyzer\EBC Analyzer.exe"
set "ISCC=%LOCALAPPDATA%\Programs\Inno Setup 6\ISCC.exe"
if not exist "%ISCC%" set "ISCC=%ProgramFiles(x86)%\Inno Setup 6\ISCC.exe"

where py >nul 2>&1 || (echo   Python was not found on PATH. & pause & exit /b 1)
where ffmpeg >nul 2>&1 || (echo   ffmpeg is not on PATH, so it cannot be packaged. & pause & exit /b 1)

echo.
echo   [1/3] Cutting the logo into the shapes the app uses...
py -3 "%HERE%make_assets.py" || (echo   Asset step failed. & pause & exit /b 1)

echo.
echo   [2/3] Building the app. A few minutes.
py -3 -m PyInstaller --clean --noconfirm ^
   --distpath "%STAGE%\dist" --workpath "%STAGE%\work" ^
   "%HERE%ebc_analyzer.spec" || (echo. & echo   Build failed - the log above says why. & pause & exit /b 1)
if not exist "%APP%" (echo   Build finished but %APP% is not there. & pause & exit /b 1)

if not exist "%ISCC%" (
  echo.
  echo   Inno Setup is not installed, so the installer cannot be built.
  echo       winget install JRSoftware.InnoSetup
  echo.
  echo   The app itself is ready, and can be handed over as a folder:
  echo       %STAGE%\dist\EBC Analyzer\
  pause & exit /b 1
)

echo.
echo   [3/3] Packing the installer. Several minutes - it is compressing ~885 MB.
"%ISCC%" "%HERE%ebc_analyzer.iss" || (echo. & echo   Packing failed. & pause & exit /b 1)

echo.
echo   Done.
echo.
echo     installer   %HERE%Setup EBC Analyzer 1.1.exe
echo     app folder  %STAGE%\dist\EBC Analyzer\
echo.
echo   Send the installer to anyone. They double-click it; nothing else is needed on
echo   their machine - no Python, no ffmpeg, no administrator.
echo.
echo   Windows will warn "Windows protected your PC" the first time, because the
echo   installer is not signed with a paid certificate. More info -^> Run anyway.
echo.
pause
