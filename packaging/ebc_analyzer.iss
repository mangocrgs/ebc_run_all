; Installer for EBC Analyzer.  Built by packaging/build.bat, or:
;
;     "%LOCALAPPDATA%\Programs\Inno Setup 6\ISCC.exe" packaging\ebc_analyzer.iss
;
; Turns the built app folder into one file to send someone.  What comes out is
;
;     packaging\Setup EBC Analyzer 1.0.exe
;
; which installs to the user's own AppData - no administrator, no UAC prompt, nothing
; to approve for a researcher installing on a lab machine they do not own.  It carries
; Python, OpenCV, MediaPipe, SciPy, matplotlib and ffmpeg inside it, so a machine that
; has never had any of them runs it.
;
; Build the app first: this packs what PyInstaller left in the staging folder below, it
; does not create it.

#define AppName    "EBC Analyzer"
#define AppVersion "1.0"
#define Publisher  "Cerebral Dynamics, Plasticity & Learning"
#define ExeName    "EBC Analyzer.exe"

; Where build.bat leaves the built app.  Deliberately NOT the folder this installs into -
; if the two were the same, a build would be overwriting the copy people have installed.
#define SourceDir  SourcePath + "..\build\dist\EBC Analyzer"

[Setup]
AppId={{7C4B1E2A-9D3F-4A61-B5C8-EBC0A1D2F3E4}
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppName} {#AppVersion}
AppPublisher={#Publisher}
VersionInfoVersion={#AppVersion}

; Per-user install: no administrator rights, so this works on a locked-down lab machine.
PrivilegesRequired=lowest
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
DisableDirPage=auto

OutputDir={#SourcePath}
OutputBaseFilename=Setup {#AppName} {#AppVersion}
SetupIconFile={#SourcePath}\..\assets\ebc.ico
UninstallDisplayIcon={app}\{#ExeName}
WizardStyle=modern

; ~885 MB of mostly-already-compressed libraries.  LZMA at max wins enough to be worth
; the minutes it costs, since this is built rarely and downloaded often.
Compression=lzma2/max
SolidCompression=yes

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"
Name: "french";  MessagesFile: "compiler:Languages\French.isl"

[Files]
Source: "{#SourceDir}\*"; DestDir: "{app}"; Flags: recursesubdirs createallsubdirs ignoreversion

[Icons]
Name: "{group}\{#AppName}";           Filename: "{app}\{#ExeName}"; Comment: "Eyeblink conditioning, scored from video"
Name: "{group}\Uninstall {#AppName}"; Filename: "{uninstallexe}"
Name: "{autodesktop}\{#AppName}";     Filename: "{app}\{#ExeName}"; Tasks: desktopicon; Comment: "Eyeblink conditioning, scored from video"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"

[Run]
Filename: "{app}\{#ExeName}"; Description: "{cm:LaunchProgram,{#AppName}}"; Flags: nowait postinstall skipifsilent
