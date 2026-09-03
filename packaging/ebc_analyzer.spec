# PyInstaller recipe for EBC Analyzer.  Build it with packaging/build.bat, or:
#
#     py -3 -m PyInstaller --clean --noconfirm packaging/ebc_analyzer.spec
#
# What comes out is dist/EBC Analyzer/, a folder holding "EBC Analyzer.exe" and
# everything it needs.  A folder rather than one file on purpose: mediapipe and OpenCV
# together are most of a gigabyte, and a one-file build unpacks all of it to a temporary
# folder on every launch - once per pipeline stage, three stages at a time.  Cold start
# goes from two seconds to the better part of a minute, and each stage pays it again.
#
# Two things about this app decide the rest of the file:
#
#   1. The pipeline runs as processes, not imports.  Nothing imports ebc_eyes; something
#      *runs* it.  So the .py files travel as data and ebc_launch runs them by name, and
#      every module they import has to be named in hiddenimports because no import
#      statement in the frozen graph points at them.
#   2. ffmpeg travels with the app.  It is ~206 MB, most of what makes this large, and it
#      is carried anyway: the point of the app is that whoever is handed it has nothing
#      to install, and "install ffmpeg, then add it to PATH" is exactly the wall that
#      stops.  It is taken from whichever ffmpeg is on PATH on the build machine, and
#      ebc_launch puts it back in front of PATH at run time.
import os
import shutil

from PyInstaller.utils.hooks import collect_all

ROOT = os.path.abspath(os.path.join(SPECPATH, ".."))                    # noqa: F821

# Stage scripts are run, never imported: they ship as source and so do their imports.
STAGES = ["ebc_run_all.py", "ebc_locate.py", "ebc_stimulus.py", "ebc_triage.py",
          "ebc_protocol.py", "ebc_eyes.py", "ebc_score.py", "ebc_figures.py",
          "ebc_export_csv.py", "ebc_workbooks.py", "ebc_qc.py",
          "ebc_app.py", "ebc_config.py", "ebc_paths.py", "ebc_signal.py", "ebc_video.py"]

datas = [(os.path.join(ROOT, s), ".") for s in STAGES]
datas += [(os.path.join(ROOT, "ebc_app_ui.html"), "."),
          (os.path.join(ROOT, "assets"), "assets")]

hiddenimports = ["cv2", "mediapipe", "numpy", "scipy", "scipy.signal", "scipy.stats",
                 "scipy.ndimage", "scipy.interpolate", "matplotlib", "matplotlib.pyplot",
                 "openpyxl", "PIL", "PIL.Image", "tkinter", "tkinter.filedialog",
                 # the app's own window, and the Edge WebView2 backend that draws it
                 "webview", "webview.platforms.edgechromium", "clr", "clr_loader"]
binaries = []


def ffmpeg_files():
    """ffmpeg.exe, ffprobe.exe and the DLLs they need, from the build machine's PATH.

    ffplay and its SDL are left behind - this app never plays anything back, and they are
    ~20 MB of nothing.  A static ffmpeg build has no DLLs to find and simply yields the
    two executables.
    """
    exe = shutil.which("ffmpeg")
    if not exe:
        raise SystemExit(
            "\n  ffmpeg is not on PATH, so it cannot be packaged into the app.\n"
            "  Install a build from https://www.gyan.dev/ffmpeg/builds/, add its bin\n"
            "  folder to PATH, open a new terminal and build again.\n")
    d = os.path.dirname(exe)
    out = []
    for f in os.listdir(d):
        low = f.lower()
        if "ffplay" in low or low.startswith("sdl"):
            continue
        if low in ("ffmpeg.exe", "ffprobe.exe") or low.endswith(".dll"):
            out.append((os.path.join(d, f), "."))
    return out


binaries += ffmpeg_files()

# mediapipe carries its face-landmark graphs and .tflite weights as data files next to
# the package.  Without them the eyelid pass imports cleanly and then finds no face.
for pkg in ("mediapipe", "cv2", "matplotlib", "scipy",
            # pywebview ships JavaScript it injects into the page, and pythonnet ships
            # the .NET runtime config and Python.Runtime.dll that reach WebView2.  None
            # of it is reachable by following imports.
            "webview", "clr_loader", "pythonnet"):
    d, b, h = collect_all(pkg)
    datas += d
    binaries += b
    hiddenimports += h

a = Analysis(                                                           # noqa: F821
    [os.path.join(ROOT, "ebc_launch.py")],
    pathex=[ROOT],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=["PyQt5", "PyQt6", "PySide2", "PySide6", "IPython", "jupyter", "notebook",
              "pytest", "pandas.tests", "matplotlib.backends.backend_qt5agg"],
    noarchive=False,
)
pyz = PYZ(a.pure)                                                       # noqa: F821

exe = EXE(                                                              # noqa: F821
    pyz, a.scripts, [],
    exclude_binaries=True,
    name="EBC Analyzer",
    icon=os.path.join(ROOT, "assets", "ebc.ico"),
    # A console, deliberately.  It is where a stopped run prints why, and the window the
    # app tells people to leave open and to close to stop it.
    console=True,
    debug=False,
    strip=False,
    upx=False,
)
coll = COLLECT(                                                         # noqa: F821
    exe, a.binaries, a.datas,
    strip=False, upx=False,
    name="EBC Analyzer",
)
