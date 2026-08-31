"""The one door into EBC Analyzer, and the only thing that knows it might be an .exe.

Almost nothing here happens in one process.  The folder dialog runs on its own so a Tk
that misbehaves cannot take the web server down with it; each video stage runs on its own
so three recordings can decode at once.  From source every one of those is a Python
handed a script:

    python ebc_stimulus.py studies/thomas.json cs_us_1

Frozen into an .exe there is no Python to hand it to and no script on disk to hand over.
sys.executable is this app, so each of those processes becomes a re-launch of the app -
which then has to be told which of its jobs it has been started to do.  This is where it
is told:

    EBC Analyzer.exe                           the web app - what a double-click does
    EBC Analyzer.exe --pick-folder [dir]       the folder dialog
    EBC Analyzer.exe --stage <script> [args]   one pipeline stage

The stage scripts travel inside the executable and run from there unchanged, as
__main__, exactly as the command line above runs them.  Nothing in the pipeline knows
whether it was started by a Python or by the app, which is the point: one set of scripts,
proven from the command line, is what the .exe actually runs.

Run from source this file is simply the app:  python ebc_launch.py  ==  python ebc_app.py
"""
import os
import sys

# Frozen, the bundled files are unpacked somewhere temporary and _MEIPASS says where.
BASE = getattr(sys, "_MEIPASS", None) or os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE)

STAGE, PICK = "--stage", "--pick-folder"


def pick_folder():
    """The native folder dialog, as its own short-lived process.

    It reports back the way the app expects: the chosen path on stdout, nothing at all if
    the dialog was cancelled, and exit 2 with "no-tk" on stderr if this Python was built
    without tkinter - which the app turns into "type the path in instead" rather than a
    dead button.
    """
    try:
        import tkinter
        from tkinter import filedialog
    except Exception as e:                                             # noqa: BLE001
        sys.stderr.write("no-tk: %s" % e)
        return 2
    initial = sys.argv[2] if len(sys.argv) > 2 else ""
    root = tkinter.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    kw = {"title": "Choose the folder that holds your recordings", "mustexist": True}
    if initial:
        kw["initialdir"] = initial
    path = filedialog.askdirectory(**kw)
    root.destroy()
    sys.stdout.write(path or "")
    return 0


def run_stage():
    """Run one pipeline script as though it had been started by name from a shell."""
    import runpy
    if len(sys.argv) < 3:
        sys.exit("--stage needs the name of a pipeline script, e.g. --stage ebc_score.py")
    script = os.path.basename(sys.argv[2])          # a name, never a path from outside
    path = os.path.join(BASE, script)
    if not os.path.isfile(path):
        sys.exit("!! %s is not part of this build (looked in %s)" % (script, BASE))
    sys.argv = [path] + sys.argv[3:]
    runpy.run_path(path, run_name="__main__")
    return 0


def use_bundled_ffmpeg():
    """Put the ffmpeg that travels with the app ahead of anything installed on PATH.

    The packaged app carries ffmpeg.exe and ffprobe.exe so that someone who is handed it
    has nothing at all to install.  Doing it by prepending to PATH, rather than by
    teaching every call site a second way to find them, means the rest of the code -
    ebc_video calling "ffmpeg", check_env calling shutil.which - keeps working unchanged
    and keeps reporting the truth.  The environment is inherited, so each stage process
    gets the same ffmpeg without being told.

    Run from source there is nothing bundled and this does nothing: PATH decides, as it
    always has.
    """
    if os.path.isfile(os.path.join(BASE, "ffmpeg.exe")):
        os.environ["PATH"] = BASE + os.pathsep + os.environ.get("PATH", "")


def main():
    # A frozen app that starts processes has to say so before it starts any, or on
    # Windows each child re-runs the app instead of the work.
    import multiprocessing
    multiprocessing.freeze_support()
    use_bundled_ffmpeg()

    job = sys.argv[1] if len(sys.argv) > 1 else ""
    if job == PICK:
        return pick_folder()
    if job == STAGE:
        return run_stage()
    import ebc_app
    return ebc_app.main()


def helper_cmd(job, *args):
    """The command line for one of the app's own helper processes.

    Imported by ebc_app and ebc_run_all so that the difference between "a Python and a
    script" and "the .exe again, with a switch" is written down once.
    """
    args = [str(a) for a in args]
    if getattr(sys, "frozen", False):
        return [sys.executable, job, *args]
    if job == STAGE:
        # From source a stage is only ever its own script, run by name - the same command
        # line the README documents, so the app runs what a person would have typed.
        return [sys.executable, os.path.join(BASE, args[0]), *args[1:]]
    return [sys.executable, os.path.join(BASE, "ebc_launch.py"), job, *args]


if __name__ == "__main__":
    sys.exit(main() or 0)
