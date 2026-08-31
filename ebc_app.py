"""EBC Analyzer - a local browser app for the eyeblink-conditioning pipeline.

Start it and a page opens in your browser.  Press "Browse..." to pick the folder that
holds the recordings, tick the videos, say what each one is, press Run.  Progress
streams live; workbooks, figures and CSVs appear as download links.

Nothing is uploaded and nothing leaves this machine: the server listens on 127.0.0.1
only, and videos are read in place from wherever you point it - a 4 GB file is never
copied.

    python ebc_app.py                    # opens http://127.0.0.1:<port>
    python ebc_app.py --port 8765        # fixed port
    python ebc_app.py --dir "D:/EBC/Bob" # start the folder browser here
    python ebc_app.py --no-browser       # don't open a browser

The app is a front end for ebc_run_all.py and nothing more: it writes a study file from
what you ticked, runs the pipeline on it, and reads the progress back.  Anything it can
do can be done from the command line with the same study file, which it leaves in the
output folder as <study>.json.

Every failure the app can see is reported as three things - what went wrong, the detail
behind it, and what to do about it - so a stopped run never leaves you guessing.
"""
import http.server
import importlib.util
import json
import os
import re
import secrets
import shutil
import socket
import socketserver
import string
import subprocess
import sys
import threading
import time
import traceback
import urllib.parse
import webbrowser

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ebc_config as C                                    # noqa: E402
from ebc_launch import STAGE, PICK, helper_cmd            # noqa: E402
from ebc_paths import BASE                                # noqa: E402

VIDEO_EXT = {".mp4", ".mov", ".mkv", ".avi", ".m4v", ".mts", ".webm"}
TOKEN = secrets.token_urlsafe(16)

VERSION = C.VERSION
FROZEN = getattr(sys, "frozen", False)

# Remembered between sessions: the last folder browsed and a short history, so the app
# opens where the work is instead of where the scripts happen to live.
PREFS_PATH = os.path.join(os.path.expanduser("~"), ".ebc_analyzer.json")
MAX_RECENT = 8

# what fraction of one recording's work each stage is, for the per-video bars
W_STIM, W_EYES = 0.62, 0.38

# rough timings, only ever shown as "roughly": the LED read is decode-bound and so
# scales with file size; the eyelid pass is a short seek per trial and so does not.
MIN_PER_GB = 3.0
MIN_PER_RECORDING = 4.0

LOCK = threading.Lock()


def blank_state():
    return {"running": False, "phase": "idle", "videos": {}, "order": [], "log": [],
            "error": None, "started": None, "finished": None, "cancel": False,
            "out": None, "triage": None, "phase_label": "idle", "notes": []}


STATE = blank_state()
PROC = {"p": None}
PICK_LOCK = threading.Lock()
SEEN_LINES = set()          # lines already shown by the log tailer, to avoid echoes


def log(msg):
    with LOCK:
        STATE["log"].append("%s  %s" % (time.strftime("%H:%M:%S"), msg))
        del STATE["log"][:-600]


def note(text, kind="info", hint=""):
    """A message the page shows as a banner - not an error, but worth reading."""
    with LOCK:
        STATE["notes"].append({"text": text, "kind": kind, "hint": hint})
        del STATE["notes"][:-12]


def fail(error, detail="", hint=""):
    """The shape every failure takes: what went wrong, the detail, what to do."""
    return {"error": error, "detail": detail, "hint": hint}


def set_video(tag, **kw):
    with LOCK:
        STATE["videos"].setdefault(tag, {"pct": 0.0, "stage": "", "detail": "",
                                         "done": False, "busy": False, "warn": ""})
        STATE["videos"][tag].update(kw)


# --------------------------------------------------------------------------
# preferences: where the user was last working
# --------------------------------------------------------------------------
def load_prefs():
    try:
        with open(PREFS_PATH, encoding="utf-8") as fh:
            p = json.load(fh)
        if not isinstance(p, dict):
            raise ValueError("not an object")
    except (OSError, ValueError):
        return {"recent": [], "last": None}
    p.setdefault("recent", [])
    p.setdefault("last", None)
    p["recent"] = [d for d in p["recent"] if isinstance(d, str) and os.path.isdir(d)]
    return p


def save_prefs(p):
    try:
        with open(PREFS_PATH, "w", encoding="utf-8") as fh:
            json.dump(p, fh, indent=1)
    except OSError:
        pass                      # a read-only home is not worth failing the app over


def remember(path):
    if not path or not os.path.isdir(path):
        return
    p = load_prefs()
    p["last"] = path
    p["recent"] = [path] + [d for d in p["recent"]
                            if os.path.normcase(d) != os.path.normcase(path)]
    del p["recent"][MAX_RECENT:]
    save_prefs(p)


def has_video(path):
    try:
        with os.scandir(path) as it:
            for e in it:
                if e.is_file() and os.path.splitext(e.name)[1].lower() in VIDEO_EXT:
                    return True
    except OSError:
        pass
    return False


def start_dir(explicit=None):
    """Open the folder browser somewhere useful.

    The scripts' own folder is the one place a recording is never kept, so starting
    there - as the app used to - shows an empty file list and no obvious way forward.
    """
    for cand in (explicit, load_prefs().get("last")):
        if cand and os.path.isdir(cand):
            return os.path.abspath(cand)
    here = BASE
    for _ in range(3):                       # the videos usually sit just above
        up = os.path.dirname(here)
        if up == here:
            break
        if has_video(up):
            return up
        here = up
    for cand in (os.path.join(os.path.expanduser("~"), "Videos"),
                 os.path.join(os.path.expanduser("~"), "Desktop"),
                 os.path.dirname(BASE), os.path.expanduser("~")):
        if os.path.isdir(cand):
            return os.path.abspath(cand)
    return BASE


# --------------------------------------------------------------------------
# is this machine able to run the pipeline at all?
# --------------------------------------------------------------------------
MODULES = [("cv2", "opencv-python", "reads video frames"),
           ("mediapipe", "mediapipe", "finds the face and the eyelids"),
           ("numpy", "numpy", "the numbers"),
           ("scipy", "scipy", "smooths the closure traces"),
           ("matplotlib", "matplotlib", "draws the figures"),
           ("openpyxl", "openpyxl", "writes the Excel workbooks"),
           ("PIL", "pillow", "puts the figures into the workbooks")]


def check_env():
    """Everything the pipeline needs, checked before a four-hour run finds out for us."""
    items, missing_pkgs = [], []
    for mod, pkg, why in MODULES:
        try:
            ok = importlib.util.find_spec(mod) is not None
        except (ImportError, ValueError):
            ok = False
        if not ok:
            missing_pkgs.append(pkg)
        items.append({"name": pkg, "kind": "package", "ok": ok, "why": why})
    # The packaged app carries ffmpeg and puts it in front of PATH, so "found" here can
    # mean either "installed on this computer" or "came with the app".  Worth telling
    # apart: only one of them is a thing the user could break by uninstalling something.
    bundled = False
    for exe, why in (("ffmpeg", "decodes the recordings"),
                     ("ffprobe", "reads size, frame rate and duration")):
        where = shutil.which(exe)
        own = bool(where) and os.path.dirname(os.path.abspath(where)) == BASE
        bundled = bundled or own
        items.append({"name": exe, "kind": "program", "ok": bool(where),
                      "why": why, "where": where or "", "bundled": own})
    # The packaged app carries all seven inside it, so a missing one is a broken build to
    # re-download, not something a researcher can pip install.
    pip_cmd = ""
    if missing_pkgs and not FROZEN:
        pip_cmd = '"%s" -m pip install %s' % (sys.executable, " ".join(missing_pkgs))
    ffmpeg_missing = not shutil.which("ffmpeg") or not shutil.which("ffprobe")
    ffmpeg_hint = ("ffmpeg and ffprobe must be on PATH. Install a build from "
                   "https://www.gyan.dev/ffmpeg/builds/, add its bin folder to PATH, then "
                   "close this window and start the app again so it picks up the new PATH.")
    hint = ""
    if missing_pkgs and FROZEN:
        hint = ("This copy of EBC Analyzer is incomplete - those parts should be inside "
                "it. Download the app again." + ("\n\n" + ffmpeg_hint if ffmpeg_missing else ""))
    elif missing_pkgs and ffmpeg_missing:
        hint = ("Install the missing Python packages with the command below, then install "
                "ffmpeg from https://www.gyan.dev/ffmpeg/builds/ and add its bin folder "
                "to PATH.")
    elif missing_pkgs:
        hint = "Run this in a command prompt, then press Check again:\n    " + pip_cmd
    elif ffmpeg_missing:
        hint = ffmpeg_hint
    return {"ok": not missing_pkgs and not ffmpeg_missing, "items": items,
            "pip": pip_cmd, "hint": hint, "python": sys.executable, "frozen": FROZEN,
            "version": VERSION, "ffmpeg": shutil.which("ffmpeg") or "",
            "ffmpeg_bundled": bundled,
            "python_version": "%d.%d.%d" % sys.version_info[:3]}


# --------------------------------------------------------------------------
# the native folder picker
# --------------------------------------------------------------------------
def pick_folder(initial):
    """Open a real Windows folder dialog.

    It runs out of process so a Tk that misbehaves cannot take the web server down with
    it, and so the dialog owns its own event loop.  The dialog itself lives in
    ebc_launch, which is what knows how to start one of these whether the app is running
    from source or out of an .exe.
    """
    if not PICK_LOCK.acquire(blocking=False):
        return fail("A folder dialog is already open.", "",
                    "Finish with the dialog that is already on screen - it may be behind "
                    "this browser window - then try again.")
    try:
        kw = {}
        if os.name == "nt":
            kw["creationflags"] = 0x08000000        # CREATE_NO_WINDOW
        r = subprocess.run(helper_cmd(PICK, initial or ""),
                           capture_output=True, text=True, timeout=600, **kw)
    except subprocess.TimeoutExpired:
        return fail("The folder dialog was left open too long.", "",
                    "Press Browse again and choose a folder within ten minutes, or type "
                    "the folder path into the box instead.")
    except OSError as e:
        return fail("The folder dialog could not be opened.", str(e),
                    "Type or paste the folder path into the box and press Open instead.")
    finally:
        PICK_LOCK.release()
    if r.returncode == 2 or "no-tk" in (r.stderr or ""):
        return fail("This Python has no folder dialog (tkinter is not installed).",
                    (r.stderr or "").strip(),
                    "Type or paste the folder path into the box and press Open instead. "
                    "To get the dialog back, re-install Python with the 'tcl/tk and IDLE' "
                    "option ticked.")
    if r.returncode != 0:
        return fail("The folder dialog stopped unexpectedly.",
                    (r.stderr or "").strip()[-400:],
                    "Type or paste the folder path into the box and press Open instead.")
    path = (r.stdout or "").strip()
    if not path:
        return {"cancelled": True}
    if not os.path.isdir(path):
        return fail("That folder could not be opened.", path,
                    "Choose a folder on this computer, not a shortcut or a library.")
    return {"path": os.path.abspath(path)}


# --------------------------------------------------------------------------
# reading the pipeline's own output back
# --------------------------------------------------------------------------
RE_TAG = re.compile(r"^\s*\[(\w+)\]\s*(.*)$")
RE_TRIAL = re.compile(r"trial\s+(\d+)\s*/\s*(\d+)")
RE_TRACKED = re.compile(r"(\d+)\s*/\s*(\d+)\s+trials tracked")
RE_STAGE = re.compile(r"^>>>\s+ebc_(\w+)\.py")
RE_FRAMES = re.compile(r"(\d+)\s+frames\s+([\d.]+)s")

# key -> label.  The key is what ebc_run_all prints (">>> ebc_<key>.py"), and the page
# weights its overall progress bar by it, so these must stay in step with STAGES there.
PHASE = {"locate": "finding the stimulator box", "stimulus": "reading the LEDs",
         "triage": "checking the LED signal quality",
         "protocol": "building trials", "eyes": "tracking eyelids",
         "score": "scoring", "figures": "drawing figures",
         "export_csv": "writing tables", "workbooks": "building workbooks",
         "qc": "quality-check pages",
         "starting": "starting", "checking": "checking this computer",
         "done": "done", "error": "stopped", "idle": "idle"}

# What the person should be doing while each stage runs.  Shown under the progress bar.
PHASE_ADVICE = {
    "checking": "Making sure ffmpeg and the Python packages are here before anything "
                "long starts.",
    "starting": "Writing the study file and starting the pipeline.",
    "locate": "One quick survey pass per recording. Under a minute each.",
    "stimulus": "The slow part - roughly 3 minutes per GB of video. Safe to leave this "
                "running and come back.",
    "triage": "Judging each CS channel. Any warning appears here in a moment.",
    "protocol": "Turning pulses into trials and checking the block structure.",
    "eyes": "A short seek per trial - roughly 4 minutes per recording.",
    "score": "Seconds.",
    "figures": "Seconds.",
    "export_csv": "Seconds.",
    "workbooks": "Seconds.",
    "qc": "Drawing the LED check pages - open these first when it finishes.",
    "done": "Open every qc_leds_<name>.png before trusting the numbers.",
    "error": "Read the message above, then the log. Nothing you already had was "
             "overwritten.",
    "idle": "",
}


def load_triage():
    """ebc_triage.py has just judged the CS channels; show the verdict straight away."""
    out = STATE.get("out")
    if not out:
        return
    p = os.path.join(out, "_work", "triage.json")
    try:
        with open(p, encoding="utf-8") as fh:
            t = json.load(fh)
    except (OSError, ValueError):
        return
    with LOCK:
        STATE["triage"] = t


def on_line(line):
    """Turn one line of ebc_run_all output into progress."""
    m = RE_STAGE.match(line)
    if m:
        stage = m.group(1)
        with LOCK:
            STATE["phase"] = stage
            STATE["phase_label"] = PHASE.get(stage, stage)
        if stage in ("protocol", "score"):     # triage has run by now
            load_triage()
        return
    m = RE_TAG.match(line)
    if not m:
        return
    tag, rest = m.group(1), m.group(2)

    if "pulses already read" in rest or "traces already present" in rest:
        set_video(tag, stage="already done", detail="cached from an earlier run",
                  pct=1.0, done=True, busy=False)
        return
    if "no trials" in rest:
        set_video(tag, stage="no trials", detail="nothing to track", pct=1.0,
                  done=True, busy=False,
                  warn="No trials were recovered from this recording.")
        return
    if "!!" in rest:
        set_video(tag, warn=rest.split("!!", 1)[1].strip())
        return
    if "role=" in rest:
        f = RE_FRAMES.search(rest)
        set_video(tag, stage="reading the LEDs", pct=0.02, busy=True,
                  detail=("%.0f s of video" % float(f.group(2))) if f else "")
        return
    if "survey:" in rest:
        set_video(tag, stage="reading the LEDs", pct=0.04, busy=True,
                  detail="survey done")
        return
    if "read window" in rest or "CS LED anchor" in rest:
        set_video(tag, stage="reading the LEDs", pct=0.05, busy=True, detail="")
        return
    if "pulses accepted" in rest:
        set_video(tag, stage="reading the LEDs", pct=W_STIM * 0.9, busy=True,
                  detail=rest.strip().split("  ")[0])
        return
    if rest.startswith("->"):
        set_video(tag, stage="LEDs done", pct=W_STIM, detail="", busy=False)
        return
    if "face box" in rest:
        set_video(tag, stage="tracking eyelids", pct=W_STIM, detail="face located",
                  busy=True)
        return
    t = RE_TRACKED.search(rest)
    if t:
        set_video(tag, stage="done", pct=1.0, detail="%s trials" % t.group(1),
                  done=True, busy=False)
        return
    t = RE_TRIAL.search(rest)
    if t:
        frac = int(t.group(1)) / max(int(t.group(2)), 1)
        set_video(tag, stage="tracking eyelids", pct=W_STIM + W_EYES * frac, busy=True,
                  detail="trial %s of %s" % (t.group(1), t.group(2)))


def feed(line, from_tail=False):
    """One line of pipeline output: into the log, and into the progress."""
    line = line.rstrip()
    if not line:
        return
    if line.startswith(("W0", "I0", "INFO:", "WARNING:")) or "feedback" in line:
        return
    key = line.strip()
    if from_tail:
        SEEN_LINES.add(key)
    elif key in SEEN_LINES:
        on_line(line)              # already shown live by the tailer; don't repeat it
        return
    log(line)
    on_line(line)


def tail_logs(out, stop):
    """Read the per-recording log files as they grow.

    ebc_run_all gives each recording its own log file and only echoes it once that
    recording has finished, so without this the page sits silent for ten minutes and
    then jumps.  Tailing is what makes 'trial 34 of 100' appear while it is happening.
    """
    logs = os.path.join(out, "_work", "logs")
    pos = {}

    def sweep():
        try:
            names = [n for n in os.listdir(logs) if n.endswith(".log")]
        except OSError:
            return
        for n in names:
            p = os.path.join(logs, n)
            try:
                with open(p, "rb") as fh:
                    fh.seek(pos.get(n, 0))
                    data = fh.read()
            except OSError:
                continue
            cut = data.rfind(b"\n")
            if cut < 0:
                continue                   # a half-written line; wait for the rest
            pos[n] = pos.get(n, 0) + cut + 1
            for line in data[:cut + 1].decode("utf-8", "replace").splitlines():
                feed(line, from_tail=True)

    while not stop.is_set():
        sweep()
        stop.wait(1.0)
    sweep()                                # one last look for the closing lines


def run_pipeline(cfg_path, out, force):
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
    cmd = helper_cmd(STAGE, "ebc_run_all.py", "--config", cfg_path)
    if force:
        cmd.append("--force")
    try:
        p = subprocess.Popen(cmd, cwd=BASE, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                             errors="replace", bufsize=1, env=env)
    except OSError as e:
        raise PipelineError(fail(
            "The analysis could not be started.", str(e),
            "ebc_run_all.py must sit next to ebc_app.py. Check that it is in:\n    "
            + BASE))
    PROC["p"] = p
    stop = threading.Event()
    t = threading.Thread(target=tail_logs, args=(out, stop), daemon=True)
    t.start()
    try:
        for line in p.stdout:
            feed(line)
        p.wait()
    finally:
        stop.set()
        t.join(timeout=5)
        PROC["p"] = None
    return p.returncode


class PipelineError(Exception):
    """Carries a fail() dict so the page gets the three-part message unchanged."""

    def __init__(self, f):
        super().__init__(f["error"])
        self.fail = f


# --------------------------------------------------------------------------
# checking what was asked for, before anything long starts
# --------------------------------------------------------------------------
BAD_NAME = re.compile(r'[\\/:*?"<>|]')
ONLINE_ONLY = 0x00400000 | 0x00040000        # RECALL_ON_DATA_ACCESS | RECALL_ON_OPEN


def is_online_only(path):
    """A OneDrive file that is not actually on this disk yet.

    ffmpeg opening one either stalls for minutes while it downloads or fails outright,
    so it is worth saying before the run rather than after.
    """
    try:
        att = os.stat(path).st_file_attributes
    except (OSError, AttributeError):
        return False
    return bool(att & ONLINE_ONLY)


def preflight(body):
    """Everything that can be checked in a second, checked in a second.

    Returns None when the run may proceed, or a fail() describing the first real
    problem.  Softer worries are pushed to the page as notes instead.
    """
    items = body.get("items") or []
    if not items:
        return fail("No recordings are ticked.", "",
                    "Tick at least one video in step 1, then press Run analysis.")

    env = check_env()
    if not env["ok"]:
        bad = ", ".join(i["name"] for i in env["items"] if not i["ok"])
        return fail("This computer is missing something the analysis needs: " + bad,
                    "Checked with " + env["python"], env["hint"])

    study = (body.get("study") or "").strip()
    if not study:
        return fail("The study name is empty.", "",
                    "Type a name in step 2 - it becomes part of every output file name, "
                    "so the participant's name or code is the usual choice.")
    if BAD_NAME.search(study):
        return fail("The study name has a character Windows will not put in a file name.",
                    'Not allowed:  \\ / : * ? " < > |',
                    "Use letters, digits, spaces, - and _ only.")

    folders = {os.path.normcase(os.path.dirname(i["path"])) for i in items}
    if len(folders) > 1:
        return fail("The ticked recordings are in more than one folder.",
                    "\n".join(sorted(folders)),
                    "One run covers one participant, and one participant's recordings "
                    "live in one folder. Untick the ones from the other folder and run "
                    "them separately.")

    missing = [i["label"] for i in items if not os.path.isfile(i["path"])]
    if missing:
        return fail("%d ticked recording(s) are no longer there." % len(missing),
                    ", ".join(missing),
                    "The folder changed since you opened it. Press Open to re-read the "
                    "folder, then tick them again.")

    for i in items:
        try:
            with open(i["path"], "rb") as fh:
                fh.read(1)
        except OSError as e:
            return fail("A ticked recording cannot be read: " + i["label"], str(e),
                        "The file may be open in another program, or on a drive that "
                        "has gone away. Close anything using it and try again.")

    cloud = [i["label"] for i in items if is_online_only(i["path"])]
    if cloud:
        note("%d recording(s) are stored online only and must download first: %s"
             % (len(cloud), ", ".join(cloud)), "warn",
             "This can add a long wait before any progress shows. To avoid it, "
             "right-click them in File Explorer and choose 'Always keep on this device'.")

    video_dir = os.path.dirname(items[0]["path"])
    out = (body.get("out_dir") or "").strip() or os.path.join(video_dir, "analysis_EBC")
    try:
        os.makedirs(out, exist_ok=True)
        probe = os.path.join(out, ".ebc_write_test")
        with open(probe, "w") as fh:
            fh.write("ok")
        os.remove(probe)
    except OSError as e:
        return fail("The output folder cannot be written to.", "%s\n%s" % (out, e),
                    "Choose a folder you can write to, or close anything that has it "
                    "locked - Excel holding a workbook open is the usual cause.")

    try:
        free_gb = shutil.disk_usage(out).free / 1e9
        if free_gb < 2:
            return fail("There is not enough free disk space.",
                        "%.1f GB free on the drive holding %s" % (free_gb, out),
                        "Free up at least 2 GB. The cache in _work is the largest thing "
                        "the analysis writes, and it can be deleted after a run.")
        if free_gb < 8:
            note("Only %.1f GB free on that drive." % free_gb, "warn",
                 "The run should fit, but the _work cache grows with the number of "
                 "recordings. Keep an eye on it.")
    except OSError:
        pass

    if "conditioning" not in [i.get("role") for i in items]:
        note("No conditioning chapter is ticked.", "warn",
             "The acquisition figure and the conditioning workbook come from "
             "conditioning chapters only. If that is deliberate, ignore this.")
    for i in items:
        if i.get("anchor") == "us" and i.get("role") in C.NO_US_ROLES:
            return fail("'%s' is set to take trials from the US LED, but its role "
                        "delivers no US." % i["label"], "role = " + str(i.get("role")),
                        "Set 'Trials from' back to Automatic for this recording, or "
                        "change its role if it really does contain paired trials.")
    return None


def estimate_minutes(items):
    gb = sum(i.get("size", 0) or 0 for i in items) / 1e9
    return MIN_PER_GB * gb + MIN_PER_RECORDING * len(items)


# --------------------------------------------------------------------------
# running
# --------------------------------------------------------------------------
def build_study(body):
    """Turn what was ticked in the browser into a study file the pipeline understands."""
    items = body["items"]
    study = (body.get("study") or "study").strip() or "study"
    video_dir = os.path.dirname(items[0]["path"])
    out = (body.get("out_dir") or "").strip() or os.path.join(video_dir, "analysis_EBC")

    proto = dict(C.DEFAULT_PROTOCOL)
    proto.update({k: float(v) for k, v in (body.get("nominal") or {}).items() if v})

    order = {r: 0 for r in C.ROLES}
    recs = []
    for it in items:
        role = it["role"] if it["role"] in C.ROLES else "conditioning"
        order[role] += 1
        rec = {"tag": it["tag"], "file": os.path.basename(it["path"]),
               "label": it["label"], "role": role, "order": order[role]}
        if it.get("anchor") in ("cs", "us"):
            rec["anchor"] = it["anchor"]
        recs.append(rec)

    cfg = {"study": study, "video_dir": video_dir, "out_dir": out,
           "protocol": proto, "recordings": recs}
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, "%s.json" % study)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=1)
    C.load(path)          # fail here, with a readable message, not part-way through a run
    return cfg, path


def set_error(f):
    with LOCK:
        STATE["error"] = f
        STATE["phase"] = "error"
        STATE["phase_label"] = PHASE["error"]
    log("!! " + f["error"])
    for line in str(f.get("detail") or "").splitlines():
        log("   " + line)
    for line in str(f.get("hint") or "").splitlines():
        log(">> " + line)


def runner(body):
    try:
        items = body["items"]
        SEEN_LINES.clear()
        with LOCK:
            STATE.update(running=True, phase="checking", phase_label=PHASE["checking"],
                         error=None, log=[], triage=None, notes=[],
                         started=time.time(), finished=None, cancel=False,
                         order=[i["tag"] for i in items],
                         videos={i["tag"]: {"label": i["label"], "pct": 0.0,
                                            "stage": "waiting its turn", "detail": "",
                                            "done": False, "busy": False, "warn": ""}
                                 for i in items})
        log("checking this computer before starting anything long")
        f = preflight(body)
        if f:
            return set_error(f)
        log("all good: ffmpeg, the Python packages, the files and the output folder")

        with LOCK:
            STATE["phase"] = "starting"
            STATE["phase_label"] = PHASE["starting"]
        try:
            cfg, cfg_path = build_study(body)
        except SystemExit as e:
            return set_error(fail(
                "The study file was refused by the pipeline.", str(e),
                "Fix the recording this names in step 1, then press Run again."))
        except OSError as e:
            return set_error(fail(
                "The study file could not be written.", str(e),
                "Check that the output folder is writable and the disk is not full."))
        with LOCK:
            STATE["out"] = cfg["out_dir"]
        remember(cfg["video_dir"])

        p = cfg["protocol"]
        log("study '%s'  |  CS %.0f ms, US onset %.0f ms, US %.0f ms, %d+%d x %d blocks"
            % (cfg["study"], p["cs_ms"], p["us_onset_ms"], p["us_dur_ms"],
               p["paired_per_block"], p["cs_only_per_block"], p["n_blocks"]))
        log("study file: " + cfg_path)
        log("results will appear in: " + cfg["out_dir"])
        for r in cfg["recordings"]:
            log("   %s  %s%s" % (r["label"], r["role"],
                                 "  (trials from the US LED)"
                                 if r.get("anchor") == "us" else ""))
        log("roughly %d minutes of work, minus anything already cached"
            % round(estimate_minutes(items)))

        rc = run_pipeline(cfg_path, cfg["out_dir"], body.get("force"))
        load_triage()
        if STATE["cancel"]:
            return set_error(fail(
                "Stopped at your request.", "",
                "Nothing was lost: finished recordings stay cached, so pressing Run "
                "again carries on from where this left off."))
        if rc != 0:
            tail = [ln for ln in STATE["log"][-60:]
                    if "!!" in ln or "Error" in ln or "Traceback" in ln]
            return set_error(fail(
                "The analysis stopped part-way through (exit code %d)." % rc,
                "\n".join(tail[-6:]) or "The last lines of the log say why.",
                "Read the log below from the bottom up. The stage that failed names "
                "itself on the '>>> ebc_<stage>.py' line above the error. Everything "
                "that finished is cached, so fixing the cause and pressing Run again "
                "does not redo the work that succeeded."))
        with LOCK:
            STATE["phase"] = "done"
            STATE["phase_label"] = PHASE["done"]
        log("finished - open every qc_leds_<name>.png before trusting the numbers")
    except PipelineError as e:
        set_error(e.fail)
    except Exception as e:  # noqa: BLE001
        set_error(fail("The app hit an unexpected problem.",
                       "%s: %s" % (type(e).__name__, e),
                       "This is a fault in the app rather than in your data. The detail "
                       "above, and the log below, are what is needed to fix it."))
        log(traceback.format_exc().strip())
    finally:
        with LOCK:
            STATE["running"] = False
            STATE["finished"] = time.time()


# --------------------------------------------------------------------------
# http
# --------------------------------------------------------------------------
def natkey(name):
    """Sort names the way a person reads them: CSUS 2 < CSUS3 < CSUS 4."""
    return [int(x) if x.isdigit() else re.sub(r"[\s_\-]+", "", x.lower())
            for x in re.split(r"(\d+)", name)]


def guess_role(stem):
    for pat, role in C._PATTERNS:
        if re.search(pat, stem, re.I):
            return role
    return "conditioning"


def make_tag(stem, used):
    t = C._slug(stem)
    base, i = t, 2
    while t in used:
        t = "%s%d" % (base, i)
        i += 1
    used.add(t)
    return t


def crumbs(path):
    """Every folder on the way here, so any of them is one click away."""
    out, p = [], path
    while True:
        head, tail = os.path.split(p)
        if not tail:
            out.append({"name": p, "path": p})
            break
        out.append({"name": tail, "path": p})
        if head == p:
            break
        p = head
    return list(reversed(out))


def list_dir(path):
    typed = (path or "").strip().strip('"').strip("'")
    if not typed:
        path = start_dir()
    else:
        # A path pasted out of an e-mail or a chat window arrives with line breaks and
        # stray spaces in it; a path typed by hand often arrives drive-relative.
        path = os.path.abspath(os.path.expanduser(" ".join(typed.split())))
    drives = ["%s:\\" % d for d in string.ascii_uppercase
              if os.path.exists("%s:\\" % d)] if os.name == "nt" else ["/"]
    base = {"path": path, "dirs": [], "videos": [], "drives": drives,
            "crumbs": crumbs(path), "up": None, "recent": load_prefs()["recent"],
            "skipped": []}
    # Saying only where the app ended up is unhelpful when that is not where the person
    # thought they were pointing it, so show both whenever they differ.
    where = path if not typed or os.path.normcase(typed) == os.path.normcase(path) \
        else "you typed:  %s\nwhich means: %s" % (typed, path)

    if not os.path.exists(path):
        base.update(fail(
            "There is no folder at that path.", where,
            "Check the spelling, or press Browse and pick the folder from a dialog. A "
            "path copied out of File Explorer's address bar works too. A path that does "
            "not start with a drive letter and a backslash is taken as relative to the "
            "app's own folder."))
        return base
    if not os.path.isdir(path):
        base.update(fail(
            "That is a file, not a folder.", where,
            "Point the app at the folder that contains the recordings, not at one "
            "recording."))
        return base

    try:
        entries = sorted(os.scandir(path), key=lambda e: natkey(e.name))
    except PermissionError:
        base.update(fail(
            "Windows will not let this app read that folder.", path,
            "Pick a folder under your own user account, or run the app as the user who "
            "owns this one."))
        return base
    except OSError as e:
        base.update(fail(
            "That folder could not be read.", "%s\n%s" % (path, e),
            "If it is on a network drive or an external disk, check it is still "
            "connected, then press Open again."))
        return base

    dirs, vids, skipped = [], [], []
    for e in entries:
        if e.name.startswith((".", "$")) or e.name == "__pycache__":
            continue
        try:
            if e.is_dir():
                dirs.append({"name": e.name, "path": e.path})
            elif os.path.splitext(e.name)[1].lower() in VIDEO_EXT:
                vids.append({"name": e.name, "size": e.stat().st_size,
                             "cloud": is_online_only(e.path)})
        except OSError:
            skipped.append(e.name)      # one bad entry must not lose the whole listing

    used = set()
    for v in vids:
        stem = os.path.splitext(v["name"])[0]
        v["tag"] = make_tag(stem, used)
        v["label"] = stem
        v["role"] = guess_role(stem)
        v["path"] = os.path.join(path, v["name"])   # built here, not in the browser

    # Which sub-folders are worth clicking into.  One scandir each, stopped at the first
    # video, so a folder of a hundred participants still lists instantly - but a folder
    # of thousands would not, and the marker is a convenience, so it is dropped there.
    if len(dirs) <= 200:
        for d in dirs:
            d["has"] = has_video(d["path"])

    up = os.path.dirname(path)
    base.update({"dirs": dirs, "videos": vids, "up": up if up != path else None,
                 "skipped": skipped,
                 "suggest": os.path.basename(path.rstrip("\\/")) or ""})
    if not vids:
        with_video = [d["name"] for d in dirs if d.get("has")][:6]
        base["empty_hint"] = (
            "No recordings in this folder. These sub-folders have some: "
            + ", ".join(with_video) if with_video else
            "No recordings here, and no sub-folder of it has any either. The app looks "
            "for " + ", ".join(sorted(VIDEO_EXT)) + " files.")
    if skipped:
        base["warn"] = ("%d item(s) in this folder could not be read and were left out."
                        % len(skipped))
    remember(path)
    return base


QC_GROUP = "LED check pages - open these first"


def results(out):
    groups = {QC_GROUP: [], "Workbooks": [], "Figures": [], "Data": []}
    if not out or not os.path.isdir(out):
        return groups
    try:
        names = sorted(os.listdir(out))
    except OSError:
        return groups
    for n in names:
        p = os.path.join(out, n)
        try:
            if not os.path.isfile(p):
                continue
            ext = os.path.splitext(n)[1].lower()
            g = (QC_GROUP if n.startswith("qc_leds")
                 else "Workbooks" if ext == ".xlsx"
                 else "Figures" if ext == ".png"
                 else "Data" if ext == ".csv" else None)
            if g:
                groups[g].append({"name": n, "size": os.path.getsize(p),
                                  "mtime": os.path.getmtime(p)})
        except OSError:
            continue
    return groups


def open_in_explorer(target):
    try:
        if not os.path.isdir(target):
            return fail("That folder is not there.", target,
                        "It is created when a run starts. If a run did finish, the "
                        "folder may have been moved since.")
        if os.name == "nt":
            os.startfile(target)  # noqa: S606
        elif sys.platform == "darwin":
            subprocess.Popen(["open", target])
        else:
            subprocess.Popen(["xdg-open", target])
    except Exception as e:  # noqa: BLE001
        return fail("The folder could not be opened.", str(e),
                    "Copy this path into File Explorer instead:\n    " + target)
    return {"ok": True, "path": target}


class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json; charset=utf-8", extra=None):
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode()
        elif isinstance(body, str):
            body = body.encode()
        try:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass                      # the tab was closed mid-reply; nothing to do

    def _authed(self, q):
        return (self.headers.get("X-Token") == TOKEN) or (q.get("t", [None])[0] == TOKEN)

    def _stale(self):
        return fail(
            "This page is out of date and the app no longer recognises it.", "",
            "The app makes a new key each time it starts, so a page left open from a "
            "previous run stops working. Close this tab and open the link printed in "
            "the black command window.")

    def _oops(self):
        return fail("The app hit an unexpected problem answering the page.",
                    traceback.format_exc().strip().splitlines()[-1],
                    "The black command window behind this browser has the full detail. "
                    "Reloading the page is usually enough.")

    def do_GET(self):
        try:
            self._get()
        except Exception:  # noqa: BLE001
            self._send(500, self._oops())

    def _get(self):
        u = urllib.parse.urlparse(self.path)
        q = urllib.parse.parse_qs(u.query)
        if u.path in ("/", "/index.html"):
            page = os.path.join(BASE, "ebc_app_ui.html")
            try:
                html = open(page, encoding="utf-8").read()
            except OSError as e:
                return self._send(
                    500,
                    "<h1>ebc_app_ui.html is missing</h1><p>%s</p>"
                    "<p>It has to sit next to ebc_app.py, in %s</p>" % (e, BASE),
                    "text/html; charset=utf-8")
            return self._send(200, html.replace("__TOKEN__", TOKEN),
                              "text/html; charset=utf-8")
        # The lab's logo, and the tab icon the browser asks for on its own - neither can
        # carry the token, and neither is worth protecting.
        if u.path == "/favicon.ico":
            return self._asset("ebc.ico")
        if u.path.startswith("/assets/"):
            return self._asset(u.path[len("/assets/"):])
        if u.path.startswith("/api/") and not self._authed(q):
            return self._send(403, self._stale())
        if u.path == "/api/browse":
            return self._send(200, list_dir(q.get("path", [""])[0]))
        if u.path == "/api/check_env":
            return self._send(200, check_env())
        if u.path == "/api/hello":
            return self._send(200, {"base": BASE, "start": start_dir(),
                                    "python": sys.executable,
                                    "python_version": "%d.%d.%d" % sys.version_info[:3],
                                    "version": VERSION, "frozen": FROZEN,
                                    "protocol": C.DEFAULT_PROTOCOL,
                                    "min_per_gb": MIN_PER_GB,
                                    "min_per_recording": MIN_PER_RECORDING})
        if u.path == "/api/status":
            with LOCK:
                s = json.loads(json.dumps(STATE))
            s["results"] = results(s.get("out"))
            s["advice"] = PHASE_ADVICE.get(s.get("phase"), "")
            return self._send(200, s)
        if u.path == "/api/download":
            return self._download(q)
        if u.path == "/api/open_folder":
            target = q.get("path", [""])[0] or STATE.get("out")
            if not target:
                return self._send(200, fail(
                    "There is no output folder yet.", "",
                    "It is created when a run starts, next to the recordings, and is "
                    "called analysis_EBC."))
            return self._send(200, open_in_explorer(target))
        return self._send(404, fail(
            "The page asked for something this app does not have.", u.path,
            "Reload the page. If it keeps happening, ebc_app_ui.html and ebc_app.py "
            "are from different versions - re-download both."))

    ASSET_TYPES = {".png": "image/png", ".ico": "image/x-icon", ".svg": "image/svg+xml"}

    def _asset(self, name):
        """One of the logo files from assets/, by name.

        Only a bare file name with a known image extension is ever looked up, so this
        cannot be walked out of the folder, and a missing one is a quiet 404: a logo that
        failed to load should leave a gap in the page, not an error banner over the work.
        """
        name = os.path.basename(urllib.parse.unquote(name))
        ctype = self.ASSET_TYPES.get(os.path.splitext(name)[1].lower())
        path = os.path.join(BASE, "assets", name)
        if not ctype or not os.path.isfile(path):
            return self._send(404, b"", "application/octet-stream")
        try:
            with open(path, "rb") as fh:
                body = fh.read()
        except OSError:
            return self._send(404, b"", "application/octet-stream")
        return self._send(200, body, ctype)

    def _download(self, q):
        out = STATE.get("out")
        name = os.path.basename(q.get("f", [""])[0])
        p = os.path.join(out or "", name)
        if not out or not os.path.isfile(p):
            return self._send(404, fail(
                "That file is not there any more.", p,
                "It may have been moved or deleted since the run finished. Press Open "
                "output folder to see what is actually in there."))
        try:
            size = os.path.getsize(p)
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(size))
            self.send_header("Content-Disposition", 'attachment; filename="%s"' % name)
            self.end_headers()
            with open(p, "rb") as f:
                while True:
                    chunk = f.read(1 << 20)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
        except (BrokenPipeError, ConnectionResetError):
            pass
        except OSError as e:
            return self._send(500, fail(
                "That file could not be read.", str(e),
                "It may be open in Excel. Close it and try again."))
        return None

    def do_POST(self):
        try:
            self._post()
        except Exception:  # noqa: BLE001
            self._send(500, self._oops())

    def _post(self):
        u = urllib.parse.urlparse(self.path)
        q = urllib.parse.parse_qs(u.query)
        if not self._authed(q):
            return self._send(403, self._stale())
        n = int(self.headers.get("Content-Length", 0))
        try:
            body = json.loads(self.rfile.read(n) or b"{}")
        except ValueError as e:
            return self._send(400, fail("The page sent something unreadable.", str(e),
                                        "Reload the page and try again."))
        if u.path == "/api/pick_folder":
            return self._send(200, pick_folder(body.get("initial") or start_dir()))
        if u.path == "/api/run":
            if STATE["running"]:
                return self._send(409, fail(
                    "A run is already going.", STATE.get("phase_label") or "",
                    "Wait for it to finish, or press Stop first."))
            if not body.get("items"):
                return self._send(400, fail(
                    "No recordings are ticked.", "",
                    "Tick at least one video in step 1, then press Run analysis."))
            threading.Thread(target=runner, args=(body,), daemon=True).start()
            return self._send(200, {"ok": True})
        if u.path == "/api/stop":
            with LOCK:
                STATE["cancel"] = True
            p = PROC.get("p")
            if p and p.poll() is None:
                try:
                    p.terminate()
                except OSError:
                    pass
            log("stop requested - waiting for the current step to end")
            return self._send(200, {"ok": True})
        if u.path == "/api/reset":
            if STATE["running"]:
                return self._send(409, fail("A run is still going.", "",
                                            "Press Stop first."))
            with LOCK:
                STATE.update(blank_state())
            return self._send(200, {"ok": True})
        return self._send(404, fail(
            "The page asked for something this app does not have.", u.path,
            "Reload the page."))


class Server(socketserver.ThreadingTCPServer):
    daemon_threads = True
    allow_reuse_address = True


def free_port(pref=0):
    if pref:
        return pref
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def arg(name, default=None):
    if name in sys.argv:
        i = sys.argv.index(name) + 1
        if i < len(sys.argv):
            return sys.argv[i]
    return default


def hide_console():
    """Drop the black command window once the app window is up.

    Only hidden, never freed: the console's handles stay valid, and this app hands those
    handles to every child process it starts.  Freeing the console would leave them
    invalid and break the pipeline in the same way importing mediapipe once did.

    It is hidden only after the window exists, so a failure to start still has somewhere
    to print itself.
    """
    if os.name != "nt":
        return
    try:
        import ctypes
        hwnd = ctypes.windll.kernel32.GetConsoleWindow()
        if hwnd:
            ctypes.windll.user32.ShowWindow(hwnd, 0)          # SW_HIDE
    except Exception:                                          # noqa: BLE001
        pass                       # a cosmetic step; never worth failing a run over


def open_window(url):
    """Show the app in its own window.  True if it did, False to fall back to a browser.

    This is a desktop application, so it gets a desktop window - no tab, no address bar,
    nothing to accidentally navigate away from.  The page inside is the same page the
    server has always served; only the frame around it has changed.

    Windows draws it with the Edge WebView2 runtime, which ships with Edge and is on
    every current machine.  If it is somehow not, that is not a reason to fail: the
    browser still works, and saying so beats a window that never appears.
    """
    try:
        import webview
    except ImportError:
        print("  (pywebview is not installed, so there is no app window.)")
        return False
    # Sized from the screen, and generously, because the window's pixels are not the
    # page's pixels: Windows is commonly scaled to 125%, which leaves the page four
    # fifths of the width the window was given.  pywebview reports screen size in those
    # scaled units while sizing windows in real ones, so rather than convert between
    # them the page is simply laid out narrow enough (940) to fit whatever this yields.
    w, h = 1400, 940
    try:
        s = webview.screens[0]
        w = max(1000, min(w, int(s.width * 0.86)))
        h = max(680, min(h, int(s.height * 0.88)))
    except Exception:                                          # noqa: BLE001
        pass                                   # no screen info; the defaults are sane
    try:
        webview.create_window("EBC Analyzer", url, width=w, height=h,
                              min_size=(900, 600), text_select=True)
        threading.Timer(2.0, hide_console).start()
        webview.start()
        return True
    except Exception as e:                                     # noqa: BLE001
        print("  The app window could not be opened: %s" % e)
        return False


def main():
    if arg("--dir"):
        remember(os.path.abspath(arg("--dir")))

    env = check_env()
    if not env["ok"]:
        print("\n  Before anything else - this computer is missing:\n")
        for i in env["items"]:
            if not i["ok"]:
                print("      %-16s %s" % (i["name"], i["why"]))
        print("\n  " + (env["hint"] or "").replace("\n", "\n  "))
        print("\n  The app will still open, and will say the same thing on the page.\n")

    try:
        port = int(arg("--port", 0) or 0)
    except ValueError:
        sys.exit("\n  --port needs a number, for example:"
                 "\n      python ebc_app.py --port 8765\n")
    try:
        port = free_port(port)
        srv = Server(("127.0.0.1", port), Handler)
    except OSError as e:
        sys.exit("\n  The app could not open port %d: %s\n"
                 "\n  Something else is using it - most likely another copy of this app"
                 "\n  already running. Close the other black window, or start this one"
                 "\n  on a different port:"
                 "\n      python ebc_app.py --port 8766\n" % (port, e))

    url = "http://127.0.0.1:%d/?t=%s" % (port, TOKEN)
    print("\n  EBC Analyzer %s  -  %s" % (VERSION, C.LAB))
    print("  running.\n")
    print("    %s\n" % url)
    print("  Videos are read in place - nothing is uploaded or copied.\n")

    # The server answers the window; it is not the window.  It runs behind, because the
    # window has to own the main thread - that is where a GUI event loop must live.
    threading.Thread(target=srv.serve_forever, daemon=True).start()

    if "--no-browser" in sys.argv:
        print("  Serving only. Press Ctrl+C to stop.\n")
        try:
            threading.Event().wait()
        except KeyboardInterrupt:
            print("\n  stopped.")
        return

    if not open_window(url):
        print("  Opening in your browser instead.")
        print("  If nothing opened, copy the whole line above - the ?t=... part")
        print("  included - into your browser's address bar.\n")
        print("  Leave this window open while it works; close it to stop the app.\n")
        webbrowser.open(url)
        try:
            threading.Event().wait()
        except KeyboardInterrupt:
            print("\n  stopped.")


if __name__ == "__main__":
    main()
