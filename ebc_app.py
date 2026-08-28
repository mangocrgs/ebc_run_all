"""EBC Analyzer - a local browser app for the eyeblink-conditioning pipeline.

Start it and a page opens in your browser.  Pick a folder, tick the videos, say what
each one is, press Run.  Progress streams live; workbooks, figures and CSVs appear as
download links.

Nothing is uploaded and nothing leaves this machine: the server listens on 127.0.0.1
only, and videos are read in place from wherever you point it - a 4 GB file is never
copied.

    python ebc_app.py              # opens http://127.0.0.1:<port>
    python ebc_app.py --port 8765  # fixed port
    python ebc_app.py --no-browser # don't open a browser

The app is a front end for ebc_run_all.py and nothing more: it writes a study file from
what you ticked, runs the pipeline on it, and reads the progress back.  Anything it can
do can be done from the command line with the same study file, which it leaves in the
output folder as <study>.json.
"""
import http.server
import json
import os
import re
import secrets
import socket
import socketserver
import string
import subprocess
import sys
import threading
import time
import urllib.parse
import webbrowser

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ebc_config as C                                    # noqa: E402
from ebc_paths import BASE                                # noqa: E402

VIDEO_EXT = {".mp4", ".mov", ".mkv", ".avi", ".m4v", ".mts", ".webm"}
TOKEN = secrets.token_urlsafe(16)

# what fraction of one recording's work each stage is, for the per-video bars
W_STIM, W_EYES = 0.62, 0.38

LOCK = threading.Lock()
STATE = {"running": False, "phase": "idle", "videos": {}, "order": [], "log": [],
         "error": None, "started": None, "finished": None, "cancel": False, "out": None}
PROC = {"p": None}


def log(msg):
    with LOCK:
        STATE["log"].append("%s  %s" % (time.strftime("%H:%M:%S"), msg))
        del STATE["log"][:-400]


def set_video(tag, **kw):
    with LOCK:
        STATE["videos"].setdefault(tag, {"pct": 0.0, "stage": "", "detail": "", "done": False})
        STATE["videos"][tag].update(kw)


# --------------------------------------------------------------------------
# reading the pipeline's own output back
# --------------------------------------------------------------------------
RE_TAG = re.compile(r"^\s*\[(\w+)\]\s*(.*)$")
RE_TRIAL = re.compile(r"trial\s+(\d+)\s*/\s*(\d+)")
RE_TRACKED = re.compile(r"(\d+)\s*/\s*(\d+)\s+trials tracked")
RE_STAGE = re.compile(r"^>>>\s+ebc_(\w+)\.py")

PHASE = {"locate": "finding the stimulator box", "stimulus": "reading the LEDs",
         "protocol": "building trials", "eyes": "tracking eyelids",
         "score": "scoring", "figures": "figures", "export": "tables",
         "workbooks": "workbooks", "qc": "quality-check pages"}


def on_line(line):
    """Turn one line of ebc_run_all output into progress."""
    m = RE_STAGE.match(line)
    if m:
        with LOCK:
            STATE["phase"] = PHASE.get(m.group(1), m.group(1))
        return
    m = RE_TAG.match(line)
    if not m:
        return
    tag, rest = m.group(1), m.group(2)

    if "pulses already read" in rest or "traces already present" in rest:
        set_video(tag, stage="cached", detail="already processed", pct=1.0, done=True)
        return
    if "read window" in rest or "CS LED anchor" in rest:
        set_video(tag, stage="reading the LEDs", pct=0.05, detail="")
        return
    if "pulses accepted" in rest:
        set_video(tag, stage="reading the LEDs", pct=W_STIM * 0.9,
                  detail=rest.strip().split("  ")[0])
        return
    if rest.startswith("->"):
        set_video(tag, stage="LEDs done", pct=W_STIM, detail="")
        return
    if "face box" in rest:
        set_video(tag, stage="tracking eyelids", pct=W_STIM, detail="face located")
        return
    t = RE_TRACKED.search(rest)
    if t:
        set_video(tag, stage="done", pct=1.0, detail="%s trials" % t.group(1), done=True)
        return
    t = RE_TRIAL.search(rest)
    if t:
        frac = int(t.group(1)) / max(int(t.group(2)), 1)
        set_video(tag, stage="tracking eyelids", pct=W_STIM + W_EYES * frac,
                  detail="trial %s of %s" % (t.group(1), t.group(2)))


def run_pipeline(cfg_path, force):
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
    cmd = [sys.executable, os.path.join(BASE, "ebc_run_all.py"), "--config", cfg_path]
    if force:
        cmd.append("--force")
    p = subprocess.Popen(cmd, cwd=BASE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, encoding="utf-8", errors="replace", bufsize=1, env=env)
    PROC["p"] = p
    for line in p.stdout:
        line = line.rstrip()
        if not line or line.startswith(("W0", "I0", "INFO:", "WARNING:")) or "feedback" in line:
            continue
        log(line)
        on_line(line)
    p.wait()
    PROC["p"] = None
    return p.returncode


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
        if it.get("anchor") == "us":
            rec["anchor"] = "us"
        recs.append(rec)

    cfg = {"study": study, "video_dir": video_dir, "out_dir": out,
           "protocol": proto, "recordings": recs}
    os.makedirs(out, exist_ok=True)
    path = os.path.join(out, "%s.json" % study)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(cfg, fh, indent=1)
    C.load(path)          # fail here, with a readable message, not part-way through a run
    return cfg, path


def runner(body):
    try:
        items = body["items"]
        with LOCK:
            STATE.update(running=True, phase="starting", error=None, log=[],
                         started=time.time(), finished=None, cancel=False,
                         order=[i["tag"] for i in items],
                         videos={i["tag"]: {"label": i["label"], "pct": 0.0, "stage": "queued",
                                            "detail": "", "done": False} for i in items})
        cfg, cfg_path = build_study(body)
        with LOCK:
            STATE["out"] = cfg["out_dir"]
        p = cfg["protocol"]
        log("study '%s'  |  CS %.0f ms, US onset %.0f ms, US %.0f ms, %d+%d x %d blocks"
            % (cfg["study"], p["cs_ms"], p["us_onset_ms"], p["us_dur_ms"],
               p["paired_per_block"], p["cs_only_per_block"], p["n_blocks"]))
        log("study file: " + cfg_path)
        for r in cfg["recordings"]:
            log("   %s  %s%s" % (r["label"], r["role"],
                                 "  (trials from the US LED)" if r.get("anchor") == "us" else ""))

        rc = run_pipeline(cfg_path, body.get("force"))
        if STATE["cancel"]:
            raise RuntimeError("stopped")
        if rc != 0:
            raise RuntimeError("the pipeline stopped with exit code %d - see the log above" % rc)
        with LOCK:
            STATE["phase"] = "done"
        log("finished")
    except SystemExit as e:                 # a study file the pipeline refuses, said plainly
        with LOCK:
            STATE["error"] = str(e)
            STATE["phase"] = "error"
        log("!! " + str(e))
    except Exception as e:  # noqa: BLE001
        with LOCK:
            STATE["error"] = str(e)
            STATE["phase"] = "error"
        log("!! " + str(e))
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


def list_dir(path):
    if not path:
        path = BASE
    path = os.path.abspath(path)
    dirs, vids = [], []
    try:
        for e in sorted(os.scandir(path), key=lambda e: natkey(e.name)):
            if e.name.startswith((".", "$")):
                continue
            if e.is_dir():
                dirs.append({"name": e.name, "path": e.path})
            elif os.path.splitext(e.name)[1].lower() in VIDEO_EXT:
                vids.append({"name": e.name, "size": e.stat().st_size})
    except OSError as e:
        return {"error": str(e), "path": path, "dirs": [], "videos": []}
    used = set()
    for v in vids:
        stem = os.path.splitext(v["name"])[0]
        v["tag"] = make_tag(stem, used)
        v["label"] = stem
        v["role"] = guess_role(stem)
        v["path"] = os.path.join(path, v["name"])   # built here, not in the browser
    up = os.path.dirname(path)
    drives = ["%s:\\" % d for d in string.ascii_uppercase
              if os.path.exists("%s:\\" % d)] if os.name == "nt" else ["/"]
    return {"path": path, "up": up if up != path else None, "dirs": dirs,
            "videos": vids, "drives": drives}


def results(out):
    groups = {"Workbooks": [], "Figures": [], "Data": []}
    if not out or not os.path.isdir(out):
        return groups
    for n in sorted(os.listdir(out)):
        p = os.path.join(out, n)
        if not os.path.isfile(p):
            continue
        ext = os.path.splitext(n)[1].lower()
        g = ("Workbooks" if ext == ".xlsx" else "Figures" if ext == ".png"
             else "Data" if ext == ".csv" else None)
        if g:
            groups[g].append({"name": n, "size": os.path.getsize(p),
                              "mtime": os.path.getmtime(p)})
    return groups


class Handler(http.server.BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):
        pass

    def _send(self, code, body, ctype="application/json; charset=utf-8", extra=None):
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode()
        elif isinstance(body, str):
            body = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        for k, v in (extra or {}).items():
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _authed(self, q):
        return (self.headers.get("X-Token") == TOKEN) or (q.get("t", [None])[0] == TOKEN)

    def do_GET(self):
        u = urllib.parse.urlparse(self.path)
        q = urllib.parse.parse_qs(u.query)
        if u.path in ("/", "/index.html"):
            html = open(os.path.join(BASE, "ebc_app_ui.html"), encoding="utf-8").read()
            return self._send(200, html.replace("__TOKEN__", TOKEN), "text/html; charset=utf-8")
        if u.path.startswith("/api/") and not self._authed(q):
            return self._send(403, {"error": "bad token"})
        if u.path == "/api/browse":
            return self._send(200, list_dir(q.get("path", [""])[0]))
        if u.path == "/api/status":
            with LOCK:
                s = json.loads(json.dumps(STATE))
            s["results"] = results(s.get("out"))
            return self._send(200, s)
        if u.path == "/api/download":
            out = STATE.get("out")
            name = os.path.basename(q.get("f", [""])[0])
            p = os.path.join(out or "", name)
            if not out or not os.path.isfile(p):
                return self._send(404, {"error": "not found"})
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
            return None
        if u.path == "/api/open_folder":
            out = STATE.get("out")
            try:
                if not out:
                    raise RuntimeError("nothing has been run yet")
                if os.name == "nt":
                    os.startfile(out)  # noqa: S606
                else:
                    subprocess.Popen(["xdg-open", out])
            except Exception as e:  # noqa: BLE001
                return self._send(200, {"ok": False, "error": str(e)})
            return self._send(200, {"ok": True})
        return self._send(404, {"error": "no such endpoint"})

    def do_POST(self):
        u = urllib.parse.urlparse(self.path)
        q = urllib.parse.parse_qs(u.query)
        if not self._authed(q):
            return self._send(403, {"error": "bad token"})
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n) or b"{}")
        if u.path == "/api/run":
            if STATE["running"]:
                return self._send(409, {"error": "already running"})
            if not body.get("items"):
                return self._send(400, {"error": "no videos selected"})
            threading.Thread(target=runner, args=(body,), daemon=True).start()
            return self._send(200, {"ok": True})
        if u.path == "/api/stop":
            with LOCK:
                STATE["cancel"] = True
            p = PROC.get("p")
            if p and p.poll() is None:
                p.terminate()
            return self._send(200, {"ok": True})
        return self._send(404, {"error": "no such endpoint"})


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


def main():
    port = 0
    if "--port" in sys.argv:
        port = int(sys.argv[sys.argv.index("--port") + 1])
    port = free_port(port)
    url = "http://127.0.0.1:%d/?t=%s" % (port, TOKEN)
    srv = Server(("127.0.0.1", port), Handler)
    print("\n  EBC Analyzer is running.\n")
    print("    %s\n" % url)
    print("  Videos are read in place - nothing is uploaded or copied.")
    print("  Leave this window open while it works; close it to stop the app.\n")
    if "--no-browser" not in sys.argv:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n  stopped.")


if __name__ == "__main__":
    main()
