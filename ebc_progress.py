"""One-line live progress readout across the video pipelines."""
# --- portable paths -------------------------------------------------------
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ebc_paths import BASE, OUT, WORK          # noqa: E402
os.chdir(WORK)                                  # cache + intermediates live here
# --------------------------------------------------------------------------
import re, time

SESS = [("csus1", "CSUS 1"), ("csus2", "CSUS 2"), ("csus3", "CSUS 3"), ("csus4", "CSUS 4")]
# milestone file -> (label, expected seconds for the stage that produces it)
STAGES = [("_blue.npz", "LED scan", 230),
          ("_yellow.npz", "CS scan", 365),
          ("_facebox.json", "face box", 45),
          ("_traces.json", "eyelids", 200),
          ("_result.json", "metrics", 5)]
TOTAL = sum(s[2] for s in STAGES)


def mt(p):
    return os.path.getmtime(p) if os.path.exists(p) else None


def log_detail(tag):
    """Exact counts if this session's log is unbuffered and readable."""
    f = f"{tag}.log"
    if not os.path.exists(f):
        return ""
    try:
        txt = open(f, encoding="utf-8", errors="ignore").read()
    except Exception:
        return ""
    m = re.findall(r"stage(\d) (\d+)/(\d+)", txt)
    t = re.findall(r"trial\s+(\d+)/(\d+)", txt)
    if t:
        return f" {t[-1][0]}/{t[-1][1]} trials"
    if m:
        return f" {int(m[-1][1])/1000:.0f}k/{int(m[-1][2])/1000:.0f}k frames"
    return ""


def session_state(tag, start_hint):
    done_at = mt(f"{tag}_result.json")
    if done_at:
        return 1.0, "done", 0.0
    prev = start_hint
    elapsed_before = 0.0
    for suf, label, exp in STAGES:
        p = f"{tag}{suf}"
        m = mt(p)
        if m is None:
            if prev is None:
                return 0.0, "queued", TOTAL
            frac_in = min((time.time() - prev) / exp, 0.98)
            done_s = elapsed_before + frac_in * exp
            return done_s / TOTAL, label, max(TOTAL - done_s, 0)
        elapsed_before += exp
        prev = m
    return 1.0, "done", 0.0


def bar(f, w=14):
    n = int(round(f * w))
    return "#" * n + "." * (w - n)


starts = {}
prev_done = None
for tag, name in SESS:
    starts[tag] = prev_done
    r = mt(f"{tag}_result.json")
    prev_done = r if r else None

parts, remain = [], 0.0
for tag, name in SESS:
    f, label, rem = session_state(tag, starts[tag])
    remain += rem
    if label == "done":
        parts.append(f"{name} [done]")
    elif label == "queued":
        parts.append(f"{name} [queued]")
    else:
        parts.append(f"{name} [{bar(f)}] {f*100:3.0f}% {label}{log_detail(tag)}")
overall = sum(session_state(t, starts[t])[0] for t, _ in SESS) / len(SESS)
eta = int(remain // 60)
print(f"{time.strftime('%H:%M:%S')}  " + "  |  ".join(parts) +
      f"  ||  overall {overall*100:.0f}%  ETA ~{eta} min")
sys.stdout.flush()
