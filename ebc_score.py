"""Score every trial on one pooled closure scale, using the known protocol.

Protocol (given by the experimenter, and confirmed by the data):
  * a block = 9 paired CS-US trials followed by 1 CS-only trial
  * 10 blocks of conditioning
  * CS = 400 ms, US = 50 ms, the two co-terminate, so US onset = 350 ms
  * a few CS-only trials afterwards = extinction

The block structure is used as a detection filter: a genuine CS presentation lasts
~400 ms, so detections far from that duration are LED flicker, not stimuli. After
filtering, the recovered sequence is exactly 9+1 x 10, which is the check that the
filter is right rather than merely convenient.

CS-only trials are scored but held apart: they never enter the session summary or the
conditioning scatter, because a trial with no US is a different measurement.
"""
# --- portable paths -------------------------------------------------------
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ebc_paths import BASE, OUT, WORK          # noqa: E402
os.chdir(WORK)                                  # cache + intermediates live here
# --------------------------------------------------------------------------
import json, os, numpy as np
from scipy.signal import savgol_filter

# The app writes run_config.json; without one these defaults reproduce the
# original Marie session, so the scripts still work standalone from the CLI.
DEFAULT_CFG = {
    "study": "Marie",
    "nominal": {"cs_ms": 400.0, "us_onset_ms": 350.0, "us_dur_ms": 50.0},
    "groups": {"conditioning": [["csus1", "CSUS 1"], ["csus2", "CSUS 2"], ["csus3", "CSUS 3"]],
               "test": [["csus4", "CSUS 4"]]},
}
CFG = DEFAULT_CFG
if os.path.exists("run_config.json"):
    CFG = json.load(open("run_config.json", encoding="utf-8"))
    for k, v in DEFAULT_CFG.items():
        CFG.setdefault(k, v)

COND = [tuple(x) for x in CFG["groups"].get("conditioning", [])]
TEST = [tuple(x) for x in CFG["groups"].get("test", [])]
ALL = [s for s in COND + TEST if os.path.exists(f"{s[0]}_result.json")]
CS_DUR = float(CFG["nominal"]["cs_ms"])
US_ONSET = float(CFG["nominal"]["us_onset_ms"])
US_DUR = float(CFG["nominal"]["us_dur_ms"])
OFFSET = {}      # filled in below, once durations are known
DUR_MIN, DUR_MAX, MIN_ISI = 330.0, 470.0, 6.0     # CS detection acceptance
PRE_MS = 300.0
MAIN, RESET, PARTIAL = 0.40, 0.20, 0.15

meta = {t: json.load(open(f"{t}_result.json")) for t, _ in ALL}
raw = {t: json.load(open(f"{t}_traces.json")) for t, _ in ALL}

# Conditioning recordings are consecutive chapters of one session, so a trial's
# position on the continuous clock is its time plus the durations of the chapters
# before it. Derived here rather than hard-coded, so any set of chapters works.
_acc = 0.0
for _t, _ in COND:
    if _t in meta:
        OFFSET[_t] = _acc
        _acc += float(meta[_t]["duration_s"])


def sm(x):
    x = np.array(x, float)
    m = np.isfinite(x)
    if m.sum() < len(x):
        x = np.interp(np.arange(len(x)), np.where(m)[0], x[m])
    return savgol_filter(x, 5, 2)


E = {}
for tag, _ in ALL:
    for k, v in raw[tag].items():
        E[(tag, int(k))] = (sm(v["er"]) + sm(v["el"])) / 2
CLOSED = float(np.percentile([e.min() for e in E.values()], 10))
print(f"pooled full-closure EAR reference = {CLOSED:.4f}  ({len(E)} trials, {len(ALL)} recordings)")

# ---- US artefact filter: paired blue events teach the pulse signature ----------
pd_ = []
for t, _ in ALL:
    fps = meta[t]["fps"]
    cs = {c for a, b, c in meta[t]["cs_events"] if c >= 0}
    pd_ += [(b - a + 1) / fps * 1000 for a, b in meta[t]["us_events"] if a in cs]
US_MEAS = float(np.median(pd_))
for t, _ in ALL:
    fps = meta[t]["fps"]
    cs = {c for a, b, c in meta[t]["cs_events"] if c >= 0}
    keep, drop = [], []
    for a, b in meta[t]["us_events"]:
        dur = (b - a + 1) / fps * 1000
        (keep if (a in cs or abs(dur - US_MEAS) <= 0.5 * US_MEAS) else drop).append([a, b])
    meta[t]["us_events"], meta[t]["us_artifacts"] = keep, drop
print(f"US pulse measured at {US_MEAS:.1f} ms (nominal {US_DUR}); "
      f"{sum(len(meta[t]['us_artifacts']) for t,_ in ALL)} blue transients rejected")

# ---- CS detection filter, then block structure --------------------------------
def order(sessions, use_offset):
    out = []
    for tag, name in sessions:
        fps = meta[tag]["fps"]
        for a, b, c in meta[tag]["cs_events"]:
            out.append(dict(tag=tag, name=name, frame=a, t=a / fps + (OFFSET.get(tag, 0) if use_offset else 0),
                            dur=(b - a + 1) / fps * 1000, paired=c >= 0))
    out.sort(key=lambda z: z["t"])
    return out


def accept(seq):
    keep = []
    for s in seq:
        if not (DUR_MIN <= s["dur"] <= DUR_MAX):
            continue
        if keep and s["t"] - keep[-1]["t"] < MIN_ISI:
            continue
        keep.append(s)
    return keep


cond_all = order([s for s in COND if s in ALL], True)
cond = accept(cond_all)
runs, run = [], 0
for s in cond:
    if s["paired"]:
        run += 1
    else:
        runs.append(run); run = 0
strict = runs == [9] * 10 and run == 0
print(f"CS detections: {len(cond_all)} -> {len(cond)} accepted "
      f"({sum(s['paired'] for s in cond)} paired + {sum(not s['paired'] for s in cond)} CS-only); "
      f"runs before each CS-only {runs}; strict 9+1 x 10 = {strict}")
for i, s in enumerate(cond):
    s["block"] = i // 10 + 1
    s["in_block"] = i % 10 + 1
test_all = order([s for s in TEST if s in ALL], False)
test = accept(test_all)
for i, s in enumerate(test):
    s["block"] = None; s["in_block"] = i + 1
print(f"extinction (CSUS 4): {len(test_all)} -> {len(test)} accepted CS-only")

VALID = {(s["tag"], s["frame"]): s for s in cond + test}
frame_to_trial = {}
for tag, _ in ALL:
    fps = meta[tag]["fps"]
    for r in meta[tag]["rows"]:
        frame_to_trial[(tag, int(round(r["cs_onset_video_s"] * fps)))] = r["trial"]

# ---- per-trial metrics --------------------------------------------------------
out_rows = []
for tag, name in ALL:
    FPS = meta[tag]["fps"]
    MS = 1000.0 / FPS
    PRE = int(round(PRE_MS / MS))
    old = {r["trial"]: r for r in meta[tag]["rows"]}
    traces = {}
    for key, s in sorted(VALID.items(), key=lambda kv: kv[1]["t"]):
        if key[0] != tag:
            continue
        ti = frame_to_trial.get(key)
        if ti is None or str(ti) not in raw[tag]:
            continue
        e = E[(tag, ti)]
        oref = float(np.percentile(e, 85))
        span = max(oref - CLOSED, 1e-6)
        C = np.clip((oref - e) / span, -0.3, 1.4)
        t = (np.arange(len(e)) - PRE) * MS
        traces[ti] = dict(t=t.tolist(), C=C.tolist())
        pre = (t >= -300) & (t < -30)
        q = e[pre][C[pre] < 0.25]
        sd = (1.4826 * np.median(np.abs(q - np.median(q))) / span) if len(q) > 5 else .03
        thr = max(5 * sd, PARTIAL)
        preflag = bool(C[pre].max() > 0.30)
        inprog = bool(C[PRE] > 0.30)
        w = np.where((t >= 0) & (t <= 1000))[0]
        exc, i = [], w[0]
        while i <= w[-1]:
            if C[i] > thr:
                st = i
                while st > 0 and C[st - 1] < C[st] and C[st - 1] > 0.04:
                    st -= 1
                j = i
                while j < len(C) - 1 and C[j + 1] > RESET:
                    j += 1
                pk = st + int(np.argmax(C[st:j + 1]))
                r50 = pk
                while r50 < len(C) - 1 and C[r50] > 0.5 * C[pk]:
                    r50 += 1
                ro = pk
                while ro < len(C) - 1 and C[ro] > RESET:
                    ro += 1
                exc.append(dict(on=float(t[st]), pk=float(t[pk]), amp=float(C[pk]),
                                r50=float(t[r50]), end=float(t[ro]), dur=float(t[ro] - t[st]),
                                rise=float((C[pk] - C[st]) / max(t[pk] - t[st], MS))))
                i = j + 1
            else:
                i += 1
        full = [b for b in exc if b["amp"] >= MAIN]
        part = [b for b in exc if b["amp"] < MAIN]
        b1 = full[0] if full else (exc[0] if exc else None)

        def klass(on, moving):
            if moving:
                return "in-progress at CS"
            return ("alpha/startle <100ms" if on < 100 else
                    "CR (100-350ms)" if on < US_ONSET else "UR (>=350ms)")

        cls = klass(b1["on"], inprog) if b1 else None
        # --- note 3: if the first event is an alpha blink or a lid already moving,
        #     look for a LATER blink in the same window - there may still be a real
        #     response (a CR or a UR) hiding behind the artefact.
        obscured = bool(b1 and (inprog or b1["on"] < 100))
        b2 = full[1] if (obscured and len(full) > 1) else None
        sec_cls = klass(b2["on"], False) if b2 else None

        def at(ms):
            return float(C[min(PRE + int(round(ms / MS)), len(C) - 1)])

        a_us, a_off, a_end = at(US_ONSET), at(CS_DUR), at(1000)
        o = old[ti]
        out_rows.append(dict(
            block_kind="conditioning" if tag != "csus4" else "extinction",
            trial_type="CS-US" if s["paired"] else "CS-only",
            block=s["block"], trial_in_block=s["in_block"],
            session=tag, session_name=name, session_trial=ti,
            cs_onset_video_s=o["cs_onset_video_s"],
            cs_onset_block_s=round(s["t"], 3) if tag != "csus4" else None,
            cs_duration_measured_ms=round(s["dur"], 1),
            quality=(("pre-CS blink" if preflag else "") +
                     (" | lid closing at CS" if inprog else "")) or "clean",
            n_full_blinks=len(full), n_partial_movements=len(part),
            blink_onset_ms=None if not b1 else round(b1["on"], 1),
            peak_closure_ms=None if not b1 else round(b1["pk"], 1),
            peak_closure_pct=None if not b1 else round(b1["amp"] * 100, 1),
            closing_speed_pct_per_ms=None if not b1 else round(b1["rise"] * 100, 2),
            closure_duration_ms=None if not b1 else round(b1["dur"], 1),
            reopen_half_ms=None if not b1 else round(b1["r50"], 1),
            reopen_full_ms=None if not b1 else round(b1["end"], 1),
            closure_at_US_pct=round(a_us * 100, 1),
            closure_at_CSoff_pct=round(a_off * 100, 1),
            closure_at_1000ms_pct=round(a_end * 100, 1),
            closed_at_US=bool(a_us >= 0.50),
            reopened_before_US=bool(b1 is not None and a_us < 0.30),
            response_class=cls,
            first_response_obscured="yes" if obscured else "",
            secondary_onset_ms=None if not b2 else round(b2["on"], 1),
            secondary_peak_pct=None if not b2 else round(b2["amp"] * 100, 1),
            secondary_class=sec_cls,
            scored_onset_ms=(round(b2["on"], 1) if b2 else (None if not b1 else round(b1["on"], 1))),
            scored_class=(sec_cls if b2 else cls),
            all_blink_onsets_ms=";".join(f"{b['on']:.0f}" for b in full),
            all_blink_amps_pct=";".join(f"{b['amp']*100:.0f}" for b in full),
            inter_blink_ms=";".join(f"{full[x+1]['on']-full[x]['on']:.0f}" for x in range(len(full) - 1)),
            partial_movement_ms=";".join(f"{b['on']:.0f}({b['amp']*100:.0f}%)" for b in part)))
    meta[tag]["traces"] = {str(k): v for k, v in traces.items()}

out_rows.sort(key=lambda r: (r["block_kind"] != "conditioning", r["session"], r["session_trial"]))
n = {"CS-US": 0, "CS-only": 0, "ext": 0}
for r in out_rows:
    if r["block_kind"] == "extinction":
        n["ext"] += 1; r["gidx"] = n["ext"]
    else:
        n[r["trial_type"]] += 1; r["gidx"] = n[r["trial_type"]]
for tag, _ in ALL:
    meta[tag]["rows"] = [r for r in out_rows if r["session"] == tag]
    meta[tag]["cs_dur_ms"] = CS_DUR
    meta[tag]["us_ms"] = US_ONSET if tag != "csus4" else None
    meta[tag]["us_dur_ms"] = US_DUR if tag != "csus4" else None
json.dump(dict(closed_ref=CLOSED, meta=meta, strict_blocks=bool(strict),
               study=CFG.get("study", "study"), offsets=OFFSET,
               nominal=dict(cs_ms=CS_DUR, us_onset_ms=US_ONSET, us_dur_ms=US_DUR),
               measured=dict(us_dur_ms=round(US_MEAS, 1)),
               groups=[["conditioning", [t for t, _ in COND if t in meta]],
                       ["test", [t for t, _ in TEST if t in meta]]]),
          open("merged.json", "w"))
json.dump(out_rows, open("merged_rows.json", "w"))

print()
for kind, tt in (("conditioning", "CS-US"), ("conditioning", "CS-only"), ("extinction", "CS-only")):
    rs = [r for r in out_rows if r["block_kind"] == kind and r["trial_type"] == tt]
    if not rs:
        continue
    sc = [r for r in rs if r["scored_class"] not in (None, "in-progress at CS")]
    cr = [r for r in sc if r["scored_class"] == "CR (100-350ms)"]
    rec = [r for r in rs if r["first_response_obscured"] == "yes" and r["secondary_onset_ms"] is not None]
    o = [r["scored_onset_ms"] for r in cr]
    print(f"{kind:>12} / {tt:<8}: {len(rs):>3} trials, {len(sc):>3} scoreable, {len(cr):>3} CR"
          + (f" ({len(cr)/len(sc)*100:.0f}%), mean onset {np.mean(o):.0f} ms" if o else "")
          + f"   [{len(rec)} recovered via a later blink]")
