"""Score every trial on one pooled closure scale.

    python ebc_score.py <config.json>

Reads the recovered trials and the eyelid traces; writes merged.json / merged_rows.json.

The closure scale is pooled across every recording of the participant, so a two-minute
extinction clip and a nine-minute conditioning chapter are on the same axis and their
amplitudes can be compared.  CS-only probes and the baselines are scored exactly like
paired trials but kept out of the acquisition summaries, because a trial with no US is
a different measurement.
"""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.signal import savgol_filter

import ebc_config as C
from ebc_paths import work_dir

MAIN, RESET, PARTIAL = 0.40, 0.20, 0.15
BASE_FROM, BASE_TO = -300.0, -30.0
SEARCH_MS = 1000.0


def smooth(x):
    x = np.array([np.nan if v is None else v for v in x], float)
    m = np.isfinite(x)
    if m.sum() < 4:
        return None
    if m.sum() < len(x):
        x = np.interp(np.arange(len(x)), np.where(m)[0], x[m])
    return savgol_filter(x, 5, 2)


def excursions(C_, t, thr):
    """Every upward excursion of the closure trace inside the response window.

    The onset is walked back down the rising edge to where the movement actually began,
    not to where it crossed the threshold.
    """
    w = np.where((t >= 0) & (t <= SEARCH_MS))[0]
    if not len(w):
        return []
    out, i = [], w[0]
    MS = t[1] - t[0]
    while i <= w[-1]:
        if C_[i] > thr:
            s = i
            while s > 0 and C_[s - 1] < C_[s] and C_[s - 1] > 0.04:
                s -= 1
            j = i
            while j < len(C_) - 1 and C_[j + 1] > RESET:
                j += 1
            pk = s + int(np.argmax(C_[s:j + 1]))
            r50 = pk
            while r50 < len(C_) - 1 and C_[r50] > 0.5 * C_[pk]:
                r50 += 1
            ro = pk
            while ro < len(C_) - 1 and C_[ro] > RESET:
                ro += 1
            out.append(dict(on=float(t[s]), pk=float(t[pk]), amp=float(C_[pk]),
                            r50=float(t[r50]), end=float(t[ro]), dur=float(t[ro] - t[s]),
                            rise=float((C_[pk] - C_[s]) / max(t[pk] - t[s], MS))))
            i = j + 1
        else:
            i += 1
    return out


def classify(onset_ms, us_onset_ms, moving, anchored_on_us):
    if moving:
        return "in-progress at stimulus"
    if anchored_on_us:
        return "UR to the puff" if onset_ms >= 20 else "alpha/startle <20ms"
    if onset_ms < 100:
        return "alpha/startle <100ms"
    if onset_ms < us_onset_ms:
        return "CR (100-%dms)" % round(us_onset_ms)
    return "UR (>=%dms)" % round(us_onset_ms)


def main():
    cfg = C.load(sys.argv[1] if len(sys.argv) > 1 else None)
    wdir = work_dir(cfg)
    proto = cfg["protocol"]
    US_ONSET, CS_DUR = proto["us_onset_ms"], proto["cs_ms"]

    with open(os.path.join(wdir, "trials.json"), encoding="utf-8") as fh:
        TR = json.load(fh)
    traces = {}
    for rec in cfg["recordings"]:
        f = os.path.join(wdir, rec["tag"] + "_traces.json")
        if os.path.exists(f):
            with open(f, encoding="utf-8") as fh:
                traces[rec["tag"]] = json.load(fh)

    fps = {s["tag"]: s["fps"] for s in TR["sessions"]}
    E = {}
    for tag, d in traces.items():
        for k, v in d.items():
            a, b = smooth(v["er"]), smooth(v["el"])
            if a is None and b is None:
                continue
            E[(tag, int(k))] = (a if b is None else b if a is None else (a + b) / 2)
    if not E:
        raise SystemExit("no eyelid traces - run ebc_eyes.py first")
    CLOSED = float(np.percentile([e.min() for e in E.values()], 10))
    print("pooled full-closure EAR reference = %.4f  (%d trials, %d recordings)"
          % (CLOSED, len(E), len(traces)))

    rows, keep_traces = [], {}
    for tr in TR["trials"]:
        tag, ti = tr["session"], tr["session_trial"]
        if (tag, ti) not in E:
            continue
        v = traces[tag][str(ti)]
        e = E[(tag, ti)]
        MS = 1000.0 / fps[tag]
        PRE = int(v["pre"])
        k0 = int(v["k0"])                       # LED onset re-found inside this window
        t = (np.arange(len(e)) - k0) * MS
        oref = float(np.percentile(e, 85))      # blink-robust open-eye reference
        span = max(oref - CLOSED, 1e-6)
        Cl = np.clip((oref - e) / span, -0.3, 1.4)
        keep_traces.setdefault(tag, {})[str(ti)] = dict(
            t=[round(float(x), 2) for x in t], C=[round(float(x), 4) for x in Cl])

        pre = (t >= BASE_FROM) & (t < BASE_TO)
        q = e[pre][Cl[pre] < 0.25]
        sd = (1.4826 * np.median(np.abs(q - np.median(q))) / span) if len(q) > 5 else 0.03
        thr = max(5 * sd, PARTIAL)
        preflag = bool(pre.any() and Cl[pre].max() > 0.30)
        inprog = bool(Cl[k0] > 0.30)
        exc = excursions(Cl, t, thr)
        full = [b for b in exc if b["amp"] >= MAIN]
        part = [b for b in exc if b["amp"] < MAIN]
        b1 = full[0] if full else (exc[0] if exc else None)
        on_us = tr["trial_type"] == "US-only"

        cls = classify(b1["on"], US_ONSET, inprog, on_us) if b1 else None
        # if the first event is an alpha blink or the lid was already moving, look for a
        # later blink in the same window - a real CR or UR may sit behind the artefact
        obscured = bool(b1 and (inprog or (not on_us and b1["on"] < 100)))
        b2 = full[1] if (obscured and len(full) > 1) else None
        sec = classify(b2["on"], US_ONSET, False, on_us) if b2 else None

        def at(ms):
            return float(Cl[min(max(k0 + int(round(ms / MS)), 0), len(Cl) - 1)])

        row = dict(
            study=cfg["study"], role=tr["role"], session=tag,
            session_name=tr["session_name"], session_trial=ti,
            trial_type=tr["trial_type"], block=tr.get("block"),
            trial_in_block=tr.get("trial_in_block"), global_trial=tr.get("global_trial"),
            cs_onset_video_s=tr["cs_onset_s"], us_onset_video_s=tr["us_onset_s"],
            session_clock_s=tr.get("session_clock_s"),
            cs_duration_ms=tr["cs_duration_ms"], us_duration_ms=tr["us_duration_ms"],
            cs_timing=tr.get("cs_timing", "measured from CS LED"),
            measured_isi_ms=tr["isi_ms"],
            alignment_error_ms=v.get("align_error_ms"),
            face_tracked_pct=round(100 * v.get("face_ok", 0), 1),
            quality=(("pre-CS blink" if preflag else "")
                     + (" | lid closing at onset" if inprog else "")
                     + (" | truncated window" if tr.get("truncated") else "")
                     + (" | face lost >20%" if v.get("face_ok", 1) < 0.8 else "")).strip(" |")
                    or "clean",
            n_full_blinks=len(full), n_partial_movements=len(part),
            blink_onset_ms=None if not b1 else round(b1["on"], 1),
            peak_closure_ms=None if not b1 else round(b1["pk"], 1),
            peak_closure_pct=None if not b1 else round(b1["amp"] * 100, 1),
            closing_speed_pct_per_ms=None if not b1 else round(b1["rise"] * 100, 2),
            closure_duration_ms=None if not b1 else round(b1["dur"], 1),
            reopen_half_ms=None if not b1 else round(b1["r50"], 1),
            reopen_full_ms=None if not b1 else round(b1["end"], 1),
            closure_at_US_pct=round(at(US_ONSET) * 100, 1),
            closure_at_CSoff_pct=round(at(CS_DUR) * 100, 1),
            closure_at_1000ms_pct=round(at(1000) * 100, 1),
            closed_at_US=bool(at(US_ONSET) >= 0.50),
            reopened_before_US=bool(b1 is not None and at(US_ONSET) < 0.30),
            response_class=cls,
            first_response_obscured="yes" if obscured else "",
            secondary_onset_ms=None if not b2 else round(b2["on"], 1),
            secondary_peak_pct=None if not b2 else round(b2["amp"] * 100, 1),
            secondary_class=sec,
            scored_onset_ms=(round(b2["on"], 1) if b2 else
                             (None if not b1 else round(b1["on"], 1))),
            scored_class=(sec if b2 else cls),
            all_blink_onsets_ms=";".join("%.0f" % b["on"] for b in full),
            all_blink_amps_pct=";".join("%.0f" % (b["amp"] * 100) for b in full),
            inter_blink_ms=";".join("%.0f" % (full[k + 1]["on"] - full[k]["on"])
                                    for k in range(len(full) - 1)),
            partial_movement_ms=";".join("%.0f(%.0f%%)" % (b["on"], b["amp"] * 100)
                                         for b in part))
        rows.append(row)

    order = {r: i for i, r in enumerate(C.ROLES)}
    rows.sort(key=lambda r: (order[r["role"]], r["session"], r["session_trial"]))
    seen = {}
    for r in rows:
        k = (r["role"], r["trial_type"])
        seen[k] = seen.get(k, 0) + 1
        r["group_index"] = seen[k]

    merged = dict(study=cfg["study"], protocol=proto, closed_ref=CLOSED,
                  sessions=TR["sessions"], checks=TR["checks"], offsets=TR["offsets"],
                  traces=keep_traces,
                  recordings=[{k: rec[k] for k in ("tag", "label", "role", "order")}
                              for rec in cfg["recordings"]])
    with open(os.path.join(wdir, "merged.json"), "w", encoding="utf-8") as fh:
        json.dump(merged, fh)
    with open(os.path.join(wdir, "merged_rows.json"), "w", encoding="utf-8") as fh:
        json.dump(rows, fh)

    print()
    for role, tt in (("conditioning", "CS-US"), ("conditioning", "CS-only"),
                     ("extinction", "CS-only"), ("baseline_cs", "CS-only"),
                     ("baseline_us", "US-only")):
        rs = [r for r in rows if r["role"] == role and r["trial_type"] == tt]
        if not rs:
            continue
        sc = [r for r in rs if r["scored_class"] not in (None, "in-progress at stimulus")]
        cr = [r for r in sc if str(r["scored_class"]).startswith("CR")]
        rec = [r for r in rs if r["first_response_obscured"] == "yes"
               and r["secondary_onset_ms"] is not None]
        o = [r["scored_onset_ms"] for r in cr]
        line = "%12s / %-8s: %3d trials, %3d scoreable, %3d CR" % (role, tt, len(rs), len(sc), len(cr))
        if o:
            line += " (%.0f%%), mean CR onset %.0f ms" % (len(cr) / len(sc) * 100, float(np.mean(o)))
        print(line + "   [%d recovered behind an artefact]" % len(rec))


if __name__ == "__main__":
    main()
