"""Turning a one-dimensional LED signal into stimulus events.

The LED lens is coloured even when dark, so the resting level is not zero and a fixed
threshold does not transfer between recordings.  What does transfer is the *switch*:
the signal is bimodal, and the level that separates the two modes is the only sensible
place to put the threshold.  A Schmitt trigger then keeps a flickering edge from
splitting one pulse into several.
"""
import numpy as np


def levels(sig, hi_pct=99.9, lo_pct=50.0):
    """Resting and lit levels of a bimodal LED signal."""
    lo = float(np.percentile(sig, lo_pct))
    hi = float(np.percentile(sig, hi_pct))
    return lo, hi


def schmitt(sig, on_thr, off_thr):
    """Runs of samples above on_thr, extended outward while still above off_thr."""
    above = sig > off_thr
    trig = sig > on_thr
    if not trig.any():
        return []
    idx = np.where(above)[0]
    runs, s, p = [], idx[0], idx[0]
    for k in idx[1:]:
        if k > p + 1:
            runs.append((s, p)); s = k
        p = k
    runs.append((s, p))
    return [(a, b) for a, b in runs if trig[a:b + 1].any()]


def detect(sig, fps, nominal_ms, tol=0.35, frac_on=0.55, frac_off=0.35,
           min_gap_s=0.0, hi_pct=99.9):
    """Find LED pulses in `sig` and label the ones that match `nominal_ms`.

    Returns (events, info).  Every pulse found is returned, with `ok` saying whether it
    is within tolerance of the nominal duration - nothing is silently dropped, so the
    rejects can be counted and shown.
    """
    sig = np.asarray(sig, float)
    lo, hi = levels(sig, hi_pct)
    contrast = hi - lo
    info = dict(rest_level=round(lo, 1), lit_level=round(hi, 1), contrast=round(contrast, 1))
    if contrast < 25:
        info["reason"] = "no bimodal LED signal (contrast < 25)"
        return [], info
    on_thr = lo + frac_on * contrast
    off_thr = lo + frac_off * contrast
    info.update(on_threshold=round(on_thr, 1), off_threshold=round(off_thr, 1))
    dmin = nominal_ms * (1 - tol)
    dmax = nominal_ms * (1 + tol)
    ev = []
    for a, b in schmitt(sig, on_thr, off_thr):
        dur = (b - a + 1) / fps * 1000.0
        ev.append(dict(a=int(a), b=int(b), t=a / fps, dur_ms=dur,
                       peak=float(sig[a:b + 1].max()),
                       ok=bool(dmin <= dur <= dmax)))
    if min_gap_s > 0:                       # a second pulse this soon is LED flicker
        last = -1e9
        for e in ev:
            if e["ok"]:
                if e["t"] - last < min_gap_s:
                    e["ok"] = False
                    e["reason"] = f"within {min_gap_s:g}s of the previous accepted pulse"
                else:
                    last = e["t"]
    info["n_raw"] = len(ev)
    info["n_ok"] = sum(e["ok"] for e in ev)
    return ev, info


def pulse_stats(ev):
    ok = [e for e in ev if e["ok"]]
    if not ok:
        return {}
    d = np.array([e["dur_ms"] for e in ok])
    t = np.array([e["t"] for e in ok])
    it = np.diff(t)
    return dict(n=len(ok), dur_med_ms=round(float(np.median(d)), 1),
                dur_min_ms=round(float(d.min()), 1), dur_max_ms=round(float(d.max()), 1),
                iti_med_s=round(float(np.median(it)), 2) if len(it) else None,
                iti_min_s=round(float(it.min()), 2) if len(it) else None,
                iti_max_s=round(float(it.max()), 2) if len(it) else None)
