"""Measure eyelid aperture in a window around every trial.

    python ebc_eyes.py <config.json> <tag>

Each window is cut with the anchoring LED *and* the face inside the same crop, so the
stimulus onset is re-detected inside the window rather than trusted from the seek.  That
makes the alignment good to one frame regardless of how the container seeks - at 119.88
fps, 8.34 ms.

Aperture is the eye aspect ratio from MediaPipe FaceMesh, divided by eye width, so it
survives head movement and changes of camera distance.  Writes <tag>_traces.json.
"""
import os
import sys
import json
import time
import warnings

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

import ebc_config as C
from ebc_paths import work_dir
from ebc_video import probe, frames, still, crop_box

warnings.filterwarnings("ignore")

# The trial window is not a constant.  It comes from ebc_config.window(), which reads it
# off the protocol: a trace design has to keep tracking well past the CS to catch a US
# that arrives half a second after it is over.  The lab's delay numbers give back
# 300 / 1150 ms, which is what this app has always used, so nothing already tracked moves.
FACE_MARGIN = 70
LED_HALF = 26            # half-size of the LED patch carried inside the trial crop
N_FACE_SAMPLES = 26

R_EYE = [33, 160, 158, 133, 153, 144]
L_EYE = [362, 385, 387, 263, 373, 380]


def log(tag, *a):
    print("[%s]" % tag, *a, flush=True)


def ear(p, ix):
    p1, p2, p3, p4, p5, p6 = [p[i] for i in ix]
    return (np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)) / (2 * np.linalg.norm(p1 - p4) + 1e-9)


def face_box(path, tag, wdir, W, H, fps, nfr):
    """Union of the face landmarks over frames sampled across the whole recording.

    Sampling the whole recording, not just the start, means a camera that is re-aimed
    part way through still leaves the face inside the box.
    """
    f = os.path.join(wdir, tag + "_facebox.json")
    if os.path.exists(f):
        with open(f, encoding="utf-8") as fh:
            return json.load(fh)
    import mediapipe as mp
    fm = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1,
                                         refine_landmarks=True, min_detection_confidence=0.4)
    xs, xe, ys, ye = [], [], [], []
    for t in np.linspace(1.5, max(nfr / fps - 2.0, 2.0), N_FACE_SAMPLES):
        buf = still(path, float(t), W, H)
        if buf is None:
            continue
        img = np.frombuffer(buf, np.uint8).reshape(H, W, 3)
        r = fm.process(img)
        if not r.multi_face_landmarks:
            continue
        lm = np.array([[l.x * W, l.y * H] for l in r.multi_face_landmarks[0].landmark])
        xs.append(lm[:, 0].min()); xe.append(lm[:, 0].max())
        ys.append(lm[:, 1].min()); ye.append(lm[:, 1].max())
    fm.close()
    if not xs:
        raise SystemExit(tag + ": no face found in any sampled frame")
    box = dict(x0=float(min(xs)), x1=float(max(xe)), y0=float(min(ys)), y1=float(max(ye)),
               n=len(xs))
    with open(f, "w", encoding="utf-8") as fh:
        json.dump(box, fh)
    return box


def main():
    cfg = C.load(sys.argv[1])
    tag = sys.argv[2]
    rec = next(r for r in cfg["recordings"] if r["tag"] == tag)
    wdir = work_dir(cfg)
    out_f = os.path.join(wdir, tag + "_traces.json")
    if os.path.exists(out_f) and "--force" not in sys.argv:
        log(tag, "traces already present")
        return

    with open(os.path.join(wdir, "trials.json"), encoding="utf-8") as fh:
        TR = json.load(fh)
    trials = [t for t in TR["trials"] if t["session"] == tag]
    if not trials:
        log(tag, "no trials")
        json.dump({}, open(out_f, "w"))
        return
    with open(os.path.join(wdir, tag + "_stim.json"), encoding="utf-8") as fh:
        stim = json.load(fh)

    path = rec["path"]
    W, H, fps, nfr = probe(path)
    MS = 1000.0 / fps
    PRE_MS, POST_MS, _ = C.window(cfg["protocol"])
    PRE = int(round(PRE_MS / MS))
    POST = int(round(POST_MS / MS))
    NW = PRE + POST
    log(tag, "trial window -%.0f to +%.0f ms  (%s)"
        % (PRE_MS, POST_MS, C.design(cfg["protocol"])["label"].lower()))

    fb = face_box(path, tag, wdir, W, H, fps, nfr)
    log(tag, "face box x[%.0f,%.0f] y[%.0f,%.0f] from %d samples"
        % (fb["x0"], fb["x1"], fb["y0"], fb["y1"], fb["n"]))

    us_anchored = rec.get("anchor", "cs") == "us" and rec["role"] != "baseline_us"
    anchor_led = "blue" if (rec["role"] == "baseline_us" or us_anchored) else "yellow"
    # A US-anchored trial is cut around the *inferred* CS onset, so the blue flash sits
    # us_onset_ms into the window.  Find it there and step back to keep k0 meaning what it
    # means everywhere else: the index of CS onset.  Nothing downstream has to change.
    us_lag = int(round(cfg["protocol"]["us_onset_ms"] / MS)) if us_anchored else 0
    led = stim["leds"].get(anchor_led) or list(stim["leds"].values())[0]
    if led.get("position"):                       # where the LED actually lit up
        lx, ly = led["position"]["x"], led["position"]["y"]
        # a camera moved mid-recording leaves the LED in two places; carry both
        wander = max(led["position"]["spread_x"], led["position"]["spread_y"]) // 2
    else:
        bx, by, bw, bh = led["box"]
        lx, ly, wander = bx + bw // 2, by + bh // 2, max(bw, bh) // 2
    led_half = LED_HALF + wander

    fx0 = max(0, fb["x0"] - FACE_MARGIN); fx1 = min(W, fb["x1"] + FACE_MARGIN)
    fy0 = max(0, fb["y0"] - FACE_MARGIN); fy1 = min(H, fb["y1"] + FACE_MARGIN)
    cx0, cy0, cw, ch = crop_box(min(fx0, lx - led_half), min(fy0, ly - led_half),
                                max(fx1, lx + led_half), max(fy1, ly + led_half), W, H)
    FS = (slice(int(fy0) - cy0, min(int(fy1) - cy0, ch)), slice(int(fx0) - cx0, min(int(fx1) - cx0, cw)))
    LS = (slice(max(0, ly - led_half - cy0), min(ch, ly + led_half - cy0)),
          slice(max(0, lx - led_half - cx0), min(cw, lx + led_half - cx0)))
    log(tag, "trial crop %dx%d@(%d,%d)  anchor=%s LED at (%d,%d)" % (cw, ch, cx0, cy0, anchor_led, lx, ly))

    import cv2
    import mediapipe as mp
    fsz = cw * ch * 3
    out = {}
    t0 = time.time()
    for ti, tr in enumerate(trials, 1):
        f0 = tr["anchor_frame"]
        buf = [np.frombuffer(x, np.uint8).reshape(ch, cw, 3)
               for x in frames(path, "crop=%d:%d:%d:%d" % (cw, ch, cx0, cy0), fsz,
                               ss=(f0 - PRE) / fps, n=NW)]
        if len(buf) < NW * 0.85:
            log(tag, "  trial %d: only %d/%d frames, skipped" % (ti, len(buf), NW))
            continue
        A = np.stack(buf)
        patch = A[:, LS[0], LS[1], :].astype(np.int16)
        half = (patch[:, :, :, 1] + patch[:, :, :, 2]) >> 1
        m = (half - patch[:, :, :, 0]) if anchor_led == "yellow" else (patch[:, :, :, 0] - half)
        m = np.where(A[:, LS[0], LS[1], :].max(axis=3) >= 150, m, -128)
        s = m.reshape(len(A), -1).max(axis=1).astype(float)
        rest = float(np.percentile(s[:max(PRE // 2, 5)], 50))
        lit = float(s.max())
        on = np.where(s > rest + 0.55 * (lit - rest))[0] if lit - rest > 25 else []
        # When the anchor LED never lights in this window there is nothing to align to,
        # and k0 falls back to where the stimulus file said the trial was.  That fallback
        # used to report align_error_ms = (PRE - PRE) * MS = exactly 0.0 - the same value
        # a perfectly verified alignment produces, and the one number a reader would use
        # to decide the trial is trustworthy.  An unverified alignment now says so.
        verified = len(on) > 0
        k0 = int(on[0]) - us_lag if verified else PRE

        fm = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1,
                                             refine_landmarks=True,
                                             min_detection_confidence=0.4,
                                             min_tracking_confidence=0.4)
        er = np.full(len(A), np.nan)
        el = np.full(len(A), np.nan)
        for i in range(len(A)):
            big = cv2.resize(A[i][FS[0], FS[1]], None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            r = fm.process(cv2.cvtColor(big, cv2.COLOR_BGR2RGB))
            if not r.multi_face_landmarks:
                continue
            hh, ww = big.shape[:2]
            p = np.array([[l.x * ww, l.y * hh] for l in r.multi_face_landmarks[0].landmark])
            er[i] = ear(p, R_EYE)
            el[i] = ear(p, L_EYE)
        fm.close()
        out[str(tr["session_trial"])] = dict(
            k0=k0, pre=PRE, n=len(A), anchor_frame=int(f0),
            align_error_ms=round((k0 - PRE) * MS, 2) if verified else None,
            alignment=("measured from the %s LED" % anchor_led) if verified else
                      ("NOT VERIFIED - the %s LED never lit in this window, so the trial "
                       "sits where the stimulus file put it" % anchor_led),
            led_rest=round(rest, 1), led_lit=round(lit, 1),
            face_ok=float(np.isfinite(er).mean()),
            er=[None if not np.isfinite(v) else round(float(v), 6) for v in er],
            el=[None if not np.isfinite(v) else round(float(v), 6) for v in el])
        if ti % 10 == 0 or ti == len(trials):
            log(tag, "  trial %3d/%d  align %+.1f ms  face %.0f%%  [%.0fs]"
                % (ti, len(trials), (k0 - PRE) * MS, np.isfinite(er).mean() * 100, time.time() - t0))
    with open(out_f, "w", encoding="utf-8") as fh:
        json.dump(out, fh)
    ae = [abs(v["align_error_ms"]) for v in out.values() if v["align_error_ms"] is not None]
    unver = [k for k, v in out.items() if v["align_error_ms"] is None]
    if unver:
        log(tag, "  !! %d trial(s) could not be aligned - the anchor LED never lit in "
                 "their window: %s" % (len(unver), ", ".join(unver[:8])))
    log(tag, "%d/%d trials tracked, median |alignment error| %.1f ms, max %.1f ms -> %s_traces.json"
        % (len(out), len(trials), float(np.median(ae)) if ae else 0,
           float(np.max(ae)) if ae else 0, tag))


if __name__ == "__main__":
    main()
