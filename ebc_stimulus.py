"""Find the two stimulus LEDs and read their exact on/off times.

    python ebc_stimulus.py <config.json> <tag>

Two passes over the recording:

  A  survey   one subsampled decode at 480x270.  For every 6x6 block it keeps the
              strongest yellow and blue value per sampled frame, so each block has a
              time course.  A stimulus LED is not the brightest or the yellowest thing
              in a lit room - it is the thing that *switches*, between two well
              separated levels, in pulses that all last the protocol's duration.  Blocks
              are ranked on exactly that, which is why a table lamp, a wooden mask or a
              painting never wins.  Run by ebc_locate.py, which turns the ranking into
              one box position per recording.

  B  read     one full-rate decode of a small full-resolution window on the box.  All
              reported timing comes from this pass, so it is good to one frame - at
              119.88 fps, 8.34 ms.

The US LED is never searched for across the frame.  At 50 ms it is one or two frames in
the survey, indistinguishable from noise; instead its window is pinned beside the CS LED,
where it physically is.  That also keeps the read window small, which is what makes pass
B cheap.
"""
import os
import sys
import json
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

import ebc_config as C
from ebc_paths import work_dir
from ebc_video import probe, frames, crop_box
from ebc_signal import detect, pulse_stats

SURVEY_W, SURVEY_H, BLK = 480, 270, 6
SURVEY_FPS = 30.0
BRIGHT_MIN = 165          # a lit LED is bright; this rejects dark bluish shadow

CS_HALF = 34              # half-size of the patch read around a confident CS LED
CS_PAD = 60               # extra margin when the position is uncertain
# where the US LED may sit relative to the CS LED: it is a few centimetres to one side
# on the same panel, so a box this size around the CS LED always contains it.
US_BOX = dict(left=90, right=190, up=80, down=80)
# a recording with no CS at all inherits the box position from the rest of the session,
# so its window has to allow for the camera having been re-aimed in between
US_BOX_INHERITED = dict(left=190, right=260, up=150, down=150)


def log(tag, *a):
    print("[%s]" % tag, *a, flush=True)


def _metrics(img):
    """(yellowness, blueness) of a BGR frame, with dark pixels knocked out."""
    b = img[:, :, 0].astype(np.int16)
    half = (img[:, :, 1].astype(np.int16) + img[:, :, 2]) >> 1
    y = half - b
    dark = img.max(axis=2) < BRIGHT_MIN
    y[dark] = -128
    nb = b - half
    nb[dark] = -128
    return y, nb


# ----------------------------------------------------------------- pass A: survey
def survey(path, tag, wdir):
    f = os.path.join(wdir, tag + "_survey.npz")
    if os.path.exists(f):
        return np.load(f)
    W, H, fps, _ = probe(path)
    BW, BH = SURVEY_W // BLK, SURVEY_H // BLK
    fsz = SURVEY_W * SURVEY_H * 3
    vf = "fps=%g,scale=%d:%d:flags=area" % (SURVEY_FPS, SURVEY_W, SURVEY_H)
    YB, BB = [], []
    t0 = time.time()
    for raw in frames(path, vf, fsz):
        img = np.frombuffer(raw, np.uint8).reshape(SURVEY_H, SURVEY_W, 3)
        y, nb = _metrics(img)
        YB.append(y.reshape(BH, BLK, BW, BLK).max(axis=(1, 3)).astype(np.int16))
        BB.append(nb.reshape(BH, BLK, BW, BLK).max(axis=(1, 3)).astype(np.int16))
    if not YB:
        raise SystemExit(tag + ": no frames decoded from " + path)
    np.savez_compressed(f, yb=np.stack(YB), bb=np.stack(BB), w=W, h=H, fps=fps,
                        survey_fps=SURVEY_FPS, blk=BLK, sw=SURVEY_W, sh=SURVEY_H)
    log(tag, "survey: %d sampled frames in %.0fs" % (len(YB), time.time() - t0))
    return np.load(f)


def rank_blocks(S, survey_fps, nominal_ms, kx, ky, blk, min_gap_s):
    """Score every block as a candidate CS LED, best first."""
    n, BH, BW = S.shape
    out = []
    for r in range(BH):
        for c in range(BW):
            sig = S[:, r, c].astype(np.float32)
            ev, info = detect(sig, survey_fps, nominal_ms, tol=0.55, min_gap_s=min_gap_s)
            ok = [e for e in ev if e["ok"]]
            if len(ok) < 2 or not info.get("contrast"):
                continue
            d = np.array([e["dur_ms"] for e in ok])
            if d.sum() / 1000.0 > 0.15 * (n / survey_fps):
                continue                              # lit too often to be a stimulus
            dcv = float(np.std(d) / max(np.mean(d), 1e-9))
            fit = float(np.exp(-abs(np.median(d) - nominal_ms) / (0.30 * nominal_ms)))
            score = info["contrast"] * fit * np.exp(-2.0 * dcv) * (1 + np.log1p(len(ok)))
            out.append(dict(score=float(score), contrast=info["contrast"], n_ok=len(ok),
                            n_raw=len(ev), dur_med=float(np.median(d)), dur_cv=dcv,
                            x=int(round((c + .5) * blk * kx)),
                            y=int(round((r + .5) * blk * ky))))
    out.sort(key=lambda z: -z["score"])
    return out


def merge_adjacent(cands, radius=40):
    """One LED lights several neighbouring blocks; keep the best of each cluster."""
    kept = []
    for c in cands:
        if all(abs(c["x"] - k["x"]) > radius or abs(c["y"] - k["y"]) > radius for k in kept):
            kept.append(c)
    return kept


# ----------------------------------------------------------------- pass B: read
def boxes(anchor, wants, W, H, confident, consensus=None):
    """Sub-windows to read: a patch on the CS LED, a wider one beside it for the US.

    When the box position for this recording is not certain - a clip with only two CS
    presentations, or none at all - the CS window is widened to span both the local
    estimate and the study consensus, so the LED is inside it either way.  A recording
    whose camera was re-aimed part way through is covered by the same widening.
    """
    ax, ay = anchor
    b = {}
    if wants.get("yellow"):
        if confident or not consensus:
            b["yellow"] = (ax - CS_HALF, ay - CS_HALF, ax + CS_HALF, ay + CS_HALF)
        else:
            kx, ky = consensus
            b["yellow"] = (min(ax, kx) - CS_PAD, min(ay, ky) - CS_PAD,
                           max(ax, kx) + CS_PAD, max(ay, ky) + CS_PAD)
    if wants.get("blue"):
        m = US_BOX_INHERITED if not confident else US_BOX
        b["blue"] = (ax - m["left"], ay - m["up"], ax + m["right"], ay + m["down"])
    for k, (x0, y0, x1, y1) in list(b.items()):
        b[k] = (max(0, int(x0)), max(0, int(y0)), min(W, int(x1)), min(H, int(y1)))
    return b


def read_window(path, tag, wdir, sub, W, H, fps):
    """Full-rate, full-resolution signal: per frame, the strongest pixel in each sub-window."""
    f = os.path.join(wdir, tag + "_led.npz")
    if os.path.exists(f):
        return np.load(f)
    x0, y0, w, h = crop_box(min(b[0] for b in sub.values()), min(b[1] for b in sub.values()),
                            max(b[2] for b in sub.values()), max(b[3] for b in sub.values()), W, H)
    cols = {}
    for k, (bx0, by0, bx1, by1) in sub.items():
        cols[k] = (slice(max(0, by0 - y0), min(h, by1 - y0)),
                   slice(max(0, bx0 - x0), min(w, bx1 - x0)))
    fsz = w * h * 3
    sig = dict((k, []) for k in sub)
    pos = dict((k, []) for k in sub)
    t0 = time.time()
    n = 0
    for raw in frames(path, "crop=%d:%d:%d:%d" % (w, h, x0, y0), fsz):
        img = np.frombuffer(raw, np.uint8).reshape(h, w, 3)
        y, nb = _metrics(img)
        for k in sub:
            sy, sx = cols[k]
            m = (y if k == "yellow" else nb)[sy, sx]
            i = int(np.argmax(m))
            sig[k].append(m.flat[i])
            pos[k].append(i)
        n += 1
    log(tag, "read window %dx%d@(%d,%d): %d frames in %.0fs" % (w, h, x0, y0, n, time.time() - t0))
    kw = {}
    for k in sub:
        sy, sx = cols[k]
        kw["sig_" + k] = np.array(sig[k], np.int16)
        kw["pos_" + k] = np.array(pos[k], np.int32)
        kw["box_" + k] = np.array([x0 + sx.start, y0 + sy.start,
                                   sx.stop - sx.start, sy.stop - sy.start])
    np.savez_compressed(f, x0=x0, y0=y0, w=w, h=h, fps=fps, n=n, **kw)
    return np.load(f)


def lit_position(led, key, on_thr):
    """Where in its sub-window the LED actually was while lit, in full-res pixels."""
    p = led["pos_" + key]
    s = led["sig_" + key].astype(float)
    bx, by, bw, bh = [int(v) for v in led["box_" + key]]
    lit = s > on_thr
    if lit.sum() < 3:
        return None
    xs = (p[lit] % bw) + bx
    ys = (p[lit] // bw) + by
    return dict(x=int(np.median(xs)), y=int(np.median(ys)),
                spread_x=int(np.percentile(xs, 95) - np.percentile(xs, 5)),
                spread_y=int(np.percentile(ys, 95) - np.percentile(ys, 5)))


def main():
    cfg = C.load(sys.argv[1])
    tag = sys.argv[2]
    rec = next(r for r in cfg["recordings"] if r["tag"] == tag)
    wdir = work_dir(cfg)
    if os.path.exists(os.path.join(wdir, tag + "_stim.json")) and "--force" not in sys.argv:
        log(tag, "pulses already read")
        return
    proto = cfg["protocol"]
    path = rec["path"]
    W, H, fps, nfr = probe(path)
    log(tag, "%s  %dx%d  %.3f fps  %d frames  %.1fs  role=%s"
        % (rec["file"], W, H, fps, nfr, nfr / fps if fps else 0, rec["role"]))

    with open(os.path.join(wdir, "leds.json"), encoding="utf-8") as fh:
        _L = json.load(fh)
    LOC, CONSENSUS = _L["leds"], _L.get("consensus")
    if tag not in LOC:
        raise SystemExit(tag + ": not in leds.json - run ebc_locate.py first")
    spot = LOC[tag]
    anchor = (spot["x"], spot["y"])
    confident = bool(spot.get("confident")) or spot["source"] == "config"
    wants = {"yellow": rec["role"] != "baseline_us",
             "blue": rec["role"] in ("conditioning", "baseline_us")}
    sub = boxes(anchor, wants, W, H, confident, CONSENSUS)
    log(tag, "CS LED anchor (%d,%d) [%s]  windows %s"
        % (anchor[0], anchor[1], spot["source"],
           {k: "%dx%d@(%d,%d)" % (v[2] - v[0], v[3] - v[1], v[0], v[1]) for k, v in sub.items()}))

    led = read_window(path, tag, wdir, sub, W, H, fps)
    out = dict(tag=tag, file=rec["file"], label=rec["label"], role=rec["role"],
               order=rec.get("order", 1), width=W, height=H, fps=fps,
               n_frames=int(led["n"]), duration_s=round(int(led["n"]) / fps, 3),
               anchor=dict(x=anchor[0], y=anchor[1], source=spot["source"],
                           confident=confident),
               leds={}, events={}, warnings=[])
    for key, nom, tol in (("yellow", proto["cs_ms"], proto["cs_tol"]),
                          ("blue", proto["us_dur_ms"], proto["us_tol"])):
        if key not in sub:
            continue
        sig = led["sig_" + key].astype(float)
        gap = proto["min_iti_s"] if key == "yellow" else 0.0
        ev, info = detect(sig, fps, nom, tol=tol, min_gap_s=gap)
        st = pulse_stats(ev)
        entry = dict(box=[int(v) for v in led["box_" + key]], signal=info, stats=st)
        if info.get("on_threshold") is not None:
            entry["position"] = lit_position(led, key, info["on_threshold"])
        out["leds"][key] = entry
        out["events"][key] = [dict(frame=e["a"], t=round(e["t"], 4),
                                   dur_ms=round(e["dur_ms"], 1), ok=e["ok"],
                                   reason=e.get("reason", "")) for e in ev]
        log(tag, "  %-6s %s/%s pulses accepted  rest=%s lit=%s thr=%s"
            % (key, info.get("n_ok", 0), info.get("n_raw", 0), info.get("rest_level"),
               info.get("lit_level"), info.get("on_threshold")))
        log(tag, "         %s  at %s" % (st, entry.get("position")))
        # the checks that say "this signal is a stimulus LED, not a noisy patch of wall"
        if info.get("contrast", 0) < 60:
            out["warnings"].append("%s: weak contrast (%.0f) - LED may be mis-located"
                                   % (key, info.get("contrast", 0)))
        if st and st.get("iti_med_s") is not None and st["iti_med_s"] < 3.0:
            out["warnings"].append("%s: accepted pulses only %.2fs apart - this looks like "
                                   "noise, not stimuli" % (key, st["iti_med_s"]))
        if entry.get("position") and max(entry["position"]["spread_x"],
                                         entry["position"]["spread_y"]) > 40:
            out["warnings"].append("%s: lit pixel wanders %dx%d px - not a point source"
                                   % (key, entry["position"]["spread_x"],
                                      entry["position"]["spread_y"]))
    for w in out["warnings"]:
        log(tag, "  !! " + w)
    with open(os.path.join(wdir, tag + "_stim.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    log(tag, "-> " + tag + "_stim.json")


if __name__ == "__main__":
    main()
