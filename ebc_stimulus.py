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
the survey, indistinguishable from noise; instead it is looked for in a window around the
CS LED, since the two are centimetres apart on the same panel.  That window is symmetric:
which side of the CS LED the US LED appears on depends on how the box was turned and
where the camera stood, so it is measured per participant and never assumed.  What keeps
the wide window honest is that every pulse must come from the same spot as the others -
a flash 200 px away is a reflection, and is rejected as one.
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

# Window sizes are in pixels of a 1920-wide frame and scaled to whatever the recording
# actually is, so a camera at 2.7K or 4K does not silently read a window a third of the
# intended size.
REF_W = 1920.0
CS_HALF = 34              # half-size of the patch read around a confident CS LED
CS_PAD = 60               # extra margin when the position is uncertain

# Where the US LED may sit relative to the CS LED.  It is a few centimetres away on the
# same panel - but WHICH WAY is a property of how the box was turned and where the
# camera stood, not of the protocol, and it has to be allowed to differ from one
# participant, one session or one recording to the next.  So the window is symmetric:
# it reaches as far to the left as to the right, and as far up as down, and the side
# the LED is actually on is measured afterwards rather than assumed here.
#
# A window this size sees a good deal besides the LED, so a pulse is also required to
# come from the same spot as the others (`gate_positions`); that, not a tight window,
# is what keeps a reflection off a bottle from being read as a stimulus.
US_REACH = dict(x=190, y=110)
# a recording with no CS at all inherits the box position from the rest of the session,
# so its window has to allow for the camera having been re-aimed in between
US_REACH_INHERITED = dict(x=280, y=180)
# two pulses of one LED are this close together; anything further away in the window is
# something else that flashed
SAME_SPOT_PX = 45


def log(tag, *a):
    print("[%s]" % tag, *a, flush=True)


def known_us_offset(wdir, tag):
    """The CS-to-US offset already measured in another recording of this participant.

    The first recording to be read searches both sides of the CS LED.  Once one of them
    has found the US LED, the rest are told where to look - still a wide window, but
    centred on the answer instead of on a guess.  Recordings are read a few at a time,
    so this fills in as the stage goes and is complete by the second run.
    """
    seen = []
    for fn in sorted(os.listdir(wdir)):
        if not fn.endswith("_stim.json") or fn == tag + "_stim.json":
            continue
        try:
            with open(os.path.join(wdir, fn), encoding="utf-8") as fh:
                d = json.load(fh)
        except (OSError, ValueError):
            continue
        if d.get("us_offset"):
            seen.append(d["us_offset"])
    if not seen:
        return None
    return [int(np.median([o[0] for o in seen])), int(np.median([o[1] for o in seen]))]


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
def boxes(anchor, wants, W, H, confident, consensus=None, us_offset=None):
    """Sub-windows to read: a patch on the CS LED, and one around it for the US.

    When the box position for this recording is not certain - a clip with only two CS
    presentations, or none at all - the CS window is widened to span both the local
    estimate and the study consensus, so the LED is inside it either way.  A recording
    whose camera was re-aimed part way through is covered by the same widening.

    The US window is centred on the CS LED and symmetric, so it does not care which side
    of the panel the US LED is on.  Once another recording of the same participant has
    *measured* that offset, `us_offset` recentres the window on where the LED will be -
    the search stays as wide, it just no longer has to be blind.
    """
    ax, ay = anchor
    s = max(W / REF_W, 0.5)
    b = {}
    if wants.get("yellow"):
        half = CS_HALF * s
        if confident or not consensus:
            b["yellow"] = (ax - half, ay - half, ax + half, ay + half)
        else:
            kx, ky = consensus
            pad = CS_PAD * s
            b["yellow"] = (min(ax, kx) - pad, min(ay, ky) - pad,
                           max(ax, kx) + pad, max(ay, ky) + pad)
    if wants.get("blue"):
        m = US_REACH_INHERITED if not confident else US_REACH
        rx, ry = m["x"] * s, m["y"] * s
        cx, cy = ax, ay
        if us_offset:                      # measured elsewhere in this participant
            cx, cy = ax + us_offset[0], ay + us_offset[1]
            rx, ry = max(rx * 0.6, 70 * s), max(ry * 0.6, 70 * s)
        b["blue"] = (min(ax, cx) - rx, min(ay, cy) - ry, max(ax, cx) + rx, max(ay, cy) + ry)
    for k, (x0, y0, x1, y1) in list(b.items()):
        b[k] = (max(0, int(x0)), max(0, int(y0)), min(W, int(x1)), min(H, int(y1)))
    return b


def read_window(path, tag, wdir, sub, W, H, fps):
    """Full-rate, full-resolution signal: per frame, the strongest pixel in each sub-window."""
    f = os.path.join(wdir, tag + "_led.npz")
    if os.path.exists(f):
        # A cache is a convenience, never a dependency.  This one is written at the end
        # of a pass that takes ten minutes, so an interrupted run - Ctrl-C, a timeout, a
        # full disk - leaves a truncated file behind, and numpy raises BadZipFile on it.
        # Reading the video again costs time; refusing to run costs the participant.
        try:
            cached = np.load(f)
            cached.files
        except Exception as e:
            log(tag, "  cached window read is unusable (%s) - reading the recording again"
                % type(e).__name__)
            try:
                os.remove(f)
            except OSError:
                pass
            cached = None
    else:
        cached = None
    if cached is not None:
        # The cache is keyed on the tag alone, but what it holds is one particular pair
        # of windows.  They can legitimately differ between runs - a measured US offset
        # aims the blue window, a re-run of ebc_locate moves the anchor - and reusing the
        # old read while the log prints the new window is a quiet lie about what was
        # measured.  So the windows are stored with the signal and checked here.
        want = {k: [int(v) for v in b] for k, b in sub.items()}
        got = {k[4:]: [int(v) for v in cached[k]] for k in cached.files
               if k.startswith("req_")}
        if not got:
            log(tag, "  cached window read predates this check - reusing it; "
                     "pass --force if the LED positions have changed")
            return cached
        if got == want:
            return cached
        log(tag, "  the window has changed since this recording was read "
                 "(%s -> %s); reading it again" % (got, want))
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
    for k, b in sub.items():
        kw["req_" + k] = np.array([int(v) for v in b])
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


def event_positions(led, key, events):
    """Where each pulse lit up, in full-resolution frame coordinates.

    Not the brightest pixel of the peak frame, which is what this used to be.  A bright
    LED SATURATES - Marie's US LED sits at 255 for the whole pulse - and as soon as any
    other pixel in the window saturates too, the two tie and `argmax` picks between them
    on raster order, i.e. arbitrarily.  The position then jumps 200 px between one real
    pulse and the next, and `gate_positions` throws the pulse out for lighting up in the
    wrong place.  On Marie that discarded a quarter of her puffs and turned every one of
    those paired trials into a CS-only probe: 36 probes recovered where the protocol has
    10.

    So take the whole pulse, not one frame of it, and return the medoid - the observed
    position closest to all the others.  A tie that wins on some frames and loses on the
    rest cannot carry the answer, and because the medoid is one of the observed points it
    is always somewhere the window actually lit up, never an average of two places.
    """
    p = led["pos_" + key]
    bx, by, bw, _ = [int(v) for v in led["box_" + key]]
    n = len(p)
    out, jitter = [], []
    for e in events:
        a, b = int(e["a"]), int(e["b"])
        b = max(a, min(b, n - 1))
        idx = p[a:b + 1]
        pts = [(bx + int(i) % bw, by + int(i) // bw) for i in idx]
        if len(pts) == 1:
            out.append(pts[0])
            jitter.append(0.0)
            continue
        xs = np.array([q[0] for q in pts], float)
        ys = np.array([q[1] for q in pts], float)
        # medoid: the observed point with the smallest total distance to the others
        d = np.abs(xs[:, None] - xs[None, :]) + np.abs(ys[:, None] - ys[None, :])
        out.append(pts[int(np.argmin(d.sum(axis=1)))])
        jitter.append(float(max(np.ptp(xs), np.ptp(ys))))
    return out, jitter


def gate_positions(events, pos, radius, key, min_ref=3):
    """Reject pulses that lit up somewhere else in the window.

    An LED does not move.  Once a handful of pulses agree on a spot, a "pulse" 200 px
    away is a reflection, a screen, a phone or someone's white sleeve catching the sun -
    and with a window wide enough to find the US LED on either side of the CS LED, there
    is room in it for all of those.  Rejecting on position is what lets the window be
    that wide: the alternative is a tight window that assumes which side the LED is on.

    The reference spot is the densest cluster of accepted pulses, not their mean, so a
    few strays cannot drag it off the LED.  Nothing is deleted - a rejected pulse keeps
    its reason and still appears in stimulus_events.csv.
    """
    ok = [i for i, e in enumerate(events) if e["ok"]]
    if len(ok) < min_ref:
        return None, 0
    best, best_n = None, -1
    for i in ok:
        x, y = pos[i]
        n = sum(1 for j in ok
                if abs(pos[j][0] - x) <= radius and abs(pos[j][1] - y) <= radius)
        if n > best_n:
            best, best_n = i, n
    if best_n < min_ref:
        return None, 0
    near = [j for j in ok if abs(pos[j][0] - pos[best][0]) <= radius
            and abs(pos[j][1] - pos[best][1]) <= radius]
    cx = int(np.median([pos[j][0] for j in near]))
    cy = int(np.median([pos[j][1] for j in near]))
    n_cut = 0
    for j in ok:
        d = ((pos[j][0] - cx) ** 2 + (pos[j][1] - cy) ** 2) ** 0.5
        if d > radius:
            events[j]["ok"] = False
            events[j]["reason"] = ("lit %d px from where the %s LED is - a reflection or "
                                   "another light, not the stimulus" % (d, key))
            n_cut += 1
    return (cx, cy), n_cut


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
    LOC = _L["leds"]
    if tag not in LOC:
        raise SystemExit(tag + ": not in leds.json - run ebc_locate.py first")
    spot = LOC[tag]
    # Widening the CS window has to reach towards where the box was in THIS recording's
    # part of the session, not towards a study-wide average.  ebc_locate works that out
    # per recording (`near_xy`, the nearest cluster in time); the study-wide figure is
    # only the fallback for a leds.json written before it did.
    CONSENSUS = spot.get("near_xy") or _L.get("consensus")
    anchor = (spot["x"], spot["y"])
    confident = bool(spot.get("confident")) or spot["source"] == "config"
    # The blue channel is read for every role, including the ones the protocol says
    # deliver no US.  It costs one wider crop and it is the only way to notice that a
    # recording labelled `extinction` is in fact full of puffs - which is what a
    # mislabelled file looks like from the outside.  It changes no trial: ebc_protocol.py
    # still builds CS-only trials for those roles whatever the blue channel saw.
    wants = {"yellow": rec["role"] != "baseline_us", "blue": True}
    # if another recording of this participant has already measured which side of the CS
    # LED the US LED sits on, start from there; otherwise search both sides equally
    us_off = rec.get("us_offset") or known_us_offset(wdir, tag)
    sub = boxes(anchor, wants, W, H, confident, CONSENSUS, us_off)
    log(tag, "CS LED anchor (%d,%d) [%s]  windows %s%s"
        % (anchor[0], anchor[1], spot["source"],
           {k: "%dx%d@(%d,%d)" % (v[2] - v[0], v[3] - v[1], v[0], v[1]) for k, v in sub.items()},
           "" if not us_off else "  US LED expected %+d,%+d from the CS LED" % tuple(us_off)))

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
        # one LED, one place: a pulse from elsewhere in the window is not this stimulus
        pos, jitter = event_positions(led, key, ev)
        radius = SAME_SPOT_PX * max(W / REF_W, 0.5)
        # Rejecting a pulse for lighting up in the wrong place assumes there IS a right
        # place - that the channel has a point source whose position one pulse can
        # measure.  Where the brightest pixel wanders as far WITHIN a single pulse as it
        # does between pulses, the position carries no information and the gate is not a
        # filter, it is a coin toss.  Marie's blue channel is like that (the peak moves
        # 85-170 px inside one 58 ms puff, against 8-20 px for every channel where the
        # LED really is a point), and gating it threw away a quarter of her puffs and
        # turned those paired trials into CS-only probes.  So measure first, and where
        # the position is meaningless keep every pulse and say that reflections cannot
        # be screened out here.
        steady = float(np.median(jitter)) if jitter else 0.0
        if steady >= radius:
            spot_xy, n_moved = None, 0
            out["warnings"].append(
                "%s: no point source - the brightest pixel moves %d px within a single "
                "pulse, so pulses cannot be screened on position and a reflection in "
                "this window would be read as a stimulus.  Check qc_leds_%s.png"
                % (key, int(steady), tag))
            log(tag, "  %-6s position is not usable (moves %d px within one pulse) - "
                     "keeping every pulse, no position screening" % (key, int(steady)))
        else:
            spot_xy, n_moved = gate_positions(ev, pos, radius, key)
        if n_moved:
            info["n_ok"] = sum(e["ok"] for e in ev)
            log(tag, "  %-6s %d pulse(s) rejected: lit somewhere else in the window"
                % (key, n_moved))
        st = pulse_stats(ev)
        entry = dict(box=[int(v) for v in led["box_" + key]], signal=info, stats=st)
        if info.get("on_threshold") is not None:
            entry["position"] = lit_position(led, key, info["on_threshold"])
        if spot_xy:
            # The spread has to be recomputed from the pulses that were KEPT.  The one
            # lit_position() measured is over every lit pixel in the window, including
            # the reflections the gate has just thrown out - carrying it over would
            # describe a scatter that is no longer there and raise "not a point source"
            # against an LED that is one.
            kept = [q for q, e in zip(pos, ev) if e["ok"]]
            entry["position"] = dict(entry.get("position") or {},
                                     x=spot_xy[0], y=spot_xy[1], source="accepted pulses",
                                     spread_x=int(max(q[0] for q in kept) -
                                                  min(q[0] for q in kept)) if kept else 0,
                                     spread_y=int(max(q[1] for q in kept) -
                                                  min(q[1] for q in kept)) if kept else 0)
            entry["n_off_spot"] = n_moved
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
    # Which side of the CS LED the US LED turned out to be on.  Nothing assumes it; it
    # is measured here, carried to the other recordings of the participant so their
    # windows can be aimed rather than searched, and checked across them by ebc_triage.
    y_pos = (out["leds"].get("yellow") or {}).get("position")
    u_pos = (out["leds"].get("blue") or {}).get("position")
    if y_pos and u_pos and y_pos.get("x") is not None and u_pos.get("x") is not None:
        dx, dy = u_pos["x"] - y_pos["x"], u_pos["y"] - y_pos["y"]
        out["us_offset"] = [int(dx), int(dy)]
        out["us_side"] = ("left" if dx < 0 else "right") if abs(dx) >= abs(dy) else                          ("above" if dy < 0 else "below")
        log(tag, "  US LED sits %d px to the %s of the CS LED (offset %+d,%+d)"
            % (max(abs(dx), abs(dy)), out["us_side"], dx, dy))
        if max(abs(dx), abs(dy)) > 0.16 * W:
            out["warnings"].append(
                "the two LEDs are %d px apart, which is far for one panel - check "
                "qc_leds_%s.png that both markers are on the box" % (max(abs(dx), abs(dy)), tag))
    for w in out["warnings"]:
        log(tag, "  !! " + w)
    with open(os.path.join(wdir, tag + "_stim.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    log(tag, "-> " + tag + "_stim.json")


if __name__ == "__main__":
    main()
