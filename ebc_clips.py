"""Cut a short video of one prototypical trial of each kind, with the numbers on it.

    python ebc_clips.py <config.json>

Four clips per participant - a CR, a ?CR, a CS-only probe and a UR - each showing the
recording's own pixels with the eye ratio that was measured off them, the eyelid trace
being drawn as it happens, and the running blink count.  They exist to answer the only
question a table of onsets cannot: does the number the scorer produced match what the
video shows?  Somebody who has never read this code should be able to watch six seconds
and say yes or no.

Everything drawn is read back from the run, never recomputed here.  The crop is the one
ebc_eyes measured in (ebc_eyes.geometry), the eye ratios are the per-frame values it
wrote, the closure trace is the one ebc_score pooled and scored, and the class, onset and
blink count are the row that went into the workbook.  A clip that showed anything else
would be a second opinion rather than evidence.

Played at a quarter speed: the whole trial is under a second and a half at 119.88 fps,
and a blink at real time is a frame or two of nothing much.
"""
import os
import sys
import csv
import json
import subprocess

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import ebc_config as C
import ebc_eyes
import ebc_score                 # for the one blink criterion, not to re-run any of it
from ebc_paths import work_dir, out_dir
from ebc_video import probe, frames

P = C.PALETTE
OUT_FPS = 30.0                  # what the file plays at
# How fast the trial runs on screen, as a fraction of real time.  It has to be stated
# rather than inherited: one output frame per recorded frame gives 0.25x on a 119.88 fps
# camera and 0.15x on Charles's 200 fps one, so "the clips" were not one speed at all.
# 0.125 puts a 60 ms lid closure on screen for half a second, which is what makes the
# blink watchable rather than merely present.  --speed 0.5 / 0.25 / 0.1 overrides it.
SPEED = 0.125
HOLD_S = 0.9                    # a still at the end, so the last frame can be read
W, H = 1280, 800
HEAD_H, MID_H, PLOT_H = 104, 452, 208
VID_W = 830                     # the video panel; the readouts take the rest
PAD = 14


# ------------------------------------------------------------------ which trials
# One clip per kind, in the order somebody would want to watch them: what a conditioned
# response looks like, then the case the scorer will not call either way, then the same
# person with no puff to react to, then the reflex the whole thing is measured against.
#
# `prefix` is matched against the start of the class string, which is how every module
# here recognises a class.  `roles` is the order recordings are searched in: a
# conditioning trial is the one worth showing, and a clip drawn from extinction says so
# on its own header rather than pretending otherwise.
ROLE_WORD = {"conditioning": "conditioning", "extinction": "extinction",
             "baseline_cs": "CS-only baseline", "baseline_us": "US-only baseline"}

KINDS = [
    ("CR", "CR", "CS-US", ("conditioning",),
     "A conditioned response: the blink began after the CS and before the puff could "
     "have caused it."),
    ("qCR", "?CR", "CS-US", ("conditioning", "extinction"),
     "The uncertain band: the blink began after the puff arrived but while the CS was "
     "still on, so it is neither a definitive CR nor definitely a reaction to the puff."),
    ("CSonly", None, "CS-only", ("conditioning", "extinction", "baseline_cs"),
     "A CS-only trial: no puff was delivered, so any blink here is a response to the CS "
     "alone."),
    ("UR", "UR", "CS-US", ("conditioning",),
     "A reaction to the puff: the blink began late enough that the puff itself can have "
     "caused it."),
]


def candidates(rows, prefix, trial_type, roles, win):
    """Trials of one kind that a clip could honestly be cut from.

    Only trials the scorer stands behind: a clip of a trial marked 'score by hand' would
    be showing the reader a number the run itself does not vouch for.
    """
    out = []
    for r in rows:
        if trial_type and r["trial_type"] != trial_type:
            continue
        if r["role"] not in roles:
            continue
        if r["needs_manual_scoring"] or r["scored_onset_ms"] is None:
            continue
        if not C.is_scoreable(r["scored_class"], win):
            continue
        if prefix and not str(r["scored_class"]).startswith(prefix):
            continue
        if prefix is None and not C.is_response(r["scored_class"], win):
            continue
        out.append(r)
    return out


def prototypical(cand, roles):
    """The most ORDINARY member of a set, which is not the most striking one.

    A clip picked for being impressive teaches the reader what the best case looks like
    and nothing about what was scored.  So the score below is a distance from the middle
    of the set - how far this trial's onset is from the median onset, in robust SDs of
    the set itself - and everything added to it is a reason to distrust the trial rather
    than a reason to prefer it: a shallower blink than the set's typical one, a face that
    was not tracked the whole way, a quality flag, a recording further down the role
    order.  Lowest score wins, and the margin is reported so a near-tie is visible.
    """
    on = np.array([r["scored_onset_ms"] for r in cand], float)
    pk = np.array([r["peak_closure_pct"] or 0.0 for r in cand], float)
    med = float(np.median(on))
    sd = float(1.4826 * np.median(np.abs(on - med))) or float(np.std(on)) or 1.0
    pmed = max(float(np.median(pk)), 1.0)
    score = []
    for r, o, p in zip(cand, on, pk):
        s = abs(o - med) / sd
        s += 1.5 * max(0.0, (pmed - p) / pmed)              # a shallow blink is unclear
        # Below the app's own full-blink criterion it is not a blink, it is a lid
        # flicker that happened to be the first event in the window.  Such a trial is
        # scored like any other and belongs in the tables; putting it on a clip captioned
        # "this is what a response looks like" is a different claim, and a false one.
        # The penalty is larger than a whole role step so a clear probe from extinction
        # beats a marginal one from conditioning - which is the only case it decides.
        s += 2.5 if p < ebc_score.MAIN * 100.0 else 0.0
        s += 0.05 * max(0.0, 99.0 - (r["face_tracked_pct"] or 0.0))
        s += 0.0 if r["quality"] == "clean" else 1.0
        s += 2.0 * roles.index(r["role"])                   # conditioning first
        score.append(s)
    k = int(np.argmin(score))
    order = sorted(range(len(cand)), key=lambda i: score[i])
    why = ("closest to the middle of the %d trial(s) this participant has of this kind: "
           "onset %.0f ms against a median of %.0f, peak closure %.0f%% against %.0f%%, "
           "face tracked %.0f%% of the window"
           % (len(cand), on[k], med, pk[k], pmed, cand[k]["face_tracked_pct"] or 0.0))
    runner = (score[order[1]] - score[order[0]]) if len(order) > 1 else None
    return cand[k], why, runner


# ------------------------------------------------------------------ drawing
def font(size, bold=False):
    """The house face, or the nearest thing this machine has."""
    for name in (["seguisb.ttf", "segoeui.ttf"] if bold else ["segoeui.ttf"]) + \
                (["arialbd.ttf"] if bold else ["arial.ttf"]) + ["DejaVuSans.ttf"]:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def rgb(name):
    h = P[name].lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def mix(a, b, f):
    return tuple(round(x + (y - x) * f) for x, y in zip(rgb(a), rgb(b)))


def cls_colour(cls):
    """The same colour a class is drawn in everywhere else in this app."""
    c = str(cls)
    if c.startswith("?CR"):
        return rgb("us_mid")
    if c.startswith("CR"):
        return rgb("cr")
    if c.startswith("alpha"):
        return rgb("cs")
    if c.startswith("UR"):
        return rgb("ur")
    return rgb("muted")


def fit(img, box_w, box_h):
    """Scale a frame into its panel without distorting it, and say where it landed."""
    k = min(box_w / img.width, box_h / img.height)
    w, h = max(1, int(img.width * k)), max(1, int(img.height * k))
    return img.resize((w, h), Image.LANCZOS), k


class Plot:
    """The eyelid trace panel: one transform, computed once, used by every frame."""

    def __init__(self, t, win, us0, cs_ms, us_anchored):
        self.x0, self.x1 = PAD + 92, W - PAD - 12
        self.y0, self.y1 = HEAD_H + MID_H + 26, HEAD_H + MID_H + PLOT_H - 34
        self.t0, self.t1 = float(t[0]), float(t[-1])
        self.win, self.us0, self.cs_ms, self.us_anchored = win, us0, cs_ms, us_anchored

    def px(self, ms):
        f = (float(ms) - self.t0) / max(self.t1 - self.t0, 1e-9)
        return self.x0 + f * (self.x1 - self.x0)

    def py(self, pct):
        f = min(max(float(pct), -10.0), 110.0) / 100.0
        return self.y1 - f * (self.y1 - self.y0)

    def background(self, d, row):
        """Everything that does not move: the bands, the stimuli, the axes."""
        d.rectangle([self.x0, self.y0, self.x1, self.y1], fill=rgb("surface"),
                    outline=rgb("rule"))
        w = self.win
        if not self.us_anchored:
            # the CR window, drawn where it is rather than described in a caption
            d.rectangle([self.px(w["lo_ms"]), self.y0, self.px(w["hi_ms"]), self.y1],
                        fill=mix("surface", "cr", .10))
            d.text(((self.px(w["lo_ms"]) + self.px(w["hi_ms"])) / 2, self.y0 + 4),
                   "CR window", font=font(12, True), fill=rgb("cr"), anchor="ma")
            if w.get("qcr_hi_ms"):
                d.rectangle([self.px(w["hi_ms"]), self.y0, self.px(w["qcr_hi_ms"]),
                             self.y1], fill=mix("surface", "us_mid", .13))
                if self.px(w["qcr_hi_ms"]) - self.px(w["hi_ms"]) > 34:
                    d.text(((self.px(w["hi_ms"]) + self.px(w["qcr_hi_ms"])) / 2,
                            self.y0 + 20), "?CR", font=font(12, True),
                           fill=rgb("us_mid"), anchor="ma")
        for pct in (0, 50, 100):
            y = self.py(pct)
            d.line([self.x0, y, self.x1, y], fill=rgb("grid"))
            d.text((self.x0 - 8, y), "%d%%" % pct, font=font(13), fill=rgb("muted"),
                   anchor="rm")
        marks = ([(0.0, "US", "us")] if self.us_anchored
                 else [(0.0, "CS on", "cs"), (self.us0, "US", "us"),
                       (self.cs_ms, "CS off", "cs")])
        for ms, lbl, colr in marks:
            if not (self.t0 <= ms <= self.t1):
                continue
            x = self.px(ms)
            d.line([x, self.y0, x, self.y1], fill=rgb(colr), width=2)
            d.text((x + 4, self.y0 + 3), lbl, font=font(13, True), fill=rgb(colr))
        if row["scored_onset_ms"] is not None and self.t0 <= row["scored_onset_ms"] <= self.t1:
            x = self.px(row["scored_onset_ms"])
            colr = cls_colour(row["scored_class"])
            d.line([x, self.y0, x, self.y1], fill=colr, width=3)
            d.text((x + 5, self.y1 - 17), "scored onset  %.0f ms" % row["scored_onset_ms"],
                   font=font(13, True), fill=colr)
        # the last tick is dropped when it would sit on top of the axis label
        ticks = [self.t0, 0.0, 250.0, 500.0, 750.0, 1000.0, 1250.0, 1500.0]
        for ms in ticks:
            if self.t0 <= ms <= self.t1 - 90:
                d.text((self.px(ms), self.y1 + 6), "%d" % round(ms), font=font(12),
                       fill=rgb("muted"), anchor="ma")
        d.text((self.x0 - 8, self.y0 - 20), "eyelid closure", font=font(13, True),
               fill=rgb("ink"), anchor="lm")
        d.text((self.x1, self.y1 + 6), "ms from %s onset"
               % ("US" if self.us_anchored else "CS"), font=font(12),
               fill=rgb("muted"), anchor="ra")

    def trace(self, d, t, closure, i):
        """The trace: faint where it has not been reached yet, solid where it has."""
        pts = [(self.px(tv), self.py(cv * 100.0)) for tv, cv in zip(t, closure)]
        d.line(pts, fill=rgb("link"), width=2, joint="curve")
        if i >= 1:
            d.line(pts[:i + 1], fill=rgb("trace"), width=3, joint="curve")
        x, y = pts[min(i, len(pts) - 1)]
        d.ellipse([x - 5, y - 5, x + 5, y + 5], fill=rgb("ink"))
        d.line([x, self.y0, x, self.y1], fill=rgb("ink"), width=1)


def led_signal(A, sl, kind):
    """How strongly one LED is lit in every frame - the same expression ebc_eyes uses.

    Frames arrive from ffmpeg as BGR, so channel 0 is blue.  Yellow is measured as the
    green-and-red mean above blue and the US LED as blue above it; a pixel that is not
    bright at all is floored, so a dark corner of the patch cannot win the maximum.
    """
    patch = A[:, sl[0], sl[1], :].astype(np.int16)
    half = (patch[:, :, :, 1] + patch[:, :, :, 2]) >> 1
    m = (half - patch[:, :, :, 0]) if kind == "yellow" else (patch[:, :, :, 0] - half)
    m = np.where(A[:, sl[0], sl[1], :].max(axis=3) >= 150, m, -128)
    return m.reshape(len(A), -1).max(axis=1).astype(float)


def other_led(geo, cw, ch):
    """Where the LED this trial was NOT anchored on sits inside the same crop.

    The crop was built around the anchor LED, so the other one is usually just inside it
    and sometimes half out.  Returned only when a patch of it is genuinely in frame:
    a lamp driven by four pixels of the edge of an LED would be a guess dressed as a
    measurement, and nothing at all is better than that.
    """
    kind = "blue" if geo["anchor_led"] == "yellow" else "yellow"
    led = (geo["stim"]["leds"] or {}).get(kind) or {}
    pos = led.get("position")
    if pos:
        lx, ly = int(pos["x"]), int(pos["y"])
    elif led.get("box"):
        bx, by, bw, bh = led["box"]
        lx, ly = bx + bw // 2, by + bh // 2
    else:
        return None
    cx0, cy0 = geo["crop"][0], geo["crop"][1]
    h = geo["led_half"]
    sl = (slice(max(0, ly - h - cy0), min(ch, ly + h - cy0)),
          slice(max(0, lx - h - cx0), min(cw, lx + h - cx0)))
    if (sl[0].stop - sl[0].start) < h or (sl[1].stop - sl[1].start) < h:
        return None
    return sl


def lit_frames(sig, pre, rest=None, hot=None):
    """Which frames that LED is on in.  Same 55%-of-the-rise rule as the alignment."""
    if not len(sig):
        return np.zeros(0, bool)
    rest = float(np.percentile(sig[:max(pre // 2, 5)], 50)) if rest is None else rest
    hot = float(sig.max()) if hot is None else hot
    if hot - rest <= 25:
        return np.zeros(len(sig), bool)
    return sig > rest + 0.55 * (hot - rest)


def readouts(d, x, y, row, er, el, closure, t_ms, nblinks, total, lamps):
    """The numbers, beside the picture they were measured from."""
    f_lbl, f_big, f_sm = font(14), font(34, True), font(15)
    d.text((x, y), "EYE RATIO  (EAR, the measure that was scored)", font=font(13, True),
           fill=rgb("accent"))
    y += 24
    for name, v in (("right eye", er), ("left eye", el),
                    ("mean  ->  scored", None if er is None and el is None
                     else np.nanmean([v for v in (er, el) if v is not None]))):
        d.text((x, y), name, font=f_lbl, fill=rgb("muted"))
        d.text((x + 250, y), "%.4f" % v if v is not None and np.isfinite(v) else "not tracked",
               font=f_sm if v is not None else font(14, True),
               fill=rgb("ink") if v is not None else rgb("alert"), anchor="ra")
        y += 22
    y += 10
    d.text((x, y), "EYELID CLOSURE", font=font(13, True), fill=rgb("accent"))
    d.text((x + 250, y - 6), "%.0f %%" % (closure * 100.0), font=f_big,
           fill=rgb("ink"), anchor="ra")
    y += 44
    bar_w = 250
    d.rectangle([x, y, x + bar_w, y + 12], fill=rgb("sunken"), outline=rgb("rule"))
    d.rectangle([x, y, x + bar_w * min(max(closure, 0.0), 1.0), y + 12],
                fill=rgb("trace"))
    y += 30
    d.text((x, y), "BLINKS COUNTED", font=font(13, True), fill=rgb("accent"))
    d.text((x + 250, y - 6), "%d of %d" % (nblinks, total), font=f_big,
           fill=rgb("ink") if nblinks else rgb("faint"), anchor="ra")
    y += 46
    d.text((x, y), "time from stimulus", font=f_lbl, fill=rgb("muted"))
    d.text((x + 250, y), "%+.0f ms" % t_ms, font=font(16, True), fill=rgb("ink"),
           anchor="ra")
    y += 30
    # The stimulus LEDs, read off these very pixels rather than taken from the protocol:
    # this is the row that says the trial really is where the analysis put it.
    d.text((x, y), "STIMULUS LEDS, read off this frame", font=font(13, True),
           fill=rgb("accent"))
    y += 24
    for name, on, colr in lamps:
        d.text((x + 26, y), name, font=f_lbl, fill=rgb("ink") if on else rgb("muted"))
        d.ellipse([x, y + 1, x + 17, y + 18],
                  fill=rgb(colr) if on else rgb("sunken"), outline=rgb("rule"))
        d.text((x + 250, y), "on" if on else "off", font=font(14, True),
               fill=rgb(colr) if on else rgb("faint"), anchor="ra")
        y += 24
    return y


def header(d, study, row, kind_note, source_note):
    d.rectangle([0, 0, W, HEAD_H], fill=rgb("surface"))
    d.line([0, HEAD_H, W, HEAD_H], fill=rgb("rule"), width=2)
    colr = cls_colour(row["scored_class"])
    d.rectangle([0, 0, 8, HEAD_H], fill=colr)
    d.text((PAD + 10, 12), "%s   %s   trial %d of the recording%s"
           % (study, row["session_name"], row["session_trial"],
              "   block %s" % row["block"] if row["block"] else ""),
           font=font(15), fill=rgb("muted"))
    d.text((PAD + 10, 34), "%s   -   scored onset %.0f ms"
           % (row["scored_class"], row["scored_onset_ms"]), font=font(23, True),
           fill=colr)
    d.text((PAD + 10, 66), kind_note, font=font(13), fill=rgb("ink"))
    d.text((W - PAD, 12), source_note, font=font(12), fill=rgb("muted"), anchor="ra")
    d.text((W - PAD, 34), "peak closure %.0f%%   ·   %d blink(s) in the window   ·   "
                          "face tracked %.0f%%"
           % (row["peak_closure_pct"] or 0.0, row["n_full_blinks"] or 0,
              row["face_tracked_pct"] or 0.0),
           font=font(12), fill=rgb("muted"), anchor="ra")


def footer(d, text):
    d.rectangle([0, H - 36, W, H], fill=rgb("sunken"))
    d.line([0, H - 36, W, H - 36], fill=rgb("rule"))
    d.text((PAD, H - 18), text, font=font(12), fill=rgb("muted"), anchor="lm")


# ------------------------------------------------------------------ one clip
def playback(n, fps, speed):
    """Which rendered frame each output frame shows, for a chosen speed.

    The overlay is drawn once per RECORDED frame - that is the expensive part and it is
    the same picture however slowly it is played - and the speed is a matter of which of
    those frames each output frame repeats.  Time-based rather than an integer repeat
    count, so any speed works and none of them drifts.
    """
    n_out = max(1, int(round(n / fps / speed * OUT_FPS)))
    return [min(n - 1, int(round(i * (n - 1) / max(n_out - 1, 1)))) for i in range(n_out)]


def encode(path, frames_of):
    """Hand the finished frames to ffmpeg.  H.264 if it has it, MPEG-4 if it does not.

    `frames_of` is called to get a fresh iterator, because the fallback codec has to send
    them all again - and because the frames are yielded rather than held twice.
    """
    for codec, extra in (("libx264", ["-preset", "medium", "-crf", "18",
                                      "-pix_fmt", "yuv420p", "-movflags", "+faststart"]),
                         ("mpeg4", ["-q:v", "3"])):
        cmd = ["ffmpeg", "-v", "error", "-nostdin", "-y", "-f", "rawvideo",
               "-pix_fmt", "rgb24", "-s", "%dx%d" % (W, H), "-r", "%g" % OUT_FPS,
               "-i", "-", "-c:v", codec] + extra + [path]
        p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            for im in frames_of():
                p.stdin.write(im.tobytes())
            p.stdin.close()
        except (BrokenPipeError, OSError):
            pass
        err = p.stderr.read().decode("utf-8", "replace").strip()
        p.stderr.close()
        p.wait()
        if p.returncode == 0 and os.path.exists(path) and os.path.getsize(path) > 1000:
            return codec
        if codec == "libx264":
            print("   H.264 not available (%s) - falling back to MPEG-4"
                  % (err.splitlines()[-1] if err else "no reason given"), flush=True)
        else:
            raise SystemExit("!! ffmpeg could not write %s: %s" % (path, err))
    return None


def render(cfg, M, row, kind, note, out_path, wdir, win, why, speed=SPEED):
    """One clip: the trial's own frames, with what was measured off them drawn on."""
    tag = row["session"]
    rec = next(r for r in cfg["recordings"] if r["tag"] == tag)
    Wf, Hf, fps, nfr = probe(rec["path"])
    geo = ebc_eyes.geometry(cfg, rec, wdir, Wf, Hf, fps, nfr)
    cx0, cy0, cw, ch = geo["crop"]

    with open(os.path.join(wdir, tag + "_traces.json"), encoding="utf-8") as fh:
        TJ = json.load(fh)[str(row["session_trial"])]
    TRC = M["traces"][tag][str(row["session_trial"])]
    t, closure = TRC["t"], TRC["C"]
    er, el = TJ["er"], TJ["el"]
    n = min(TJ["n"], len(t), len(closure), len(er), len(el))
    f0, pre = TJ["anchor_frame"], TJ["pre"]

    proto = C.fill(M["protocol"])
    us_anchored = row["trial_type"] == "US-only"
    plot = Plot(t[:n], win, proto["us_onset_ms"], proto["cs_ms"], us_anchored)
    # Every blink the scorer found, so the count on screen rises on the frames it rose on
    onsets = [float(x) for x in str(row["all_blink_onsets_ms"] or "").split(";") if x]
    total = int(row["n_full_blinks"] or 0)
    source = "%s  ·  %s  ·  aligned on the %s LED" % (
        os.path.basename(rec["file"]), ROLE_WORD.get(rec["role"], rec["role"]),
        geo["anchor_led"])
    foot = ("PLAYING AT x%g OF REAL TIME (%.0fx slow motion).  Window %+.0f to %+.0f ms "
            "around the stimulus, %d frames recorded at %.2f fps.  Every number here is "
            "read back from the run, not recomputed for this clip.  EBC Analyzer %s"
            % (speed, 1.0 / speed, t[0], t[n - 1], n, fps, C.VERSION))

    buf = list(frames(rec["path"], "crop=%d:%d:%d:%d" % (cw, ch, cx0, cy0),
                      cw * ch * 3, ss=(f0 - pre) / fps, n=n))
    if len(buf) < n * 0.85:
        print("   only %d of %d frames came back for %s trial %d - clip skipped"
              % (len(buf), n, tag, row["session_trial"]), flush=True)
        return None
    n = min(n, len(buf))

    A = np.stack([np.frombuffer(b, np.uint8).reshape(ch, cw, 3) for b in buf[:n]])
    # The anchor LED, read with the same expression and the same rest / lit thresholds
    # ebc_eyes aligned this trial on, so the lamp lights on exactly the frames the
    # alignment was taken from rather than on a second opinion about them.
    boxes = [(geo["LS"], geo["anchor_led"],
              lit_frames(led_signal(A, geo["LS"], geo["anchor_led"]), pre,
                         float(TJ["led_rest"]), float(TJ["led_lit"])))]
    # And the other LED where the crop happens to contain it - it usually does, the two
    # sit ~44 px apart.  Seeing the puff arrive is most of the point of a UR clip, and
    # this is the puff itself rather than the protocol's idea of when it should be.
    other = other_led(geo, cw, ch)
    if other is not None:
        kind_o = "blue" if geo["anchor_led"] == "yellow" else "yellow"
        boxes.append((other, kind_o,
                      lit_frames(led_signal(A, other, kind_o), pre)))
    LAMP = {"yellow": ("CS  ·  yellow LED", "cs"), "blue": ("US  ·  blue LED", "us")}

    base = Image.new("RGB", (W, H), rgb("paper"))
    bd = ImageDraw.Draw(base)
    bd.rectangle([0, HEAD_H, W, HEAD_H + MID_H], fill=rgb("ink"))
    bd.rectangle([VID_W, HEAD_H, W, HEAD_H + MID_H], fill=rgb("surface"))
    bd.rectangle([0, HEAD_H + MID_H, W, H - 36], fill=rgb("paper"))
    header(bd, M["study"], row, note, source)
    footer(bd, foot)
    plot.background(bd, row)

    out = []
    for i in range(n):
        im = base.copy()
        d = ImageDraw.Draw(im)
        fr = Image.fromarray(A[i][:, :, ::-1])          # ffmpeg gives BGR
        small, k = fit(fr, VID_W - 2 * PAD, MID_H - 2 * PAD)
        ox = PAD + (VID_W - 2 * PAD - small.width) // 2
        oy = HEAD_H + PAD + (MID_H - 2 * PAD - small.height) // 2
        im.paste(small, (ox, oy))
        # what was looked at, on the picture: the face crop the eye ratio came from and
        # each stimulus LED, outlined where it is and brightened when it is on
        marks = [(geo["FS"], rgb("faint"), "face tracked here")]
        marks += [(sl, rgb(LAMP[kd][1]) if on[i] else rgb("faint"),
                   LAMP[kd][0].replace("  ·  ", " "))
                  for sl, kd, on in boxes]
        # The two LED boxes sit ~44 px apart and overlap at this scale, so their labels
        # go on opposite sides of them rather than on top of each other.
        for mi, (sl, colr, lbl) in enumerate(marks):
            box = [ox + sl[1].start * k, oy + sl[0].start * k,
                   ox + sl[1].stop * k, oy + sl[0].stop * k]
            d.rectangle(box, outline=colr, width=2)
            d.text((box[0], box[1] - 15 if mi % 2 == 0 else box[3] + 3), lbl,
                   font=font(12), fill=colr)
        nb = sum(1 for o in onsets if o <= t[i])
        readouts(d, VID_W + PAD + 10, HEAD_H + PAD + 6, row,
                 er[i], el[i], closure[i], t[i], nb, total,
                 [(LAMP[kd][0], bool(on[i]), LAMP[kd][1]) for _, kd, on in boxes])
        plot.trace(d, t[:n], closure[:n], i)
        out.append(im)
    order = playback(n, fps, speed) + [n - 1] * int(round(HOLD_S * OUT_FPS))
    codec = encode(out_path, lambda: (out[j] for j in order))
    print("   %-8s %-13s trial %-3d %-24s -> %s (%s, %.1fs at x%g)"
          % (kind, row["session_name"], row["session_trial"], row["scored_class"],
             os.path.basename(out_path), codec, len(order) / OUT_FPS, speed), flush=True)
    return dict(kind=kind, file=os.path.basename(out_path), session=tag,
                session_name=row["session_name"], session_trial=row["session_trial"],
                role=rec["role"], recording=rec["file"], block=row["block"],
                trial_type=row["trial_type"], scored_class=row["scored_class"],
                scored_onset_ms=row["scored_onset_ms"],
                peak_closure_pct=row["peak_closure_pct"],
                n_full_blinks=row["n_full_blinks"],
                face_tracked_pct=row["face_tracked_pct"],
                at_in_video=row["cs_onset_video_s"] if row["cs_onset_video_s"] is not None
                else row["us_onset_video_s"], chosen_because=why)


def main():
    cfg = C.load(sys.argv[1] if len(sys.argv) > 1 else None)
    wdir, odir = work_dir(cfg), out_dir(cfg)
    with open(os.path.join(wdir, "merged.json"), encoding="utf-8") as fh:
        M = json.load(fh)
    with open(os.path.join(wdir, "merged_rows.json"), encoding="utf-8") as fh:
        ROWS = json.load(fh)
    win = M.get("cr_window") or C.cr_window(C.fill(M["protocol"]))
    args = sys.argv[2:]
    only = [a for a in args if not a.startswith("-")]
    speed = SPEED
    if "--speed" in args:
        try:
            speed = float(args[args.index("--speed") + 1])
        except (IndexError, ValueError):
            sys.exit("--speed needs a number: the fraction of real time to play at, "
                     "e.g. --speed 0.25")
        if not 0.01 <= speed <= 1.0:
            sys.exit("--speed must be between 0.01 and 1.0 (1.0 is real time)")
        only = [a for a in only if a != args[args.index("--speed") + 1]]
    print("clips play at x%g of real time (%.0fx slow motion)" % (speed, 1.0 / speed),
          flush=True)

    made, missing = [], []
    for kind, prefix, tt, roles, note in KINDS:
        if only and kind not in only:
            continue
        cand = candidates(ROWS, prefix, tt, roles, win)
        if not cand:
            # Said out loud, and not quietly replaced with something else.  "This
            # participant produced no ?CR the scorer would stand behind" is a result;
            # a clip of the nearest other thing, captioned as a ?CR, is not.
            missing.append((kind, note))
            print("   %-8s no trial of this kind - nothing to show" % kind, flush=True)
            continue
        row, why, margin = prototypical(cand, list(roles))
        if margin is not None and margin < 0.15:
            why += (" - a close call: another trial was almost exactly as typical, so "
                    "nothing turns on this one being the one shown")
        got = render(cfg, M, row, kind, note,
                     os.path.join(odir, "clip_%s.mp4" % kind), wdir, win, why, speed)
        if got:
            made.append(dict(got, speed=speed))

    with open(os.path.join(odir, "clips.json"), "w", encoding="utf-8") as fh:
        json.dump(dict(study=cfg["study"], speed=speed, clips=made,
                       missing=[dict(kind=k, note=n) for k, n in missing]), fh, indent=1)
    if made:
        keys = list(made[0])
        with open(os.path.join(odir, "clips_index.csv"), "w", encoding="utf-8",
                  newline="") as fh:
            w = csv.DictWriter(fh, keys)
            w.writeheader()
            w.writerows(made)
    print("%d clip(s) in %s" % (len(made), odir), flush=True)


if __name__ == "__main__":
    main()
