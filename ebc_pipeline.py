"""Generalised eyeblink-conditioning video scorer.

usage: python ebc_pipeline.py "<video path>" <tag>

Stages (each cached to <tag>_*.npy/json so the script can be resumed):
  1  locate the stimulator box + detect the blue (US) LED      [full decode @ 320x180]
  2  locate the yellow (CS) LED using the blue events as anchors [seek windows]
  3  cache a tiny full-res ROI on the yellow LED               [full decode]
  4  yellow (CS) events, paired with blue
  5  face bounding box
  6  per-trial eyelid tracking (MediaPipe FaceMesh -> EAR)
  7  blink metrics
"""
# --- portable paths -------------------------------------------------------
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ebc_paths import BASE, OUT, WORK, video   # noqa: E402
os.chdir(WORK)                                  # cache + intermediates live here
# --------------------------------------------------------------------------
import subprocess, sys, os, json, time, warnings
import numpy as np
import cv2

warnings.filterwarnings("ignore")
VID = sys.argv[1]
if not os.path.isabs(VID):
    VID = video(VID)          # a bare file name means "next to the scripts"
TAG = sys.argv[2]
PRE_MS, POST_MS = 300.0, 1150.0
MAIN, RESET, PARTIAL = 0.40, 0.20, 0.15


def log(*a):
    print(f"[{TAG}]", *a, flush=True)


def probe():
    q = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height,r_frame_rate,nb_frames", "-of", "default=noprint_wrappers=1", VID]
    d = dict(l.split("=") for l in subprocess.run(q, capture_output=True, text=True).stdout.strip().splitlines())
    num, den = d["r_frame_rate"].split("/")
    return int(d["width"]), int(d["height"]), float(num) / float(den), int(d["nb_frames"])


W_, H_, FPS, NFR = probe()
MS = 1000.0 / FPS
PRE = int(round(PRE_MS / MS))
POST = int(round(POST_MS / MS))
log(f"{os.path.basename(VID)}  {W_}x{H_}  {FPS:.3f} fps  {NFR} frames  {NFR/FPS:.1f} s")


def stream(vf, fsz, ss=None, n=None):
    """yield raw bgr24 frames for a filtergraph."""
    cmd = ["ffmpeg", "-v", "error"]
    if ss is not None:
        cmd += ["-ss", f"{ss:.5f}"]
    cmd += ["-i", VID]
    if n is not None:
        cmd += ["-frames:v", str(n)]
    cmd += ["-vf", vf, "-f", "rawvideo", "-pix_fmt", "bgr24", "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=10 ** 8)
    while True:
        b = p.stdout.read(fsz)
        if len(b) < fsz:
            break
        yield b
    p.stdout.close()
    p.wait()


def groups(mask, gap):
    idx = np.where(mask)[0]
    out = []
    if not len(idx):
        return out
    s = prev = idx[0]
    for k in idx[1:]:
        if k - prev > gap:
            out.append((s, prev))
            s = k
        prev = k
    out.append((s, prev))
    return out


# ---------------------------------------------------------------- 1. blue LED
f_blue = f"{TAG}_blue.npz"
if not os.path.exists(f_blue):
    SW, SH = 320, 180
    fsz = SW * SH * 3
    mx = np.zeros(NFR + 64, np.int16)
    ax = np.zeros(NFR + 64, np.int16)
    ay = np.zeros(NFR + 64, np.int16)
    cnt = np.zeros(NFR + 64, np.int32)
    t0 = time.time()
    i = 0
    ema = None
    alpha = 1.0 / (3.0 * FPS)          # ~3 s background; a 58 ms flash barely moves it
    for b in stream(f"scale={SW}:{SH}", fsz):
        img = np.frombuffer(b, np.uint8).reshape(SH, SW, 3).astype(np.float32)
        bl = img[:, :, 0] - 0.5 * (img[:, :, 1] + img[:, :, 2])
        if ema is None:
            ema = bl.copy()
        dbl = bl - ema                  # transient blueness, immune to static blue objects
        m = float(dbl.max())
        mx[i] = int(min(m, 32000))
        cnt[i] = int((dbl > 30).sum())
        if m > 30:
            k = int(np.argmax(dbl))
            ay[i], ax[i] = k // SW, k % SW
        ema += (bl - ema) * (0.05 if i < 200 else alpha)
        i += 1
        if i % 15000 == 0:
            log(f"  stage1 {i}/{NFR}  {time.time()-t0:.0f}s")
    n = i
    np.savez(f_blue, mx=mx[:n], ax=ax[:n], ay=ay[:n], cnt=cnt[:n], fps=FPS, sw=SW, sh=SH)
    log(f"stage1 done {n} frames {time.time()-t0:.0f}s  blue max p50={np.median(mx[:n])} max={mx[:n].max()}")
d = np.load(f_blue)
mx, ax, ay, cnt = d["mx"], d["ax"], d["ay"], d["cnt"]
NF = len(mx)
thr_b = max(40, int(np.percentile(mx, 99.8) * 0.45))
bg_groups = [g for g in groups((mx > thr_b) & (cnt > 8), int(0.25 * FPS))
             if 2 <= g[1] - g[0] <= int(0.40 * FPS)]      # a US pulse is short, not a lighting change
sel = np.zeros(NF, bool)
for a, b in bg_groups:
    sel[a:b + 1] = True
bx = float(np.median(ax[sel])) * (W_ / d["sw"])
by = float(np.median(ay[sel])) * (H_ / d["sh"])
log(f"blue: {len(bg_groups)} events, thr={thr_b}, box at full-res ({bx:.0f},{by:.0f}), "
    f"dur {np.mean([(b-a+1)/FPS*1000 for a,b in bg_groups]):.1f} ms")

# ---------------------------------------------------------------- 2. yellow LED, detected independently
# The yellow LED is found as a transient in its own right, NOT anchored to the blue
# events - in some blocks the two stimuli are unpaired, so anchoring would fail.
f_yel = f"{TAG}_yellow.npz"
BW, BH = 340, 220
BX = int(np.clip(round(bx) - BW / 2, 0, W_ - BW))
BY = int(np.clip(round(by) - BH / 2, 0, H_ - BH))
if not os.path.exists(f_yel):
    fsz = BW * BH * 3
    ymx = np.zeros(NFR + 64, np.int16)
    yax = np.zeros(NFR + 64, np.int16)
    yay = np.zeros(NFR + 64, np.int16)
    ycnt = np.zeros(NFR + 64, np.int32)
    ema = None
    alpha = 1.0 / (5.0 * FPS)
    t0 = time.time()
    i = 0
    for b in stream(f"crop={BW}:{BH}:{BX}:{BY}", fsz):
        img = np.frombuffer(b, np.uint8).reshape(BH, BW, 3).astype(np.float32)
        yl_ = 0.5 * (img[:, :, 2] + img[:, :, 1]) - img[:, :, 0]
        brt = img.max(axis=2)
        if ema is None:
            ema = yl_.copy()
        dyl = np.where(brt > 150, yl_ - ema, 0.0)
        m = float(dyl.max())
        ymx[i] = int(min(m, 32000))
        ycnt[i] = int((dyl > 45).sum())
        if m > 45:
            k = int(np.argmax(dyl))
            yay[i], yax[i] = k // BW, k % BW
        ema += (yl_ - ema) * (0.05 if i < 200 else alpha)
        i += 1
        if i % 15000 == 0:
            log(f"  stage2 {i}/{NFR}  {time.time()-t0:.0f}s")
    np.savez(f_yel, mx=ymx[:i], ax=yax[:i], ay=yay[:i], cnt=ycnt[:i], bx0=BX, by0=BY, bw=BW, bh=BH)
    log(f"stage2 done {i} frames {time.time()-t0:.0f}s")
dy_ = np.load(f_yel)
ymx, yax, yay, ycnt = dy_["mx"], dy_["ax"], dy_["ay"], dy_["cnt"]
thr_y = max(50, float(np.percentile(ymx, 99)) * 0.55)
ye = [g for g in groups((ymx > thr_y) & (ycnt >= 3), int(0.20 * FPS))
      if (g[1] - g[0]) >= int(0.05 * FPS)]
ysel = np.zeros(len(ymx), bool)
for a, b in ye:
    ysel[a:b + 1] = True
yx = float(np.median(yax[ysel])) + BX
yy = float(np.median(yay[ysel])) + BY
log(f"yellow LED at full-res ({yx:.0f},{yy:.0f})  thr={thr_y:.0f}  {len(ye)} raw events")

# ---------------------------------------------------------------- 3. pair CS with US
RW, RH = 64, 40
RX = int(np.clip(round(yx) - RW / 2, 0, W_ - RW))
RY = int(np.clip(round(yy) - RH / 2, 0, H_ - RH))
blue_on = np.array([g[0] for g in bg_groups])
used = set()
trials = []
for a, b in ye:
    nb = blue_on[(blue_on >= a) & (blue_on - a <= int(1.5 * FPS))]
    k = int(nb[0]) if len(nb) else -1
    if k >= 0:
        used.add(k)
    trials.append((a, b, k))
trials = np.array(trials)
us_only = [int(k) for k in blue_on if k not in used]
paired = trials[trials[:, 2] >= 0]
isi = float(np.median((paired[:, 2] - paired[:, 0]) / FPS * 1000)) if len(paired) else np.nan
durs = (trials[:, 1] - trials[:, 0] + 1) / FPS * 1000
cs_dur = float(np.median(durs))
np.save(f"{TAG}_trials.npy", trials)
log(f"CS trials: {len(trials)}  ({len(paired)} paired with US, {len(trials)-len(paired)} CS-only)  "
    f"US-only events: {len(us_only)}  CS-US={isi:.1f} ms  CS dur~{cs_dur:.1f} ms")

# ---------------------------------------------------------------- 5. face bbox
import mediapipe as mp

f_face = f"{TAG}_facebox.json"
if not os.path.exists(f_face):
    fm = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1,
                                         refine_landmarks=True, min_detection_confidence=0.4)
    xs, ys, xe, ye_ = [], [], [], []
    for t in np.linspace(2, NFR / FPS - 3, 22):
        r = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{t:.3f}", "-i", VID, "-frames:v", "1",
                            "-f", "rawvideo", "-pix_fmt", "rgb24", "-"], stdout=subprocess.PIPE)
        if len(r.stdout) < W_ * H_ * 3:
            continue
        img = np.frombuffer(r.stdout[:W_ * H_ * 3], np.uint8).reshape(H_, W_, 3)
        res = fm.process(img)
        if not res.multi_face_landmarks:
            continue
        lm = np.array([[l.x * W_, l.y * H_] for l in res.multi_face_landmarks[0].landmark])
        xs.append(lm[:, 0].min()); xe.append(lm[:, 0].max())
        ys.append(lm[:, 1].min()); ye_.append(lm[:, 1].max())
    fm.close()
    json.dump(dict(x0=min(xs), x1=max(xe), y0=min(ys), y1=max(ye_), n=len(xs)), open(f_face, "w"))
fb = json.load(open(f_face))
log(f"face bbox x[{fb['x0']:.0f},{fb['x1']:.0f}] y[{fb['y0']:.0f},{fb['y1']:.0f}] from {fb['n']} samples")

MG = 60
FX0 = int(np.clip(fb["x0"] - MG, 0, W_ - 2))
FX1 = int(np.clip(fb["x1"] + MG, 2, W_))
FY0 = int(np.clip(fb["y0"] - MG, 0, H_ - 2))
FY1 = int(np.clip(fb["y1"] + MG, 2, H_))
# combined crop: face box + yellow LED ROI, so CS onset is re-verified inside every window.
# Width/height are rounded to multiples of 16 and the origin to an even pixel: an odd crop
# width gives an odd rawvideo row stride, which desynchronises the frame reads and shears
# every frame.
CX0 = (min(FX0, RX) // 2) * 2
CY0 = (min(FY0, RY) // 2) * 2
CX1 = max(FX1, RX + RW); CY1 = max(FY1, RY + RH)
CW2 = min(((CX1 - CX0 + 15) // 16) * 16, W_ - CX0)
CH2 = min(((CY1 - CY0 + 15) // 16) * 16, H_ - CY0)
CW2 -= CW2 % 16 if CW2 % 16 and CX0 + CW2 > W_ else 0
CH2 -= CH2 % 2
LEDs = (slice(RY - CY0, RY - CY0 + RH), slice(RX - CX0, RX - CX0 + RW))
FACEs = (slice(FY0 - CY0, FY1 - CY0), slice(FX0 - CX0, FX1 - CX0))
log(f"window crop {CW2}x{CH2} at ({CX0},{CY0})")

# ---------------------------------------------------------------- 6. eyelid tracking
Rl = [33, 160, 158, 133, 153, 144]
Ll = [362, 385, 387, 263, 373, 380]


def ear(p, ix):
    p1, p2, p3, p4, p5, p6 = [p[i] for i in ix]
    return (np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)) / (2 * np.linalg.norm(p1 - p4) + 1e-9)


f_tr = f"{TAG}_traces.json"
if not os.path.exists(f_tr):
    fsz = CW2 * CH2 * 3
    NW = PRE + POST
    out = {}
    t0 = time.time()
    for ti, (f0, f1, fb_) in enumerate(trials, 1):
        frames = [np.frombuffer(x, np.uint8).reshape(CH2, CW2, 3)
                  for x in stream(f"crop={CW2}:{CH2}:{CX0}:{CY0}", fsz, ss=(f0 - PRE) / FPS, n=NW)]
        if len(frames) < NW * 0.9:
            log(f"  trial {ti}: only {len(frames)} frames, skipped")
            continue
        A = np.stack(frames)
        led = A[:, LEDs[0], LEDs[1], :].astype(np.int16)
        ym = (led[:, :, :, 2] + led[:, :, :, 1]) // 2 - led[:, :, :, 0]
        br = A[:, LEDs[0], LEDs[1], :].max(axis=3)
        ys_ = np.where(br > 150, ym - np.median(ym[:PRE // 2], axis=0), 0).reshape(len(A), -1).max(axis=1)
        on = np.where(ys_ > max(40.0, ys_.max() * 0.5))[0]
        k0 = int(on[0]) if len(on) else PRE
        fm = mp.solutions.face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1,
                                             refine_landmarks=True, min_detection_confidence=0.4,
                                             min_tracking_confidence=0.4)
        er = np.full(len(A), np.nan); el = np.full(len(A), np.nan)
        for i in range(len(A)):
            sub = A[i][FACEs[0], FACEs[1]]
            big = cv2.resize(sub, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            r = fm.process(cv2.cvtColor(big, cv2.COLOR_BGR2RGB))
            if not r.multi_face_landmarks:
                continue
            hh, ww = big.shape[:2]
            p = np.array([[l.x * ww, l.y * hh] for l in r.multi_face_landmarks[0].landmark])
            er[i] = ear(p, Rl); el[i] = ear(p, Ll)
        fm.close()
        out[ti] = dict(k0=k0, f0=int(f0), er=er.tolist(), el=el.tolist(),
                       paired=bool(fb_ >= 0), n=len(A))
        log(f"  trial {ti:>2}/{len(trials)} k0={k0} (exp {PRE}) face_ok={np.isfinite(er).mean()*100:.0f}% "
            f"[{time.time()-t0:.0f}s]")
    json.dump(out, open(f_tr, "w"))

# ---------------------------------------------------------------- 7. metrics
from scipy.signal import savgol_filter

D = json.load(open(f_tr))
US_PRESENT = bool(np.isfinite(isi))
US_MS = float(isi) if US_PRESENT else 350.4      # nominal, for the response-class boundary only
CS_OFF = float(cs_dur)
med_dur = float(np.median(durs))
onsets_all = trials[:, 0]


def sm(x):
    x = np.array(x, float)
    m = np.isfinite(x)
    if m.sum() < len(x):
        x = np.interp(np.arange(len(x)), np.where(m)[0], x[m])
    return savgol_filter(x, 5, 2)


E = {int(k): (sm(v["er"]) + sm(v["el"])) / 2 for k, v in D.items()}
CLOSED = float(np.percentile([e.min() for e in E.values()], 10))
rows, traces = [], {}
for ti in sorted(E):
    e = E[ti]; v = D[str(ti)]
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
    exc = []
    i = w[0]
    while i <= w[-1]:
        if C[i] > thr:
            s = i
            while s > 0 and C[s - 1] < C[s] and C[s - 1] > 0.04:
                s -= 1
            j = i
            while j < len(C) - 1 and C[j + 1] > RESET:
                j += 1
            pk = s + int(np.argmax(C[s:j + 1]))
            r50 = pk
            while r50 < len(C) - 1 and C[r50] > 0.5 * C[pk]:
                r50 += 1
            ro = pk
            while ro < len(C) - 1 and C[ro] > RESET:
                ro += 1
            exc.append(dict(on=float(t[s]), pk=float(t[pk]), amp=float(C[pk]), r50=float(t[r50]),
                            end=float(t[ro]), dur=float(t[ro] - t[s]),
                            rise=float((C[pk] - C[s]) / max(t[pk] - t[s], MS))))
            i = j + 1
        else:
            i += 1
    full = [b for b in exc if b["amp"] >= MAIN]
    part = [b for b in exc if b["amp"] < MAIN]
    b1 = full[0] if full else (exc[0] if exc else None)

    def at(ms):
        return float(C[min(PRE + int(round(ms / MS)), len(C) - 1)])

    a_us, a_off, a_end = at(US_MS), at(CS_OFF), at(1000)
    cls = None
    if b1:
        cls = ("in-progress at CS" if inprog else "alpha/startle <100ms" if b1["on"] < 100
               else "CR (100-350ms)" if b1["on"] < US_MS else "UR (>=350ms)")
    d_ms = float(durs[ti - 1])
    prev_isi = float((onsets_all[ti - 1] - onsets_all[ti - 2]) / FPS) if ti > 1 else float("nan")
    irregular = (abs(d_ms - med_dur) > 0.40 * med_dur) or (ti > 1 and prev_isi < 5.0)
    q = ("pre-CS blink" if preflag else "") + (" | lid closing at CS" if inprog else "")
    if irregular:
        q = (q + " | irregular CS").lstrip(" |")
    rows.append(dict(
        session=TAG, trial=ti, type="CS-US" if v["paired"] else "CS-only",
        cs_onset_video_s=round(v["f0"] / FPS, 3),
        cs_duration_ms=round(d_ms, 1),
        prev_cs_interval_s=None if not np.isfinite(prev_isi) else round(prev_isi, 2),
        quality=q or "clean",
        n_full_blinks=len(full), n_partial_movements=len(part),
        blink_onset_ms=None if not b1 else round(b1["on"], 1),
        peak_closure_ms=None if not b1 else round(b1["pk"], 1),
        peak_closure_pct=None if not b1 else round(b1["amp"] * 100, 1),
        closing_speed_pct_per_ms=None if not b1 else round(b1["rise"] * 100, 2),
        closure_duration_ms=None if not b1 else round(b1["dur"], 1),
        reopen_half_ms=None if not b1 else round(b1["r50"], 1),
        reopen_full_ms=None if not b1 else round(b1["end"], 1),
        closure_at_US_pct=round(a_us * 100, 1), closure_at_CSoff_pct=round(a_off * 100, 1),
        closure_at_1000ms_pct=round(a_end * 100, 1),
        closed_at_US=bool(a_us >= 0.50), reopened_before_US=bool(b1 is not None and a_us < 0.30),
        response_class=cls,
        all_blink_onsets_ms=";".join(f"{b['on']:.0f}" for b in full),
        all_blink_amps_pct=";".join(f"{b['amp']*100:.0f}" for b in full),
        inter_blink_ms=";".join(f"{full[k+1]['on']-full[k]['on']:.0f}" for k in range(len(full) - 1)),
        partial_movement_ms=";".join(f"{b['on']:.0f}({b['amp']*100:.0f}%)" for b in part)))
json.dump(dict(tag=TAG, video=os.path.basename(VID), fps=FPS, nframes=int(NFR),
               duration_s=round(NFR / FPS, 3),
               us_present=US_PRESENT, us_ms=None if not US_PRESENT else round(float(isi), 1),
               cs_dur_ms=round(CS_OFF, 1),
               us_dur_ms=round(float(np.mean([(b - a + 1) / FPS * 1000 for a, b in bg_groups])), 1)
               if bg_groups else None,
               n_trials=len(trials), n_paired=int(len(paired)),
               n_us_only=len(us_only),
               us_only_s=[round(k / FPS, 3) for k in us_only],
               cs_events=[[int(a), int(b), int(c)] for a, b, c in trials],
               us_events=[[int(a), int(b)] for a, b in bg_groups],
               led_yellow=[round(yx), round(yy)], led_blue=[round(bx), round(by)],
               rows=rows, traces=traces), open(f"{TAG}_result.json", "w"))
log(f"DONE  {len(rows)} trials scored -> {TAG}_result.json")
