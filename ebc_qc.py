"""Quality-check images: prove the LEDs were found, and show what a trial looked like.

    python ebc_qc.py <config.json> leds            one page per recording: where the
                                                   LEDs were found and every pulse read
    python ebc_qc.py <config.json> trial <tag> <n> [n ...]
                                                   eye filmstrip with the measured
                                                   closure printed on each frame

The LED page is the check to look at first on a new participant.  If the marker sits on
the LED and the raster shows one clean pulse per trial, everything downstream is sound;
if it does not, put  "led_yellow": [x, y]  in the study config and re-run.
"""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import ebc_config as C
from ebc_paths import work_dir, out_dir
from ebc_video import probe, frames, still

CS_C, US_C, INK, MUT = "#B8760F", "#3A67CF", "#141922", "#59636F"


def led_page(cfg, rec, wdir, odir):
    tag = rec["tag"]
    sf = os.path.join(wdir, tag + "_stim.json")
    if not os.path.exists(sf):
        return None
    with open(sf, encoding="utf-8") as fh:
        S = json.load(fh)
    led = np.load(os.path.join(wdir, tag + "_led.npz"))
    W, H, fps, nfr = probe(rec["path"])
    # a frame with the CS LED lit, so the marker can be checked against a visible LED
    ok = [e for e in S["events"].get("yellow", []) if e["ok"]]
    t_shot = (ok[len(ok) // 2]["t"] + 0.15) if ok else min(5.0, nfr / fps / 2)
    buf = still(rec["path"], t_shot, W, H)
    img = np.frombuffer(buf, np.uint8).reshape(H, W, 3) if buf is not None else np.zeros((H, W, 3), np.uint8)

    n_sig = len(S["events"])
    fig = plt.figure(figsize=(15.0, 4.4 + 2.1 * n_sig))
    gs = fig.add_gridspec(1 + n_sig, 2, height_ratios=[3.1] + [1.0] * n_sig,
                          width_ratios=[1.55, 1], hspace=.55, wspace=.13)
    ax = fig.add_subplot(gs[0, 0])
    ax.imshow(img)
    ax.set_xlim(0, W); ax.set_ylim(H, 0)
    ax.set_title("%s  ·  %s  ·  frame at %.1f s" % (rec["label"], rec["file"], t_shot),
                 fontsize=11, loc="left")
    ax.tick_params(labelsize=7)
    az = fig.add_subplot(gs[0, 1])
    xs = [v["box"][0] for v in S["leds"].values()] + [v["box"][0] + v["box"][2] for v in S["leds"].values()]
    ys = [v["box"][1] for v in S["leds"].values()] + [v["box"][1] + v["box"][3] for v in S["leds"].values()]
    zx0, zx1 = max(0, min(xs) - 90), min(W, max(xs) + 90)
    zy0, zy1 = max(0, min(ys) - 70), min(H, max(ys) + 70)
    az.imshow(img[zy0:zy1, zx0:zx1], extent=[zx0, zx1, zy1, zy0])
    az.set_title("stimulator box, as located", fontsize=11, loc="left")
    az.tick_params(labelsize=7)
    for a in (ax, az):
        for key, col in (("yellow", CS_C), ("blue", US_C)):
            v = S["leds"].get(key)
            if not v:
                continue
            bx, by, bw, bh = v["box"]
            a.add_patch(Rectangle((bx, by), bw, bh, fill=False, ec=col, lw=1.4, ls="--"))
            p = v.get("position")
            if p:
                a.plot(p["x"], p["y"], "o", ms=13, mfc="none", mec=col, mew=2.0)
                a.plot(p["x"], p["y"], "+", ms=9, color=col, mew=1.4)
    az.set_xlim(zx0, zx1); az.set_ylim(zy1, zy0)

    i = -1
    for key, col, nom in (("yellow", CS_C, cfg["protocol"]["cs_ms"]),
                          ("blue", US_C, cfg["protocol"]["us_dur_ms"])):
        if key not in S["events"]:
            continue
        i += 1
        a = fig.add_subplot(gs[1 + i, :])
        sig = led["sig_" + key].astype(float)
        t = np.arange(len(sig)) / fps
        step = max(1, len(sig) // 24000)
        a.plot(t[::step], sig[::step], lw=.6, color=col, alpha=.85)
        inf = S["leds"][key]["signal"]
        for lvl, lab, ls in ((inf.get("rest_level"), "rest", ":"),
                             (inf.get("on_threshold"), "threshold", "-"),
                             (inf.get("lit_level"), "lit", ":")):
            if lvl is not None:
                a.axhline(lvl, color=INK, lw=.9, ls=ls, alpha=.55)
                a.annotate(lab, (t[-1], lvl), fontsize=7.5, color=MUT, va="center",
                           xytext=(4, 0), textcoords="offset points")
        acc = [e for e in S["events"][key] if e["ok"]]
        rej = [e for e in S["events"][key] if not e["ok"]]
        ymax = max(sig.max(), 1)
        a.plot([e["t"] for e in acc], [ymax * 1.12] * len(acc), "v", ms=5, color=col)
        if rej:
            a.plot([e["t"] for e in rej], [ymax * 1.12] * len(rej), "x", ms=5, color=MUT, mew=1.2)
        st = S["leds"][key]["stats"] or {}
        a.set_title("%s LED  ·  %s  ·  %d accepted, %d rejected  ·  duration %s ms  ·  ITI %s s"
                    % ("CS (yellow)" if key == "yellow" else "US (blue)",
                       "nominal %g ms" % nom, len(acc), len(rej),
                       st.get("dur_med_ms", "-"), st.get("iti_med_s", "-")),
                    fontsize=10, loc="left")
        a.set_xlim(0, t[-1] if len(t) else 1)
        a.set_ylim(min(-20, sig.min()), ymax * 1.25)
        a.set_ylabel("%sness" % key, fontsize=9)
        a.tick_params(labelsize=8)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
        if i == n_sig - 1:
            a.set_xlabel("time in recording, s", fontsize=10)
    sub = "anchor (%d,%d) from %s" % (S["anchor"]["x"], S["anchor"]["y"], S["anchor"]["source"])
    if S.get("warnings"):
        sub += "   ⚠  " + "; ".join(S["warnings"])
    fig.suptitle("%s — stimulus detection check   |   %s" % (cfg["study"], sub),
                 fontsize=12.5, x=.012, ha="left", y=.985)
    fig.subplots_adjust(left=.045, right=.965, top=.90, bottom=.07)
    p = os.path.join(odir, "qc_leds_%s.png" % tag)
    fig.savefig(p, dpi=135)
    plt.close(fig)
    return p


def filmstrip(cfg, tag, trial_ids, wdir, odir):
    import cv2
    import mediapipe as mp
    rec = next(r for r in cfg["recordings"] if r["tag"] == tag)
    with open(os.path.join(wdir, "merged.json"), encoding="utf-8") as fh:
        M = json.load(fh)
    with open(os.path.join(wdir, "merged_rows.json"), encoding="utf-8") as fh:
        ROWS = json.load(fh)
    with open(os.path.join(wdir, tag + "_facebox.json"), encoding="utf-8") as fh:
        fb = json.load(fh)
    rows = {r["session_trial"]: r for r in ROWS if r["session"] == tag}
    W, H, fps, _ = probe(rec["path"])
    MS = 1000.0 / fps
    PRE = int(round(300.0 / MS))
    X0 = max(0, int(fb["x0"]) - 70); X1 = min(W, int(fb["x1"]) + 70)
    Y0 = max(0, int(fb["y0"]) - 70); Y1 = min(H, int(fb["y1"]) + 70)
    CW = (X1 - X0) // 16 * 16; CH = (Y1 - Y0) // 2 * 2
    fsz = CW * CH * 3
    fm = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1,
                                         refine_landmarks=True)
    EYE = [33, 133, 160, 158, 144, 153, 362, 263, 385, 387, 380, 373, 246, 466, 7, 249]
    us0 = cfg["protocol"]["us_onset_ms"]
    out_files = []
    for ti in trial_ids:
        r = rows.get(ti)
        if not r:
            print("no trial %d in %s" % (ti, tag)); continue
        f0 = int(round(r["cs_onset_video_s"] * fps)) if r["cs_onset_video_s"] is not None \
            else int(round(r["us_onset_video_s"] * fps))
        A = [np.frombuffer(x, np.uint8).reshape(CH, CW, 3)
             for x in frames(rec["path"], "crop=%d:%d:%d:%d" % (CW, CH, X0, Y0), fsz,
                             ss=(f0 - PRE) / fps, n=PRE + 138)]
        if not A:
            print("no frames for trial %d" % ti); continue
        A = np.stack(A)
        r0 = fm.process(cv2.cvtColor(A[0], cv2.COLOR_BGR2RGB))
        if not r0.multi_face_landmarks:
            print("no face on the first frame of trial %d" % ti); continue
        lm = np.array([[l.x * CW, l.y * CH] for l in r0.multi_face_landmarks[0].landmark])[EYE]
        x1, x2 = max(0, int(lm[:, 0].min()) - 24), min(CW, int(lm[:, 0].max()) + 24)
        y1, y2 = max(0, int(lm[:, 1].min()) - 20), min(CH, int(lm[:, 1].max()) + 20)
        tr = M["traces"][tag][str(ti)]
        t = np.array(tr["t"]); Cl = np.array(tr["C"])
        tiles = []
        for ms in range(-150, 850, 50):
            i = int(np.argmin(np.abs(t - ms)))
            if i >= len(A):
                break
            crop = cv2.resize(A[i][y1:y2, x1:x2], None, fx=1.7, fy=1.7, interpolation=cv2.INTER_CUBIC)
            pad = np.full((crop.shape[0] + 34, crop.shape[1], 3), 25, np.uint8)
            pad[34:] = crop
            col = (0, 220, 255) if ms < 0 else ((80, 255, 80) if ms < us0 else (255, 140, 80))
            cv2.putText(pad, "%+d ms" % ms, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, .42, col, 1, cv2.LINE_AA)
            cv2.putText(pad, "close %.0f%%" % (Cl[i] * 100), (4, 29), cv2.FONT_HERSHEY_SIMPLEX,
                        .40, (230, 230, 230), 1, cv2.LINE_AA)
            tiles.append(pad)
        per = 5
        img_rows = []
        for k in range(0, len(tiles), per):
            ch = tiles[k:k + per]
            while len(ch) < per:
                ch.append(np.zeros_like(tiles[0]))
            img_rows.append(np.hstack(ch))
        out = np.vstack(img_rows)
        cv2.putText(out, "%s trial %d - %s - onset %s ms  (green = before the US, orange = after)"
                    % (tag, ti, r["scored_class"], r["scored_onset_ms"]),
                    (6, out.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1, cv2.LINE_AA)
        p = os.path.join(odir, "qc_%s_t%d.png" % (tag, ti))
        cv2.imwrite(p, out)
        out_files.append(p)
        print("wrote %s  class=%s onset=%s" % (os.path.basename(p), r["scored_class"],
                                               r["scored_onset_ms"]))
    fm.close()
    return out_files


def main():
    cfg = C.load(sys.argv[1])
    what = sys.argv[2] if len(sys.argv) > 2 else "leds"
    wdir, odir = work_dir(cfg), out_dir(cfg)
    if what == "leds":
        for rec in cfg["recordings"]:
            p = led_page(cfg, rec, wdir, odir)
            if p:
                print("wrote " + os.path.basename(p))
    elif what == "trial":
        filmstrip(cfg, sys.argv[3], [int(x) for x in sys.argv[4:]], wdir, odir)
    else:
        raise SystemExit("usage: ebc_qc.py <config> leds | trial <tag> <n> [n ...]")


if __name__ == "__main__":
    main()
