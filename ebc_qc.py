"""Render an eye filmstrip for one trial and print the measured closure beside it.

usage: python qc_gen.py <tag> <video path> <trial> [trial ...]
"""
# --- portable paths -------------------------------------------------------
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ebc_paths import BASE, OUT, WORK, video   # noqa: E402
os.chdir(WORK)                                  # cache + intermediates live here
# --------------------------------------------------------------------------
import subprocess, sys, json, numpy as np, cv2, mediapipe as mp, warnings
warnings.filterwarnings("ignore")

TAG, VID = sys.argv[1], sys.argv[2]
if not os.path.isabs(VID):
    VID = video(VID)          # a bare file name means "next to the scripts"
TRIALS = [int(x) for x in sys.argv[3:]]
M = json.load(open("merged.json"))
meta = M["meta"][TAG]  # noqa
FPS = meta["fps"]
MS = 1000.0 / FPS
PRE = int(round(300.0 / MS))
fb = json.load(open(f"{TAG}_facebox.json"))
X0 = max(0, int(fb["x0"]) - 70); X1 = min(1920, int(fb["x1"]) + 70)
Y0 = max(0, int(fb["y0"]) - 70); Y1 = min(1080, int(fb["y1"]) + 70)
CW = (X1 - X0) // 16 * 16; CH = (Y1 - Y0) // 2 * 2
fsz = CW * CH * 3
fm = mp.solutions.face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True)
EYE = [33, 133, 160, 158, 144, 153, 362, 263, 385, 387, 380, 373, 246, 466, 7, 249]
rows = {r["session_trial"]: r for r in meta["rows"]}

for ti in TRIALS:
    f0 = None
    for r in meta["rows"]:
        if r["session_trial"] == ti:
            f0 = int(round(r["cs_onset_video_s"] * FPS))
    if f0 is None:
        print("no trial", ti); continue
    p = subprocess.Popen(["ffmpeg", "-v", "error", "-ss", f"{(f0-PRE)/FPS:.5f}", "-i", VID,
                          "-frames:v", str(PRE + 138), "-vf", f"crop={CW}:{CH}:{X0}:{Y0}",
                          "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], stdout=subprocess.PIPE, bufsize=10**8)
    A = []
    while True:
        b = p.stdout.read(fsz)
        if len(b) < fsz:
            break
        A.append(np.frombuffer(b, np.uint8).reshape(CH, CW, 3))
    p.stdout.close(); p.wait()
    if not A:
        print("no frames for trial", ti); continue
    A = np.stack(A)
    r0 = fm.process(cv2.cvtColor(A[0], cv2.COLOR_BGR2RGB))
    if not r0.multi_face_landmarks:
        print("no face on first frame of trial", ti); continue
    lm = np.array([[l.x * CW, l.y * CH] for l in r0.multi_face_landmarks[0].landmark])[EYE]
    x1, x2 = int(lm[:, 0].min()) - 24, int(lm[:, 0].max()) + 24
    y1, y2 = int(lm[:, 1].min()) - 20, int(lm[:, 1].max()) + 20
    x1, y1 = max(0, x1), max(0, y1); x2, y2 = min(CW, x2), min(CH, y2)
    tr = meta["traces"][str(ti)]
    t = np.array(tr["t"]); C = np.array(tr["C"])
    tiles = []
    for ms in range(-150, 850, 50):
        i = int(np.argmin(np.abs(t - ms)))
        if i >= len(A):
            break
        crop = cv2.resize(A[i][y1:y2, x1:x2], None, fx=1.7, fy=1.7, interpolation=cv2.INTER_CUBIC)
        pad = np.full((crop.shape[0] + 34, crop.shape[1], 3), 25, np.uint8)
        pad[34:] = crop
        col = (0, 220, 255) if ms < 0 else ((80, 255, 80) if ms < 350 else (255, 140, 80))
        cv2.putText(pad, f"{ms:+d}ms", (4, 14), cv2.FONT_HERSHEY_SIMPLEX, .42, col, 1, cv2.LINE_AA)
        cv2.putText(pad, f"close {C[i]*100:.0f}%", (4, 29), cv2.FONT_HERSHEY_SIMPLEX, .40,
                    (230, 230, 230), 1, cv2.LINE_AA)
        tiles.append(pad)
    per = 5
    rimg = []
    for k in range(0, len(tiles), per):
        ch = tiles[k:k + per]
        while len(ch) < per:
            ch.append(np.zeros_like(tiles[0]))
        rimg.append(np.hstack(ch))
    out = np.vstack(rimg)
    rr = rows[ti]
    cv2.putText(out, f"{TAG} trial {ti} - {rr['scored_class']} - onset {rr['scored_onset_ms']} ms"
                     f"  (green = CS 0-350ms, orange = after US)",
                (6, out.shape[0] - 8), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.imwrite(os.path.join(OUT, f"qc_{TAG}_t{ti}.png"), out)
    print(f"wrote qc_{TAG}_t{ti}.png  class={rr['scored_class']} onset={rr['scored_onset_ms']}")
