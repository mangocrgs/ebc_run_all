"""Thin ffmpeg helpers shared by every stage."""
import subprocess

FFMPEG = ["ffmpeg", "-v", "error", "-nostdin"]


def probe(path):
    q = ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height,r_frame_rate,nb_frames", "-of", "default=noprint_wrappers=1", path]
    txt = subprocess.run(q, capture_output=True, text=True).stdout.strip()
    if not txt:
        raise SystemExit(f"ffprobe found no video stream in {path}")
    d = dict(l.split("=", 1) for l in txt.splitlines())
    num, den = d["r_frame_rate"].split("/")
    fps = float(num) / float(den)
    nfr = int(d["nb_frames"]) if d.get("nb_frames", "N/A").isdigit() else 0
    return int(d["width"]), int(d["height"]), fps, nfr


def frames(path, vf, fsz, ss=None, n=None, pix="bgr24"):
    """Yield raw frames of exactly fsz bytes for a filtergraph."""
    cmd = list(FFMPEG)
    if ss is not None:
        cmd += ["-ss", f"{ss:.5f}"]
    cmd += ["-i", path]
    if n is not None:
        cmd += ["-frames:v", str(n)]
    cmd += ["-vf", vf, "-f", "rawvideo", "-pix_fmt", pix, "-"]
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, bufsize=1 << 26)
    try:
        while True:
            b = p.stdout.read(fsz)
            if len(b) < fsz:
                break
            yield b
    finally:
        p.stdout.close()
        p.wait()


def still(path, t, w, h):
    """One full-resolution RGB frame at time t, or None."""
    cmd = FFMPEG + ["-ss", f"{t:.3f}", "-i", path, "-frames:v", "1",
                    "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    out = subprocess.run(cmd, stdout=subprocess.PIPE).stdout
    return out[:w * h * 3] if len(out) >= w * h * 3 else None


def even(v):
    return int(v) // 2 * 2


def crop_box(x0, y0, x1, y1, W, H, align=16):
    """A crop rectangle ffmpeg will accept: even origin, width a multiple of `align`.

    An odd crop width gives an odd rawvideo row stride, which desynchronises the frame
    reads and shears every frame, so this is not cosmetic.
    """
    x0 = max(0, even(x0)); y0 = max(0, even(y0))
    w = min(int(x1) - x0, W - x0); h = min(int(y1) - y0, H - y0)
    w = max(align, w - w % align); h = max(2, h - h % 2)
    if x0 + w > W:
        w = (W - x0) // align * align
    if y0 + h > H:
        h = (H - y0) // 2 * 2
    return x0, y0, w, h
