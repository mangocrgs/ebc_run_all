"""Cut the lab logo into the three shapes the app actually uses.

The source is one wide lockup on a black field - brain glyph, two-line wordmark, a
hairline rule and the tagline - which is the wrong shape for every place it has to go.
A 1983 x 793 strip in a 44 px title bar puts the tagline at four pixels tall.

So it is cut, not scaled:

    logo_mark.png   the brain glyph alone, square, black field.  Title bar, at 34 px,
                    where only the glyph survives the reduction.
    logo_full.png   the whole lockup, tight-cropped.  The footer colophon, at 360 px,
                    where the tagline is legible again.
    ebc.ico         the brain glyph on a rounded black tile, 16 - 256 px, for the
                    executable, the task bar and the browser tab.

Re-run after replacing the source:

    py -3 packaging/make_assets.py  ["path/to/logo.png"]

The band coordinates below are measured from the source, not assumed: change the file
and the measurement changes with it, which is why they are found rather than hard-coded.
"""
import os
import sys

import numpy as np
from PIL import Image, ImageDraw

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(os.path.dirname(HERE), "assets")
DEFAULT_SRC = os.path.join(os.path.expanduser("~"), "OneDrive", "Bureau", "Recherche",
                           "logo carole.png")

INK = 18          # anything darker than this is field, not artwork
GAP = 15          # a column gap this wide or wider separates two elements


def bands(mask):
    """Start and end of every run of True, longest first is not assumed - order is kept."""
    out, start = [], None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        elif not v and start is not None:
            out.append((start, i - 1))
            start = None
    if start is not None:
        out.append((start, len(mask) - 1))
    return out


def measure(a):
    """Find the glyph and the wordmark in the source, by looking rather than guessing."""
    lit = a.max(axis=2) > INK
    cols, rows = lit.any(axis=0), lit.any(axis=1)
    x0, x1 = np.argmax(cols), len(cols) - 1 - np.argmax(cols[::-1])
    y0, y1 = np.argmax(rows), len(rows) - 1 - np.argmax(rows[::-1])

    # the widest empty column band inside the artwork splits glyph from wordmark
    empty = [(s, e) for s, e in bands(~cols[x0:x1 + 1]) if e - s + 1 >= GAP]
    if not empty:
        raise SystemExit("No gap found between the glyph and the wordmark - is this the "
                         "right logo file?")
    s, e = max(empty, key=lambda b: b[1] - b[0])
    split = (x0 + s, x0 + e)
    return {"art": (int(x0), int(y0), int(x1), int(y1)),
            "mark": (int(x0), int(y0), int(x0 + s - 1), int(y1)),
            "text_x": int(x0 + e + 1)}


def square(im, pad=0.06):
    """The glyph on a square black field, with a little air around it."""
    side = int(max(im.size) * (1 + 2 * pad))
    out = Image.new("RGB", (side, side), (0, 0, 0))
    out.paste(im, ((side - im.width) // 2, (side - im.height) // 2))
    return out


def rounded(im, radius=0.20):
    """The same square, corners rounded and the outside transparent - an app icon."""
    im = im.convert("RGBA")
    m = Image.new("L", im.size, 0)
    ImageDraw.Draw(m).rounded_rectangle([0, 0, im.width - 1, im.height - 1],
                                        radius=int(im.width * radius), fill=255)
    im.putalpha(m)
    return im


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SRC
    if not os.path.isfile(src):
        raise SystemExit("Source logo not found:\n    %s\n\nPass the path as an argument."
                         % src)
    im = Image.open(src).convert("RGB")
    m = measure(np.asarray(im).astype(int))
    os.makedirs(OUT, exist_ok=True)

    full = im.crop((m["art"][0] - 8, m["art"][1] - 8, m["art"][2] + 9, m["art"][3] + 9))
    full = full.resize((1100, round(1100 * full.height / full.width)), Image.LANCZOS)
    full.save(os.path.join(OUT, "logo_full.png"), optimize=True)

    mark = square(im.crop((m["mark"][0], m["mark"][1], m["mark"][2] + 1, m["mark"][3] + 1)))
    mark.resize((512, 512), Image.LANCZOS).save(os.path.join(OUT, "logo_mark.png"),
                                                optimize=True)

    icon = rounded(mark.resize((512, 512), Image.LANCZOS))
    icon.save(os.path.join(OUT, "ebc.ico"),
              sizes=[(s, s) for s in (16, 24, 32, 48, 64, 128, 256)])

    print("  measured  glyph x%d-%d, wordmark from x%d" %
          (m["mark"][0], m["mark"][2], m["text_x"]))
    for n in ("logo_full.png", "logo_mark.png", "ebc.ico"):
        p = os.path.join(OUT, n)
        print("  wrote     %-16s %6.1f kB" % (n, os.path.getsize(p) / 1e3))


if __name__ == "__main__":
    main()
