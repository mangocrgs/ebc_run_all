"""Check the analyser against the hand scoring, on the things that carry the result.

    python compare_manual.py

Each participant's folder holds a workbook scored by hand, frame by frame at 120 fps.
This matches those trials to the analyser's output on the session clock and compares the
two scorings.

WHAT IS BEING TESTED, AND WHAT IS NOT
-------------------------------------
A constant offset between the two scorers is not a finding.  A person marks the frame
where they can see the lid has moved; the analyser walks back down the rising edge to
where the movement began.  Those are different definitions of "onset" by construction,
and a fixed difference between them subtracts out of everything that matters.

What must agree is the part that carries the science:

  the learning curve   CR rate block by block.  If the two scorers see the same
                       acquisition, the analyser is measuring conditioning.
  the spread           SD of the onset distribution.  A scorer that agrees on the mean
                       but not the spread is not measuring the same quantity - it is
                       adding noise, and noise is what a rate cannot survive.

Both scorings are put through the same classification rule (ebc_score.classify), so the
comparison is of the measurement and nothing else.
"""
import csv
import datetime
import os
import sys

import numpy as np
import openpyxl

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ebc_score as S

V = r"C:/Users/marga/OneDrive/Bureau/Recherche/EBC/Video"
STUDIES = [
    ("Carole", V + "/Carole/data Carole 08.09.xlsx", "Sheet1", V + "/analysis_EBC/Carole"),
    ("Thomas", V + "/Thomas/data Thomas.xlsx", "Data brut", V + "/analysis_EBC/Thomas"),
    ("Marie", V + "/Marie/data Marie.xlsx", "Data brut", V + "/Marie/analysis_EBC"),
]
US = 350.0
MATCH_TOL_S = 6.0


def klass(onset, us_delivered):
    """The pipeline's own rule, applied to a hand-read onset as well as a measured one."""
    if onset is None:
        return None
    return S.classify(onset, US, False, False, us_delivered)


def is_cr(cls):
    return cls is not None and cls.startswith("CR")


def scoreable(cls):
    return (cls is not None and cls != "in-progress at stimulus"
            and not cls.startswith("spontaneous"))


def read_manual(path, sheet):
    """Trial rows from a hand-scoring workbook: timestamp, onset in ms after CS onset."""
    ws = openpyxl.load_workbook(path, data_only=True)[sheet]
    out = []
    for r in ws.iter_rows(values_only=True):
        lab, ts, _frame, ms = (list(r) + [None] * 4)[:4]
        if lab is None:
            continue
        lab = str(lab).strip()
        if not (lab.isdigit() or lab.upper().startswith("UN")):
            continue
        t = (ts.hour * 60 + ts.minute + ts.second / 60.0) if isinstance(ts, datetime.time) else None
        out.append(dict(t=t, onset=float(ms) if isinstance(ms, (int, float)) and ms else None))
    return out


def read_auto(d):
    rows = []
    for fn, paired in (("trials_conditioning_CSUS.csv", True),
                       ("trials_conditioning_CSonly.csv", False)):
        p = os.path.join(d, fn)
        if not os.path.exists(p):
            continue
        for r in csv.DictReader(open(p, encoding="utf-8-sig")):
            o = r["scored_onset_ms"]
            rows.append(dict(t=float(r["session_clock_s"]), paired=paired,
                             block=int(r["block"]) if r.get("block") else None,
                             onset=float(o) if o not in ("", "None") else None))
    rows.sort(key=lambda a: a["t"])
    return rows


def match(man, auto):
    """Pair hand-scored trials with analyser trials by time on the session clock."""
    pairs, used = [], set()
    for m in [x for x in man if x["t"] is not None]:
        best, bd = None, 1e9
        for i, a in enumerate(auto):
            if i in used:
                continue
            dt = abs(a["t"] - m["t"])
            if dt < bd:
                best, bd = i, dt
        if best is not None and bd <= MATCH_TOL_S:
            used.add(best)
            pairs.append((m, auto[best]))
    return pairs


def curve(pairs, key):
    """CR rate per block, from one scorer's onsets, binned by the analyser's blocks."""
    per = {}
    for m, a in pairs:
        if a["block"] is None:
            continue
        onset = (m if key == "hand" else a)["onset"]
        cls = klass(onset, a["paired"])
        if not scoreable(cls):
            continue
        per.setdefault(a["block"], []).append(is_cr(cls))
    return {b: (100.0 * sum(v) / len(v), len(v)) for b, v in sorted(per.items()) if v}


def report(name, pairs):
    both = [(m, a) for m, a in pairs if m["onset"] and a["onset"] is not None]
    print("=" * 76)
    print("%s   %d trials matched, %d scored by both" % (name, len(pairs), len(both)))
    if len(both) < 8:
        print("  too few to compare")
        return None

    h = np.array([m["onset"] for m, a in both])
    c = np.array([a["onset"] for m, a in both])

    print()
    print("  onset distribution        hand      analyser     difference")
    print("    mean                  %6.1f      %6.1f      %+6.1f ms" % (h.mean(), c.mean(), c.mean() - h.mean()))
    print("    SD                    %6.1f      %6.1f      %+6.1f ms  <- must match"
          % (h.std(ddof=1), c.std(ddof=1), c.std(ddof=1) - h.std(ddof=1)))
    print("    IQR                   %6.1f      %6.1f" % (
        np.percentile(h, 75) - np.percentile(h, 25), np.percentile(c, 75) - np.percentile(c, 25)))
    sd_ratio = c.std(ddof=1) / max(h.std(ddof=1), 1e-9)
    print("    SD ratio  analyser/hand  %.2f   (1.00 = same spread)" % sd_ratio)

    # once the constant offset is removed, how much scatter is left?
    resid = (c - h) - np.median(c - h)
    print("    scatter left after removing the constant offset:  SD %.1f ms" % resid.std(ddof=1))

    ch, cc = curve(both, "hand"), curve(both, "auto")
    common = sorted(set(ch) & set(cc))
    print()
    print("  learning curve, CR%% per block")
    print("    block      " + "".join("%6d" % b for b in common))
    print("    n          " + "".join("%6d" % ch[b][1] for b in common))
    print("    hand       " + "".join("%6.0f" % ch[b][0] for b in common))
    print("    analyser   " + "".join("%6.0f" % cc[b][0] for b in common))
    r = None
    if len(common) >= 3:
        a1 = np.array([ch[b][0] for b in common])
        a2 = np.array([cc[b][0] for b in common])
        if a1.std() > 0 and a2.std() > 0:
            r = float(np.corrcoef(a1, a2)[0, 1])
        print("    per-block difference: mean %+.0f pts, largest %.0f pts%s"
              % ((a2 - a1).mean(), np.abs(a2 - a1).max(),
                 "" if r is None else ",  correlation r = %.2f" % r))

    hs = [klass(m["onset"], a["paired"]) for m, a in both]
    cs = [klass(a["onset"], a["paired"]) for m, a in both]
    hr = 100.0 * sum(is_cr(x) for x in hs) / sum(scoreable(x) for x in hs)
    cr = 100.0 * sum(is_cr(x) for x in cs) / sum(scoreable(x) for x in cs)
    print()
    print("  overall CR rate     hand %.0f%%     analyser %.0f%%     (%+.0f pts)" % (hr, cr, cr - hr))
    return dict(name=name, sd_ratio=sd_ratio, r=r, hand=hr, auto=cr,
                bias=float(np.median(c - h)), n=len(both))


def main():
    out = []
    for name, xl, sheet, d in STUDIES:
        if not (os.path.exists(xl) and os.path.exists(d)):
            print("%s: skipped, nothing to compare" % name)
            continue
        auto = read_auto(d)
        if not auto:
            print("%s: skipped, no analyser output yet" % name)
            continue
        r = report(name, match(read_manual(xl, sheet), auto))
        if r:
            out.append(r)

    if not out:
        return
    print("=" * 76)
    print("%-9s %6s %10s %10s %9s %9s %8s" %
          ("", "n", "SD ratio", "curve r", "hand CR", "auto CR", "offset"))
    for r in out:
        print("%-9s %6d %10.2f %10s %8.0f%% %8.0f%% %7.0fms" %
              (r["name"], r["n"], r["sd_ratio"],
               "-" if r["r"] is None else "%.2f" % r["r"], r["hand"], r["auto"], r["bias"]))
    print()
    print("Read the SD ratio and the curve correlation.  A constant offset is a difference")
    print("of definition between the two scorers and subtracts out; a spread that does not")
    print("match, or a curve that does not track, is the analyser measuring something else.")


if __name__ == "__main__":
    main()
