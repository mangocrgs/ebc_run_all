"""Compare the hand scoring against what EBC Analyzer produced, trial by trial.

    python compare_manual.py

The hand sheets hold, per conditioning trial, a timestamp and the blink onset in ms
after CS onset, read off frame by frame at 120 fps.  Trials are matched to the analyser's
output by time on the session clock, and BOTH sets of onsets are then put through the
same classification rule, so the comparison is of the measurement and nothing else.
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
    ("Carole", V + "/Carole/data Carole 08.09.xlsx", "Sheet1",
     V + "/analysis_EBC/Carole"),
    ("Thomas", V + "/Thomas/data Thomas.xlsx", "Data brut",
     V + "/analysis_EBC/Thomas"),
    ("Marie", V + "/Marie/data Marie.xlsx", "Data brut",
     V + "/Marie/analysis_EBC"),
]
LO, HI = S.RESP_OFFSET_MS, 350.0 + S.RESP_OFFSET_MS
FRAME_MS = 1000.0 / 119.88


def klass(o):
    if o is None:
        return None
    if o < LO:
        return "alpha"
    if o > S.RESP_MAX_MS:
        return "spont"
    return "CR" if o < HI else "UR"


def read_manual(path, sheet):
    ws = openpyxl.load_workbook(path, data_only=True)[sheet]
    out = []
    for r in ws.iter_rows(values_only=True):
        lab, ts, _fr, ms = (list(r) + [None] * 4)[:4]
        if lab is None:
            continue
        lab = str(lab).strip()
        if not (lab.isdigit() or lab.upper().startswith("UN")):
            continue
        t = (ts.hour * 60 + ts.minute + ts.second / 60.0) if isinstance(ts, datetime.time) else None
        out.append(dict(t=t, unpaired=lab.upper().startswith("UN"),
                        onset=float(ms) if isinstance(ms, (int, float)) and ms else None))
    return out


def read_auto(d):
    out = []
    for fn in ("trials_conditioning_CSUS.csv", "trials_conditioning_CSonly.csv"):
        p = os.path.join(d, fn)
        if not os.path.exists(p):
            continue
        for r in csv.DictReader(open(p, encoding="utf-8-sig")):
            o = r["scored_onset_ms"]
            out.append(dict(t=float(r["session_clock_s"]), cls=r["scored_class"],
                            onset=float(o) if o not in ("", "None") else None))
    out.sort(key=lambda a: a["t"])
    return out


def match(man, auto, tol=6.0):
    pairs, used = [], set()
    for m in [x for x in man if x["t"] is not None]:
        best, bd = None, 1e9
        for i, a in enumerate(auto):
            if i in used:
                continue
            dt = abs(a["t"] - m["t"])
            if dt < bd:
                best, bd = i, dt
        if best is not None and bd <= tol:
            used.add(best)
            pairs.append((m, auto[best]))
    return pairs


def report(name, man, auto, pairs):
    both = [(m, a) for m, a in pairs if m["onset"] and a["onset"] is not None]
    print("=" * 74)
    print("%s   %d hand-scored trials, %d from the analyser, %d matched, %d with an "
          "onset from both" % (name, len(man), len(auto), len(pairs), len(both)))
    if not both:
        print("  nothing to compare")
        return None
    d = np.array([a["onset"] - m["onset"] for m, a in both])
    print("  onset difference (analyser - hand): median %+6.1f ms   mean %+6.1f ms   "
          "SD %5.1f ms" % (np.median(d), d.mean(), d.std(ddof=1)))
    for lim, what in ((FRAME_MS, "1 frame"), (2 * FRAME_MS, "2 frames"), (50.0, "50 ms")):
        print("     within %-8s %3d/%d  (%3.0f%%)"
              % (what, int((abs(d) <= lim).sum()), len(d), 100 * (abs(d) <= lim).mean()))
    mk = [klass(m["onset"]) for m, a in both]
    ak = [klass(a["onset"]) for m, a in both]
    agree = sum(x == y for x, y in zip(mk, ak))
    print("  same rule applied to both, CR window %.0f-%.0f ms:" % (LO, HI))
    print("     class agreement  %d/%d = %.0f%%" % (agree, len(mk), 100 * agree / len(mk)))
    ks = ["alpha", "CR", "UR", "spont"]
    print("     hand vs analyser " + "".join("%8s" % k for k in ks))
    for k in ks:
        row = [sum(1 for x, y in zip(mk, ak) if x == k and y == j) for j in ks]
        if sum(row):
            print("     %-15s" % k + "".join("%8d" % v for v in row))
    mcr, acr = mk.count("CR"), ak.count("CR")
    print("     CR rate   hand %d/%d = %2.0f%%     analyser %d/%d = %2.0f%%"
          % (mcr, len(mk), 100 * mcr / len(mk), acr, len(ak), 100 * acr / len(ak)))
    return d


def main():
    alld = []
    for name, xl, sheet, d in STUDIES:
        if not (os.path.exists(xl) and os.path.exists(d)):
            print("%s: skipped (missing %s)" % (name, xl if not os.path.exists(xl) else d))
            continue
        man, auto = read_manual(xl, sheet), read_auto(d)
        if not auto:
            print("%s: skipped - no analyser output yet" % name)
            continue
        r = report(name, man, auto, match(man, auto))
        if r is not None:
            alld.append(r)
    if alld:
        a = np.concatenate(alld)
        print("=" * 74)
        print("ALL PARTICIPANTS  n=%d   median %+.1f ms   mean %+.1f ms   SD %.1f ms"
              % (len(a), np.median(a), a.mean(), a.std(ddof=1)))
        print("  within 1 frame %.0f%%   within 2 frames %.0f%%   within 50 ms %.0f%%"
              % (100 * (abs(a) <= FRAME_MS).mean(), 100 * (abs(a) <= 2 * FRAME_MS).mean(),
                 100 * (abs(a) <= 50).mean()))


if __name__ == "__main__":
    main()
