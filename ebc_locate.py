"""Decide where the stimulator box is in every recording of a study.

    python ebc_locate.py <config.json>

The CS LED is the one that can be found from scratch: it is lit for 400 ms at a time,
tens of times, and nothing else in a room does that.  The US LED cannot - a 50 ms flash
is one or two frames in a subsampled survey, which is not enough to tell it from sensor
noise or someone walking past.  So only the CS LED is searched for here, and the US LED
is found later inside a window pinned to it: the two sit a couple of centimetres apart
on the same box, and that relationship holds however the camera is aimed.

Recordings where the CS never fires - the US-only baseline - inherit the box position
from the rest of the participant's session, which is why this is a study-level step and
not a per-recording one.

Writes <out>/_work/leds.json.  Any entry can be overridden per recording in the study
config with  "led_yellow": [x, y].
"""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

import ebc_config as C
from ebc_paths import work_dir
from ebc_stimulus import survey, rank_blocks, merge_adjacent

CONFIDENT_MIN_PULSES = 5
CONFIDENT_MAX_CV = 0.25
CONFIDENT_MARGIN = 1.5      # top candidate must beat the runner-up by this factor
CONSENSUS_RADIUS = 280      # px; how far the box may sit from where the study says


def cluster(points, radius=CONSENSUS_RADIUS):
    """Group positions that are in the same place, biggest group first.

    Taking one median over the whole study assumes the box never moved.  When it did -
    the camera re-aimed between recordings, the tripod nudged, the box carried round to
    the participant's other side - a median lands between the two positions, which is
    where the box has never been, and every recording that inherits it inherits a wrong
    answer.  Clustering keeps the two positions apart so each recording can inherit the
    one that belongs to it.
    """
    groups = []
    for p in points:
        for g in groups:
            if (p["x"] - g[0]["x"]) ** 2 + (p["y"] - g[0]["y"]) ** 2 <= radius ** 2:
                g.append(p)
                break
        else:
            groups.append([p])
    groups.sort(key=lambda g: -len(g))
    return [dict(x=int(np.median([p["x"] for p in g])), y=int(np.median([p["y"] for p in g])),
                 tags=[p["tag"] for p in g], n=len(g)) for g in groups]


def main():
    cfg = C.load(sys.argv[1] if len(sys.argv) > 1 else None)
    wdir = work_dir(cfg)
    proto = cfg["protocol"]
    res = {}
    for rec in cfg["recordings"]:
        tag = rec["tag"]
        if not os.path.exists(rec["path"]):
            print("-- %s: file not found, skipped" % tag)
            continue
        over = rec.get("led_yellow")
        if over:
            res[tag] = dict(x=int(over[0]), y=int(over[1]), source="config",
                            confident=True, candidates=[])
            print("%-12s CS LED (%4d,%4d)  from config" % (tag, over[0], over[1]))
            continue
        if rec["role"] == "baseline_us":
            res[tag] = None          # filled from the consensus below
            continue
        d = survey(rec["path"], tag, wdir)
        kx, ky = int(d["w"]) / int(d["sw"]), int(d["h"]) / int(d["sh"])
        cands = merge_adjacent(rank_blocks(d["yb"], float(d["survey_fps"]), proto["cs_ms"],
                                           kx, ky, int(d["blk"]), proto["min_iti_s"]))
        if not cands:
            res[tag] = None
            print("%-12s CS LED not found in the survey" % tag)
            continue
        top = cands[0]
        margin = top["score"] / cands[1]["score"] if len(cands) > 1 else 99.0
        conf = (top["n_ok"] >= CONFIDENT_MIN_PULSES and top["dur_cv"] <= CONFIDENT_MAX_CV
                and margin >= CONFIDENT_MARGIN)
        res[tag] = dict(x=top["x"], y=top["y"], source="survey", confident=bool(conf),
                        n_pulses=top["n_ok"], dur_med_ms=round(top["dur_med"], 1),
                        dur_cv=round(top["dur_cv"], 3), score=round(top["score"], 1),
                        margin=round(margin, 2),
                        candidates=[{k: (round(v, 2) if isinstance(v, float) else v)
                                     for k, v in c.items()} for c in cands[:6]])
        print("%-12s CS LED (%4d,%4d)  %2d pulses  %.0f ms  cv=%.2f  margin=%.1fx  %s"
              % (tag, top["x"], top["y"], top["n_ok"], top["dur_med"], top["dur_cv"],
                 margin, "confident" if conf else "LOW CONFIDENCE"))

    good = [dict(v, tag=t) for t, v in res.items() if v and v["confident"]]
    if not good:
        raise SystemExit("no recording gave a confident CS LED position; "
                         "set led_yellow in the study config")

    # Where the box was, as one or more PLACES rather than one average.  A study where the
    # camera was re-aimed between takes has two of them, and a median over both lands
    # between them - where the box has never been - so every recording that inherits it
    # inherits a position that is wrong for both takes.
    groups = cluster(good)
    cx, cy = groups[0]["x"], groups[0]["y"]
    if len(groups) == 1:
        print("\nstudy box position: (%d,%d) from %d confident recording(s)"
              % (cx, cy, groups[0]["n"]))
    else:
        print("\nthe box was in %d different places in this study - a recording that "
              "cannot find it inherits the one nearest it in time:" % len(groups))
        for g in groups:
            print("   (%4d,%4d)  %d recording(s): %s"
                  % (g["x"], g["y"], g["n"], ", ".join(g["tags"])))

    # When each recording was made, so "nearest" means nearest in time rather than
    # nearest in the config file.  ebc_timeline puts elapsed_s there; without it (a study
    # whose files carry no camera dates) fall back to the order they are listed in.
    when = {}
    for i, rec in enumerate(cfg["recordings"]):
        e = rec.get("elapsed_s")
        when[rec["tag"]] = float(e) if e is not None else float(i) * 600.0

    def nearest_group(tag):
        """The place the box was in, in the recording closest in time to this one."""
        if len(groups) == 1 or tag not in when:
            return groups[0]
        best, best_dt = groups[0], None
        for g in groups:
            dts = [abs(when[tag] - when[t]) for t in g["tags"] if t in when]
            if not dts:
                continue
            if best_dt is None or min(dts) < best_dt:
                best, best_dt = g, min(dts)
        return best

    for tag, v in res.items():
        g = nearest_group(tag)
        gx, gy = g["x"], g["y"]
        why = "" if len(groups) == 1 else " (where it was in %s)" % ", ".join(g["tags"])
        if v is None:
            res[tag] = dict(x=gx, y=gy, source="consensus", confident=False,
                            candidates=[], near_xy=[gx, gy])
            print("%-12s CS LED (%4d,%4d)  inherited%s" % (tag, gx, gy, why))
            continue
        v["near_xy"] = [gx, gy]
        if v["confident"] or v["source"] == "config":
            continue
        near = [c for c in v["candidates"]
                if (c["x"] - gx) ** 2 + (c["y"] - gy) ** 2 <= CONSENSUS_RADIUS ** 2]
        if near:
            best = max(near, key=lambda c: c["score"])
            if (best["x"], best["y"]) != (v["x"], v["y"]):
                print("%-12s CS LED moved to (%4d,%4d): the better-scoring (%d,%d) is %d px "
                      "from the rest of the session" %
                      (tag, best["x"], best["y"], v["x"], v["y"],
                       int(((v["x"] - gx) ** 2 + (v["y"] - gy) ** 2) ** .5)))
            v.update(x=best["x"], y=best["y"], source="survey+consensus")
        else:
            print("%-12s no candidate within %d px of (%d,%d); using that position%s"
                  % (tag, CONSENSUS_RADIUS, gx, gy, why))
            v.update(x=gx, y=gy, source="consensus")

    out = dict(study=cfg["study"], consensus=[cx, cy], clusters=groups, leds=res)
    with open(os.path.join(wdir, "leds.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print("\n-> leds.json")


if __name__ == "__main__":
    main()
