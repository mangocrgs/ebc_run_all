"""Turn detected LED pulses into trials, and check them against the protocol.

    python ebc_protocol.py <config.json>

The protocol is used as a *test*, never as an instruction.  Trials are built from what
the LEDs actually did; the expected structure - nine paired trials then one CS-only
probe, ten times over - is then compared against the recovered sequence and the
agreement (or the exact disagreement) is reported.  Nothing is renumbered to fit.

Writes <out>/_work/trials.json and prints the protocol report.
"""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

import ebc_config as C
from ebc_paths import work_dir


def load_stim(cfg, wdir):
    out = []
    for rec in cfg["recordings"]:
        f = os.path.join(wdir, rec["tag"] + "_stim.json")
        if os.path.exists(f):
            with open(f, encoding="utf-8") as fh:
                out.append(json.load(fh))
        else:
            print("-- %s: no _stim.json, skipped" % rec["tag"])
    order = {r: i for i, r in enumerate(C.ROLES)}
    out.sort(key=lambda s: (order[s["role"]], s.get("order", 1)))
    return out


def pair_cs_us(cs, us, proto):
    """Attach to each CS the US that falls inside its window, if any.

    The window runs from CS onset to CS offset plus a little slack, so a US that is
    delivered at the nominal 350 ms, or anywhere else inside the CS, is found - and a
    blue flash seconds later is not mistaken for one.
    """
    slack = 0.12
    lim = (proto["cs_ms"] + 120.0) / 1000.0
    free = [u for u in us if u["ok"]]
    used = set()
    pairs = []
    for c in cs:
        hit = None
        for i, u in enumerate(free):
            if i in used:
                continue
            dt = u["t"] - c["t"]
            if -slack <= dt <= lim:
                hit = (i, u)
                break
        if hit:
            used.add(hit[0])
            pairs.append((c, hit[1]))
        else:
            pairs.append((c, None))
    unpaired_us = [u for i, u in enumerate(free) if i not in used]
    return pairs, unpaired_us


def build(cfg):
    wdir = work_dir(cfg)
    proto = cfg["protocol"]
    stim = load_stim(cfg, wdir)
    by_tag = {s["tag"]: s for s in stim}

    # conditioning chapters share one clock: a trial's position in the session is its
    # time in its own file plus the length of every chapter before it
    offset, acc = {}, 0.0
    for s in stim:
        if s["role"] == "conditioning":
            offset[s["tag"]] = acc
            acc += float(s["duration_s"])

    trials, report = [], []
    for s in stim:
        cs = [e for e in s["events"].get("yellow", []) if e["ok"]]
        us_all = s["events"].get("blue", [])
        pairs, unpaired = pair_cs_us(cs, us_all, proto)
        fps = s["fps"]
        dur = s["duration_s"]
        n_par = sum(1 for _, u in pairs if u)
        rows = []
        if s["role"] == "baseline_us":
            for i, u in enumerate([u for u in us_all if u["ok"]], 1):
                rows.append(dict(session=s["tag"], session_name=s["label"], role=s["role"],
                                 session_trial=i, trial_type="US-only",
                                 cs_onset_s=None, cs_frame=None, cs_duration_ms=None,
                                 us_onset_s=round(u["t"], 4), us_frame=u["frame"],
                                 us_duration_ms=u["dur_ms"], isi_ms=None,
                                 anchor_frame=u["frame"], anchor_s=round(u["t"], 4)))
        else:
            for i, (c, u) in enumerate(pairs, 1):
                rows.append(dict(session=s["tag"], session_name=s["label"], role=s["role"],
                                 session_trial=i,
                                 trial_type="CS-US" if u else "CS-only",
                                 cs_onset_s=round(c["t"], 4), cs_frame=c["frame"],
                                 cs_duration_ms=c["dur_ms"],
                                 us_onset_s=round(u["t"], 4) if u else None,
                                 us_frame=u["frame"] if u else None,
                                 us_duration_ms=u["dur_ms"] if u else None,
                                 isi_ms=round((u["t"] - c["t"]) * 1000, 1) if u else None,
                                 anchor_frame=c["frame"], anchor_s=round(c["t"], 4)))
        for r in rows:
            if s["tag"] in offset:
                r["session_clock_s"] = round(r["anchor_s"] + offset[s["tag"]], 3)
            r["truncated"] = bool(r["anchor_s"] + 1.2 > dur or r["anchor_s"] < 0.35)
        trials += rows
        report.append(dict(tag=s["tag"], label=s["label"], role=s["role"],
                           duration_s=dur, fps=fps,
                           n_cs=len(cs), n_paired=n_par, n_cs_only=len(cs) - n_par,
                           n_us_unpaired=len(unpaired), n_trials=len(rows),
                           cs_rejected=sum(1 for e in s["events"].get("yellow", []) if not e["ok"]),
                           us_rejected=sum(1 for e in us_all if not e["ok"]),
                           isi_ms=_median([r["isi_ms"] for r in rows if r["isi_ms"] is not None]),
                           truncated=sum(r["truncated"] for r in rows)))

    # ---- block structure over the conditioning group, in the order it was run --------
    cond = [t for t in trials if t["role"] == "conditioning"]
    cond.sort(key=lambda t: t.get("session_clock_s", t["anchor_s"]))
    # A block is what the protocol says it is: a run of paired trials closed by a CS-only
    # probe.  Numbering by the probes rather than by counting off ten trials at a time means
    # a block that came up short does not shift every block after it.
    b, k = 1, 0
    runs, run = [], 0
    for i, t in enumerate(cond):
        k += 1
        t["global_trial"] = i + 1
        t["block"] = b
        t["trial_in_block"] = k
        if t["trial_type"] == "CS-US":
            run += 1
        else:
            runs.append(run)
            run = 0
            b, k = b + 1, 0
    tail = run
    want_runs = [proto["paired_per_block"]] * proto["n_blocks"]
    strict = (runs == want_runs and tail == 0)

    for role in ("extinction", "baseline_cs", "baseline_us"):
        g = [t for t in trials if t["role"] == role]
        g.sort(key=lambda t: (t["session"], t["anchor_s"]))
        for i, t in enumerate(g):
            t["global_trial"] = i + 1
            t.setdefault("block", None)
            t.setdefault("trial_in_block", i + 1)

    check = dict(
        expected_trials=C.expected_trials(proto),
        found_conditioning_trials=len(cond),
        expected_paired=proto["paired_per_block"] * proto["n_blocks"],
        found_paired=sum(1 for t in cond if t["trial_type"] == "CS-US"),
        expected_cs_only=proto["cs_only_per_block"] * proto["n_blocks"],
        found_cs_only=sum(1 for t in cond if t["trial_type"] == "CS-only"),
        paired_runs_before_each_probe=runs,
        trailing_paired_without_probe=tail,
        strict_block_structure=bool(strict),
        expected_runs=want_runs,
        short_blocks=[{"block": i + 1, "paired": n,
                       "expected": proto["paired_per_block"]}
                      for i, n in enumerate(runs) if n != proto["paired_per_block"]],
        blocks_found=len(runs),
    )
    out = dict(study=cfg["study"], protocol=proto, offsets=offset,
               sessions=report, checks=check, trials=trials)
    with open(os.path.join(wdir, "trials.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    return out


def _median(v):
    return round(float(np.median(v)), 1) if v else None


def print_report(res):
    print("\n%-14s %-13s %8s %6s %7s %8s %7s %8s %9s" %
          ("recording", "role", "dur (s)", "CS", "paired", "CS-only", "US un-", "CS rej", "ISI (ms)"))
    for s in res["sessions"]:
        print("%-14s %-13s %8.1f %6d %7d %8d %7d %8d %9s" %
              (s["tag"], s["role"], s["duration_s"], s["n_cs"], s["n_paired"],
               s["n_cs_only"], s["n_us_unpaired"], s["cs_rejected"],
               "-" if s["isi_ms"] is None else "%.1f" % s["isi_ms"]))
    c = res["checks"]
    p = res["protocol"]
    print("\nProtocol check  (%d blocks of %d paired + %d CS-only)" %
          (p["n_blocks"], p["paired_per_block"], p["cs_only_per_block"]))
    print("  conditioning trials   %d found / %d expected" %
          (c["found_conditioning_trials"], c["expected_trials"]))
    print("  paired CS-US          %d found / %d expected" % (c["found_paired"], c["expected_paired"]))
    print("  CS-only probes        %d found / %d expected" % (c["found_cs_only"], c["expected_cs_only"]))
    print("  paired run before each probe: %s" % c["paired_runs_before_each_probe"])
    if c["trailing_paired_without_probe"]:
        print("  %d paired trials after the last probe (no probe closed the block)"
              % c["trailing_paired_without_probe"])
    for sb in c.get("short_blocks", []):
        print("  block %d has %d paired trials, not %d - the CS-only probe closed it early"
              % (sb["block"], sb["paired"], sb["expected"]))
    print("  blocks recovered      %d / %d expected" % (c["blocks_found"], p["n_blocks"]))
    print("  strict %d+%d x %d structure: %s" %
          (p["paired_per_block"], p["cs_only_per_block"], p["n_blocks"],
           "YES" if c["strict_block_structure"] else "NO - the deviations are listed above"))


if __name__ == "__main__":
    cfg = C.load(sys.argv[1] if len(sys.argv) > 1 else None)
    print_report(build(cfg))
