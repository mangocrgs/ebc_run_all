"""Score every trial on one pooled closure scale.

    python ebc_score.py <config.json>

Reads the recovered trials and the eyelid traces; writes merged.json / merged_rows.json.

The closure scale is pooled across every recording of the participant, so a two-minute
extinction clip and a nine-minute conditioning chapter are on the same axis and their
amplitudes can be compared.  CS-only probes and the baselines are scored exactly like
paired trials but kept out of the acquisition summaries, because a trial with no US is
a different measurement.

The US-only baseline is scored first and separately, because the CR window is built from
it: the mean and SD of the unconditioned onsets give the latency below which no blink can
be a reaction to the stimulus that preceded it, and both edges of the CR window sit that
far after their own stimulus.  See ebc_config.cr_window().
"""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
from scipy.signal import savgol_filter

import ebc_config as C
from ebc_paths import work_dir

MAIN, RESET, PARTIAL = 0.40, 0.20, 0.15
BASE_FROM, BASE_TO = -300.0, -30.0
# How far past the stimulus a response is looked for is not a constant: a trace protocol
# puts the US - and so the UR - hundreds of milliseconds later than a delay one.  It comes
# from ebc_config.window(), which gives back the historical 1000 ms for the delay numbers
# this lab has been running.


def smooth(x):
    x = np.array([np.nan if v is None else v for v in x], float)
    m = np.isfinite(x)
    if m.sum() < 4:
        return None
    if m.sum() < len(x):
        x = np.interp(np.arange(len(x)), np.where(m)[0], x[m])
    return savgol_filter(x, 5, 2)


def excursions(C_, t, thr, search_ms):
    """Every upward excursion of the closure trace inside the response window.

    The onset is walked back down the rising edge to where the movement actually began,
    not to where it crossed the threshold.  `search_ms` is how far past CS onset the
    window runs, which the protocol decides.
    """
    w = np.where((t >= 0) & (t <= search_ms))[0]
    if not len(w):
        return []
    out, i = [], w[0]
    MS = t[1] - t[0]
    while i <= w[-1]:
        if C_[i] > thr:
            s = i
            while s > 0 and C_[s - 1] < C_[s] and C_[s - 1] > 0.04:
                s -= 1
            j = i
            while j < len(C_) - 1 and C_[j + 1] > RESET:
                j += 1
            pk = s + int(np.argmax(C_[s:j + 1]))
            r50 = pk
            while r50 < len(C_) - 1 and C_[r50] > 0.5 * C_[pk]:
                r50 += 1
            ro = pk
            while ro < len(C_) - 1 and C_[ro] > RESET:
                ro += 1
            out.append(dict(on=float(t[s]), pk=float(t[pk]), amp=float(C_[pk]),
                            r50=float(t[r50]), end=float(t[ro]), dur=float(t[ro] - t[s]),
                            rise=float((C_[pk] - C_[s]) / max(t[pk] - t[s], MS))))
            i = j + 1
        else:
            i += 1
    return out


def classify(onset_ms, win, moving, anchored_on_us):
    """What a blink at this latency is, given where this study's CR window sits.

    `win` is ebc_config.cr_window(): its two edges are the earliest latency at which the
    CS could have caused a blink, and the earliest at which the US could have.  Both are
    the measured reflex latency past their own stimulus when a US-only baseline was there
    to measure it, and fall back to the protocol's startle cut-off and the bare US onset
    when it was not.  Nothing here knows which of the two happened - it reads the window.
    """
    if moving:
        return win["moving_label"]
    if anchored_on_us:
        # No CS exists in this recording, so there is nothing to anticipate: every blink
        # is unconditioned, and the only cut is the one below which nothing can be a
        # response at all.  These are the trials the window itself is measured from, so
        # they are never classified with it.
        return win["ur_puff_label"] if onset_ms >= 20 else win["alpha_us_label"]
    if onset_ms < win["lo_ms"]:
        return win["alpha_label"]
    if onset_ms < win["hi_ms"]:
        return win["cr_label"]
    return win["ur_label"]


def reflex_from_us_only(rows, k=None, min_n=None):
    """How soon after the puff the unconditioned blink actually starts, measured.

    Given the scored US-only baseline trials, the mean and SD of their onsets and the
    latency mean - k*SD below which no blink can be a reaction to a stimulus delivered at
    time zero.  This is the number the CR window is then built from, at both ends.

    Only trials the scorer stands behind are used.  A US-only trial with the lid already
    moving at the puff has its "onset" hundreds of ms late - the second-look rule found a
    later blink - and one of those in a baseline of five would drag the mean out and blow
    the SD up until mean - 1.5 SD went negative.  Those trials are named in the returned
    dict rather than quietly dropped.
    """
    k = C.REFLEX_K if k is None else k
    min_n = C.REFLEX_MIN_N if min_n is None else min_n
    used, skipped = [], []
    for r in rows:
        why = None
        if r["scored_onset_ms"] is None:
            why = "no blink was detected"
        elif r["needs_manual_scoring"]:
            why = r["manual_scoring_because"]
        elif r["first_response_obscured"] == "yes":
            why = "the first movement in the window was an artefact"
        elif not str(r["scored_class"]).startswith("UR"):
            why = "scored %s, which is too soon to be the reflex" % r["scored_class"]
        (skipped if why else used).append(
            dict(session_name=r["session_name"], session_trial=r["session_trial"],
                 onset_ms=r["scored_onset_ms"], because=why))
    # One bad trial ruins a mean and an SD.  In a 35-trial US-only baseline a single
    # onset of 826 ms - a spontaneous blink scored long after the reflex was missed -
    # took the SD from 13 ms to 152 and drove mean - 1.5 SD to -129 ms, which is no
    # window at all.  So before the mean and the SD asked for are taken, onsets more
    # than REFLEX_OUTLIER_SD robust SDs from the median are set aside and named: the median and
    # the MAD are unmoved by an outlier, which is the whole point of using them to find
    # one.  Below REFLEX_OUTLIER_MIN_N there are too few onsets for a median to mean anything
    # and every one is kept.
    if len(used) >= C.REFLEX_OUTLIER_MIN_N:
        v = np.array([u["onset_ms"] for u in used], float)
        med = float(np.median(v))
        rsd = float(1.4826 * np.median(np.abs(v - med)))
        if rsd > 0:
            keep = []
            for u in used:
                if abs(u["onset_ms"] - med) > C.REFLEX_OUTLIER_SD * rsd:
                    u["because"] = ("%.0f ms is %.1f robust SDs from the %.0f ms median of "
                                    "this baseline - too far out to be the same reflex"
                                    % (u["onset_ms"], abs(u["onset_ms"] - med) / rsd, med))
                    skipped.append(u)
                else:
                    keep.append(u)
            used = keep

    o = [u["onset_ms"] for u in used]
    out = dict(n=len(o), k=k, used=used, skipped=skipped,
               onsets=[round(float(x), 1) for x in o],
               sessions=sorted({r["session_name"] for r in rows}))
    if len(o) < min_n:
        out.update(onset_ms=None, mean_ms=None, sd_ms=None,
                   why=("this study has no scored US-only baseline" if not rows else
                        "only %d of the %d US-only trial(s) could be used, and a mean and "
                        "an SD need at least %d" % (len(o), len(rows), min_n)))
        return out
    mean = float(np.mean(o))
    sd = float(np.std(o, ddof=1))
    ms = mean - k * sd
    out.update(mean_ms=round(mean, 1), sd_ms=round(sd, 1), onset_ms=round(ms, 1))
    if ms <= 0:
        out["why"] = ("the US-only onsets scatter too widely (%.0f +- %.0f ms over %d "
                      "trials) for mean - %.1f SD to land after the puff"
                      % (mean, sd, len(o), k))
    return out


# How far the CS onset re-found inside a trial window may sit from where the seek put it
# before the alignment is no longer worth trusting.  Two frames at 119.88 fps is 17 ms;
# past 25 ms something other than seek jitter is going on.
ALIGN_LIMIT_MS = 25.0
FACE_FLOOR_PCT = 80.0


def manual_reasons(row, win):
    """Why this trial cannot be left to the automatic score, in the user's words.

    Empty means the trial was scored cleanly.  Each entry is a fact about *this* trial -
    something measured in it that the scorer cannot see past - not a doubt about the
    recording as a whole, which triage already reports separately.
    """
    why = []
    # a US-only recording has no CS: naming one there describes a stimulus that was
    # never delivered
    stim = "the puff came" if row["trial_type"] == "US-only" else "the CS came on"
    if row["scored_class"] == win["moving_label"]:
        why.append("the lid was already closing when %s, and no later blink was found "
                   "to score instead" % stim)
    elif row["first_response_obscured"] == "yes" and row["secondary_onset_ms"] is None:
        why.append("the only movement in the window began within %.0f ms of the CS - too "
                   "soon for the CS to have caused it - and no later blink was found "
                   "behind it" % win["lo_ms"])
    if "truncated window" in (row["quality"] or ""):
        why.append("the recording ends before the trial window does, so the response may "
                   "be cut off")
    if row["face_tracked_pct"] < FACE_FLOOR_PCT:
        why.append("the face was tracked in only %.0f%% of the window"
                   % row["face_tracked_pct"])
    ae = row.get("alignment_error_ms")
    if ae is not None and abs(ae) > ALIGN_LIMIT_MS:
        why.append("the CS onset found inside the window is %+.0f ms from where the seek "
                   "put it, so time zero is uncertain" % ae)
    return why


def clock(seconds):
    """A position in its own recording, in the form a video player shows."""
    if seconds is None:
        return ""
    m, sec = divmod(float(seconds), 60.0)
    return "%d:%06.3f" % (int(m), sec)


def measure(tr, v, e, ms_per_frame, closed, search_ms):
    """Everything measured inside one trial window that the CR window does not touch.

    Kept apart from the labelling because the CR window is not known until the US-only
    baseline has been measured, and that baseline is measured from trials scored exactly
    like every other one.  So every trial is measured first, and labelled afterwards.
    """
    MS = ms_per_frame
    k0 = int(v["k0"])                       # LED onset re-found inside this window
    t = (np.arange(len(e)) - k0) * MS
    oref = float(np.percentile(e, 85))      # blink-robust open-eye reference
    span = max(oref - closed, 1e-6)
    Cl = np.clip((oref - e) / span, -0.3, 1.4)

    pre = (t >= BASE_FROM) & (t < BASE_TO)
    q = e[pre][Cl[pre] < 0.25]
    sd = (1.4826 * np.median(np.abs(q - np.median(q))) / span) if len(q) > 5 else 0.03
    thr = max(5 * sd, PARTIAL)
    exc = excursions(Cl, t, thr, search_ms)
    full = [b for b in exc if b["amp"] >= MAIN]
    part = [b for b in exc if b["amp"] < MAIN]
    return dict(tr=tr, v=v, t=t, C=Cl, k0=k0, MS=MS, full=full, part=part,
                first=(full[0] if full else (exc[0] if exc else None)),
                preflag=bool(pre.any() and Cl[pre].max() > 0.30),
                inprog=bool(Cl[k0] > 0.30),
                on_us=tr["trial_type"] == "US-only")


def build_row(m, win, study, proto, des):
    """One measured trial, labelled against this study's CR window."""
    tr, v, Cl, k0, MS = m["tr"], m["v"], m["C"], m["k0"], m["MS"]
    US_ONSET, CS_DUR = float(proto["us_onset_ms"]), float(proto["cs_ms"])
    full, part, b1, on_us = m["full"], m["part"], m["first"], m["on_us"]
    inprog = m["inprog"]

    cls = classify(b1["on"], win, inprog, on_us) if b1 else None
    # if the first event is too early to be a response to anything, or the lid was
    # already moving, look for a later blink in the same window - a real CR or UR may sit
    # behind the artefact.  "Too early" is the CR window's own lower edge, so in a study
    # with a US-only baseline it is the measured reflex latency, not a fixed 100 ms.
    obscured = bool(b1 and (inprog or (not on_us and b1["on"] < win["lo_ms"])))
    b2 = full[1] if (obscured and len(full) > 1) else None
    sec = classify(b2["on"], win, False, on_us) if b2 else None

    def at(ms):
        return float(Cl[min(max(k0 + int(round(ms / MS)), 0), len(Cl) - 1)])

    row = dict(
        study=study, role=tr["role"], session=tr["session"],
        session_name=tr["session_name"], session_trial=tr["session_trial"],
        trial_type=tr["trial_type"], block=tr.get("block"),
        trial_in_block=tr.get("trial_in_block"), global_trial=tr.get("global_trial"),
        cs_onset_video_s=tr["cs_onset_s"], us_onset_video_s=tr["us_onset_s"],
        session_clock_s=tr.get("session_clock_s"),
        cs_duration_ms=tr["cs_duration_ms"], us_duration_ms=tr["us_duration_ms"],
        cs_timing=tr.get("cs_timing", "measured from CS LED"),
        block_closed_by=tr.get("block_closed_by"),
        measured_isi_ms=tr["isi_ms"],
        alignment_error_ms=v.get("align_error_ms"),
        face_tracked_pct=round(100 * v.get("face_ok", 0), 1),
        quality=(("pre-CS blink" if m["preflag"] else "")
                 + (" | lid closing at onset" if inprog else "")
                 + (" | truncated window" if tr.get("truncated") else "")
                 + (" | face lost >20%" if v.get("face_ok", 1) < 0.8 else "")).strip(" |")
                or "clean",
        n_full_blinks=len(full), n_partial_movements=len(part),
        blink_onset_ms=None if not b1 else round(b1["on"], 1),
        peak_closure_ms=None if not b1 else round(b1["pk"], 1),
        peak_closure_pct=None if not b1 else round(b1["amp"] * 100, 1),
        closing_speed_pct_per_ms=None if not b1 else round(b1["rise"] * 100, 2),
        closure_duration_ms=None if not b1 else round(b1["dur"], 1),
        reopen_half_ms=None if not b1 else round(b1["r50"], 1),
        reopen_full_ms=None if not b1 else round(b1["end"], 1),
        closure_at_US_pct=round(at(US_ONSET) * 100, 1),
        closure_at_CSoff_pct=round(at(CS_DUR) * 100, 1),
        # In a trace protocol the CS is long over before the US arrives, so how closed
        # the lid was across the empty gap is a measurement in its own right.  It is
        # None for a delay design, where there is no gap to sample.
        closure_at_midgap_pct=(round(at((CS_DUR + US_ONSET) / 2) * 100, 1)
                               if des["kind"] == "trace" else None),
        closure_at_1000ms_pct=round(at(1000) * 100, 1),
        closed_at_US=bool(at(US_ONSET) >= 0.50),
        reopened_before_US=bool(b1 is not None and at(US_ONSET) < 0.30),
        response_class=cls,
        first_response_obscured="yes" if obscured else "",
        secondary_onset_ms=None if not b2 else round(b2["on"], 1),
        secondary_peak_pct=None if not b2 else round(b2["amp"] * 100, 1),
        secondary_class=sec,
        scored_onset_ms=(round(b2["on"], 1) if b2 else
                         (None if not b1 else round(b1["on"], 1))),
        scored_class=(sec if b2 else cls),
        all_blink_onsets_ms=";".join("%.0f" % b["on"] for b in full),
        all_blink_amps_pct=";".join("%.0f" % (b["amp"] * 100) for b in full),
        inter_blink_ms=";".join("%.0f" % (full[k + 1]["on"] - full[k]["on"])
                                for k in range(len(full) - 1)),
        partial_movement_ms=";".join("%.0f(%.0f%%)" % (b["on"], b["amp"] * 100)
                                     for b in part))
    why = manual_reasons(row, win)
    row["needs_manual_scoring"] = "yes" if why else ""
    row["manual_scoring_because"] = "; ".join(why)
    return row


def main():
    cfg = C.load(sys.argv[1] if len(sys.argv) > 1 else None)
    wdir = work_dir(cfg)
    proto = cfg["protocol"]
    US_ONSET, CS_DUR = proto["us_onset_ms"], proto["cs_ms"]
    DES = C.design(proto)
    _, _, SEARCH_MS = C.window(proto)
    print("%s: CS %.0f ms, US %.0f ms at %.0f ms (%s); responses searched to %.0f ms"
          % (DES["label"], CS_DUR, proto["us_dur_ms"], US_ONSET, DES["short"], SEARCH_MS))

    with open(os.path.join(wdir, "trials.json"), encoding="utf-8") as fh:
        TR = json.load(fh)
    traces = {}
    for rec in cfg["recordings"]:
        f = os.path.join(wdir, rec["tag"] + "_traces.json")
        if os.path.exists(f):
            with open(f, encoding="utf-8") as fh:
                traces[rec["tag"]] = json.load(fh)

    fps = {s["tag"]: s["fps"] for s in TR["sessions"]}
    E = {}
    for tag, d in traces.items():
        for k, v in d.items():
            a, b = smooth(v["er"]), smooth(v["el"])
            if a is None and b is None:
                continue
            E[(tag, int(k))] = (a if b is None else b if a is None else (a + b) / 2)
    if not E:
        raise SystemExit("no eyelid traces - run ebc_eyes.py first")
    CLOSED = float(np.percentile([e.min() for e in E.values()], 10))
    print("pooled full-closure EAR reference = %.4f  (%d trials, %d recordings)"
          % (CLOSED, len(E), len(traces)))

    keep_traces = {}
    MEAS = []
    for tr in TR["trials"]:
        tag, ti = tr["session"], tr["session_trial"]
        if (tag, ti) not in E:
            continue
        v = traces[tag][str(ti)]
        m = measure(tr, v, E[(tag, ti)], 1000.0 / fps[tag], CLOSED, SEARCH_MS)
        keep_traces.setdefault(tag, {})[str(ti)] = dict(
            t=[round(float(x), 2) for x in m["t"]],
            C=[round(float(x), 4) for x in m["C"]])
        MEAS.append(m)

    # ---- where the CR window sits, measured before anything is called a CR ----------
    # The US-only trials are scored first, against a window they do not use: a recording
    # with no CS has nothing to anticipate, so its labels come from the puff alone.  The
    # reflex latency measured across them then sets the window every other trial is read
    # against.  With no US-only baseline in the study, cr_window() falls back to the
    # protocol's startle cut-off and the run scores exactly as it did before.
    FALLBACK = C.cr_window(proto)
    us_rows = [build_row(m, FALLBACK, cfg["study"], proto, DES)
               for m in MEAS if m["on_us"]]
    REFLEX = reflex_from_us_only(us_rows)
    WIN = C.cr_window(proto, REFLEX)
    if WIN["measured"]:
        print("reflex onset measured from %d US-only trial(s): %.0f +- %.0f ms after the "
              "puff, mean - %.1f SD = %.0f ms"
              % (REFLEX["n"], REFLEX["mean_ms"], REFLEX["sd_ms"], REFLEX["k"],
                 REFLEX["onset_ms"]))
    else:
        print("reflex onset not measured (%s); falling back to the protocol's %.0f ms "
              "startle cut-off" % (REFLEX.get("why") or "no US-only baseline",
                                   float(proto["alpha_ms"])))
    print("CR window = %.0f-%.0f ms from CS onset  [%s]"
          % (WIN["lo_ms"], WIN["hi_ms"], WIN["cr_label"]))

    rows = [build_row(m, WIN, cfg["study"], proto, DES) for m in MEAS]

    order = {r: i for i, r in enumerate(C.ROLES)}
    rows.sort(key=lambda r: (order[r["role"]], r["session"], r["session_trial"]))
    seen = {}
    for r in rows:
        k = (r["role"], r["trial_type"])
        seen[k] = seen.get(k, 0) + 1
        r["group_index"] = seen[k]

    # Trials the scorer will not stand behind, gathered where the page and the console can
    # both read them.  The paired conditioning trials come first because those are the
    # measurement - a probe or a baseline trial matters less, and is listed after.
    manual = []
    for r in rows:
        if not r["needs_manual_scoring"]:
            continue
        manual.append(dict(
            session=r["session"], session_name=r["session_name"], role=r["role"],
            trial_type=r["trial_type"], session_trial=r["session_trial"],
            global_trial=r.get("global_trial"), block=r.get("block"),
            at_s=r["cs_onset_video_s"] if r["cs_onset_video_s"] is not None
            else r["us_onset_video_s"],
            at=clock(r["cs_onset_video_s"] if r["cs_onset_video_s"] is not None
                     else r["us_onset_video_s"]),
            session_clock_s=r.get("session_clock_s"),
            scored_class=r["scored_class"], scored_onset_ms=r["scored_onset_ms"],
            because=r["manual_scoring_because"]))
    key_role = {r: i for i, r in enumerate(C.ROLES)}
    manual.sort(key=lambda m: (0 if (m["role"] == "conditioning"
                                     and m["trial_type"] == "CS-US") else 1,
                               key_role[m["role"]], m["session"], m["session_trial"]))
    n_paired = sum(1 for m in manual
                   if m["role"] == "conditioning" and m["trial_type"] == "CS-US")

    merged = dict(study=cfg["study"], protocol=proto, design=DES, closed_ref=CLOSED,
                  cr_window=WIN, reflex=REFLEX,
                  manual_review=dict(trials=manual, n_total=len(manual),
                                     n_conditioning_paired=n_paired,
                                     n_scoreable=sum(1 for r in rows
                                                     if not r["needs_manual_scoring"])),
                  sessions=TR["sessions"], checks=TR["checks"], offsets=TR["offsets"],
                  traces=keep_traces,
                  recordings=[{k: rec[k] for k in ("tag", "label", "role", "order")}
                              for rec in cfg["recordings"]])
    with open(os.path.join(wdir, "merged.json"), "w", encoding="utf-8") as fh:
        json.dump(merged, fh)
    with open(os.path.join(wdir, "merged_rows.json"), "w", encoding="utf-8") as fh:
        json.dump(rows, fh)

    print()
    for role, tt in (("conditioning", "CS-US"), ("conditioning", "CS-only"),
                     ("extinction", "CS-only"), ("baseline_cs", "CS-only"),
                     ("baseline_us", "US-only")):
        rs = [r for r in rows if r["role"] == role and r["trial_type"] == tt]
        if not rs:
            continue
        sc = [r for r in rs if r["scored_class"] not in (None, WIN["moving_label"])]
        cr = [r for r in sc if str(r["scored_class"]).startswith("CR")]
        rec = [r for r in rs if r["first_response_obscured"] == "yes"
               and r["secondary_onset_ms"] is not None]
        o = [r["scored_onset_ms"] for r in cr]
        line = "%12s / %-8s: %3d trials, %3d scoreable, %3d CR" % (role, tt, len(rs), len(sc), len(cr))
        if o:
            line += " (%.0f%%), mean CR onset %.0f ms" % (len(cr) / len(sc) * 100, float(np.mean(o)))
        print(line + "   [%d recovered behind an artefact]" % len(rec))

    if manual:
        print("\n%d trial(s) could not be scored with confidence and should be read off "
              "the video by hand:" % len(manual))
        if n_paired:
            print("  %d of them are paired CS-US conditioning trials - the measurement "
                  "itself." % n_paired)
        shown = 0
        for m in manual:
            if shown >= 25:
                print("  ... and %d more, all listed in trials_to_score_by_hand.csv"
                      % (len(manual) - shown))
                break
            print("  %-14s trial %-3d at %-10s %-8s %s"
                  % (m["session_name"], m["session_trial"], m["at"],
                     "block %s" % m["block"] if m["block"] else "", m["because"]))
            shown += 1


if __name__ == "__main__":
    main()
