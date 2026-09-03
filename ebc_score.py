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


def classify(onset_ms, win, moving, anchored_on_us, us_delivered=True):
    """What a blink at this latency is, given where this study's CR window sits.

    `us_delivered` says whether a puff arrived on THIS trial.  The window's upper edge is
    the earliest the puff could have caused a blink, so on a probe, an extinction trial
    or a CS-only baseline - none of which deliver one - that edge cannot mean "UR".  It
    still bounds the CR, because a probe exists to be comparable with the paired trials
    around it; a response past it is a late response.  Calling every blink on a no-US
    trial a CR is the mirror of calling them all URs: a CS-only baseline would report a
    100% false-positive rate by construction, and extinction, whose whole job is to show
    responding decay, could only ever read 100%.

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
        return win["cr_label"] if us_delivered else win["cr_no_us_label"]
    return win["ur_label"] if us_delivered else win["late_no_us_label"]


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


BANNER = "=" * 78


def no_us_baseline(cfg, proto, reflex, win):
    """No measured reflex: say what that costs, and make the choice the user's.

    The pipeline will not pick for you.  Running on the protocol's standard window is a
    defensible thing to do and it is what happens if you do nothing - but it is a
    different measurement from one made against this person's own reflex, and a workbook
    that does not say which it was cannot be defended later.
    """
    print(BANNER)
    print("NO US-ONLY BASELINE - THE CR WINDOW IS NOT MEASURED FOR THIS PARTICIPANT")
    print(BANNER)
    print("  %s." % (reflex.get("why") or "there is no US-only recording in this study"))
    print("""
  The CR window is normally the participant's own blink reflex: the mean onset of their
  unconditioned blinks minus 1.5 SD, applied after the CS and again after the US.  With
  no US-only baseline there is nothing to measure it from, and you have to choose:

    1  MEASURE IT BY HAND.  Read a dozen unconditioned blinks off the US-only video,
       frame by frame, and put the latency in the study file:

           "protocol": { ..., "reflex_ms": 43 }

       The run then uses that number and every workbook says it was entered by hand.

    2  USE THE STANDARD WINDOW.  %.0f-%.0f ms after CS onset - the protocol's startle
       cut-off to the nominal US onset.  This is what is being used right now.  It
       treats both stimuli as if a blink could follow them instantly, so a blink in the
       first %.0f ms after the puff is counted as a reaction to the puff when it may be
       a late CR.

  Option 2 is running.  It is a real choice with a real cost, not a default to be
  passed over - and comparing a participant scored one way with one scored the other is
  comparing two different measurements.  Record ten to fifteen US-only trials in the
  next session and neither option is needed.""" % (
        win["lo_ms"], win["hi_ms"], float(proto["us_onset_ms"]) - win["lo_ms"]))
    print(BANNER)


def cs_baseline(cfg, rows, win):
    """Step one: what the CS alone does, before any pairing has happened.

    Two different things come out of a CS-only baseline, and they are not
    interchangeable:

      startle        a blink that begins BEFORE the CR window opens - sooner after the
                     CS than any stimulus can be responded to.  It is a reaction to the
                     onset of the stimulus itself, not to what the stimulus predicts.
                     It is what the window's lower edge exists to exclude, and a
                     participant who does it often will have paired trials thrown out
                     for the same reason, so it is flagged here rather than discovered
                     block by block.
      CS-alone       a blink INSIDE the CR window on a recording where no US is ever
      responding     delivered.  There has been no pairing, so none of it can be
                     conditioned: this is the false-positive rate, and it is the number
                     the conditioning CR rate has to be read against.

    Both are counted, neither is judged.  There is no threshold here that says how much
    startle is too much - that is a fact about a laboratory and a population, not about
    this recording, and inventing one would put a pass/fail label on a number the
    analyser has no standing to grade.  The rates are stated, and stated again next to
    every CR rate downstream.

    Returns the summary, or None when there is no CS-only baseline to summarise.
    """
    got = [r for r in rows if r["role"] == "baseline_cs"]
    if not got:
        no_cs_baseline(cfg)
        return None

    scoreable = [r for r in got if r["scored_class"] not in (None, win["moving_label"])]
    startle = [r for r in scoreable if r["scored_class"] == win["alpha_label"]]
    inwin = [r for r in scoreable if r["scored_class"] == win["cr_no_us_label"]]
    late = [r for r in scoreable if r["scored_class"] == win["late_no_us_label"]]
    n = len(scoreable)
    pct = lambda k: (100.0 * k / n) if n else None

    out = dict(n_trials=len(got), n_scoreable=n,
               n_startle=len(startle), n_in_window=len(inwin), n_late=len(late),
               startle_pct=pct(len(startle)), false_positive_pct=pct(len(inwin)),
               startle_onsets_ms=[r["scored_onset_ms"] for r in startle
                                  if r["scored_onset_ms"] is not None],
               window_lo_ms=win["lo_ms"], window_hi_ms=win["hi_ms"],
               sessions=sorted({r["session_name"] for r in got}))
    o = [r["scored_onset_ms"] for r in inwin if r["scored_onset_ms"] is not None]
    if o:
        out["in_window_mean_ms"] = round(float(np.mean(o)), 1)
        out["in_window_sd_ms"] = round(float(np.std(o, ddof=1)), 1) if len(o) > 1 else None

    print("  %s: %d trial(s), %d scoreable" %
          (", ".join(out["sessions"]), out["n_trials"], n))
    if not n:
        print("  none of them could be scored, so the CS alone tells us nothing here.")
        return out
    print("  startle (blink before %.0f ms, too soon to be a response)   %3d / %-3d  %5.1f%%"
          % (win["lo_ms"], len(startle), n, out["startle_pct"]))
    print("  in the CR window with no US anywhere - false positives      %3d / %-3d  %5.1f%%"
          % (len(inwin), n, out["false_positive_pct"]))
    print("  later than the window closes                                %3d / %-3d  %5.1f%%"
          % (len(late), n, pct(len(late))))

    if startle:
        print()
        print(BANNER)
        print("STARTLE TO THE CS ALONE - %d of %d baseline trials (%.0f%%)"
              % (len(startle), n, out["startle_pct"]))
        print(BANNER)
        if out["startle_onsets_ms"]:
            print("  onsets: %s ms - all before the CR window opens at %.0f ms."
                  % (", ".join("%.0f" % v for v in sorted(out["startle_onsets_ms"])),
                     win["lo_ms"]))
        print("""
  These blinks began too soon after the CS for the CS to have caused them, so they are
  not conditioned responses and they are not counted as any.  The flag is here because
  the same reaction will occur on paired trials, where it does two things: it is
  excluded from the CR count, and a lid already moving when the puff arrives can hide
  the response that follows it.  Expect this participant's paired trials to carry more
  in-progress-at-stimulus exclusions than usual, and read the CR rate as a fraction of
  the trials that survived rather than of the trials that were run.""")
        print(BANNER)

    if inwin:
        print()
        print("  %.0f%% of CS-alone trials produced a blink inside the CR window with no US"
              % out["false_positive_pct"])
        print("  in the recording.  Every conditioning CR rate below is to be read against")
        print("  that number, not against zero.")
    return out


def no_cs_baseline(cfg):
    """No CS-only baseline: the false-positive rate is unknown, and cannot be guessed."""
    # A CS-only recording that triage threw out is NOT the same thing as a study that
    # never had one, and saying so would be a lie about the session.  The effective
    # config keeps what it dropped, and why, under its own key - so look there before
    # telling anyone their study has no baseline.
    listed = [r for r in cfg["recordings"] if r.get("role") == "baseline_cs"]
    dropped = [r for r in cfg.get("excluded", []) if r.get("role") == "baseline_cs"]
    print(BANNER)
    print("NO CS-ONLY BASELINE - THE FALSE-POSITIVE RATE IS UNKNOWN")
    print(BANNER)
    if dropped:
        for r in dropped:
            print("  %s WAS recorded, and was left out of this run:"
                  % (r.get("file") or r.get("tag")))
            print("    %s" % (r.get("excluded_because")
                              or "no reason was recorded, which is itself worth chasing"))
    elif listed:
        print("  %s is in the study file but produced no scored trials - its CS LED"
              % ", ".join(r["file"] for r in listed))
        print("  could not be read, so the trial times could not be recovered.")
    else:
        print("  There is no CS-only recording in this study.")
    print("""
  The CS-only baseline is what says whether the tone alone makes this person blink.
  Without it there is nothing to read the conditioning CR rate against: a 60% CR rate
  means one thing next to a 5% false-positive rate and something else entirely next to
  a 40% one.

  This one cannot be recovered automatically.  A CS-only recording delivers no US, so
  when its CS LED is unreadable there is no second channel to take the trial times from.

    SCORE IT BY HAND.  Open the recording, read the onset of each CS presentation and
    of any blink that follows, and record them beside the analyser's numbers.  Until
    that exists, quote the conditioning CR rate with the false-positive rate stated as
    unknown - not as zero.""")
    print(BANNER)


def order_check(wdir):
    """Step three: are the CSUS takes in the order their names claim?

    ebc_timeline reads the camera clock and renumbers the recordings when the file names
    disagree with it.  That happens in the first stage of a run that takes the better
    part of an hour, and its warning has scrolled a long way off the screen by the time
    the numbers it moved arrive.  So it is re-stated here, next to the block numbers it
    decided, and returned so merged.json can carry it into the workbooks.
    """
    path = os.path.join(wdir, "timeline.json")
    try:
        with open(path, encoding="utf-8") as fh:
            tl = json.load(fh)
    except (OSError, ValueError):
        print("  no timeline.json in this run, so the recording order was never checked")
        print("  against the camera clock.  The order below is the order of the names.")
        return None
    changed = tl.get("order_changed") or []
    if not changed:
        print("  the file names and the camera clock agree - nothing was reordered.")
        return changed
    print(BANNER)
    print("ORDER CORRECTED - %d recording(s) are not where their names put them"
          % len(changed))
    print(BANNER)
    for c in changed:
        print("  %-28s %-13s was #%s by name, is #%s by the clock"
              % (str(c.get("file"))[:28], c.get("role"), c.get("was"), c.get("now")))
    print("""
  The conditioning takes are laid end to end on one clock, so their order is what sets
  every block boundary and every trial number below.  The order used is the camera's,
  not the names'.  If the names are right and the clock is not - a camera whose date was
  never set, or two cameras in one session - then the blocks below are wrong and this
  run should not be quoted.  Check the recorded times in the timeline table before
  using these numbers.""")
    print(BANNER)
    return changed


def summarise(rows, role, tt, win, label):
    """One line of the trial summary: how many, how many scoreable, how many CR."""
    rs = [r for r in rows if r["role"] == role and r["trial_type"] == tt]
    if not rs:
        return None
    sc = [r for r in rs if r["scored_class"] not in (None, win["moving_label"])]
    cr = [r for r in sc if str(r["scored_class"]).startswith("CR")]
    rec = [r for r in rs if r["first_response_obscured"] == "yes"
           and r["secondary_onset_ms"] is not None]
    o = [r["scored_onset_ms"] for r in cr if r["scored_onset_ms"] is not None]
    line = "  %-32s %3d trials, %3d scoreable, %3d CR" % (label, len(rs), len(sc), len(cr))
    if o and sc:
        line += " (%.0f%%), mean onset %.0f ms" % (len(cr) / len(sc) * 100,
                                                   float(np.mean(o)))
        if len(o) > 1:
            line += " +- %.0f SD" % float(np.std(o, ddof=1))
    print(line + "   [%d recovered behind an artefact]" % len(rec))
    return dict(role=role, trial_type=tt, n=len(rs), n_scoreable=len(sc), n_cr=len(cr))


def build_row(m, win, study, proto, des):
    """One measured trial, labelled against this study's CR window."""
    tr, v, Cl, k0, MS = m["tr"], m["v"], m["C"], m["k0"], m["MS"]
    US_ONSET, CS_DUR = float(proto["us_onset_ms"]), float(proto["cs_ms"])
    full, part, b1, on_us = m["full"], m["part"], m["first"], m["on_us"]
    inprog = m["inprog"]

    # Only a paired CS-US trial and a US-only baseline trial actually deliver a puff.
    us_here = tr["trial_type"] in ("CS-US", "US-only")
    cls = classify(b1["on"], win, inprog, on_us, us_here) if b1 else None
    # if the first event is too early to be a response to anything, or the lid was
    # already moving, look for a later blink in the same window - a real CR or UR may sit
    # behind the artefact.  "Too early" is the CR window's own lower edge, so in a study
    # with a US-only baseline it is the measured reflex latency, not a fixed 100 ms.
    obscured = bool(b1 and (inprog or (not on_us and b1["on"] < win["lo_ms"])))
    b2 = full[1] if (obscured and len(full) > 1) else None
    sec = classify(b2["on"], win, False, on_us, us_here) if b2 else None

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

    # A reflex latency typed into the study file stands in for one that could not be
    # measured.  It is not a default and it is not silent: it is a value someone read off
    # the video themselves, and it is recorded as such everywhere the window is quoted.
    manual_ms = proto.get("reflex_ms")
    if not REFLEX.get("onset_ms") and manual_ms is not None:
        REFLEX = dict(REFLEX, onset_ms=float(manual_ms), n=0, k=None,
                      why="entered by hand in the study file (reflex_ms)", by_hand=True)
    WIN = C.cr_window(proto, REFLEX)

    rows = [build_row(m, WIN, cfg["study"], proto, DES) for m in MEAS]

    # ---- read in the order the session is worked through ----------------------------
    # The arithmetic above has only one possible order, and it is not this one: a blink
    # is startle, CR or UR only relative to the CR window, and that window is measured
    # from the US-only baseline, so every trial in the study - the CS-only baseline
    # included - has to be classified after the US-only trials have been.  The reading
    # order is a different thing from the dependency order, and this is the reading
    # order: what the CS alone does, how fast the reflex is, whether the recordings are
    # where their names put them, the paired conditioning, then the probes that say
    # whether what was learnt held.
    print()
    print(BANNER)
    print("STEP 1 of 5   CS-ONLY BASELINE - does the CS alone make this person blink?")
    print(BANNER)
    CS_BASE = cs_baseline(cfg, rows, WIN)

    print()
    print(BANNER)
    print("STEP 2 of 5   US-ONLY BASELINE - how fast is the reflex, and where does that")
    print("              put the CR window?")
    print(BANNER)
    if WIN["measured"] and REFLEX.get("by_hand"):
        print("  CR WINDOW FROM A HAND-ENTERED REFLEX LATENCY")
        print("  reflex_ms = %.0f ms was taken from the study file, not measured from a"
              % float(manual_ms))
        print("  US-only baseline.  The CR window is %.0f-%.0f ms.  Every workbook says so."
              % (WIN["lo_ms"], WIN["hi_ms"]))
    elif WIN["measured"]:
        print("  reflex onset measured from %d US-only trial(s): %.0f +- %.0f ms after "
              "the puff," % (REFLEX["n"], REFLEX["mean_ms"], REFLEX["sd_ms"]))
        print("  mean - %.1f SD = %.0f ms" % (REFLEX["k"], REFLEX["onset_ms"]))
        print("  CR window = CS onset + %.0f ms  ->  US onset + %.0f ms"
              % (REFLEX["onset_ms"], REFLEX["onset_ms"]))
    else:
        no_us_baseline(cfg, proto, REFLEX, WIN)
    print("  CR window = %.0f-%.0f ms from CS onset  [%s]"
          % (WIN["lo_ms"], WIN["hi_ms"], WIN["cr_label"]))

    print()
    print(BANNER)
    print("STEP 3 of 5   RECORDING ORDER - are the CSUS takes where their names put them?")
    print(BANNER)
    ORDER_CHANGED = order_check(wdir)

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
                  # step 1 and step 3 of the reading order, carried so a workbook
                  # opened weeks later still says what the CS alone did and
                  # whether the recordings were put back in a different order
                  cs_baseline=CS_BASE, order_changed=ORDER_CHANGED,
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
    print(BANNER)
    print("STEP 4 of 5   PAIRED CS-US CONDITIONING - the measurement itself")
    print(BANNER)
    summarise(rows, "conditioning", "CS-US", WIN, "paired CS-US")
    if CS_BASE and CS_BASE.get("false_positive_pct") is not None:
        print("  read against the CS-only baseline: %.0f%% of CS-alone trials produced a"
              % CS_BASE["false_positive_pct"])
        print("  blink in the same window with no US in the recording.")

    print()
    print(BANNER)
    print("STEP 5 of 5   CS-ONLY PROBES DURING CONDITIONING - did the learning hold?")
    print(BANNER)
    if not summarise(rows, "conditioning", "CS-only", WIN, "CS-only probes"):
        print("  no CS-only probes in the conditioning recordings, so there is nothing")
        print("  here to read the stability of the learning off.")

    after = [("extinction", "CS-only", "extinction"),
             ("baseline_us", "US-only", "US-only baseline")]
    shown_after = [t for t in after
                   if any(r["role"] == t[0] and r["trial_type"] == t[1] for r in rows)]
    if shown_after:
        print()
        print("Also in this study, outside the five steps:")
        for role, tt, label in shown_after:
            summarise(rows, role, tt, WIN, label)

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
