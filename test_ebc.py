"""Unit tests for the pure logic in the pipeline.

    python test_ebc.py            (or: python -m pytest test_ebc.py -q)

These cover the functions that decide what a trial IS - classification, CS-US pairing,
block numbering and pulse detection.  All of them run in milliseconds on synthetic
input, and every bug encoded here was found instead by a fifty-minute run over video.

THE RUNNER LIVES AT THE END OF THIS FILE, and it checks that the number of tests it
collected matches the number defined.  It has to: an earlier version of this file had
the runner in the middle, so tests appended after it were never even defined when the
file was run - the suite reported "19 passed" out of 38, and the twenty that never ran
included every test defending the scoring boundary.  A test suite that silently shrinks
is worse than no test suite, because it reports success either way.
"""
import io
import os
import re
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ebc_config as C
import ebc_protocol as P
import ebc_score as S

PROTO = dict(cs_ms=400.0, us_onset_ms=350.0, us_dur_ms=50.0, paired_per_block=9,
             cs_only_per_block=1, n_blocks=10, min_iti_s=5.0, cs_tol=0.35, us_tol=0.6)
US = 350.0

# The window every test below is read against: a measured reflex of mean 68.5 SD 19.7,
# which is the pooled US-only baseline of the three participants scored so far.
REFLEX = dict(n=34, mean_ms=68.5, sd_ms=19.7, k=1.5, onset_ms=68.5 - 1.5 * 19.7)
# Scoring runs on the standard window now (see cr_window_mode), so a test that means to
# exercise the measured one has to ask for it by name rather than inherit it.
WIN = C.cr_window(dict(PROTO, cr_window_mode="measured"), REFLEX)
FALLBACK = C.cr_window(PROTO)
LO, HI = WIN["lo_ms"], WIN["hi_ms"]
# The standard window: the same two numbers for every participant, with the uncertain
# band between the puff and CS offset.
STD = C.cr_window(PROTO)
SLO, SHI, SQ = STD["lo_ms"], STD["hi_ms"], STD["qcr_hi_ms"]


# ------------------------------------------------------------------ the response window
def test_the_window_is_mean_minus_one_and_a_half_sd_past_each_stimulus():
    """CR runs from CS+reflex to US+reflex, reflex = mean - 1.5 SD."""
    off = REFLEX["mean_ms"] - REFLEX["k"] * REFLEX["sd_ms"]
    assert abs(LO - off) < 0.06         # cr_window rounds both edges to 0.1 ms
    assert abs(HI - (US + off)) < 0.06


def test_one_window_decides_it_everywhere():
    """ebc_config.cr_window is the single place the rule lives; classify only reads it."""
    assert "win" in str(__import__("inspect").signature(S.classify))


def test_the_window_says_whether_it_was_measured():
    assert WIN["measured"] is True and FALLBACK["measured"] is False
    assert FALLBACK["why"]


def test_without_a_baseline_it_falls_back_to_the_protocol():
    assert FALLBACK["lo_ms"] == 100.0 and FALLBACK["hi_ms"] == US


def test_cr_window_runs_from_CS_plus_offset_to_US_plus_offset():
    assert S.classify(LO + 1, WIN, False, False, True).startswith("CR")
    assert S.classify(HI - 1, WIN, False, False, True).startswith("CR")
    assert S.classify(HI + 1, WIN, False, False, True).startswith("UR")


def test_too_early_to_be_a_response_to_the_CS():
    assert S.classify(LO - 1, WIN, False, False, True).startswith("alpha")


def test_cr_before_the_puff():
    assert S.classify(200.0, WIN, False, False, True).startswith("CR")


def test_ur_after_the_puff():
    assert S.classify(430.0, WIN, False, False, True).startswith("UR")


def test_the_old_boundary_would_have_called_this_a_UR():
    """350.4 ms - six of Carole's trials sit there, 0.4 ms after the puff."""
    assert S.classify(350.4, WIN, False, False, True).startswith("CR")


# ------------------------------------------------------- the standard window and ?CR
def test_the_standard_window_ignores_the_measured_reflex():
    """Comparability is the point: two participants must be scored on the same numbers."""
    w = C.cr_window(PROTO, REFLEX)
    assert (w["lo_ms"], w["hi_ms"]) == (100.0, US) and w["standard"] is True
    assert w["measured"] is False


def test_the_standard_window_still_records_what_the_reflex_would_have_been():
    """Discarding the measurement would make the choice of window unreviewable."""
    w = C.cr_window(PROTO, REFLEX)
    assert w["reflex"] and abs(w["reflex"]["mean_ms"] - REFLEX["mean_ms"]) < 0.01
    assert "%.0f" % REFLEX["mean_ms"] in w["why"]


def test_the_uncertain_band_runs_from_the_puff_to_CS_offset():
    assert (SHI, SQ) == (US, PROTO["cs_ms"])


def test_a_blink_after_the_puff_is_not_a_definitive_CR():
    """The whole point of the band: the puff has arrived, so a CR cannot be asserted."""
    c = S.classify(SHI + 1, STD, False, False, True)
    assert c.startswith("?CR") and not c.startswith("CR")


def test_the_uncertain_band_is_not_a_UR_either():
    """Calling it a UR asserts the opposite with no more evidence."""
    assert not S.classify(SQ - 1, STD, False, False, True).startswith("UR")


def test_past_CS_offset_it_is_a_UR_again():
    assert S.classify(SQ + 1, STD, False, False, True).startswith("UR")


def test_a_blink_before_the_puff_is_still_a_plain_CR():
    assert S.classify(SHI - 1, STD, False, False, True) == STD["cr_label"]


def test_the_uncertain_band_exists_on_a_probe_too():
    """A probe delivers no puff, so the band cannot mean "may be a UR" - but a rate that
    counted it would not be comparable with the paired trials it exists to be read
    against, which is the only reason a probe is scored on the paired window at all."""
    c = S.classify(SHI + 1, STD, False, False, us_delivered=False)
    assert c.startswith("?CR") and "UR" not in c


def test_the_measured_window_has_no_uncertain_band():
    """It already states where the puff's influence begins; a second band would contradict it."""
    assert WIN.get("qcr_hi_ms") is None
    assert S.classify(HI + 1, WIN, False, False, True).startswith("UR")


def test_a_trace_protocol_has_no_uncertain_band():
    """The CS is over before the puff arrives, so there is nothing between the two."""
    w = C.cr_window(dict(PROTO, us_onset_ms=900.0))
    assert w["qcr_hi_ms"] is None


def test_the_three_classes_do_not_overlap():
    """Every latency gets exactly one name, and no name is a prefix of another's test."""
    seen = [S.classify(x, STD, False, False, True)
            for x in (SLO - 1, SLO + 1, SHI - 1, SHI + 1, SQ - 1, SQ + 1)]
    assert [s[:4] for s in seen] == ["alph", "CR (", "CR (", "?CR ", "?CR ", "UR ("], seen


def test_moving_lid_is_untimeable():
    assert S.classify(200.0, WIN, True, False, True) == WIN["moving_label"]


# ----------------------------------------- the two ends outside which nothing is scored
# A blink before the CS had nothing to respond to, and one long enough after the puff is
# the next spontaneous blink rather than the reflex to it.  Both used to be classified
# like any other latency: the first as startle, the second as a UR - inflating the UR
# count and dragging the mean UR latency out with it.
def test_a_blink_before_the_CS_is_not_a_response_to_it():
    assert S.classify(-40.0, STD, False, False, True) == STD["before_label"]
    assert S.classify(-1.0, STD, False, False, True) == STD["before_label"]
    # and the boundary itself is still scored: 0 is the CS, not before it
    assert S.classify(0.0, STD, False, False, True) == STD["alpha_label"]


def test_a_blink_long_after_the_puff_is_not_the_reflex_to_it():
    late = STD["us_onset_ms"] + STD["late_ms"]
    assert S.classify(late + 1, STD, False, False, True) == STD["too_late_label"]
    assert S.classify(late - 1, STD, False, False, True).startswith("UR")


def test_the_same_two_ends_apply_to_a_us_only_baseline():
    """Timed from the puff there, so the cut-offs are 0 and late_ms themselves."""
    assert S.classify(-5.0, STD, False, True, True) == STD["before_us_label"]
    assert S.classify(STD["late_ms"] + 1, STD, False, True, True) == STD["too_late_label"]
    assert S.classify(67.0, STD, False, True, True) == STD["ur_puff_label"]


def test_a_trial_with_no_puff_is_bounded_the_same_way():
    """Nothing was delivered, but the US onset is still a moment in the trial, and a
    blink 200 ms past it is no more a response to the CS than on a paired trial."""
    late = STD["us_onset_ms"] + STD["late_ms"]
    assert S.classify(late + 50, STD, False, False, False) == STD["too_late_label"]


def test_the_set_aside_classes_are_the_ones_the_rates_skip():
    for lab in (STD["before_label"], STD["before_us_label"], STD["too_late_label"],
                STD["moving_label"]):
        assert not C.is_scoreable(lab, STD), lab
    for lab in (STD["cr_label"], STD["ur_label"], STD["alpha_label"], STD["qcr_label"]):
        assert C.is_scoreable(lab, STD), lab
    assert not C.is_scoreable(None, STD)


def test_the_cut_offs_do_not_move_the_CR_window():
    """They decide what is scored, never where the boundary is."""
    w = C.cr_window(dict(PROTO, late_ms=500.0))
    assert (w["lo_ms"], w["hi_ms"], w["qcr_hi_ms"]) == (STD["lo_ms"], STD["hi_ms"],
                                                        STD["qcr_hi_ms"])
    assert w["late_ms"] == 500.0
    assert S.classify(STD["us_onset_ms"] + 300, w, False, False, True).startswith("UR")


def test_us_anchored_trial_is_always_unconditioned():
    assert S.classify(67.0, WIN, False, True, True) == WIN["ur_puff_label"]


# ------------------------------------------------------------------ trials with no puff
def test_late_response_on_a_probe_is_not_a_UR():
    """A2. No puff was delivered, so there is no UR the response could be."""
    assert "UR" not in S.classify(HI + 30, WIN, False, False, us_delivered=False)


def test_a_probe_uses_the_same_CR_window_as_a_paired_trial():
    """Otherwise the probe cannot serve its purpose, which is to be comparable."""
    assert S.classify(LO + 1, WIN, False, False, False).startswith("CR")
    assert S.classify(HI - 1, WIN, False, False, False).startswith("CR")


def test_a_probe_response_after_the_window_is_not_a_CR():
    """'Every blink is a CR' is as wrong as 'every blink is a UR': a CS-only baseline
    exists to give the false-positive rate and cannot do that if it reads 100% by
    construction."""
    assert not S.classify(HI + 30, WIN, False, False, False).startswith("CR")


def test_alpha_still_wins_on_a_probe():
    assert S.classify(LO - 5, WIN, False, False, False).startswith("alpha")


# ------------------------------------------------------------------ the standing check
def test_a_no_conditioning_participant_must_not_look_like_a_learner():
    """Thomas. A clean spike 50-75 ms after his puff and almost nothing before it is
    what a correct measurement of NO conditioning looks like.  Any scoring change that
    turns him into a learner is wrong, and this is the cheapest way to notice: a window
    built on the MEDIAN reflex rather than mean-1.5SD put it at 425 ms and read him at
    76% CR.  These are his real paired-trial onsets.
    """
    onsets = ([120.0] * 2 + [180.0] + [210.0] + [310.0] + [330.0]
              + [390.0] * 10 + [410.0] * 47 + [430.0] * 17 + [510.0] + [540.0] + [560.0])
    cls = [S.classify(o, WIN, False, False, True) for o in onsets]
    scoreable = [c for c in cls if c != WIN["moving_label"]]
    cr = [c for c in scoreable if c.startswith("CR")]
    rate = len(cr) / len(scoreable)
    assert rate < 0.35, "Thomas reads %.0f%% CR - the window is in the wrong place" % (100 * rate)


# ------------------------------------------------------------------ pair_cs_us
def _cs(*ts):
    return [dict(t=t, ok=True) for t in ts]


def _us(*ts):
    return [dict(t=t, ok=True) for t in ts]


def test_us_inside_the_cs_is_paired():
    pairs, un = P.pair_cs_us(_cs(10.0), _us(10.35), PROTO)
    assert pairs[0][1] is not None and un == []


def test_us_seconds_later_is_not_paired():
    pairs, un = P.pair_cs_us(_cs(10.0), _us(14.0), PROTO)
    assert pairs[0][1] is None and len(un) == 1


def test_a_us_is_used_once():
    pairs, un = P.pair_cs_us(_cs(10.0, 10.05), _us(10.35), PROTO)
    assert sum(p[1] is not None for p in pairs) == 1


def test_rejected_us_pulses_never_pair():
    pairs, un = P.pair_cs_us(_cs(10.0), [dict(t=10.35, ok=False)], PROTO)
    assert pairs[0][1] is None and un == []


# ------------------------------------------------------------------ block numbering
def _blocks(types):
    """Run the block loop the way ebc_protocol.build does, on trial types alone."""
    cond = [dict(trial_type=t) for t in types]
    b, k, run = 1, 0, 0
    runs, closed_by = [], {}
    for t in cond:
        if t["trial_type"] == "CS-US" and run >= PROTO["paired_per_block"]:
            runs.append(run); closed_by[b] = "count"; run = 0; b, k = b + 1, 0
        k += 1
        t["block"] = b
        t["trial_in_block"] = k
        if t["trial_type"] == "CS-US":
            run += 1
        else:
            runs.append(run); closed_by[b] = "probe"; run = 0; b, k = b + 1, 0
    return runs, closed_by, cond


def test_a_clean_protocol_gives_ten_blocks():
    runs, closed, _ = _blocks((["CS-US"] * 9 + ["CS-only"]) * 10)
    assert runs == [9] * 10
    assert set(closed.values()) == {"probe"}


def test_a_missing_probe_closes_the_block_by_count():
    runs, closed, _ = _blocks(["CS-US"] * 9 + ["CS-US"] * 9 + ["CS-only"])
    assert closed[1] == "count" and closed[2] == "probe"


def test_a_lost_puff_invents_a_block():
    """B4/B8. One paired trial mis-read as a probe splits its block in two.

    Marie: the position gate discarded a quarter of her puffs, each lost puff turned a
    paired trial into a CS-only trial, and the block loop closed a block on every one -
    36 blocks recovered where the protocol has 10.
    """
    good = (["CS-US"] * 9 + ["CS-only"]) * 2
    runs_ok, _, _ = _blocks(good)
    broken = list(good)
    broken[4] = "CS-only"                      # a puff delivered but not seen
    runs_bad, _, _ = _blocks(broken)
    assert len(runs_ok) == 2
    assert len(runs_bad) == 3, "a lost puff must not silently add a block"


def test_block_count_is_reported_against_the_protocol():
    runs, _, _ = _blocks((["CS-US"] * 9 + ["CS-only"]) * 36)
    assert len(runs) != PROTO["n_blocks"], "36 blocks is not 10 and must not compare equal"


# ------------------------------------------------------------------ detect
def _square(fps, n_s, pulses, dur_ms, hi=240.0, lo=20.0):
    import numpy as np
    sig = np.full(int(fps * n_s), lo)
    for t in pulses:
        a = int(t * fps)
        sig[a:a + int(round(dur_ms / 1000.0 * fps))] = hi
    return sig


def test_detect_finds_square_pulses():
    import ebc_stimulus as X
    sig = _square(119.88, 60, [5.0, 20.0, 35.0, 50.0], 400.0)
    ev, info = X.detect(sig, 119.88, 400.0, tol=0.35, min_gap_s=5.0)
    assert sum(e["ok"] for e in ev) == 4, [e["dur_ms"] for e in ev]


def test_detect_rejects_the_wrong_duration():
    import ebc_stimulus as X
    sig = _square(119.88, 60, [5.0, 20.0], 1200.0)
    ev, info = X.detect(sig, 119.88, 400.0, tol=0.35, min_gap_s=5.0)
    assert sum(e["ok"] for e in ev) == 0


# --------------------------------------------- the reading order, and what it must say
def _cs_row(onset_ms, cls):
    """One scored CS-only baseline trial, in the shape cs_baseline() reads."""
    return dict(role="baseline_cs", trial_type="CS-only", session_name="CS ONLY",
                session_trial=1, scored_onset_ms=onset_ms, scored_class=cls,
                first_response_obscured="no", needs_manual_scoring=False)


def test_startle_in_the_cs_only_baseline_is_flagged_and_not_counted_as_a_CR():
    """A blink before the window opens is startle: flagged, and no part of any CR rate."""
    rows = [_cs_row(LO - 20, WIN["alpha_label"]),
            _cs_row(LO - 10, WIN["alpha_label"]),
            _cs_row(LO + 100, WIN["cr_no_us_label"]),
            _cs_row(HI + 200, WIN["late_no_us_label"])]
    out = S.cs_baseline(dict(recordings=[], excluded=[]), rows, WIN)
    assert out["n_scoreable"] == 4, out
    assert out["n_startle"] == 2, out
    assert out["startle_pct"] == 50.0, out
    # the one blink inside the window is the false-positive rate, and startle is not in it
    assert out["n_in_window"] == 1 and out["false_positive_pct"] == 25.0, out


def test_a_dropped_cs_only_baseline_is_not_reported_as_never_recorded():
    """Triage throwing a recording out is a different fact from it not existing.

    The effective config carries what was dropped and why; saying "there is no CS-only
    recording in this study" when there is one on the SD card sends someone looking for
    a recording they already made.
    """
    import io as _io, contextlib
    cfg = dict(recordings=[],
               excluded=[dict(file="CS ONLY.MP4", role="baseline_cs",
                              excluded_because="its CS LED could not be read")])
    buf = _io.StringIO()
    with contextlib.redirect_stdout(buf):
        out = S.cs_baseline(cfg, [], WIN)
    txt = buf.getvalue()
    assert out is None
    assert "CS ONLY.MP4" in txt and "WAS recorded" in txt, txt
    assert "There is no CS-only recording in this study." not in txt, txt


def test_a_corrected_recording_order_is_warned_about_where_the_results_are():
    """The camera disagreeing with the file names has to reach the person reading the
    numbers, not only the first minute of a fifty-minute log."""
    import io as _io, json as _json, contextlib, tempfile
    d = tempfile.mkdtemp()
    with open(os.path.join(d, "timeline.json"), "w", encoding="utf-8") as fh:
        _json.dump(dict(order_changed=[dict(file="CSUS 4.MP4", role="conditioning",
                                            was=4, now=2)]), fh)
    buf = _io.StringIO()
    with contextlib.redirect_stdout(buf):
        changed = S.order_check(d)
    txt = buf.getvalue()
    assert changed and changed[0]["file"] == "CSUS 4.MP4"
    assert "ORDER CORRECTED" in txt and "CSUS 4.MP4" in txt, txt
    assert "was #4 by name, is #2 by the clock" in txt, txt


def test_an_unchanged_order_says_so_rather_than_saying_nothing():
    import io as _io, json as _json, contextlib, tempfile
    d = tempfile.mkdtemp()
    with open(os.path.join(d, "timeline.json"), "w", encoding="utf-8") as fh:
        _json.dump(dict(order_changed=[]), fh)
    buf = _io.StringIO()
    with contextlib.redirect_stdout(buf):
        changed = S.order_check(d)
    assert changed == []
    assert "agree" in buf.getvalue()


# -------------------------------------------- what a mean blink onset is taken over
def test_a_response_is_a_CR_a_qCR_or_a_UR():
    for lbl in (STD["cr_label"], STD["qcr_label"], STD["ur_label"],
                STD["cr_no_us_label"], STD["late_no_us_label"], STD["ur_puff_label"]):
        assert C.is_response(lbl, STD), lbl


def test_a_startle_is_scoreable_but_is_not_a_response():
    """It counts in a RATE's denominator and must stay out of a mean onset."""
    for lbl in (STD["alpha_label"], STD["alpha_us_label"]):
        assert C.is_scoreable(lbl, STD), lbl
        assert not C.is_response(lbl, STD), lbl


def test_a_set_aside_trial_is_not_a_response_either():
    for lbl in STD["excluded_labels"]:
        assert not C.is_response(lbl, STD), lbl


def test_the_mean_blink_onset_includes_the_qCRs_and_the_URs():
    """The user's rule, and the reason it exists: CRs alone cannot show the shift."""
    import ebc_figures as F
    rows = [dict(block=1, scored_class=STD["cr_label"], scored_onset_ms=200.0),
            dict(block=1, scored_class=STD["qcr_label"], scored_onset_ms=380.0),
            dict(block=1, scored_class=STD["ur_label"], scored_onset_ms=440.0)]
    (b, mean, sd, n), = F.block_onset(rows, STD)
    assert (b, n) == (1, 3), (b, n)
    assert abs(mean - 340.0) < 0.01, mean          # not 200.0, which is the CRs alone


def test_the_mean_blink_onset_leaves_the_startles_out():
    import ebc_figures as F
    rows = [dict(block=1, scored_class=STD["cr_label"], scored_onset_ms=300.0),
            dict(block=1, scored_class=STD["alpha_label"], scored_onset_ms=40.0),
            dict(block=1, scored_class=STD["moving_label"], scored_onset_ms=None)]
    (_, mean, _, n), = F.block_onset(rows, STD)
    assert (mean, n) == (300.0, 1), (mean, n)


def test_a_block_can_improve_without_any_of_its_CRs_moving():
    """The whole case for the rule, as an arithmetic fact rather than an opinion.

    Two blocks whose CRs sit at exactly the same latency, but where one UR has crossed
    into the window.  Over the CRs alone nothing has changed; over every response the
    mean walks in by 25 ms, which is the learning.
    """
    import numpy as np
    import ebc_figures as F
    early = [dict(block=1, scored_class=STD["cr_label"], scored_onset_ms=250.0)] * 2
    b1 = early + [dict(block=1, scored_class=STD["ur_label"], scored_onset_ms=450.0)] * 2
    b2 = [dict(block=2, scored_class=STD["cr_label"], scored_onset_ms=250.0)] * 3 + \
         [dict(block=2, scored_class=STD["ur_label"], scored_onset_ms=450.0)]
    cr_only = [np.mean([r["scored_onset_ms"] for r in g
                        if r["scored_class"].startswith("CR")]) for g in (b1, b2)]
    assert cr_only[0] == cr_only[1] == 250.0, cr_only
    means = [F.block_onset(g, STD)[0][1] for g in (b1, b2)]
    assert abs(means[0] - 350.0) < .01 and abs(means[1] - 300.0) < .01, means


# ------------------------------------------------------- which trial a clip is cut from
def _clip_row(cls, onset, **kw):
    return dict(dict(role="conditioning", trial_type="CS-US", scored_class=cls,
                     scored_onset_ms=onset, peak_closure_pct=95.0,
                     face_tracked_pct=100.0, quality="clean",
                     needs_manual_scoring=""), **kw)


def test_a_clip_is_never_cut_from_a_trial_the_scorer_will_not_stand_behind():
    import ebc_clips as K
    rows = [_clip_row(STD["cr_label"], 200.0, needs_manual_scoring="yes")]
    assert K.candidates(rows, "CR", "CS-US", ("conditioning",), STD) == []


def test_a_missing_kind_is_reported_and_never_substituted():
    """No ?CR to show is a fact about the participant, not a gap to paper over."""
    import ebc_clips as K
    rows = [_clip_row(STD["cr_label"], 200.0), _clip_row(STD["ur_label"], 450.0)]
    assert K.candidates(rows, "?CR", "CS-US", ("conditioning", "extinction"), STD) == []


def test_the_clip_shows_the_ordinary_trial_not_the_striking_one():
    import ebc_clips as K
    rows = [_clip_row(STD["cr_label"], o) for o in (150.0, 200.0, 205.0, 210.0, 330.0)]
    got, why, _ = K.prototypical(rows, ["conditioning"])
    assert got["scored_onset_ms"] == 205.0, got["scored_onset_ms"]
    assert "median" in why


def test_a_lid_flicker_under_the_blink_criterion_loses_to_a_real_blink():
    """Even one from a recording further down the order - see prototypical()."""
    import ebc_clips as K
    flicker = _clip_row(STD["cr_no_us_label"], 200.0, trial_type="CS-only",
                        peak_closure_pct=16.0)
    real = _clip_row(STD["cr_no_us_label"], 200.0, trial_type="CS-only",
                     role="extinction", peak_closure_pct=95.0)
    got, _, _ = K.prototypical([flicker, real], ["conditioning", "extinction"])
    assert got["role"] == "extinction", got


def test_every_clip_kind_names_a_class_the_window_can_produce():
    """A kind whose prefix matches no label this app emits could never be filled."""
    import ebc_clips as K
    labels = [v for k, v in STD.items() if k.endswith("_label") and v]
    for kind, prefix, tt, roles, note in K.KINDS:
        if prefix:
            assert any(str(v).startswith(prefix) for v in labels), kind
        assert set(roles) <= set(C.ROLES), kind


def test_a_standard_window_does_not_claim_the_baseline_is_missing():
    """It was measured and then not used; saying it was never there is a different claim."""
    import ebc_figures as F
    note = F.window_note(C.cr_window(PROTO, REFLEX))
    assert "no US-only baseline" not in note, note
    assert "standard window" in note and "%d blinks" % REFLEX["n"] in note, note


def test_with_no_baseline_at_all_the_note_still_says_so():
    """Both ways of ending up without one: the standard window, and a measured one that
    had nothing to measure."""
    import ebc_figures as F
    assert "no US-only baseline" in F.window_note(C.cr_window(PROTO)), "standard"
    fell_back = C.cr_window(dict(PROTO, cr_window_mode="measured"))
    assert "no US-only baseline" in F.window_note(fell_back), "measured"


def test_every_pipeline_stage_has_a_name_on_the_progress_bar():
    """A stage the page has never heard of shows as its bare key and weights nothing.

    ebc_clips was added to the pipeline and had to be added to three separate maps to
    appear; this is the check that the next one does not have to be found by watching a
    run go blank half way through.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    runner = io.open(os.path.join(here, "ebc_run_all.py"), encoding="utf-8").read()
    ran = set(re.findall(r'run(?:_parallel)?\(\s*"ebc_(\w+)\.py"', runner))
    ran -= {"triage"}                      # started under the protocol stage's own name
    app = io.open(os.path.join(here, "ebc_app.py"), encoding="utf-8").read()
    page = io.open(os.path.join(here, "ebc_app_ui.html"), encoding="utf-8").read()
    known = set(re.findall(r'"(\w+)":', app.split("PHASE = {", 1)[1].split("}", 1)[0]))
    shown = set(re.findall(r'(\w+):"', page.split("const PHASES = {", 1)[1]
                           .split("};", 1)[0]))
    weighed = set(re.findall(r'(\w+):[.\d]', page.split("const WEIGHT = {", 1)[1]
                             .split("};", 1)[0]))
    for name, got in (("ebc_app PHASE", known), ("the page's PHASES", shown),
                      ("the page's WEIGHT", weighed)):
        assert ran <= got, "%s does not know about %s" % (name, sorted(ran - got))


# ---------------------------------------- charts Excel will actually agree to open
# Excel does not report an invalid chart.  It offers to "recover" the workbook, deletes
# every chart part in it and saves that, so the whole thing arrives on somebody else's
# machine as bare numbers with nothing anywhere saying why.  Both bugs below shipped and
# were found only by unzipping a workbook Excel had been through.
CHART_NS = "{http://schemas.openxmlformats.org/drawingml/2006/chart}"


def _chart_xml(chart):
    """One chart serialised the way it lands in the file, as text."""
    from openpyxl.xml.functions import tostring
    x = tostring(chart._write())
    return x.decode("utf-8") if isinstance(x, bytes) else x


def _plot_group(chart):
    """The <c:scatterChart> / <c:lineChart> element inside a serialised chart."""
    import xml.etree.ElementTree as ET
    pa = ET.fromstring(_chart_xml(chart)).find(CHART_NS + "chart").find(CHART_NS + "plotArea")
    return next(c for c in pa if c.tag.endswith("Chart"))


def test_a_chart_colour_is_six_hex_digits_and_a_cell_colour_is_eight():
    """The bug that made every workbook this app ever wrote refuse to open.

    A cell fill is spreadsheet markup and wants opaque ARGB.  A chart is DrawingML and
    `<a:srgbClr val>` is ST_HexColorRGB - exactly six hex digits.  Handing a chart the
    eight-digit cell colour does not draw the wrong colour and does not drop the chart:
    Excel refuses to open the FILE, and if the offer to recover it is declined, nothing
    opens at all.  Confirmed against Excel itself, one probe workbook per feature.
    """
    for name in ("cr", "cs", "us", "ur", "muted", "faint", "us_mid", "surface"):
        assert re.fullmatch(r"[0-9A-Fa-f]{6}", C.dml(name)), (name, C.dml(name))
        assert re.fullmatch(r"FF[0-9A-Fa-f]{6}", C.xl(name)), (name, C.xl(name))
    assert C.xl("cr") == "FF" + C.dml("cr")
    assert re.fullmatch(r"[0-9A-Fa-f]{6}", C.dml_blend("cs", "cr", 0.5))
    for f, want in ((0.0, C.dml("cs")), (1.0, C.dml("cr"))):
        assert C.dml_blend("cs", "cr", f).upper() == want.upper(), f


def test_no_chart_in_the_workbook_builder_is_given_a_cell_colour():
    """The rule, enforced on the source: C.xl() must not reach a chart.

    ebc_workbooks reads a config as it is imported, so the check is on its text - and
    the text is where the mistake is made, one call site at a time.
    """
    src = io.open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "ebc_workbooks.py"), encoding="utf-8").read()
    for pat in (r"LineProperties\([^)]*C\.xl\(", r"GraphicalProperties\([^)]*C\.xl\(",
                r"solidFill=C\.xl\("):
        hit = re.search(pat, src)
        assert not hit, "a chart is being given a cell colour: " + hit.group(0)


def test_error_bars_put_plus_before_minus():
    """openpyxl 3.1.5 writes them the other way round; CT_ErrBars is a sequence."""
    C.patch_openpyxl_charts()
    from openpyxl.chart.error_bar import ErrorBars
    e = list(ErrorBars.__elements__)
    assert e.index("plus") < e.index("minus"), e


def test_patching_the_element_order_twice_does_not_shuffle_it():
    C.patch_openpyxl_charts()
    from openpyxl.chart.error_bar import ErrorBars
    before = ErrorBars.__elements__
    assert C.patch_openpyxl_charts() is False
    assert ErrorBars.__elements__ == before


def test_a_scatter_chart_carries_the_style_the_format_requires():
    """CT_ScatterChart's scatterStyle is minOccurs=1 and openpyxl omits it."""
    from openpyxl.chart import ScatterChart
    assert "scatterStyle" not in _chart_xml(ScatterChart()), \
        "openpyxl started writing it by itself - the workaround can go"
    ch = ScatterChart()
    ch.scatterStyle = C.SCATTER_STYLE
    kids = [c.tag.replace(CHART_NS, "") for c in _plot_group(ch)]
    assert kids[0] == "scatterStyle", kids


def test_an_error_bar_chart_serialises_in_schema_order():
    """The end-to-end check: build one the way the workbooks do, and read the XML back."""
    from openpyxl import Workbook
    from openpyxl.chart import ScatterChart, Reference, Series
    from openpyxl.chart.error_bar import ErrorBars
    from openpyxl.chart.data_source import NumDataSource, NumRef
    C.patch_openpyxl_charts()
    wb = Workbook()
    ws = wb.active
    for i in range(1, 5):
        ws.append([i, i * 10.0, 2.0])
    s = Series(Reference(ws, min_col=2, min_row=1, max_row=4),
               Reference(ws, min_col=1, min_row=1, max_row=4))
    sd = Reference(ws, min_col=3, min_row=1, max_row=4)
    s.errBars = ErrorBars(errDir="y", errValType="cust", errBarType="both",
                          plus=NumDataSource(numRef=NumRef(f=sd)),
                          minus=NumDataSource(numRef=NumRef(f=sd)))
    ch = ScatterChart()
    ch.scatterStyle = C.SCATTER_STYLE
    ch.series.append(s)
    grp = _plot_group(ch)
    eb = grp.find(".//" + CHART_NS + "errBars")
    got = [c.tag.replace(CHART_NS, "") for c in eb]
    assert got.index("plus") < got.index("minus"), got
    kids = [c.tag.replace(CHART_NS, "") for c in grp]
    assert kids[0] == "scatterStyle", kids
    ser = [c.tag.replace(CHART_NS, "") for c in grp.find(CHART_NS + "ser")]
    order = ["idx", "order", "tx", "spPr", "marker", "dPt", "dLbls", "trendline",
             "errBars", "xVal", "yVal", "smooth"]
    seen = [order.index(g) for g in ser if g in order]
    assert seen == sorted(seen), ser


def test_every_rendered_figure_points_at_a_sheet_that_exists():
    """PNG_DATA tells the reader where a picture's numbers are; a stale name misleads.

    ebc_workbooks reads a config as it is imported, so this reads its source rather than
    running it - and a sheet name that no longer exists is the only thing here that can
    go stale.
    """
    src = io.open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "ebc_workbooks.py"), encoding="utf-8").read()
    # sheets are created on either book - wb holds the numbers, fb the figures
    made = set(re.findall(r'create_sheet\(\s*"([^"]+)"', src))
    made |= set(re.findall(r'f?b?w?b,\s*"(F\d[^"]*)"', src))
    made |= set(re.findall(r'fb,\s*"([^"]+)"', src))
    named = set()
    for block in re.findall(r'PNG_DATA = \{(.*?)\n\}', src, re.S):
        named |= set(re.findall(r'"([^"]+)"\s*[,\]]', block))
    named -= set(re.findall(r'"(\w+\.png)"', src))
    assert len(named) >= 6, named
    assert named <= made, "PNG_DATA names sheets nothing creates: %s" % (named - made)
    for want in ("F1 CR rate by block", "F4 Mean blink onset by block",
                 "F7 Mean closure by block", "Figure index", "Video clips"):
        assert want in made, want


def test_the_probe_curve_is_broken_across_blocks_with_no_probe():
    """Joining the only two scoreable probes draws a trend over blocks never measured."""
    import ebc_figures as F
    rows = [dict(block=1, scored_class=WIN["cr_no_us_label"], scored_onset_ms=200.0),
            dict(block=9, scored_class=WIN["late_no_us_label"], scored_onset_ms=600.0)]
    pr = F.block_rate(rows, "CR", WIN)
    assert [p[0] for p in pr] == [1, 9], pr
    got = {p[0]: p for p in pr}
    ys = [got[x][1] if x in got else float("nan") for x in range(1, 11)]
    gaps = [y for y in ys[1:8]]
    assert all(y != y for y in gaps), ys        # NaN != NaN: every block between is a gap


# ------------------------------------------------------------------ the suite itself
def test_every_test_defined_in_this_file_is_collected():
    """The failure this file's docstring describes, as a test.

    Counts `def test_` in the source and compares it with what the runner collected.
    If someone appends a test after the runner block again, this fails instead of the
    suite quietly reporting success over a subset.
    """
    src = open(os.path.abspath(__file__), encoding="utf-8").read()
    defined = len(re.findall(r"^def (test_\w+)", src, re.M))
    collected = len([n for n in globals() if n.startswith("test_")])
    assert collected == defined, "%d tests defined, %d collected" % (defined, collected)


# ------------------------------------------------- names that contradict the camera
import ebc_app as A


def _take(files, take="t1"):
    """rows as ebc_media.timeline writes them: one take, chapters in the order given."""
    return {f: dict(file=f, take=["t", take], chapter=i, n_chapters=len(files),
                    continues_previous=(i > 1))
            for i, f in enumerate(files, 1)}


def _vids(files):
    return [dict(name=f, path="C:/v/" + f) for f in files]


def test_a_chapter_named_for_another_role_is_flagged():
    """Carole's CSUS fin.MP4 - chapter 2 of the extinction take, named for conditioning."""
    files = ["extinction.MP4", "CSUS fin.MP4"]
    c = A.name_findings(_vids(files), _take(files))
    assert len(c) == 1, c
    assert c[0]["name"] == "CSUS fin.MP4"
    assert c[0]["claims"] == "conditioning" and c[0]["actual"] == "extinction"


def test_the_suggested_name_follows_the_chapter_it_belongs_to():
    files = ["extinction.MP4", "CSUS fin.MP4"]
    assert A.name_findings(_vids(files), _take(files))[0]["suggest"] == "extinction 2.MP4"


def test_a_baseline_name_on_an_extinction_chapter_is_flagged():
    """Marie retest's cs only.MP4, which was being used as her CS-only baseline."""
    files = ["extinction.MP4", "cs only.MP4"]
    c = A.name_findings(_vids(files), _take(files))
    assert len(c) == 1 and c[0]["actual"] == "extinction"
    assert c[0]["claims"] == "baseline_cs"


def test_chapters_that_agree_with_their_take_are_not_flagged():
    """CSUS 1/2/3 are three chapters of one conditioning take and are perfectly named."""
    files = ["CSUS 1.MP4", "CSUS 2.MP4", "CSUS 3.MP4"]
    assert A.name_findings(_vids(files), _take(files)) == []


def test_a_name_that_claims_nothing_is_offered_one():
    """Nothing to contradict, but the take still says what it is, so offer a name."""
    files = ["extinction.MP4", "GX012908.MP4"]
    c = A.name_findings(_vids(files), _take(files))
    assert len(c) == 1 and c[0]["kind"] == "unnamed"
    assert c[0]["name"] == "GX012908.MP4" and c[0]["suggest"] == "extinction 2.MP4"
    assert c[0]["actual"] == "extinction" and c[0]["claims"] is None


def test_a_contradicted_name_and_an_absent_one_are_told_apart():
    files = ["extinction.MP4", "CSUS fin.MP4", "GX012908.MP4"]
    kinds = {c["name"]: c["kind"] for c in A.name_findings(_vids(files), _take(files))}
    assert kinds == {"CSUS fin.MP4": "conflict", "GX012908.MP4": "unnamed"}


def test_the_chapter_that_names_the_take_need_not_be_the_first():
    """An unnamed chapter 1 still gets a name from the chapter that does say one."""
    files = ["GX012907.MP4", "extinction 2.MP4"]
    c = A.name_findings(_vids(files), _take(files))
    assert len(c) == 1 and c[0]["name"] == "GX012907.MP4"
    assert c[0]["suggest"] == "extinction 1.MP4" and c[0]["actual"] == "extinction"


def test_a_take_no_chapter_names_is_left_entirely_alone():
    """Chaptering says these belong together, never what they are."""
    files = ["GX012888.MP4", "GX022888.MP4", "GX032888.MP4"]
    assert A.name_findings(_vids(files), _take(files)) == []


def test_the_chapter_number_is_stripped_before_a_name_is_built():
    """Otherwise chapter 3 of a take headed 'CSUS 1' would be offered 'CSUS 1 3'."""
    assert A.base_stem("CSUS 1") == "CSUS" and A.base_stem("extinction") == "extinction"
    assert A.base_stem("CSUS3") == "CSUS" and A.base_stem("csus 2") == "csus"


def test_a_role_on_a_later_chapter_still_speaks_for_the_take():
    files = ["GX012908.MP4", "CSUS fin.MP4"]
    c = A.name_findings(_vids(files), _take(files))
    assert len(c) == 1 and c[0]["name"] == "GX012908.MP4" and c[0]["actual"] == "conditioning"


def test_single_file_takes_are_never_flagged():
    rows = {"CSUS 4.MP4": dict(file="CSUS 4.MP4", take=["t", "t9"], chapter=1, n_chapters=1)}
    assert A.name_findings(_vids(["CSUS 4.MP4"]), rows) == []


def test_a_suggested_name_never_collides_with_a_file_already_there():
    files = ["extinction.MP4", "CSUS fin.MP4"]
    vids = _vids(files + ["extinction 2.MP4"])
    rows = _take(files)
    rows["extinction 2.MP4"] = dict(file="extinction 2.MP4", take=["t", "z"], chapter=1,
                                    n_chapters=1)
    s = A.name_findings(vids, rows)[0]["suggest"]
    assert s != "extinction 2.MP4" and s.startswith("extinction ")


# ------------------------------- roles proposed from where a recording was filmed
# The take answers first (name_findings above).  Where there is no take to answer,
# between_findings asks the recordings either side.  Both sides are required: duration
# cannot stand in for them (Thomas's US-only baseline is 498 s, his extinction 58 s) and
# neither can position (Charles's folder opens with two nameless clips filmed 45 minutes
# before his first baseline).
def _solo(files):
    """rows for files that are each their own single-chapter take, in the order given."""
    return {f: dict(file=f, take=["t", "t%d" % i], chapter=1, n_chapters=1, rank=i,
                    dated=True)
            for i, f in enumerate(files, 1)}


def test_a_nameless_recording_between_two_of_one_role_is_proposed_that_role():
    files = ["CSUS 1.MP4", "GX012908.MP4", "CSUS 3.MP4"]
    f = A.between_findings(_vids(files), _solo(files))
    assert len(f) == 1, f
    assert f[0]["name"] == "GX012908.MP4" and f[0]["actual"] == "conditioning"
    assert f[0]["kind"] == "unnamed" and f[0]["between"] == ["CSUS 1.MP4", "CSUS 3.MP4"]


def test_a_name_the_recordings_either_side_contradict_is_a_conflict():
    """Both neighbours say extinction, so a name saying conditioning is wrong."""
    files = ["extinction.MP4", "CSUS 9.MP4", "extinction 3.MP4"]
    f = A.between_findings(_vids(files), _solo(files))
    assert len(f) == 1 and f[0]["kind"] == "conflict"
    assert f[0]["claims"] == "conditioning" and f[0]["actual"] == "extinction"


def test_neighbours_that_disagree_settle_nothing():
    files = ["US ONLY.MP4", "GX012908.MP4", "CSUS 1.MP4"]
    assert A.between_findings(_vids(files), _solo(files)) == []


def test_a_recording_with_nothing_known_before_it_is_left_alone():
    """Charles's GX012907/GX012908, filmed 45 min before his first baseline. Anything at
    all can precede a session, so an unbounded side settles nothing."""
    files = ["GX012907.MP4", "GX012908.MP4", "US ONLY.MP4", "CS ONLY.MP4"]
    assert A.between_findings(_vids(files), _solo(files)) == []


def test_a_recording_with_nothing_known_after_it_is_left_alone():
    files = ["extinction.MP4", "GX012999.MP4"]
    assert A.between_findings(_vids(files), _solo(files)) == []


def test_a_take_overrules_the_name_of_its_own_chapter():
    """A name already known to be wrong must not go on being evidence.  With
    `extinction.MP4` and `CSUS fin.MP4` as chapters 1 and 2 of one take, taking the
    second name at face value made it a witness for conditioning - and the recording
    between it and the real conditioning chapters, `extinction.MP4` itself, was then
    reported as the misnamed one.  Both are extinction: the take says so."""
    files = ["extinction.MP4", "CSUS fin.MP4"]
    rows = _take(files)
    for i, f in enumerate(files, 1):
        rows[f].update(rank=i, dated=True)
    known = A.settled_roles(_vids(files), rows)
    assert known == {"extinction.MP4": "extinction", "CSUS fin.MP4": "extinction"}


def test_a_take_no_chapter_names_settles_nothing():
    files = ["GX012888.MP4", "GX022888.MP4"]
    assert A.settled_roles(_vids(files), _take(files)) == {}


def test_the_real_extinction_is_not_reported_as_the_misnamed_one():
    """The whole of Carole's folder with her old name back on it: the only finding is
    CSUS fin, and it comes from the take."""
    files = ["CS ONLY.MP4", "US ONLY.MP4", "CSUS 1.MP4", "CSUS 3.MP4",
             "extinction.MP4", "CSUS fin.MP4"]
    rows = {}
    for i, f in enumerate(files, 1):
        take = "ext" if f in ("extinction.MP4", "CSUS fin.MP4") else "t%d" % i
        rows[f] = dict(file=f, take=["t", take], chapter=1, n_chapters=1, rank=i,
                       dated=True)
    rows["extinction.MP4"].update(chapter=1, n_chapters=2)
    rows["CSUS fin.MP4"].update(chapter=2, n_chapters=2)
    f = A.role_findings(_vids(files), rows)
    assert [x["name"] for x in f] == ["CSUS fin.MP4"], f


def test_a_take_names_every_chapter_of_itself():
    files = ["extinction.MP4", "GX012908.MP4"]
    known = A.settled_roles(_vids(files), _take(files))
    assert known["GX012908.MP4"] == "extinction"


def test_an_undated_recording_neither_is_placed_nor_places_anything():
    files = ["CSUS 1.MP4", "raccourci.mkv", "CSUS 3.MP4"]
    rows = _solo(files)
    rows["raccourci.mkv"]["dated"] = False
    assert A.between_findings(_vids(files), rows) == []


def test_the_take_answers_before_the_neighbours_do():
    """One recording, one finding: role_findings must not raise the same file twice."""
    files = ["extinction.MP4", "GX012908.MP4", "extinction 3.MP4"]
    rows = _take(files[:2])
    rows["extinction 3.MP4"] = dict(file="extinction 3.MP4", take=["t", "z"], chapter=1,
                                    n_chapters=1, rank=3, dated=True)
    for i, f in enumerate(files[:2], 1):
        rows[f].update(rank=i, dated=True)
    f = A.role_findings(_vids(files), rows)
    assert [x["name"] for x in f] == ["GX012908.MP4"]
    assert "same take" in f[0]["why"]


# ------------------------- a position ticked against the order the camera filmed in
def test_a_chapter_moved_below_one_filmed_later_is_flagged():
    """The user's case: move CSUS 2 under CSUS 3 and it is numbered chapter 3, which
    makes chapter 3 the one filmed first."""
    rows = _items([("CSUS 1.MP4", "conditioning"), ("CSUS 2.MP4", "conditioning"),
                   ("CSUS 3.MP4", "conditioning")])
    rows[1], rows[2] = rows[2], rows[1]            # as if the arrows had been used
    f = A.order_findings(rows)
    assert len(f) == 1, f
    assert f[0]["file"] == "CSUS 2.MP4" and f[0]["shown"] == 3
    assert f[0]["after"] == "CSUS 3.MP4" and f[0]["after_shown"] == 2


def test_the_order_the_camera_filmed_in_is_not_flagged():
    assert A.order_findings(_items([("CSUS 1.MP4", "conditioning"),
                                    ("CSUS 2.MP4", "conditioning"),
                                    ("CSUS 3.MP4", "conditioning")])) == []


def test_positions_are_counted_within_one_role():
    """A baseline between two conditioning chapters does not push their numbers about."""
    rows = _items([("CSUS 1.MP4", "conditioning"), ("US ONLY.MP4", "baseline_us"),
                   ("CSUS 2.MP4", "conditioning")])
    assert A.order_findings(rows) == []


# ------------------------- numbers in the names that run against the filming order
def test_a_chapter_numbered_after_one_filmed_later_is_flagged():
    """Rename the second conditioning chapter `CSUS 5` and nothing used to notice: both
    are conditioning, so the role check is happy, and both really are chapters of that
    take, so the take check is happy. What is left is a name claiming it was filmed
    fifth while the camera says second."""
    files = ["CSUS 1.MP4", "CSUS 5.MP4", "CSUS 3.MP4"]
    f = A.sequence_findings(_vids(files), _solo(files))
    assert len(f) == 1, f
    assert f[0]["name"] == "CSUS 3.MP4" and f[0]["number"] == 3
    assert f[0]["after"] == "CSUS 5.MP4" and f[0]["after_number"] == 5


def test_names_numbered_in_the_filming_order_are_not_flagged():
    files = ["CSUS 1.MP4", "CSUS 2.MP4", "CSUS 3.MP4", "CSUS 4.MP4"]
    assert A.sequence_findings(_vids(files), _solo(files)) == []


def test_each_role_is_numbered_on_its_own():
    """extinction 1 filmed after CSUS 3 is not a number running backwards - they are
    different roles, and each carries its own count."""
    files = ["CSUS 1.MP4", "CSUS 2.MP4", "extinction 1.MP4", "extinction 2.MP4"]
    assert A.sequence_findings(_vids(files), _solo(files)) == []


def test_a_name_with_no_number_claims_no_position():
    """`extinction.MP4` before `extinction 2.MP4` is the usual pair and says nothing
    about order; "no number" and "number one" are different claims."""
    files = ["extinction.MP4", "extinction 2.MP4"]
    assert A.sequence_findings(_vids(files), _solo(files)) == []
    assert A.name_number("extinction") is None and A.name_number("CSUS 2") == 2
    assert A.name_number("CSUS3") == 3 and A.name_number("CSUS 2 test") == 2


def test_a_camera_number_is_not_a_position():
    """`GX012908.MP4` carries a five-digit number that means nothing about order, and
    its name states no role, so it takes no part."""
    files = ["CSUS 1.MP4", "GX012908.MP4", "CSUS 2.MP4"]
    assert A.sequence_findings(_vids(files), _solo(files)) == []


def test_an_undated_recording_makes_no_claim_about_order():
    files = ["CSUS 1.MP4", "CSUS 5.MP4", "CSUS 3.MP4"]
    rows = _solo(files)
    rows["CSUS 3.MP4"]["dated"] = False
    assert A.sequence_findings(_vids(files), rows) == []


# ------------------------------------- roles that contradict the session's own clock
# name_findings asks whether the NAME agrees with the camera.  These ask the question
# that reaches the numbers: whether the ROLE the run is about to use does.  The Role
# dropdown can be set to anything at any time, including after the name panel has been
# answered and dismissed, so this is the last thing standing between a mislabelled
# recording and a run that finishes normally and is wrong.
def _items(pairs):
    """Ticked recordings as the page sends them: (file, role) in recording order."""
    return [dict(path="C:/v/" + f, label=os.path.splitext(f)[0], role=r,
                 rank=i, dated=True, recorded="2024-10-31 09:%02d:00" % i)
            for i, (f, r) in enumerate(pairs, 1)]


def test_a_conditioning_chapter_filmed_after_extinction_is_flagged():
    """Carole's CSUS fin.MP4: chapter 2 of the extinction take, roled conditioning."""
    f = A.stage_findings(_items([("extinction.MP4", "extinction"),
                                 ("CSUS fin.MP4", "conditioning")]))
    assert len(f) == 1, f
    assert f[0]["file"] == "CSUS fin.MP4" and f[0]["after"] == "extinction.MP4"
    assert "conditioning chapter cannot" in f[0]["why"]


def test_a_baseline_filmed_after_conditioning_is_flagged():
    """Marie retest's cs only.MP4, which was serving as her CS-only baseline."""
    f = A.stage_findings(_items([("CSUS 1.MP4", "conditioning"),
                                 ("cs only.MP4", "baseline_cs")]))
    assert len(f) == 1 and f[0]["file"] == "cs only.MP4"
    assert f[0]["after"] == "CSUS 1.MP4" and "before any conditioning" in f[0]["why"]


def test_a_session_filmed_in_the_usual_order_is_not_flagged():
    """Every real session on disk is exactly this, and none of them may warn."""
    assert A.stage_findings(_items([
        ("CS ONLY.MP4", "baseline_cs"), ("US ONLY.MP4", "baseline_us"),
        ("CSUS 1.MP4", "conditioning"), ("CSUS 2.MP4", "conditioning"),
        ("CSUS 3.MP4", "conditioning"), ("extinction.MP4", "extinction"),
        ("extinction 2.MP4", "extinction")])) == []


def test_the_two_baselines_are_one_stage_in_either_order():
    """Which baseline the camera saw first means nothing - neither conditions anybody."""
    assert A.stage_findings(_items([("US ONLY.MP4", "baseline_us"),
                                    ("CS ONLY.MP4", "baseline_cs")])) == []


def test_the_reference_is_the_first_recording_of_the_later_stage():
    """'filmed after extinction' has to name where extinction STARTED, not its last
    chapter, or the sentence understates how far back the role reaches."""
    f = A.stage_findings(_items([("extinction.MP4", "extinction"),
                                 ("extinction 2.MP4", "extinction"),
                                 ("CSUS fin.MP4", "conditioning")]))
    assert len(f) == 1 and f[0]["after"] == "extinction.MP4"


def test_a_role_the_app_guessed_is_not_held_against_anybody():
    """`GX012907.MP4` says nothing, so the dropdown shows the placeholder the app had to
    put there - conditioning.  Checking that against a baseline filmed later fires on
    Charles's folder and on the whole 2016 Video root, and is the app accusing somebody
    of a label the app wrote itself."""
    items = _items([("GX012907.MP4", "conditioning"), ("us only.MP4", "baseline_us")])
    items[0]["guessed"] = True
    assert A.stage_findings(items) == []
    items[0]["guessed"] = False            # the user has now said what it is
    assert len(A.stage_findings(items)) == 1


def test_a_recording_the_camera_never_dated_takes_no_part():
    """It has no place on the session clock, so it is left out rather than guessed at."""
    items = _items([("extinction.MP4", "extinction"), ("GX012908.MP4", "conditioning")])
    items[1]["dated"] = False
    assert A.stage_findings(items) == []


def test_the_check_reads_the_camera_order_not_the_row_order():
    """The arrows in step 1 reorder the table; they do not move the camera clock."""
    items = _items([("extinction.MP4", "extinction"),
                    ("CSUS fin.MP4", "conditioning")])
    items.reverse()                        # as if the user had moved the row up
    assert len(A.stage_findings(items)) == 1


def test_every_role_has_a_stage_and_a_word():
    """A role added to ebc_config and not to ROLE_STAGE would stop being checked."""
    for r in C.ROLES:
        assert r in A.ROLE_STAGE and r in A.ROLE_WORD
    assert len(A.ROLE_STAGE_WORD) == len(set(A.ROLE_STAGE.values()))


def test_the_page_checks_the_same_stages_as_the_server():
    """The page runs this rule too, so the answer arrives while the Role column is still
    being set.  Two copies of one rule is a thing that drifts, so they are compared."""
    src = io.open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "ebc_app_ui.html"), encoding="utf-8").read()
    m = re.search(r"const ROLE_STAGE = \{([^}]*)\}", src)
    assert m, "the page no longer defines ROLE_STAGE"
    page = dict((k, int(v)) for k, v in re.findall(r"(\w+):(\d+)", m.group(1)))
    assert page == A.ROLE_STAGE, "page %s, server %s" % (page, A.ROLE_STAGE)


# --------------------------------------------------- the app can still start a run
# Nothing here had ever pressed Run.  1.3 added a module-level `STAGE` dict to ebc_app,
# which shadowed the `--stage` switch it imports from ebc_launch, and every run then
# built a command line with a dict in it and died inside subprocess - while Browse, the
# folder dialog, the name panel and every other button went on working normally, so the
# app looked entirely well.  These two are cheap and they close that whole class.
def test_nothing_shadows_what_ebc_app_imports():
    """Every name ebc_app takes from ebc_launch must still BE that name at import time."""
    import ebc_launch as L
    for n in ("STAGE", "PICK", "helper_cmd", "helper_env"):
        assert getattr(A, n) is getattr(L, n), \
            "ebc_app.%s is no longer ebc_launch.%s - something shadows it" % (n, n)


def test_the_app_can_start_its_own_pipeline():
    """Actually launch the command the app builds for Run.

    The config it is pointed at does not exist, so ebc_run_all refuses it and exits at
    once; what is under test is that the command line can be launched at all, which is
    exactly what the shadowed switch broke.
    """
    d = tempfile.mkdtemp()
    try:
        rc = A.run_pipeline(os.path.join(d, "does-not-exist.json"), d, False)
    finally:
        shutil.rmtree(d, ignore_errors=True)
    assert rc != 0, "a missing config should have been refused, not run"
    assert not any("TypeError" in ln for ln in A.STATE["log"]), \
        "the pipeline could not even be started:\n  " + "\n  ".join(A.STATE["log"][-6:])


# ---------------------------------------------------------- the app page
# ebc_app_ui.html ships as one <script>, so a single unterminated string takes the WHOLE
# page down: no button is wired, and the app opens looking perfectly normal and does
# nothing at all.  That shipped in 1.2 - a newline escape that had become a real
# newline inside a string literal - and no test could see it, because nothing here
# had ever read the page.
UI = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ebc_app_ui.html")


def _chrome():
    """Any Chrome on this machine.  Counting quotes by hand cannot do this job - a quote
    inside a regex literal is legal and a naive count calls it an error - so the page is
    handed to a real JavaScript parser instead."""
    for p in (os.environ.get("CHROME"),
              r"C:\Program Files\Google\Chrome\Application\chrome.exe",
              r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"):
        if p and os.path.exists(p):
            return p
    return shutil.which("chrome") or shutil.which("google-chrome") or shutil.which("chromium")


def test_the_page_script_parses():
    """The page is one <script>: one syntax error anywhere in it and NOTHING is wired -
    no Browse button, no Run button - while the app still opens and looks entirely
    normal.  That is exactly how 1.2 shipped.  The script is wrapped in a function that
    is never called, so this asks the parser about syntax and never runs the page."""
    chrome = _chrome()
    if not chrome:
        return                      # no parser here; the check is skipped, not faked
    src = io.open(UI, encoding="utf-8").read()
    body = src[src.index("<script>") + 8:src.rindex("</script>")]
    d = tempfile.mkdtemp()
    page = os.path.join(d, "syntax.html")
    with io.open(page, "w", encoding="utf-8") as fh:
        fh.write("<!doctype html><meta charset=utf-8><script>\nfunction __never__(){\n")
        fh.write(body)
        fh.write("\n}\n</script>")
    r = subprocess.run([chrome, "--headless", "--disable-gpu", "--no-sandbox",
                        "--virtual-time-budget=2000", "--dump-dom",
                        "--enable-logging=stderr", "--v=0", "file:///" + page.replace("\\", "/")],
                       capture_output=True, text=True, timeout=120)
    shutil.rmtree(d, ignore_errors=True)
    bad = [ln for ln in (r.stderr or "").splitlines()
           if "SyntaxError" in ln or "Uncaught" in ln]
    assert not bad, "the page's script does not parse:\n  " + "\n  ".join(bad[:3])


def test_every_id_the_script_reaches_for_exists_in_the_markup():
    """$("#foo") on an element that is not there returns null, and the next property
    access throws - which again takes the whole page with it."""
    src = io.open(UI, encoding="utf-8").read()
    have = set(re.findall(r'id="([A-Za-z0-9_-]+)"', src))
    want = set(re.findall(r'\$\("#([A-Za-z0-9_-]+)"\)', src))
    assert want <= have, "referenced but never defined: %s" % sorted(want - have)


def test_the_buttons_the_page_promises_are_wired():
    src = io.open(UI, encoding="utf-8").read()
    for b in ("again", "againtop", "dorename", "keepnames", "pick", "run", "stop"):
        assert 'id="%s"' % b in src, "no #%s in the markup" % b
        assert ('$("#%s").onclick' % b) in src, "#%s is never wired" % b


if __name__ == "__main__":
    import traceback
    fns = [(n, f) for n, f in sorted(globals().items())
           if n.startswith("test_") and callable(f)]
    bad = 0
    for n, f in fns:
        try:
            f()
            print("  ok    %s" % n)
        except Exception:
            bad += 1
            print("  FAIL  %s" % n)
            traceback.print_exc()
    print("\n%d passed, %d failed, %d total" % (len(fns) - bad, bad, len(fns)))
    sys.exit(1 if bad else 0)
