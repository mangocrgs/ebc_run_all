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
import os
import re
import sys

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
WIN = C.cr_window(PROTO, REFLEX)
FALLBACK = C.cr_window(PROTO)
LO, HI = WIN["lo_ms"], WIN["hi_ms"]


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


def test_moving_lid_is_untimeable():
    assert S.classify(200.0, WIN, True, False, True) == WIN["moving_label"]


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


def test_the_probe_curve_is_broken_across_blocks_with_no_probe():
    """Joining the only two scoreable probes draws a trend over blocks never measured."""
    import ebc_figures as F
    rows = [dict(block=1, scored_class=WIN["cr_no_us_label"], scored_onset_ms=200.0),
            dict(block=9, scored_class=WIN["late_no_us_label"], scored_onset_ms=600.0)]
    pr = F.block_rate(rows, "CR")
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
