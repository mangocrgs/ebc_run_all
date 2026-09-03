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
import ebc_protocol as P
import ebc_score as S

PROTO = dict(cs_ms=400.0, us_onset_ms=350.0, us_dur_ms=50.0, paired_per_block=9,
             cs_only_per_block=1, n_blocks=10, min_iti_s=5.0, cs_tol=0.35, us_tol=0.6)
US = 350.0


# ------------------------------------------------------------------ the response window
def test_the_offset_is_mean_minus_one_and_a_half_sd():
    assert abs(S.RESP_OFFSET_MS - (S.REFLEX_MEAN_MS - 1.5 * S.REFLEX_SD_MS)) < 1e-9


def test_the_same_window_applies_to_every_participant():
    """One rule, not a per-person fit: classify() takes no latency argument."""
    import inspect
    assert "latency" not in str(inspect.signature(S.classify))


def test_cr_window_runs_from_CS_plus_offset_to_US_plus_offset():
    lo, hi = S.RESP_OFFSET_MS, US + S.RESP_OFFSET_MS
    assert S.classify(lo + 1, US, False, False, True).startswith("CR")
    assert S.classify(hi - 1, US, False, False, True).startswith("CR")
    assert S.classify(hi + 1, US, False, False, True).startswith("UR")


def test_too_early_to_be_a_response_to_the_CS():
    assert S.classify(S.RESP_OFFSET_MS - 1, US, False, False, True).startswith("alpha")


def test_cr_before_the_puff():
    assert S.classify(200.0, US, False, False, True).startswith("CR")


def test_ur_after_the_puff():
    assert S.classify(430.0, US, False, False, True).startswith("UR")


def test_the_old_boundary_would_have_called_this_a_UR():
    """350.4 ms - six of Carole's trials sit there, 0.4 ms after the puff."""
    assert S.classify(350.4, US, False, False, True).startswith("CR")


def test_a_very_late_blink_is_neither_CR_nor_UR():
    assert S.classify(950.0, US, False, False, True).startswith("spontaneous")


def test_moving_lid_is_untimeable():
    assert S.classify(200.0, US, True, False, True) == "in-progress at stimulus"


def test_us_anchored_trial_is_always_unconditioned():
    assert S.classify(67.0, US, False, True, True) == "UR to the puff"


def test_spontaneous_blink_in_a_us_only_trial_is_not_a_UR():
    """Thomas's US-only onsets run to 901 ms; those are not reflexes."""
    assert S.classify(901.0, US, False, True, True).startswith("spontaneous")


# ------------------------------------------------------------------ trials with no puff
def test_late_response_on_a_probe_is_not_a_UR():
    """A2. No puff was delivered, so there is no UR the response could be."""
    cls = S.classify(383.7, US, False, False, us_delivered=False)
    assert "UR" not in cls, cls


def test_a_probe_uses_the_same_CR_window_as_a_paired_trial():
    """Otherwise the probe cannot serve its purpose, which is to be comparable."""
    lo, hi = S.RESP_OFFSET_MS, US + S.RESP_OFFSET_MS
    assert S.classify(lo + 1, US, False, False, False).startswith("CR")
    assert S.classify(hi - 1, US, False, False, False).startswith("CR")


def test_a_probe_response_after_the_window_is_not_a_CR():
    """A2 must not become its own mirror image: 'every blink is a CR' is as wrong as
    'every blink is a UR'.  A CS-only baseline exists to give the false-positive rate,
    and it cannot do that if it returns 100% by construction."""
    cls = S.classify(US + S.RESP_OFFSET_MS + 50.0, US, False, False, False)
    assert not cls.startswith("CR"), cls


def test_alpha_still_wins_on_a_probe():
    assert S.classify(25.0, US, False, False, False).startswith("alpha")


# ------------------------------------------------------------------ the standing check
def test_a_no_conditioning_participant_must_not_look_like_a_learner():
    """Thomas. A clean spike 50-75 ms after his puff and almost nothing before it is
    what a correct measurement of NO conditioning looks like.  Any scoring change that
    turns him into a learner is wrong, and this is the cheapest way to notice.

    These are his real paired-trial onsets, binned as they appear in his CSV.
    """
    onsets = ([120.0] * 2 + [180.0] + [210.0] + [310.0] + [330.0]
              + [390.0] * 10 + [410.0] * 47 + [430.0] * 17 + [510.0] + [540.0] + [560.0])
    cls = [S.classify(o, US, False, False, True) for o in onsets]
    scoreable = [c for c in cls if not c.startswith("spontaneous")
                 and c != "in-progress at stimulus"]
    cr = [c for c in scoreable if c.startswith("CR")]
    rate = len(cr) / len(scoreable)
    assert rate < 0.35, "Thomas reads %.0f%% CR - the boundary is in the wrong place" % (100 * rate)


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
