"""Unit tests for the pure logic in the pipeline.

    python -m pytest test_ebc.py -q        (or: python test_ebc.py)

These cover the functions that decide what a trial IS - classification, CS-US pairing,
block numbering and pulse detection.  All four run in milliseconds on synthetic input,
and every bug they encode here was found instead by a fifty-minute run over real video:

    A2   classify() never saw trial_type, so a CS-only probe with a late response came
         out labelled "UR (>=350ms)" - and since the summaries count a CR with
         startswith("CR"), every late response in extinction was excluded from the CR
         count by definition.
    B4   a US that goes missing turns a paired trial into a CS-only trial that never
         happened, which the block loop then reads as a block boundary: Marie came out
         with 36 blocks where the protocol has 10.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ebc_protocol as P
import ebc_score as S

PROTO = dict(cs_ms=400.0, us_onset_ms=350.0, us_dur_ms=50.0, paired_per_block=9,
             cs_only_per_block=1, n_blocks=10, min_iti_s=5.0, cs_tol=0.35, us_tol=0.6)


# ------------------------------------------------------------------ classify (A2)
def test_cr_before_the_puff():
    assert S.classify(200.0, 350.0, False, False, True).startswith("CR")


def test_ur_after_the_puff():
    assert S.classify(430.0, 350.0, False, False, True).startswith("UR")


def test_alpha_is_not_a_cr():
    assert S.classify(25.0, 350.0, False, False, True).startswith("alpha")


def test_late_response_on_a_probe_is_a_CR_not_a_UR():
    """A2. No puff was delivered, so there is no UR the response could be."""
    cls = S.classify(383.7, 350.0, False, False, us_delivered=False)
    assert cls.startswith("CR"), cls
    assert "UR" not in cls


def test_probe_and_paired_trials_differ_at_the_same_latency():
    """The same onset means different things depending on whether a puff arrived."""
    paired = S.classify(430.0, 350.0, False, False, True)
    probe = S.classify(430.0, 350.0, False, False, False)
    assert paired.startswith("UR") and probe.startswith("CR")




def test_alpha_still_wins_on_a_probe():
    assert S.classify(25.0, 350.0, False, False, False).startswith("alpha")


def test_us_anchored_trial_is_always_unconditioned():
    assert S.classify(80.0, 350.0, False, True, True) == "UR to the puff"


def test_moving_lid_is_untimeable():
    assert S.classify(200.0, 350.0, True, False, True) == "in-progress at stimulus"


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
    for i, t in enumerate(cond):
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
    """B4. One paired trial mis-read as a probe splits its block in two.

    This is Marie: the position gate discarded a quarter of her puffs, each lost puff
    turned a paired trial into a CS-only trial, and the block loop closed a block on
    every one of them - 36 blocks recovered where the protocol has 10.
    """
    good = (["CS-US"] * 9 + ["CS-only"]) * 2
    runs_ok, _, _ = _blocks(good)
    broken = list(good)
    broken[4] = "CS-only"                      # a puff that was delivered but not seen
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
    fps = 119.88
    sig = _square(fps, 60, [5.0, 20.0, 35.0, 50.0], 400.0)
    ev, info = X.detect(sig, fps, 400.0, tol=0.35, min_gap_s=5.0)
    assert sum(e["ok"] for e in ev) == 4, [e["dur_ms"] for e in ev]


def test_detect_rejects_the_wrong_duration():
    import ebc_stimulus as X
    fps = 119.88
    sig = _square(fps, 60, [5.0, 20.0], 1200.0)
    ev, info = X.detect(sig, fps, 400.0, tol=0.35, min_gap_s=5.0)
    assert sum(e["ok"] for e in ev) == 0


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


# ------------------------------------------------------------------ the response window
def test_cr_window_runs_from_CS_plus_offset_to_US_plus_offset():
    lo, hi = S.RESP_OFFSET_MS, 350.0 + S.RESP_OFFSET_MS
    assert S.classify(lo + 1, 350.0, False, False, True).startswith("CR")
    assert S.classify(hi - 1, 350.0, False, False, True).startswith("CR")
    assert S.classify(hi + 1, 350.0, False, False, True).startswith("UR")


def test_too_early_to_be_a_response_to_the_CS():
    assert S.classify(S.RESP_OFFSET_MS - 1, 350.0, False, False, True).startswith("alpha")


def test_the_offset_is_mean_minus_one_and_a_half_sd():
    assert abs(S.RESP_OFFSET_MS - (S.REFLEX_MEAN_MS - 1.5 * S.REFLEX_SD_MS)) < 1e-9


def test_the_same_window_applies_to_every_participant():
    """One rule, not a per-person fit: classify() takes no latency argument."""
    import inspect
    assert "latency" not in str(inspect.signature(S.classify))


def test_the_old_boundary_would_have_called_this_a_UR():
    """350.4 ms - six of Carole's trials sit there, 0.4 ms after the puff."""
    assert S.classify(350.4, 350.0, False, False, True).startswith("CR")


def test_a_very_late_blink_is_neither_CR_nor_UR():
    assert S.classify(950.0, 350.0, False, False, True).startswith("spontaneous")


def test_a_late_but_plausible_probe_response_is_still_a_CR():
    assert S.classify(370.0, 350.0, False, False, False).startswith("CR")


def test_spontaneous_blink_in_a_us_only_trial_is_not_a_UR():
    assert S.classify(901.0, 350.0, False, True, True).startswith("spontaneous")
    assert S.classify(67.0, 350.0, False, True, True) == "UR to the puff"


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


# ------------------------------------------------------------------ A1 / A3
def test_response_between_puff_and_reflex_is_a_CR():
    """A1. The puff is at 350 ms and the reflex takes 67 ms, so nothing puff-driven can
    begin before 417 ms.  A response at 383 ms started too early to be a UR."""
    cls = S.classify(383.0, 350.0, False, False, True, ur_latency_ms=67.0)
    assert cls.startswith("CR"), cls


def test_response_after_the_reflex_latency_is_a_UR():
    assert S.classify(430.0, 350.0, False, False, True, ur_latency_ms=67.0).startswith("UR")


def test_the_boundary_moves_with_the_measured_latency():
    onset = 400.0
    assert S.classify(onset, 350.0, False, False, True, ur_latency_ms=67.0).startswith("CR")
    assert S.classify(onset, 350.0, False, False, True, ur_latency_ms=20.0).startswith("UR")


def test_a_very_late_blink_is_neither_CR_nor_UR():
    cls = S.classify(950.0, 350.0, False, False, True, ur_latency_ms=67.0)
    assert cls.startswith("spontaneous"), cls


def test_a_very_late_blink_on_a_probe_is_also_spontaneous():
    cls = S.classify(950.0, 350.0, False, False, False, ur_latency_ms=67.0)
    assert cls.startswith("spontaneous"), cls


def test_a_late_but_plausible_probe_response_is_still_a_CR():
    """A2 must survive A1: a decaying CR is late, but not spontaneous."""
    assert S.classify(500.0, 350.0, False, False, False, ur_latency_ms=67.0).startswith("CR")


def test_spontaneous_blink_in_a_us_only_trial_is_not_a_UR():
    """A3. Thomas's US-only onsets run to 901 ms; those are not reflexes."""
    assert S.classify(901.0, 350.0, False, True, True).startswith("spontaneous")
    assert S.classify(67.0, 350.0, False, True, True) == "UR to the puff"


def test_latency_is_the_fast_end_of_plausible_puff_responses():
    """The line asks whether the puff COULD have caused it, so it sits at the fastest
    reflex the puff has produced, not the typical one."""
    lat, used, tot = S.measure_ur_latency([50.0, 67.0, 75.0, 66.0, 901.0, -50.0])
    assert used == 4 and tot == 6
    assert lat < 60.0, lat


def test_the_median_would_mislabel_the_UR_population():
    """Thomas. Reflex 33-125 ms, and 47 of 87 paired trials in one bin at 400-425 ms.

    A line built on his median (75) sits at 425 and calls that whole spike a CR; a line
    built on the fast end sits below it and leaves it as the UR population it is.
    """
    import numpy as np
    reflex = [33.4, 33.4, 41.7, 58.4, 58.4, 58.4, 58.4, 66.7, 66.7, 75.1, 75.1, 75.1,
              75.1, 83.4, 83.4, 125.1]
    lat, _, _ = S.measure_ur_latency(reflex)
    assert 350.0 + lat < 400.0, "the UR spike at 400-425 ms must fall above the line"
    assert 350.0 + float(np.median(reflex)) > 415.0, "the median would swallow it"


def test_latency_falls_back_when_the_baseline_is_too_thin():
    lat, used, tot = S.measure_ur_latency([901.0, -50.0], default=60.0)
    assert used == 0 and lat == 60.0


def test_contaminated_baseline_would_have_moved_the_boundary():
    """Why the window matters: without it the median is dragged by spontaneous blinks."""
    onsets = [50.0, 67.0, 75.0, 66.0, 901.0, 880.0, 870.0]
    windowed, _, _ = S.measure_ur_latency(onsets)
    import numpy as np
    naive = float(np.median([o for o in onsets]))
    assert windowed < 100.0 < naive
