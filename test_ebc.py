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
