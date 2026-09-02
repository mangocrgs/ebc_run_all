"""Decide, per recording, whether the CS LED can be trusted - and what to do when it cannot.

    python ebc_triage.py <config.json>

The stimulus pass has already read both LEDs and written <tag>_stim.json.  This step reads
those numbers back and asks one question of each recording: *is the CS channel measuring
the LED, or is it measuring the room?*  From the answer it writes an effective study file
in which every recording is either

    anchored on the CS   the normal case, nothing changes
    anchored on the US   the CS LED is unreadable but the US LED is clean, so trials are
                         recovered from the US and the CS onset inferred - with a warning,
                         because an inferred onset is not a measured one
    excluded             neither LED can carry the trials.  Nothing automatic can save it
                         and the report says so instead of producing confident nonsense

Downstream stages read the effective file, so the rest of the pipeline never has to know
that any of this happened.

An explicit "anchor" in the study file is always obeyed; only "auto" (the default) is
decided here.

The thresholds below are not guesses.  They come from every recording processed so far,
where the two populations do not overlap:

    healthy CS channel     contrast 73 - 150, 80 - 100% of pulses survive the duration filter
    unreadable CS channel  contrast 30 -  43, 0.1 - 1.3% survive

A CS LED washed out by warm or dim room light does not fail cleanly.  Its resting level
climbs toward its lit level, the threshold ends up inside the noise, and the detector
reports hundreds of pulses that are then almost all rejected on duration.  Hundreds of
raw pulses with a handful accepted is the signature, and it is what these tests look for.
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ebc_config as C
from ebc_paths import work_dir

# a CS channel is suspect on any of these; two make it unusable
MIN_CONTRAST = 55.0      # healthy has been >= 73, unreadable <= 43
MIN_ACCEPT = 0.50        # healthy has been >= 0.80, unreadable <= 0.013
MIN_VS_US = 0.80         # a US only ever fires inside a CS, so CS pulses >= US pulses
CERTAIN_ACCEPT = 0.10    # this alone is damning: nine of ten "pulses" were noise

BANNER = "=" * 78

# a role is contradicted, not merely unsupported, when this many stimuli of the wrong
# kind were read - one stray pulse is noise, a dozen is a mislabelled file
WRONG_ROLE_MIN = 5

SUSPECT_NOTE = """
  These are scored normally from their CS LED: only one of the three checks was failed,
  which is not enough to overrule the channel.  But a healthy CS channel usually passes
  all three, so open qc_leds_<tag>.png and satisfy yourself the accepted pulses really
  look like stimuli before leaning on these trials."""


def examine(stim, proto, role=None):
    """What the two channels look like in one recording, and whether the CS can be used."""
    y = (stim["leds"].get("yellow") or {}).get("signal") or {}
    b = (stim["leds"].get("blue") or {}).get("signal") or {}
    n_raw, n_ok = y.get("n_raw", 0), y.get("n_ok", 0)
    us_ok = b.get("n_ok", 0)
    accept = n_ok / n_raw if n_raw else 0.0

    if role == "baseline_us":          # no CS is delivered, so there is no CS to judge
        return dict(cs_usable=True, suspect=False, reasons=[], n_cs_ok=n_ok, n_cs_raw=n_raw,
                    n_us_ok=us_ok, contrast=y.get("contrast"),
                    rest=y.get("rest_level"), lit=y.get("lit_level"))

    reasons = []
    if n_raw and accept < MIN_ACCEPT:
        reasons.append("only %d of %d detected pulses survived the duration filter (%.1f%%)"
                       % (n_ok, n_raw, 100.0 * accept))
    if y.get("contrast") is not None and y["contrast"] < MIN_CONTRAST:
        reasons.append("contrast %.0f is below %.0f (rest %.0f, lit %.0f)"
                       % (y["contrast"], MIN_CONTRAST, y.get("rest_level", 0), y.get("lit_level", 0)))
    if us_ok and n_ok < MIN_VS_US * us_ok:
        reasons.append("%d CS pulses against %d clean US pulses - a US only fires inside a CS"
                       % (n_ok, us_ok))

    cs_ok = not (len(reasons) >= 2 or (n_raw and accept < CERTAIN_ACCEPT))
    # One test failing is not enough to overrule the channel, but it is not nothing either:
    # the recording is scored normally and flagged for a human to look at.
    return dict(cs_usable=cs_ok, suspect=bool(cs_ok and reasons), reasons=reasons,
                n_cs_ok=n_ok, n_cs_raw=n_raw,
                n_us_ok=us_ok, contrast=y.get("contrast"),
                rest=y.get("rest_level"), lit=y.get("lit_level"))


def decide(rec, ex):
    """anchor / exclude / keep, for one recording."""
    role = rec["role"]
    want = rec.get("anchor", "auto")

    if role == "baseline_us":                       # never needed a CS in the first place
        return "cs", None, None
    if want in ("cs", "us"):                        # an explicit choice is obeyed
        note = None
        if want == "cs" and not ex["cs_usable"]:
            note = ("kept on the CS LED because the study file says so, but the CS channel "
                    "does not look readable - check qc_leds_%s.png" % rec["tag"])
        return want, note, None
    if ex["cs_usable"]:
        return "cs", None, None

    # the CS LED cannot be read.  What can be done depends on what the recording contains.
    if role in C.NO_US_ROLES:
        return None, None, (
            "%s delivers the CS alone, so there is no US to recover its trials from. "
            "No automatic method can score it." % rec["label"])
    if ex["n_us_ok"] < 2:
        return None, None, (
            "%s has an unreadable CS LED and only %d usable US pulse(s), so neither channel "
            "can carry the trials." % (rec["label"], ex["n_us_ok"]))
    return "us", ("trials will be recovered from the US LED and the CS onset inferred"), None


def role_check(cfg, stims):
    """Does each recording behave like the role it was given?

    The role is typed by a person at the end of a long session and it decides how the
    recording is scored, so it is worth testing against what the LEDs did.  A US is only
    ever delivered in conditioning; a recording labelled `extinction` that is full of
    puffs is a file named wrong, not an unusual extinction.  Nothing is renamed here -
    that would be guessing at intent - but the disagreement is stated.
    """
    out = []
    for rec in cfg["recordings"]:
        s = stims.get(rec["tag"])
        if not s:
            continue
        n_cs = ((s["leds"].get("yellow") or {}).get("signal") or {}).get("n_ok", 0)
        n_us = ((s["leds"].get("blue") or {}).get("signal") or {}).get("n_ok", 0)
        role, said = rec["role"], None
        if role in C.NO_US_ROLES and n_us >= WRONG_ROLE_MIN:
            said = ("%d US flashes were read in a recording labelled '%s', which delivers "
                    "the CS alone.  Either this is a conditioning recording under the "
                    "wrong name, or the blue channel is reading something else."
                    % (n_us, role))
        elif role == "conditioning" and n_cs >= WRONG_ROLE_MIN and n_us == 0:
            said = ("%d CS presentations and no US at all, in a recording labelled "
                    "'conditioning'.  That is what extinction or a CS-only baseline looks "
                    "like; check the role before reading the CR rate." % n_cs)
        elif role == "baseline_us" and n_cs >= WRONG_ROLE_MIN:
            said = ("%d CS presentations were read in a US-only baseline, which has no CS."
                    % n_cs)
        if said:
            out.append(dict(tag=rec["tag"], label=rec["label"], role=role, message=said))
    return out


def geometry_check(stims):
    """Is the US LED on the same side of the CS LED in every recording?

    Which side it is on is a fact about the rig - how the box was turned, which side of
    the participant the camera stood.  It is measured per recording rather than assumed,
    and if it changes half way through a participant then something moved between
    recordings, and every position inherited across that boundary is suspect.
    """
    seen = [(t, s["us_offset"], s.get("us_side")) for t, s in stims.items()
            if s.get("us_offset")]
    if len(seen) < 2:
        return None
    sides = {sd for _, _, sd in seen}
    xs = [o[0] for _, o, _ in seen]
    ys = [o[1] for _, o, _ in seen]
    spread = max(max(xs) - min(xs), max(ys) - min(ys))
    if len(sides) == 1 and spread <= 60:
        return None
    return ("the US LED is not in the same place relative to the CS LED in every "
            "recording (%s).  The box was turned or the camera moved between them, so "
            "check qc_leds_<tag>.png for each: a position inherited from another "
            "recording will be wrong across that change."
            % ", ".join("%s %+d,%+d" % (t, o[0], o[1]) for t, o, _ in seen))


def run(cfg):
    wdir = work_dir(cfg)
    proto = cfg["protocol"]
    out, notes, excluded, seen, suspect = [], [], [], [], []
    stims = {}

    for rec in cfg["recordings"]:
        f = os.path.join(wdir, rec["tag"] + "_stim.json")
        if not os.path.exists(f):
            out.append(dict(rec))                   # not read yet; leave it alone
            continue
        with open(f, encoding="utf-8") as fh:
            stim = json.load(fh)
        stims[rec["tag"]] = stim
        ex = examine(stim, proto, rec["role"])
        anchor, note, drop = decide(rec, ex)

        entry = dict(tag=rec["tag"], label=rec["label"], role=rec["role"], **ex)
        if drop:
            entry.update(action="excluded", message=drop)
            excluded.append(entry)
            continue
        r = dict(rec)
        r["anchor"] = anchor
        entry.update(action="anchor:" + anchor, message=note, anchor=anchor)
        seen.append(entry)
        if note:
            notes.append(entry)
        elif ex["suspect"] and anchor == "cs":
            suspect.append(entry)
        out.append(r)

    eff = dict(cfg)
    eff["recordings"] = [{k: v for k, v in r.items() if k != "path"} for r in out]
    eff_path = os.path.join(wdir, "effective_config.json")
    with open(eff_path, "w", encoding="utf-8") as fh:
        json.dump(eff, fh, indent=1)
    roles = role_check(cfg, stims)
    geom = geometry_check(stims)
    with open(os.path.join(wdir, "triage.json"), "w", encoding="utf-8") as fh:
        json.dump(dict(examined=seen, notes=notes, suspect=suspect, excluded=excluded,
                       role_disagreements=roles, geometry=geom), fh, indent=1)
    return eff_path, notes, excluded, seen, suspect, roles, geom


def report(notes, excluded, kept, suspect=(), roles=(), geom=None):
    print("\n%-14s %-13s %9s %7s %9s   %s"
          % ("recording", "role", "contrast", "CS ok", "of raw", "trials from"))
    for r in kept:
        print("%-14s %-13s %9s %7s %9s   %s"
              % (r["tag"], r["role"], "-" if r.get("contrast") is None else "%.0f" % r["contrast"],
                 r.get("n_cs_ok", "-"), r.get("n_cs_raw", "-"),
                 "US LED" if r.get("anchor") == "us" else "CS LED"))

    if notes:
        print("\n" + BANNER)
        print("WARNING - the CS LED could not be read in %d recording(s)." % len(notes))
        print(BANNER)
        for e in notes:
            print("\n  %s (%s)" % (e["label"], e["tag"]))
            for why in e["reasons"]:
                print("      - " + why)
            print("    -> %s" % e["message"])
        print("""
  What this means for the results:

    - Trial times come from the US LED, which is clean in these recordings.  The CS
      onset is INFERRED as US onset minus the protocol interval, not measured.  Checked
      against a recording where both LEDs were readable, that inference was accurate to
      a median of 0.3 ms - under one video frame - so the timing is sound.
    - CS-only probes deliver no US and CANNOT be recovered from these recordings.  Any
      block boundary that a probe would have marked is set by counting paired trials
      instead, and every trial says which it was.
    - Each trial in the workbook carries "CS timing source" so measured and inferred
      onsets are never confused.

  Open qc_leds_<tag>.png to see the LED traces this decision was based on.""")
        print(BANNER)

    if suspect:
        print("")
        print(BANNER)
        print("WORTH A LOOK - %d recording(s) passed, but not comfortably." % len(suspect))
        print(BANNER)
        for e in suspect:
            print("")
            print("  %s (%s)" % (e["label"], e["tag"]))
            for why in e["reasons"]:
                print("      - " + why)
        print(SUSPECT_NOTE)
        print(BANNER)

    if excluded:
        print("\n" + BANNER)
        print("CANNOT BE PROCESSED - %d recording(s) left out of this run." % len(excluded))
        print(BANNER)
        for e in excluded:
            print("\n  %s (%s), role %s" % (e["label"], e["tag"], e["role"]))
            for why in e["reasons"]:
                print("      - " + why)
            print("    -> " + e["message"])
        print("""
  THESE RECORDINGS MUST BE SCORED BY HAND.  This is not a setting that can be tuned:
  a CS-only or extinction recording delivers no US, so when its CS LED is unreadable
  there is no second channel to recover the trials from.

  The rest of the study has been processed normally.  Open qc_leds_<tag>.png for these
  recordings to see the CS channel, then read the trial onsets off the video by hand.""")
        print(BANNER)


def main():
    cfg = C.load(sys.argv[1] if len(sys.argv) > 1 else None)
    eff_path, notes, excluded, kept, suspect, roles, geom = run(cfg)
    report(notes, excluded, kept, suspect, roles, geom)
    if not any(r["role"] != "baseline_us" for r in kept):
        print("\n!! nothing left to score - every recording was excluded")
    print("\nwrote " + eff_path)


if __name__ == "__main__":
    main()
