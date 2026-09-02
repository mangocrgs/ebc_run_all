"""Study configuration: which recordings exist, what role each plays, what the protocol is.

Everything that changes from one participant to the next lives in a JSON file, so the
same code runs unchanged on the next dozen participants:

    python ebc_run_all.py --config studies/thomas.json

With no config, the recordings are discovered from a folder by name, which is enough
for the standard layout (CSUS 1..n, extinction, CS only, US only).
"""
import json
import os
import re

# Roles a recording can play.  The role decides what is expected of it and how it is
# scored - not the file name, and not a hard-coded tag.
#
#   conditioning  paired CS-US trials plus the CS-only probe that ends each block.
#                 Chapters of one continuous session; they concatenate on one clock.
#   extinction    CS alone after conditioning.  No US is delivered.
#   baseline_cs   CS alone before/outside conditioning.  Not part of the block structure.
#   baseline_us   US alone.  There is no CS, so trials are anchored on the US instead.
ROLES = ("conditioning", "extinction", "baseline_cs", "baseline_us")

# Which LED a recording's trials are built from.
#
#   cs   the CS (yellow) LED marks trial onset.  The default, and the only one that can
#        see a CS-only probe, because a probe delivers no US.
#   us   the US (blue) LED marks the trial and the CS onset is inferred as
#        us_onset - us_onset_ms.  Use when the CS LED is too washed out to threshold
#        reliably (ebc_qc.py leds reports the contrast).  CS-only probes are invisible
#        to this mode and the CS duration is not measured, only assumed.
#   auto the default: ebc_triage.py looks at what the LEDs actually did in this
#        recording and picks.  An explicit "cs" or "us" is always obeyed instead.
ANCHORS = ("auto", "cs", "us")

# Roles that deliver the CS alone.  US-anchoring is impossible for these - there is no US.
NO_US_ROLES = ("extinction", "baseline_cs")

NO_US_MESSAGE = """
{file}: cannot anchor on the US - this recording has none.

  Role '{role}' delivers the CS alone, by design.  There is no blue pulse to mark a
  trial and nothing from which to infer a CS onset, so if the CS (yellow) LED cannot
  be thresholded in this recording, then no automatic method can recover its trials.
  This is a property of the protocol, not a setting to tune.

  SCORE THIS RECORDING MANUALLY.  Run

      python ebc_qc.py <config> leds

  to see what the CS channel actually looks like, then read the trial onsets off the
  recording by hand.

  Remove '{file}' from the study file to process the rest.
"""

DEFAULT_PROTOCOL = {
    "cs_ms": 400.0,          # CS duration (yellow LED)
    "us_onset_ms": 350.0,    # US onset, measured from CS onset
    "us_dur_ms": 50.0,       # US duration (blue LED); co-terminates with the CS
    "paired_per_block": 9,
    "cs_only_per_block": 1,
    "n_blocks": 10,
    "min_iti_s": 5.0,        # two CS onsets closer than this cannot both be trials
    "cs_tol": 0.35,          # accept a CS whose measured duration is within +-35%
    "us_tol": 0.60,          # the US is short, so its measured duration is noisier
}

# name -> role, applied to the file stem, case-insensitive, first match wins.
#
# The names come from whoever emptied the SD card, so the patterns allow for the
# spellings that have actually turned up: `CS ONLY`, `cs only`, `CSUS3` with no space,
# `csus 2`, `CSUS fin`, and the French the lab writes half the time.  What a name says
# about the ORDER of the recordings is not used - see ebc_media.py; the camera's own
# clock is read instead, because `CSUS fin` has twice turned out to be the tail of the
# extinction take rather than the recording after `CSUS 3`.
_PATTERNS = [
    (r"\bcs\s*[-_ ]?(only|alone|seul)", "baseline_cs"),
    (r"\bus\s*[-_ ]?(only|alone|seul)", "baseline_us"),
    (r"extinct", "extinction"),
    (r"cs\s*[-_ ]?us|us\s*[-_ ]?cs|paired|appari|conditionn?(ement|ing)|acquisition", "conditioning"),
]

VIDEO_EXTS = (".mp4", ".mov", ".avi", ".mkv", ".m4v", ".mts")


def _slug(stem):
    return re.sub(r"[^a-z0-9]+", "", stem.lower()) or "rec"


def _numeric_key(stem):
    """Sort CSUS 2 before CSUS 10, and 'extinction' before 'extinction 2'."""
    m = re.findall(r"\d+", stem)
    return (int(m[-1]) if m else 1, stem.lower())


def role_of(stem):
    """The role a file name implies, or None when it implies nothing."""
    return next((r for pat, r in _PATTERNS if re.search(pat, stem, re.I)), None)


def _uniq(tag, used):
    """A tag no other recording has.  Two files can slug to one tag - `CSUS 1.MP4` and
    `CSUS1.MP4` both give `csus1` - and that used to stop the run before it started."""
    t, i = tag, 2
    while t in used:
        t, i = "%s_%d" % (tag, i), i + 1
    used.add(t)
    return t


def video_files(video_dir, exts=VIDEO_EXTS):
    return [fn for fn in sorted(os.listdir(video_dir))
            if os.path.splitext(fn)[1].lower() in exts and not fn.startswith(("~", "."))]


def unrecognised(video_dir, exts=VIDEO_EXTS):
    """Recordings in the folder whose name says nothing about what they are.

    Discovery cannot include these - a role is not a guess to be made from nothing - but
    they are the commonest way for a session to go missing (`GX012908.MP4`, straight off
    the card), so they are reported rather than passed over in silence.
    """
    return [fn for fn in video_files(video_dir, exts)
            if role_of(os.path.splitext(fn)[0]) is None]


def discover(video_dir, exts=VIDEO_EXTS):
    """Build a recording list from the file names in a folder."""
    found, used = [], set()
    for fn in video_files(video_dir, exts):
        stem = os.path.splitext(fn)[0]
        role = role_of(stem)
        if role is None:
            continue
        found.append({"tag": _uniq(_slug(stem), used), "file": fn, "label": stem.strip(),
                      "role": role, "from": "folder scan", "_key": _numeric_key(stem)})
    order = {r: 0 for r in ROLES}
    out = []
    for rec in sorted(found, key=lambda r: (ROLES.index(r["role"]), r["_key"])):
        rec.pop("_key")
        order[rec["role"]] += 1
        rec["order"] = order[rec["role"]]        # provisional: ebc_timeline.py re-dates it
        out.append(rec)
    return out


def load(path=None, video_dir=None, study=None):
    cfg = {}
    if path:
        with open(path, encoding="utf-8") as f:
            cfg = json.load(f)
        base = os.path.dirname(os.path.abspath(path))
        if cfg.get("video_dir") and not os.path.isabs(cfg["video_dir"]):
            cfg["video_dir"] = os.path.normpath(os.path.join(base, cfg["video_dir"]))
    if video_dir:
        cfg["video_dir"] = video_dir
    if study:
        cfg["study"] = study
    if not cfg.get("video_dir"):
        raise SystemExit("no video_dir: pass --videos <folder> or a --config that sets it")
    cfg["video_dir"] = os.path.abspath(cfg["video_dir"])
    cfg.setdefault("study", os.path.basename(cfg["video_dir"].rstrip("/\\")))
    cfg.setdefault("out_dir", os.path.join(cfg["video_dir"], "analysis_EBC"))
    if not os.path.isabs(cfg["out_dir"]):
        cfg["out_dir"] = os.path.join(cfg["video_dir"], cfg["out_dir"])

    proto = dict(DEFAULT_PROTOCOL)
    proto.update(cfg.get("protocol", {}))
    cfg["protocol"] = proto

    if not cfg.get("recordings"):
        cfg["recordings"] = discover(cfg["video_dir"])
    seen, kept, left_out = set(), [], []
    for rec in cfg["recordings"]:
        rec.setdefault("label", os.path.splitext(rec["file"])[0])
        rec.setdefault("tag", _slug(rec["label"]))
        if rec["role"] not in ROLES:
            raise SystemExit(f"{rec['file']}: unknown role {rec['role']!r} (expected {ROLES})")
        rec.setdefault("anchor", "auto")
        if rec["anchor"] not in ANCHORS:
            raise SystemExit(f"{rec['file']}: unknown anchor {rec['anchor']!r} (expected {ANCHORS})")
        if rec["anchor"] == "us" and rec["role"] in NO_US_ROLES:
            raise SystemExit(NO_US_MESSAGE.format(file=rec["file"], role=rec["role"]))
        # Two files whose names differ only in spacing slug to the same tag.  That is a
        # naming accident, not a reason to refuse to run: rename the second and say so.
        if rec["tag"] in seen:
            was = rec["tag"]
            rec["tag"] = _uniq(was, seen)
            print("note: %s and another recording both give the tag %r; this one is %r"
                  % (rec["file"], was, rec["tag"]))
        else:
            seen.add(rec["tag"])
        rec["path"] = os.path.join(cfg["video_dir"], rec["file"])
        # "include": false is how ebc_timeline.py records a decision to leave a file out
        # (a copy of another recording, a name that says it failed).  The reason travels
        # with it, so nothing disappears without an explanation.
        (left_out if rec.get("include") is False else kept).append(rec)
    for rec in left_out:
        print("left out: %-28s %s" % (rec["file"], rec.get("excluded_because", "include: false")))
    cfg["recordings"] = kept
    cfg["excluded"] = cfg.get("excluded", []) + [
        {k: v for k, v in r.items() if k != "path"} for r in left_out]
    if not kept:
        raise SystemExit("every recording in the study file is marked include: false")
    return cfg


def by_role(cfg, *roles):
    order = {r: i for i, r in enumerate(ROLES)}
    return sorted([r for r in cfg["recordings"] if r["role"] in roles],
                  key=lambda r: (order[r["role"]], r.get("order", 0)))


def expected_trials(proto):
    per = proto["paired_per_block"] + proto["cs_only_per_block"]
    return per * proto["n_blocks"]
