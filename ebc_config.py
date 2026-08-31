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

# Stamped on the page, on the console banner and on every workbook cover, because a
# number that reaches a paper has to be traceable to the thing that produced it.  Raise
# it whenever the scoring changes.
VERSION = "1.0"
LAB = "Cerebral Dynamics, Plasticity & Learning"

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

# name -> role, applied to the file stem, case-insensitive, first match wins
_PATTERNS = [
    (r"^\s*cs\s*[-_ ]?only", "baseline_cs"),
    (r"^\s*us\s*[-_ ]?only", "baseline_us"),
    (r"extinction", "extinction"),
    (r"^\s*cs\s*[-_ ]?us", "conditioning"),
]


def _slug(stem):
    return re.sub(r"[^a-z0-9]+", "", stem.lower()) or "rec"


def _numeric_key(stem):
    """Sort CSUS 2 before CSUS 10, and 'extinction' before 'extinction 2'."""
    m = re.findall(r"\d+", stem)
    return (int(m[-1]) if m else 1, stem.lower())


def discover(video_dir, exts=(".mp4", ".mov", ".avi", ".mkv")):
    """Build a recording list from the file names in a folder."""
    found = []
    for fn in sorted(os.listdir(video_dir)):
        stem, ext = os.path.splitext(fn)
        if ext.lower() not in exts or stem.startswith("~"):
            continue
        role = next((r for pat, r in _PATTERNS if re.search(pat, stem, re.I)), None)
        if role is None:
            continue
        found.append({"tag": _slug(stem), "file": fn, "label": stem.strip(), "role": role,
                      "_key": _numeric_key(stem)})
    order = {r: 0 for r in ROLES}
    out = []
    for rec in sorted(found, key=lambda r: (ROLES.index(r["role"]), r["_key"])):
        rec.pop("_key")
        order[rec["role"]] += 1
        rec["order"] = order[rec["role"]]
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
    seen = set()
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
        if rec["tag"] in seen:
            raise SystemExit(f"duplicate tag {rec['tag']!r}")
        seen.add(rec["tag"])
        rec["path"] = os.path.join(cfg["video_dir"], rec["file"])
    return cfg


def by_role(cfg, *roles):
    order = {r: i for i, r in enumerate(ROLES)}
    return sorted([r for r in cfg["recordings"] if r["role"] in roles],
                  key=lambda r: (order[r["role"]], r.get("order", 0)))


def expected_trials(proto):
    per = proto["paired_per_block"] + proto["cs_only_per_block"]
    return per * proto["n_blocks"]
