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
VERSION = "1.1"
LAB = "Cerebral Dynamics, Plasticity & Learning"

# ------------------------------------------------------------------------ house style
# One palette and one face, read by the app page, by the figures and by the workbooks, so
# that a CR is the same blue in a spreadsheet chart, in a PNG and on the screen.
#
# Two hues, opposite each other on the wheel, each with one job.  210 degrees is measured
# off the glyph in the lab mark: it carries structure - headings, step numbers - and it
# carries the US and the response conditioned to it.  30 degrees is its complement, and it
# is spent on the CS and on anything you are meant to press or read before trusting a
# number.  Everything else is grey, and the reds and greens only ever mean a state.
PALETTE = {
    "brand":   "#3785D2",   # the glyph blue, unmodified
    "accent":  "#2E6BA8",   # the same hue, dark enough to read as text
    "action":  "#A85408",   # the complement at 30 deg: press this, read this
    "cs":      "#9B6530",   # CS - the yellow LED
    "us":      "#30669B",   # US - the blue LED
    "cr":      "#24466F",   # the conditioned response: the US hue, driven darker
    "ur":      "#A6332F",   # the reflex to the puff
    "ok":      "#33795A",
    "warn":    "#8A6100",
    "alert":   "#A6332F",
    "ink":     "#161B23",
    "muted":   "#59636F",
    "faint":   "#93A0AE",   # a trial with nothing in it
    "link":    "#B9C2CE",   # the thread joining trials in order
    "rule":    "#D7DDE6",
    "grid":    "#EDF0F4",
    "paper":   "#E9ECF1",
    "surface": "#FFFFFF",
    "sunken":  "#F2F5F9",
    # the same hues as a wash, for a band behind text or a spreadsheet colour scale
    "us_mid":  "#6D85AE",   # the US hue at half strength, for a colour scale
    "trace":   "#4A6076",   # one trial's closure trace, drawn among the others
    "cs_soft": "#F6E7C8",
    "us_soft": "#D6E2F0",
    "cr_soft": "#E4EBF4",
    "ur_soft": "#EFCFCD",
    "flag_soft": "#FBF1E6",  # a row somebody has to look at
}

# Segoe UI is on every machine this runs on, it has a large x-height so a table stays
# legible small, and its figures line up - which matters more here than a display face
# would.  The fallbacks are for a machine that somehow lacks it.
FONT = "Segoe UI"
FONT_STACK = ["Segoe UI", "Segoe UI Variable Text", "Tahoma", "DejaVu Sans", "Arial"]


def ink(name):
    """A palette colour as matplotlib and CSS want it."""
    return PALETTE[name]


def xl(name):
    """The same colour as openpyxl wants it - ARGB, opaque."""
    return "FF" + PALETTE[name].lstrip("#")


def mpl_font(plt):
    """Set the figure face once, for every figure this run draws.

    Passed the pyplot module rather than importing it, so this file stays importable by
    the app and the launcher, neither of which draws anything.
    """
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = FONT_STACK + list(plt.rcParams["font.sans-serif"])
    plt.rcParams["axes.titlecolor"] = PALETTE["ink"]
    plt.rcParams["text.color"] = PALETTE["ink"]
    plt.rcParams["axes.labelcolor"] = PALETTE["ink"]
    plt.rcParams["xtick.color"] = PALETTE["muted"]
    plt.rcParams["ytick.color"] = PALETTE["muted"]
    plt.rcParams["axes.edgecolor"] = PALETTE["rule"]

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

# The protocol the app opens on.  It is a *default*, not a law and not anyone's house
# standard: every number here can be set per study, the app remembers the last protocol
# actually run, and the pipeline reads them rather than assuming them.  In particular
# nothing below requires the US to co-terminate with the CS - put us_onset_ms at or
# beyond cs_ms and the study becomes a trace protocol, with the gap carried through
# pairing, scoring, the figures and the workbook read-mes.
DEFAULT_PROTOCOL = {
    "cs_ms": 400.0,          # CS duration (yellow LED)
    "us_onset_ms": 350.0,    # US onset, measured from CS onset
    "us_dur_ms": 50.0,       # US duration (blue LED)
    "paired_per_block": 9,
    "cs_only_per_block": 1,
    "n_blocks": 10,
    "min_iti_s": 5.0,        # two CS onsets closer than this cannot both be trials
    "cs_tol": 0.35,          # accept a CS whose measured duration is within +-35%
    "us_tol": 0.60,          # the US is short, so its measured duration is noisier
    "alpha_ms": 100.0,       # a blink sooner than this after CS onset is startle, not a CR
    "pre_ms": 300.0,         # trial window before the anchor
    "post_ms": 0.0,          # trial window after the anchor; 0 = derived from the design
}

# Protocol keys that are counts rather than durations, so a study file saying "9.0" is
# read as 9 and a fraction is refused rather than silently floored.
COUNT_KEYS = ("paired_per_block", "cs_only_per_block", "n_blocks")

# Ready-made protocols the app offers.  There are two, named for the two designs that
# exist - delay and trace - and nothing more.  A preset is only a set of numbers:
# choosing one is exactly the same as typing them, and nothing downstream knows which
# was used.  Neither is labelled the standard one, because which protocol is standard is
# a fact about a lab and not about this program; the numbers a study was actually run
# with are what get written beside its results.
PRESETS = [
    {"id": "delay", "name": "Delay",
     "note": "The US falls inside the CS and ends with it, so there is no gap to bridge. "
             "Nine paired trials then a CS-only probe, ten times over.",
     "protocol": {"cs_ms": 400.0, "us_onset_ms": 350.0, "us_dur_ms": 50.0,
                  "paired_per_block": 9, "cs_only_per_block": 1, "n_blocks": 10}},
    {"id": "trace", "name": "Trace",
     "note": "The CS ends, nothing happens for 500 ms, then the US. The gap is what the "
             "participant has to bridge. Nine paired trials then a CS-only probe, ten "
             "times over.",
     "protocol": {"cs_ms": 400.0, "us_onset_ms": 900.0, "us_dur_ms": 50.0,
                  "paired_per_block": 9, "cs_only_per_block": 1, "n_blocks": 10}},
]


def fill(proto):
    """A protocol read back from an earlier run, with any key it predates filled in.

    merged.json carries the protocol that produced it, and an older one was written
    before some of these keys existed.  Reading it through here means a workbook or a
    figure can still be rebuilt from a run made by an earlier version of this app.
    """
    out = dict(DEFAULT_PROTOCOL)
    out.update(proto or {})
    for k in COUNT_KEYS:
        try:
            out[k] = int(float(out[k]))
        except (TypeError, ValueError):
            out[k] = DEFAULT_PROTOCOL[k]
    return out


def design(proto):
    """What kind of conditioning these three numbers describe.

    Nothing in the pipeline branches on the *name*.  It is worked out in one place so
    that the page, the figures and the workbook read-mes all describe the same protocol
    in the same words, from the same numbers.
    """
    cs = float(proto["cs_ms"])
    on = float(proto["us_onset_ms"])
    dur = float(proto["us_dur_ms"])
    us_off = on + dur
    gap = on - cs                       # > 0: the CS is over before the US arrives
    cot = abs(us_off - cs) <= 0.5

    if gap > 0.5:
        kind, label = "trace", "Trace conditioning"
        sentence = ("The CS ends %.0f ms before the US begins, so the two never overlap. "
                    "That gap is the trace interval." % gap)
        short = "%.0f ms gap between CS offset and US onset" % gap
    elif cot:
        kind, label = "delay", "Delay conditioning, co-terminating"
        sentence = "The US falls inside the CS and the two end together, at %.0f ms." % cs
        short = "US co-terminates with the CS"
    elif us_off < cs:
        kind, label = "delay", "Delay conditioning"
        sentence = ("The US falls inside the CS and is over %.0f ms before the CS ends."
                    % (cs - us_off))
        short = "US ends %.0f ms before the CS does" % (cs - us_off)
    else:
        kind, label = "delay", "Delay conditioning, overlapping"
        sentence = ("The US begins inside the CS and outlasts it by %.0f ms."
                    % (us_off - cs))
        short = "US outlasts the CS by %.0f ms" % (us_off - cs)

    return dict(kind=kind, label=label, sentence=sentence, short=short,
                coterminate=bool(cot), trace_gap_ms=round(gap, 1),
                cs_offset_ms=round(cs, 1), us_offset_ms=round(us_off, 1),
                isi_ms=round(on, 1), span_ms=round(max(cs, us_off), 1))


def window(proto):
    """(pre, post, search) in ms: the trial window, and the span searched for a response.

    All three follow the protocol rather than a constant, because a trace protocol puts
    the US - and therefore the UR - much later in the window than a delay one does.  The
    standard delay numbers give back exactly the 300 / 1150 / 1000 ms this app has always
    used, so nothing already analysed moves.
    """
    end = design(proto)["span_ms"]
    pre = float(proto.get("pre_ms") or 0.0) or 300.0
    post = float(proto.get("post_ms") or 0.0) or max(1150.0, end + 450.0)
    search = min(post - 150.0, max(1000.0, end + 300.0))
    return pre, post, max(search, 200.0)


# How far below the mean unconditioned latency the reflex is taken to begin.  1.5 SD
# puts the edge at roughly the 7th percentile of a normal spread of UR onsets: early
# enough that almost no genuine reflex is called a CR, late enough that the window is not
# thrown open to twitches that started before any stimulus could have driven them.
REFLEX_K = 1.5
# Fewest US-only trials worth taking a mean and an SD from.  Below this the SD is not an
# estimate of anything and the protocol default is used instead.
REFLEX_MIN_N = 3
# A mean and an SD have no defence against one bad trial, and a US-only baseline can hold
# one: a spontaneous blink scored hundreds of ms after the reflex was missed.  Onsets more
# than this many robust SDs from the median of the baseline are set aside before the mean
# and the SD are taken - three is the usual convention, and the median and the MAD used to
# find them are themselves unmoved by an outlier.  It is only applied once there are
# enough onsets for a median to describe anything.
REFLEX_OUTLIER_SD = 3.0
REFLEX_OUTLIER_MIN_N = 5


def cr_window(proto, reflex=None):
    """Which blink latencies count as a conditioned response, and why those ones.

    The rule used to be arithmetic on the protocol: a blink between the startle cut-off
    and the US onset was a CR.  That treats both stimuli as if a blink could follow them
    instantaneously, which no reflex does - so it called a blink 20 ms after the puff a
    UR, when 20 ms is far too soon for the puff to have caused anything, and it called a
    blink 5 ms after the CS a CR at the other end.

    When the study includes a US-only baseline the reflex latency is *measured* there -
    the mean onset of the unconditioned blinks, minus REFLEX_K SDs - and both edges of
    the window move by it.  A blink cannot have been caused by the CS until that long
    after CS onset, and cannot have been caused by the US until that long after US onset,
    so the window is

        [CS onset + reflex,  US onset + reflex]

    which in trial time - zero is CS onset - is [reflex, us_onset_ms + reflex].

    With no US-only baseline there is nothing to measure, and the protocol's own startle
    cut-off is used with the window ending at the US: exactly what this app did before.
    `measured` says which of the two happened, and every sheet and figure that names the
    window prints `why` next to it.
    """
    p = fill(proto)
    us0 = float(p["us_onset_ms"])
    r = dict(reflex or {})
    ms = r.get("onset_ms")
    if ms is not None and 0.0 < float(ms) < us0:
        lo, hi = float(ms), us0 + float(ms)
        why = ("Measured from the US-only baseline: %d unconditioned blinks began "
               "%.0f +- %.0f ms after the puff, so mean - %.1f SD = %.0f ms is the "
               "soonest a stimulus can have caused a blink. Both edges of the CR window "
               "sit that far after their own stimulus: %.0f ms after CS onset, and "
               "%.0f ms after US onset."
               % (r.get("n", 0), r.get("mean_ms", 0.0), r.get("sd_ms", 0.0),
                  r.get("k", REFLEX_K), lo, lo, lo))
        measured = True
    else:
        lo, hi = float(p["alpha_ms"]), us0
        measured = False
        why = ("Not measured - %s. The protocol's startle cut-off (%.0f ms) is used "
               "instead and the window ends at the US onset, so a blink in the first "
               "%.0f ms after the puff is still counted as a reaction to it."
               % (r.get("why") or "this study has no US-only baseline",
                  lo, us0 - lo if us0 > lo else 0.0))
    out = dict(lo_ms=round(lo, 1), hi_ms=round(hi, 1), measured=measured, why=why,
               reflex_ms=round(lo, 1) if measured else None,
               us_onset_ms=round(us0, 1), reflex=r or None)
    # The class names are built here, once, from these two numbers.  Every module that
    # has to recognise a class string reads them from here rather than spelling them out,
    # so a window that moves can never leave a sheet matching on a label that no longer
    # exists.
    out["cr_label"] = "CR (%d-%dms)" % (round(lo), round(hi))
    out["alpha_label"] = "alpha/startle <%dms" % round(lo)
    out["ur_label"] = "UR (>=%dms)" % round(hi)
    # A US-only recording has no CS, so its trials are timed from the puff and carry
    # their own two labels.
    out["ur_puff_label"] = "UR to the puff"
    out["alpha_us_label"] = "alpha/startle <20ms"
    out["moving_label"] = "in-progress at stimulus"
    return out


def pair_window_s(proto):
    """How long after a CS onset a US may fall and still belong to that CS.

    The end of the stimulus pair plus a little slack, capped at half the minimum ITI so
    a pulse can never be claimed by the trial before the one it belongs to.  For a trace
    protocol this window is long - that is the point - but it is still far shorter than
    the gap to the next trial.
    """
    hi = (design(proto)["span_ms"] + 120.0) / 1000.0
    return min(hi, 0.5 * float(proto["min_iti_s"]))


def check_protocol(proto):
    """Problems that would make these numbers impossible to analyse, in plain words.

    A protocol that is merely unusual - a trace interval, a block with no probe, twenty
    blocks instead of ten - is not a problem and is not listed here.  Only arithmetic
    that cannot describe a real session is.
    """
    bad = []
    for k in ("cs_ms", "us_onset_ms", "us_dur_ms", "min_iti_s"):
        if float(proto[k]) <= 0:
            bad.append("%s must be greater than zero." % k)
    for k in COUNT_KEYS:
        v = proto[k]
        if float(v) != int(float(v)) or int(float(v)) < 0:
            bad.append("%s must be a whole number that is not negative." % k)
    if bad:
        return bad
    d = design(proto)
    if int(proto["paired_per_block"]) + int(proto["cs_only_per_block"]) <= 0:
        bad.append("A block needs at least one trial: paired and CS-only cannot both be 0.")
    if int(proto["n_blocks"]) < 1:
        bad.append("A conditioning session needs at least one block.")
    if d["span_ms"] > float(proto["min_iti_s"]) * 1000.0 * 0.5:
        bad.append("One trial spans %.0f ms but trials are only %g s apart, so a US could "
                   "belong to either of two CS presentations. Raise the minimum ITI, or "
                   "shorten the trial." % (d["span_ms"], float(proto["min_iti_s"])))
    if not 0 < float(proto["cs_tol"]) < 1:
        bad.append("CS tolerance is a fraction between 0 and 1 (0.35 means +-35%).")
    if not 0 < float(proto["us_tol"]) < 1:
        bad.append("US tolerance is a fraction between 0 and 1 (0.60 means +-60%).")
    if float(proto["alpha_ms"]) >= float(proto["us_onset_ms"]):
        bad.append("The startle cut-off (%.0f ms) is at or after the US (%.0f ms), which "
                   "leaves no window in which a CR could be counted."
                   % (float(proto["alpha_ms"]), float(proto["us_onset_ms"])))
    _, post, _ = window(proto)
    if post < d["span_ms"] + 100.0:
        bad.append("The trial window ends %.0f ms after CS onset, before the stimuli are "
                   "over at %.0f ms." % (post, d["span_ms"]))
    return bad

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
    for k in COUNT_KEYS:
        try:
            if float(proto[k]) == int(float(proto[k])):
                proto[k] = int(float(proto[k]))
        except (TypeError, ValueError):
            raise SystemExit("protocol: %s must be a whole number, not %r" % (k, proto[k]))
    bad = check_protocol(proto)
    if bad:
        raise SystemExit("this protocol cannot be analysed:\n  - " + "\n  - ".join(bad))
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
    """How many conditioning trials this protocol says a session contains.

    Used only as the number the recovered sequence is *compared against*.  No trial is
    ever created, renumbered or dropped to make the two agree.
    """
    per = int(proto["paired_per_block"]) + int(proto["cs_only_per_block"])
    return per * int(proto["n_blocks"])
