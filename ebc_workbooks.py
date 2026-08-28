"""Build the Excel workbooks - one per group of trials that stands on its own.

    python ebc_workbooks.py <config.json>

Each workbook carries its own read-me: the protocol, how the numbers were produced and
what not to over-read, so a sheet that leaves this folder still explains itself.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import json, numpy as np
from openpyxl import Workbook
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter
from openpyxl.chart import ScatterChart, LineChart, Reference, Series
from openpyxl.chart.marker import Marker
from openpyxl.drawing.line import LineProperties
from openpyxl.chart.shapes import GraphicalProperties
from openpyxl.formatting.rule import ColorScaleRule
from openpyxl.drawing.image import Image as XLImage

import ebc_config as C
from ebc_paths import work_dir, out_dir

CFG = C.load(sys.argv[1] if len(sys.argv) > 1 else None)
WORK, OUT = work_dir(CFG), out_dir(CFG)
M = json.load(open(os.path.join(WORK, "merged.json"), encoding="utf-8"))
ALLROWS = json.load(open(os.path.join(WORK, "merged_rows.json"), encoding="utf-8"))
NOM = M["protocol"]
CHECKS = M.get("checks", {})
TAG_ORDER = [r["tag"] for r in CFG["recordings"]]
SESSMETA = {s["tag"]: s for s in M["sessions"]}
REC_FILE = {r["tag"]: r["file"] for r in CFG["recordings"]}

# the sheet code below speaks in these older names; keep them as the bridge
for _r in ALLROWS:
    _r["block_kind"] = _r["role"]
    _r["gidx"] = _r["group_index"]
    _r["cs_onset_block_s"] = _r.get("session_clock_s")
    _r["cs_duration_measured_ms"] = _r.get("cs_duration_ms")

CR_LBL = "CR (100-%.0fms)" % NOM["us_onset_ms"]
UR_LBL = "UR (>=%.0fms)" % NOM["us_onset_ms"]
ALPHA_LBL = "alpha/startle <100ms"
MOVING_LBL = "in-progress at stimulus"


def is_cr(r):
    return str(r["scored_class"]).startswith("CR")


def is_ur(r):
    return str(r["scored_class"]).startswith("UR")


def is_alpha(r):
    return str(r["scored_class"]).startswith("alpha")


def stim_events(tag):
    """Every pulse read from the LEDs of one recording, accepted or not."""
    f = os.path.join(WORK, tag + "_stim.json")
    if not os.path.exists(f):
        return []
    S = json.load(open(f, encoding="utf-8"))
    out = []
    for key, nm in (("yellow", "CS (yellow LED)"), ("blue", "US (blue LED)")):
        for e in S["events"].get(key, []):
            out.append((e["t"], nm, e))
    out.sort(key=lambda z: z[0])
    return out


INK, MUT, HDR = "FF141922", "FF59636F", "FF1F2937"
thin = Side(style="thin", color="FFDCE1E9")
OUTDIR = OUT

PROTOCOL = [
    ("Design", "Delay eyeblink conditioning. A block is %d paired CS-US trials followed by %d CS-only "
               "trial(s); %d blocks of conditioning, then CS-only trials as extinction."
               % (NOM["paired_per_block"], NOM["cs_only_per_block"], NOM["n_blocks"])),
    ("Stimuli", f"CS = yellow LED, {NOM['cs_ms']:.0f} ms. US = blue LED, {NOM['us_dur_ms']:.0f} ms. "
                f"They overlap and co-terminate, so US onset is at {NOM['us_onset_ms']:.0f} ms after CS onset."),
    ("Structure recovered", "The trial structure was recovered from the video, not imposed on it: the "
                            "LEDs were read frame by frame and the sequence that came out was then "
                            "compared with the protocol. Recovered %d paired CS-US and %d CS-only trials "
                            "against %d and %d expected; the run of paired trials before each CS-only "
                            "probe was %s. Strict %d+%d x %d structure: %s."
                            % (CHECKS.get("found_paired", 0), CHECKS.get("found_cs_only", 0),
                               CHECKS.get("expected_paired", 0), CHECKS.get("expected_cs_only", 0),
                               CHECKS.get("paired_runs_before_each_probe", []),
                               NOM["paired_per_block"], NOM["cs_only_per_block"], NOM["n_blocks"],
                               "yes" if CHECKS.get("strict_block_structure") else "NO - see the report")),
]
METHOD = [
    ("Stimulus detection", "Both LEDs were recovered from the pixels, never assumed. Each is detected as a "
                           "transient against a running background, so a static coloured object cannot "
                           "trigger it, and the two are detected independently of each other."),
    ("Detection filter", "A genuine CS lasts about %.0f ms. Pulses outside +-%.0f%% of that, or closer "
                         "than %g s to the previous accepted one, are LED flicker rather than stimuli and "
                         "are rejected. Blue pulses that do not match the US duration go the same way. "
                         "Every rejected pulse is listed in stimulus_events.csv, so nothing is dropped "
                         "silently." % (NOM["cs_ms"], NOM["cs_tol"] * 100, NOM["min_iti_s"])),
    ("Trial alignment", "Every trial window is cut with the LED and the face inside the same crop, so the CS "
                        "onset is re-detected inside each window. The alignment error measured for each trial is a column in the trial tables."),
    ("Eyelid measure", "MediaPipe FaceMesh, 478 landmarks with iris refinement, on a 2x-upscaled face crop. "
                       "Eye aspect ratio per eye, averaged. EAR is normalised by eye width, so it is robust "
                       "to head movement and camera distance."),
    ("Closure scale", "0% = a blink-robust open-eye reference (85th percentile of EAR in that window). "
                      "100% = a full-closure reference pooled over every trial of every recording, so "
                      "all blocks sit on one comparable scale. Smoothed with a 5-frame Savitzky-Golay "
                      "filter (42 ms)."),
    ("Blink criterion", "Five robust SDs above the trial's own pre-CS baseline, floor 15% closure, walked "
                        "back along the falling edge to the true onset. A separate blink must re-reach 40% "
                        "closure after first returning below 20%."),
    ("Second look after an artefact", "If the first event in the window is an alpha blink (<100 ms) or the "
                                      "lid was already moving at CS onset, the window is searched for a "
                                      "LATER blink, because a real CR or UR may sit behind the artefact. "
                                      "Where one is found it becomes the scored response; 'First response "
                                      "obscured' marks those trials and the original first blink is kept in "
                                      "its own columns."),
    ("Response classes", f"alpha / startle = onset under 100 ms. CR = onset 100-{NOM['us_onset_ms']:.0f} ms, "
                         f"i.e. the blink began before the US. UR only = onset at or after "
                         f"{NOM['us_onset_ms']:.0f} ms. Trials where the lid was moving at CS onset and no "
                         f"later blink was found cannot be timed and are excluded from the summary."),
]
CAVEATS = [
    ("Measure", "Landmark EAR is a good proxy for lid aperture but it is not EMG or a magnetic search coil. "
                "Treat the timings as the reliable quantity and absolute closure percentages as relative."),
    ("Which onset to use", "'Scored onset' is the column to analyse - it already applies the second-look "
                           "rule. 'Blink onset' is the raw first event, kept for transparency."),
    ("Quality flags", "Flagged trials are kept in the table so nothing is silently dropped; filter on "
                      "'Quality flag' = clean if you want the strictest subset."),
]

STUDY = CFG["study"]
ROLE_TITLE = {"conditioning": "delay eyeblink conditioning",
              "extinction": "extinction / post-conditioning test",
              "baseline_cs": "baseline - CS alone",
              "baseline_us": "baseline - US alone"}
ROLE_FIGS = {"conditioning": [("cond_acquisition.png", "Acquisition by block - conditioned responses "
                                                       "replace reactions to the puff"),
                              ("cond_paired_onset_scatter.png", "Blink onset per paired trial, joined in "
                                                                "order, with the block-mean learning curve"),
                              ("cond_csonly_onset_scatter.png", "The CS-only probe ending each block"),
                              ("cond_paired_overview.png", "Eyelid closure per recording - raster and "
                                                           "overlaid traces")],
             "extinction": [("ext_onset_scatter.png", "Blink onset per CS-only trial, against the "
                                                      "learned US window"),
                            ("ext_overview.png", "Eyelid closure - raster and overlaid traces")],
             "baseline_cs": [("baseline_cs_onset_scatter.png", "Blink onset per CS-only trial"),
                             ("baseline_cs_overview.png", "Eyelid closure - raster and overlaid traces")],
             "baseline_us": [("baseline_us_onset_scatter.png", "Blink latency per US-only trial"),
                             ("baseline_us_overview.png", "Eyelid closure - raster and overlaid traces")]}


def _span(tags):
    labels = [SESSMETA[t]["label"] for t in tags if t in SESSMETA]
    mins = sum(SESSMETA[t]["duration_s"] for t in tags if t in SESSMETA) / 60.0
    return labels, mins


def _book(role):
    tags = [t for t in TAG_ORDER if t in SESSMETA and SESSMETA[t]["role"] == role]
    if not tags or not any(r["role"] == role for r in ALLROWS):
        return None
    labels, mins = _span(tags)
    npair = sum(1 for r in ALLROWS if r["role"] == role and r["trial_type"] == "CS-US")
    nrest = sum(1 for r in ALLROWS if r["role"] == role and r["trial_type"] != "CS-US")
    others = [ROLE_TITLE[o] for o in C.ROLES
              if o != role and any(r["role"] == o for r in ALLROWS)]
    off = M.get("offsets", {})
    timeline = "  ".join("%s = %.0f-%.0f s." % (SESSMETA[t]["label"], off.get(t, 0),
                                                off.get(t, 0) + SESSMETA[t]["duration_s"])
                         for t in tags if t in off)
    what = [("What this covers",
             "%s: %d recording(s) (%s), %.1f min in total. %d paired CS-US trials and %d "
             "CS-only / US-only trials." % (ROLE_TITLE[role], len(labels), ", ".join(labels),
                                            mins, npair, nrest))]
    if role == "conditioning":
        what.append(("CS-only trials are separate",
                     "The CS-only probes are scored but held out of the recording summary, the block "
                     "summary and the main scatter - a trial with no US is a different measurement. "
                     "They have their own sheet and their own figure."))
        what.append(("Protocol check",
                     "%d paired and %d CS-only recovered against %d and %d expected. Run of paired "
                     "trials before each probe: %s. Strict structure: %s."
                     % (CHECKS.get("found_paired", 0), CHECKS.get("found_cs_only", 0),
                        CHECKS.get("expected_paired", 0), CHECKS.get("expected_cs_only", 0),
                        CHECKS.get("paired_runs_before_each_probe", []),
                        "yes" if CHECKS.get("strict_block_structure") else "NO")))
    if role == "extinction":
        what.append(("Why it is separate",
                     "No CS is paired with a US here, so this is extinction, not more conditioning. "
                     "Pooling it with the conditioning trials would mix two measurements."))
        what.append(("Reading the US column",
                     "No US was delivered. 'Closure at US' is sampled at %.0f ms - the interval "
                     "learned during conditioning - and the CR window is that learned interval "
                     "applied as a probe." % NOM["us_onset_ms"]))
    if role == "baseline_us":
        what.append(("What the timings mean",
                     "There is no CS, so every window is anchored on the US and the latencies are "
                     "measured from the puff. These are unconditioned responses by definition and "
                     "give the reflex baseline the CRs are read against."))
    if role == "baseline_cs":
        what.append(("What the timings mean",
                     "The CS alone, before any pairing. Blinks here are orienting or spontaneous, "
                     "not conditioned, and give the false-positive rate for the CR window."))
    what.append(("Timeline", (timeline + " 'CS onset on session clock' is the position on that "
                              "continuous clock.") if timeline else "Single recording."))
    if others:
        what.append(("Companion workbooks", "This run also produced: " + ", ".join(others) + "."))
    return dict(file="EBC_%s_%s.xlsx" % (STUDY, role),
                title="%s - %s - %s" % (STUDY, ROLE_TITLE[role], ", ".join(labels)),
                sel=(lambda rl: (lambda r: r["role"] == rl))(role),
                figs=ROLE_FIGS.get(role, []),
                what=what)


BOOKS = dict((role, b) for role, b in
             ((r, _book(r)) for r in C.ROLES) if b)


COLS = [("#", "gidx", "0", 6), ("Block", "block", "0", 7), ("Trial in block", "trial_in_block", "0", 9),
        ("Session", "session_name", None, 10), ("Trial in session", "session_trial", "0", 9),
        ("Type", "trial_type", None, 10),
        ("CS onset in video (s)", "cs_onset_video_s", "0.000", 14),
        ("CS onset on session clock (s)", "cs_onset_block_s", "0.000", 14),
        ("Alignment error (ms)", "alignment_error_ms", "0.00", 11),
        ("Face tracked (%)", "face_tracked_pct", "0.0", 11),
        ("CS duration measured (ms)", "cs_duration_measured_ms", "0.0", 13),
        ("CS timing source", "cs_timing", None, 20),
        ("Block boundary from", "block_closed_by", None, 16),
        ("Quality flag", "quality", None, 24),
        ("Blinks in 1 s window", "n_full_blinks", "0", 10),
        ("SCORED onset (ms)", "scored_onset_ms", "0.0", 13),
        ("SCORED class", "scored_class", None, 19),
        ("First response obscured", "first_response_obscured", None, 12),
        ("Raw first blink onset (ms)", "blink_onset_ms", "0.0", 13),
        ("Raw first blink class", "response_class", None, 19),
        ("Later blink onset (ms)", "secondary_onset_ms", "0.0", 13),
        ("Later blink peak (%)", "secondary_peak_pct", "0.0", 12),
        ("Later blink class", "secondary_class", None, 19),
        ("Peak closure time (ms)", "peak_closure_ms", "0.0", 13),
        ("Peak closure (%)", "peak_closure_pct", "0.0", 11),
        ("Closing speed (%/ms)", "closing_speed_pct_per_ms", "0.00", 12),
        ("Closure duration (ms)", "closure_duration_ms", "0.0", 13),
        ("Closure at US %.0f ms (%%)" % NOM["us_onset_ms"], "closure_at_US_pct", "0.0", 13),
        ("Closure at CS offset %.0f ms (%%)" % NOM["cs_ms"], "closure_at_CSoff_pct", "0.0", 14),
        ("Half-reopened (ms)", "reopen_half_ms", "0.0", 12),
        ("Fully reopened (ms)", "reopen_full_ms", "0.0", 12),
        ("Closure at 1000 ms (%)", "closure_at_1000ms_pct", "0.0", 13),
        ("Closed at US", "closed_at_US", None, 10),
        ("Reopened before US", "reopened_before_US", None, 11),
        ("All blink onsets (ms)", "all_blink_onsets_ms", None, 16),
        ("All blink amplitudes (%)", "all_blink_amps_pct", None, 16),
        ("Inter-blink interval (ms)", "inter_blink_ms", None, 14),
        ("Partial lid movements", "partial_movement_ms", None, 20)]


def style_header(ws, row=1):
    for c in range(1, ws.max_column + 1):
        cell = ws.cell(row=row, column=c)
        cell.font = Font(bold=True, size=10, color="FFFFFFFF")
        cell.fill = PatternFill("solid", fgColor=HDR)
        cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    ws.row_dimensions[row].height = 36


def widths(ws, w):
    for i, x in enumerate(w, 1):
        ws.column_dimensions[get_column_letter(i)].width = x


def scoreable(rs):
    return [r for r in rs if r["scored_class"] not in (None, MOVING_LBL)]


def write_table(ws, rows):
    for j, c in enumerate(COLS, 1):
        ws.cell(row=1, column=j, value=c[0])
    for i, src in enumerate(rows, 2):
        flagged = src["quality"] != "clean"
        for j, (hh, key, fmt, w) in enumerate(COLS, 1):
            v = src.get(key)
            if isinstance(v, bool):
                v = "yes" if v else "no"
            c = ws.cell(row=i, column=j, value=v)
            c.font = Font(size=10, color=MUT, italic=True) if flagged else Font(size=10, color=INK)
            c.alignment = Alignment(horizontal="right" if fmt else "left")
            if fmt:
                c.number_format = fmt
            c.border = Border(bottom=thin)
            if key in ("scored_onset_ms", "scored_class"):
                c.font = Font(size=10, bold=True, color=INK, italic=flagged)
    style_header(ws)
    widths(ws, [c[3] for c in COLS])
    ws.freeze_panes = "G2"
    n = len(rows) + 1
    if not rows:
        ws.cell(row=2, column=1, value="No trials of this kind in this recording group.")
        ws.cell(row=2, column=1).font = Font(size=10, color=MUT, italic=True)
        return
    ws.auto_filter.ref = "A1:" + get_column_letter(len(COLS)) + str(n)
    ws.conditional_formatting.add("L2:L" + str(n), ColorScaleRule(
        start_type="num", start_value=0, start_color="FFF6E7C8",
        mid_type="num", mid_value=175, mid_color="FFBFD0EC",
        end_type="num", end_value=400, end_color="FFE9BDB9"))
    for rng in ("U2:U", "X2:X", "Y2:Y", "AB2:AB"):
        ws.conditional_formatting.add(rng + str(n), ColorScaleRule(
            start_type="num", start_value=0, start_color="FFFFFFFF",
            end_type="num", end_value=100, end_color="FF6D85AE"))
    ws.sheet_view.showGridLines = False


def scatter_sheet(ws, rows, has_us, xlabel, title):
    lbl_us = ("US onset = %.0f ms" % NOM["us_onset_ms"]) if has_us else (
        "learned US onset = %.0f ms (none delivered)" % NOM["us_onset_ms"])
    hd = ["#", "Block", "Session", CR_LBL, ALPHA_LBL, UR_LBL,
          "lid moving at onset", "CS onset = 0 ms", lbl_us,
          "CS / US offset = %.0f ms" % NOM["cs_ms"], "Block mean onset"]
    for j, x in enumerate(hd, 1):
        ws.cell(row=1, column=j, value=x)
    colmap = {CR_LBL: 4, ALPHA_LBL: 5, UR_LBL: 6, MOVING_LBL: 7}
    bmean = {}
    for b in sorted({r["block"] for r in rows if r["block"]}):
        g = [r["scored_onset_ms"] for r in rows if r["block"] == b and r["scored_onset_ms"] is not None
             and r["scored_class"] != MOVING_LBL]
        if g:
            bmean[b] = float(np.mean(g))
    for i, r_ in enumerate(rows, 2):
        ws.cell(row=i, column=1, value=r_["gidx"])
        ws.cell(row=i, column=2, value=r_["block"])
        ws.cell(row=i, column=3, value=r_["session_name"])
        if r_["scored_onset_ms"] is not None and r_["scored_class"]:
            ws.cell(row=i, column=colmap[r_["scored_class"]], value=r_["scored_onset_ms"]).number_format = "0.0"
        ws.cell(row=i, column=8, value=0)
        ws.cell(row=i, column=9, value=NOM["us_onset_ms"])
        ws.cell(row=i, column=10, value=NOM["cs_ms"])
        if r_["block"] in bmean:
            ws.cell(row=i, column=11, value=round(bmean[r_["block"]], 1)).number_format = "0.0"
    style_header(ws)
    widths(ws, [6, 7, 10, 19, 21, 19, 17, 15, 20, 19, 15])
    ws.sheet_view.showGridLines = False
    ch = ScatterChart()
    ch.style = 2
    ch.title = title
    ch.x_axis.title = xlabel
    ch.y_axis.title = "Blink onset (ms from yellow LED / CS onset)"
    ch.height, ch.width = 13, 30 if len(rows) > 14 else 20
    ch.x_axis.scaling.min, ch.x_axis.scaling.max = 0, len(rows) + 1
    ch.y_axis.scaling.min, ch.y_axis.scaling.max = -120, 520
    xref = Reference(ws, min_col=1, min_row=2, max_row=len(rows) + 1)
    for k, colr, size in ((4, "FF2C4C86", 7), (5, "FFB8760F", 7), (6, "FFB03A32", 7), (7, "FF93A0AE", 6)):
        s = Series(Reference(ws, min_col=k, min_row=1, max_row=len(rows) + 1), xref, title_from_data=True)
        s.marker = Marker(symbol="circle", size=size)
        s.marker.graphicalProperties = GraphicalProperties(solidFill=colr)
        s.marker.graphicalProperties.line = LineProperties(solidFill=colr)
        s.graphicalProperties.line.noFill = True
        ch.series.append(s)
    for k, colr, dash, wdt in ((8, "FFB8760F", "solid", 20000), (9, "FF3A67CF", "dash", 20000),
                              (10, "FF3A67CF", "dash", 20000), (11, "FF141922", "solid", 28000)):
        s = Series(Reference(ws, min_col=k, min_row=1, max_row=len(rows) + 1), xref, title_from_data=True)
        s.marker = Marker(symbol="none")
        s.graphicalProperties.line = LineProperties(solidFill=colr, w=wdt, prstDash=dash)
        ch.series.append(s)
    ws.add_chart(ch, "M2")


def build(block):
    cfg = BOOKS[block]
    rows = sorted([r for r in ALLROWS if cfg["sel"](r)], key=lambda r: (r["trial_type"] != "CS-US", r["gidx"]))
    paired = [r for r in rows if r["trial_type"] == "CS-US"]
    csonly = [r for r in rows if r["trial_type"] == "CS-only"]
    main = paired if paired else csonly
    sess = sorted({r["session"] for r in rows}, key=lambda t: TAG_ORDER.index(t))
    wb = Workbook()

    def blockhdr(ws, r, title):
        c = ws.cell(row=r, column=1, value=title)
        c.font = Font(bold=True, size=10, color="FFFFFFFF")
        c.fill = PatternFill("solid", fgColor=HDR)
        ws.merge_cells(start_row=r, start_column=1, end_row=r, end_column=2)

    def kv(ws, r, k, v, h=46):
        ws.cell(row=r, column=1, value=k).font = Font(size=10, color=MUT)
        c = ws.cell(row=r, column=2, value=v)
        c.font = Font(size=10, bold=True, color=INK)
        c.alignment = Alignment(horizontal="left", wrap_text=True)
        ws.row_dimensions[r].height = h

    # ---------------- Read me ----------------
    ws = wb.active
    ws.title = "Read me"
    ws["A1"] = cfg["title"]
    ws["A1"].font = Font(bold=True, size=14, color=INK)
    ws.merge_cells("A1:B1")
    r = 3
    for hdr_, items in (("PROTOCOL", PROTOCOL), ("WHAT THIS WORKBOOK COVERS", cfg["what"]),
                        ("HOW THE NUMBERS WERE PRODUCED", METHOD), ("CAVEATS", CAVEATS)):
        blockhdr(ws, r, hdr_); r += 1
        for k, v in items:
            kv(ws, r, k, v); r += 1
        r += 1
    blockhdr(ws, r, "SHEETS"); r += 1
    sheets = [("Session summary", "One row per recording, paired CS-US trials only."),
              ("Block summary", "One row per block of 9 paired trials - the learning curve in numbers.")] if paired else []
    sheets += ([("Paired trials", "One row per paired CS-US trial."),
                ("CS-only trials", "The CS-only probes, scored and charted separately.")] if paired
               else [("CS-only trials", "One row per CS-only trial.")])
    sheets += [
               ("Stimulus events", "Every accepted CS and US event with frame and time."),
               ("Onset scatter", "Data behind the scatter, with a live Excel chart."),
               ("Closure traces", "Full eyelid traces, time x trial, with a live chart."),
               ("Figures", "Rendered PNG figures.")]
    for k, v in sheets:
        kv(ws, r, k, v, 20); r += 1
    widths(ws, [30, 112])
    ws.sheet_view.showGridLines = False

    # ---------------- Session summary (paired only) ----------------
    if paired:
        ws = wb.create_sheet("Session summary")
        hd = ["Session", "Video file", "Duration (s)", "Paired CS-US trials", "Scoreable", "CR n",
              "CR % of scoreable", "alpha <100 ms", "UR only", "no later blink", "Mean CR onset (ms)",
              "SD (ms)", "Median CR onset (ms)", "Mean peak closure (%)", "Mean closure at US (%)",
              "Recovered behind artefact"]
        for j, x in enumerate(hd, 1):
            ws.cell(row=1, column=j, value=x)
        ri = 2
        items = [(t, REC_FILE[t], [r for r in paired if r["session"] == t]) for t in sess]
        items.append(("ALL", " + ".join(REC_FILE[t] for t in sess), paired))
        for tag, vid, rs in items:
            if not rs:
                continue
            sc = scoreable(rs)
            cr = [r for r in sc if is_cr(r)]
            o = [r["scored_onset_ms"] for r in cr]
            name = "All pooled" if tag == "ALL" else rs[0]["session_name"]
            vals = [name, vid, round(SESSMETA[sess[0]]["duration_s"], 1) if tag != "ALL"
                    else round(sum(SESSMETA[t]["duration_s"] for t in sess), 1),
                    len(rs), len(sc), len(cr), round(len(cr) / len(sc) * 100, 1) if sc else None,
                    sum(1 for r in sc if is_alpha(r)),
                    sum(1 for r in sc if is_ur(r)),
                    len(rs) - len(sc),
                    round(float(np.mean(o)), 1) if o else None,
                    round(float(np.std(o, ddof=1)), 1) if len(o) > 1 else None,
                    round(float(np.median(o)), 1) if o else None,
                    round(float(np.mean([r["peak_closure_pct"] for r in sc if r["peak_closure_pct"]])), 1) if sc else None,
                    round(float(np.mean([r["closure_at_US_pct"] for r in sc])), 1) if sc else None,
                    sum(1 for r in rs if r["first_response_obscured"] == "yes" and r["secondary_onset_ms"])]
            if tag != "ALL":
                vals[2] = round(SESSMETA[tag]["duration_s"], 1)
            for j, v in enumerate(vals, 1):
                c = ws.cell(row=ri, column=j, value=v)
                c.font = Font(size=10, bold=(tag == "ALL"), color=INK)
                c.border = Border(bottom=thin,
                                  top=Side(style="medium", color="FF1F2937") if tag == "ALL" else None)
                c.alignment = Alignment(horizontal="left" if j <= 2 else "right")
            ri += 1
        style_header(ws)
        widths(ws, [13, 26, 11, 12, 10, 8, 12, 11, 9, 11, 13, 9, 13, 13, 14, 13])
        ws.freeze_panes = "C2"
        ws.sheet_view.showGridLines = False

        # ---------------- Block summary ----------------
        ws = wb.create_sheet("Block summary")
        hd = ["Block", "Paired trials", "Scoreable", "CR n", "CR %", "UR only n", "UR only %",
              "alpha n", "Mean scored onset (ms)", "SD (ms)", "Mean CR onset (ms)",
              "Mean peak closure (%)", "Mean closure at US (%)"]
        for j, x in enumerate(hd, 1):
            ws.cell(row=1, column=j, value=x)
        for bi, b in enumerate(sorted({r["block"] for r in paired}), 2):
            g = [r for r in paired if r["block"] == b]
            sc = scoreable(g)
            cr = [r for r in sc if is_cr(r)]
            allo = [r["scored_onset_ms"] for r in sc]
            o = [r["scored_onset_ms"] for r in cr]
            vals = [b, len(g), len(sc), len(cr), round(len(cr) / len(sc) * 100, 1) if sc else None,
                    sum(1 for r in sc if is_ur(r)),
                    round(100 * sum(1 for r in sc if is_ur(r)) / len(sc), 1) if sc else None,
                    sum(1 for r in sc if is_alpha(r)),
                    round(float(np.mean(allo)), 1) if allo else None,
                    round(float(np.std(allo, ddof=1)), 1) if len(allo) > 1 else None,
                    round(float(np.mean(o)), 1) if o else None,
                    round(float(np.mean([r["peak_closure_pct"] for r in sc if r["peak_closure_pct"]])), 1) if sc else None,
                    round(float(np.mean([r["closure_at_US_pct"] for r in sc])), 1) if sc else None]
            for j, v in enumerate(vals, 1):
                c = ws.cell(row=bi, column=j, value=v)
                c.font = Font(size=10, color=INK)
                c.border = Border(bottom=thin)
                c.alignment = Alignment(horizontal="right")
        style_header(ws)
        widths(ws, [8, 11, 10, 8, 9, 10, 10, 8, 15, 9, 15, 14, 15])
        ws.sheet_view.showGridLines = False
        n = len(set(r["block"] for r in paired)) + 1
        ws.conditional_formatting.add("E2:E" + str(n), ColorScaleRule(
            start_type="num", start_value=0, start_color="FFFFFFFF",
            end_type="num", end_value=100, end_color="FF6D85AE"))
        lc = LineChart()
        lc.title = "Learning across blocks"
        lc.x_axis.title = "Block"
        lc.y_axis.title = "CR % / mean onset (ms)"
        lc.height, lc.width = 10, 20
        lc.add_data(Reference(ws, min_col=5, min_row=1, max_row=n), titles_from_data=True)
        lc.add_data(Reference(ws, min_col=11, min_row=1, max_row=n), titles_from_data=True)
        lc.set_categories(Reference(ws, min_col=1, min_row=2, max_row=n))
        ws.add_chart(lc, "O2")

    # ---------------- trial tables ----------------
    write_table(wb.create_sheet("Paired trials" if paired else "CS-only trials"), main)
    if paired and csonly:
        write_table(wb.create_sheet("CS-only trials"), csonly)

    # ---------------- Stimulus events ----------------
    ws = wb.create_sheet("Stimulus events")
    hd = ["Recording", "Event #", "Stimulus", "Onset frame", "Onset (s)", "Duration (ms)",
          "Accepted", "Rejected because"]
    for j, x in enumerate(hd, 1):
        ws.cell(row=1, column=j, value=x)
    ri = 2
    for tag in sess:
        name = SESSMETA[tag]["label"]
        for i, (t, kind, e) in enumerate(stim_events(tag), 1):
            vals = [name, i, kind, e["frame"], round(e["t"], 4), e["dur_ms"],
                    "yes" if e["ok"] else "no",
                    e.get("reason", "") or ("" if e["ok"] else "duration off-spec")]
            for j, v in enumerate(vals, 1):
                cc = ws.cell(row=ri, column=j, value=v)
                cc.font = Font(size=10, color=INK if e["ok"] else MUT,
                               italic=not e["ok"])
                cc.border = Border(bottom=thin)
                cc.alignment = Alignment(horizontal="left" if j in (1, 3, 7, 8) else "right")
                if j == 5:
                    cc.number_format = "0.0000"
                if j == 6:
                    cc.number_format = "0.0"
            ri += 1
    style_header(ws)
    widths(ws, [22, 9, 17, 12, 12, 13, 10, 26])
    ws.freeze_panes = "C2"
    ws.sheet_view.showGridLines = False

    # ---------------- Onset scatter ----------------
    scatter_sheet(wb.create_sheet("Onset scatter"), main, bool(paired),
                  "Paired CS-US trial, in order" if paired else "CS-only trial",
                  "Blink onset per trial, relative to CS and US")
    if paired and csonly:
        scatter_sheet(wb.create_sheet("CS-only scatter"), csonly, False,
                      "CS-only probe (one per block)", "CS-only probes - blink onset")

    # ---------------- Closure traces ----------------
    ws = wb.create_sheet("Closure traces")
    t = None
    series = []
    for r in rows:
        TRC = M["traces"][r["session"]][str(r["session_trial"])]
        if t is None:
            t = TRC["t"]
        series.append((f"{r['session_name']} T{r['session_trial']} ({r['trial_type']})", TRC["C"]))
    ws.cell(row=1, column=1, value="Time from CS onset (ms)")
    for j, (nm, _) in enumerate(series, 2):
        ws.cell(row=1, column=j, value=nm)
    for i, tv in enumerate(t, 2):
        ws.cell(row=i, column=1, value=round(tv, 2)).number_format = "0.00"
        for j, (_, C) in enumerate(series, 2):
            if i - 2 < len(C):
                ws.cell(row=i, column=j, value=round(C[i - 2] * 100, 2)).number_format = "0.0"
    style_header(ws)
    widths(ws, [21] + [17] * len(series))
    ws.freeze_panes = "B2"
    ws.sheet_view.showGridLines = False
    lc = LineChart()
    lc.title = "Eyelid closure (%) - every trial, aligned to yellow LED onset"
    lc.x_axis.title = "Time from CS onset (ms)"
    lc.y_axis.title = "% eyelid closure"
    lc.height, lc.width = 12, 28
    lc.add_data(Reference(ws, min_col=2, max_col=len(series) + 1, min_row=1, max_row=len(t) + 1),
                titles_from_data=True)
    lc.set_categories(Reference(ws, min_col=1, min_row=2, max_row=len(t) + 1))
    for s in lc.series:
        s.graphicalProperties.line = LineProperties(solidFill="FF33445E", w=6000)
        s.smooth = False
    lc.x_axis.tickLblSkip = 12
    lc.legend = None
    ws.add_chart(lc, get_column_letter(len(series) + 3) + "2")

    # ---------------- Figures ----------------
    ws = wb.create_sheet("Figures")
    ws.sheet_view.showGridLines = False
    ws["A1"] = "Rendered figures"
    ws["A1"].font = Font(bold=True, size=13, color=INK)
    r = 3
    for f, cap in cfg["figs"]:
        f = os.path.join(OUT, f)
        if not os.path.exists(f):
            continue
        im = XLImage(f)
        ws.cell(row=r, column=1, value=cap).font = Font(bold=True, size=10, color=MUT)
        r += 1
        k = min(1200 / im.width, 1.0)
        im.width, im.height = int(im.width * k), int(im.height * k)
        ws.add_image(im, "A" + str(r))
        r += int(im.height / 19) + 3
    widths(ws, [115])

    out = os.path.join(OUTDIR, cfg["file"])
    try:
        wb.save(out)
    except PermissionError:
        alt = out.replace(".xlsx", "_NEW.xlsx")
        wb.save(alt)
        print(f"!! {os.path.basename(out)} is open in Excel - wrote {os.path.basename(alt)} instead")
        out = alt
    print(f"saved {out}  ({len(paired)} paired + {len(csonly)} CS-only)")


for b in (sys.argv[2:] or list(BOOKS)):
    if b in BOOKS:
        build(b)
