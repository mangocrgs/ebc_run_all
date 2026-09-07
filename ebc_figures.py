"""Figures: onset scatter, acquisition curve, closure rasters.

    python ebc_figures.py <config.json>

One set per group of trials that means something on its own - the paired conditioning
trials, the CS-only probes inside conditioning, extinction, and each baseline.
"""
import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap

import ebc_config as C
from ebc_paths import work_dir, out_dir

# Every colour and the face come from ebc_config, so a CR is the same blue here, in the
# workbook charts and on the app page.
P = C.PALETTE
CS_C, US_C, AL = P["cs"], P["us"], P["ur"]
INK, MUT, GY, LINK = P["ink"], P["muted"], P["faint"], P["link"]
CR_C, GRID, RULE = P["cr"], P["grid"], P["rule"]
# The uncertain band sits between the CR and the UR in meaning, so it is drawn between
# them in colour too - a reader should not be able to mistake it for either.
QCR_C = P.get("cr_soft") or MUT
C.mpl_font(plt)
# The closure ramp runs from the page to the ink through the US hue, so a raster and a
# scatter drawn side by side are made of the same colours.
CMAP = LinearSegmentedColormap.from_list(
    "lid", [P["surface"], "#DDE4EC", "#93A9C0", "#4A6076", P["ink"]])
TRACE, TRACE_MEAN = "#4A6076", P["ink"]


def excluded(cls, win=None):
    """Classes that are not a scored response.

    The lid was already moving at the stimulus, the blink began before it, or it came so
    long after the puff that it is the next spontaneous blink rather than a reaction to
    anything.  ebc_config.is_scoreable is the authority; the fallback below is for the
    one caller that has no window to hand.
    """
    if win is not None:
        return not C.is_scoreable(cls, win)
    c = str(cls)
    return c == "in-progress at stimulus" or c.startswith("spontaneous")


def colour(cls):
    if cls is None:
        return GY
    # Checked before "CR": the uncertain band is not a CR and must not be drawn as one.
    if cls.startswith("?CR"):
        return QCR_C
    if cls.startswith("CR"):
        return CR_C
    if cls.startswith("alpha"):
        return CS_C
    if cls.startswith("UR"):
        return AL
    return GY


def label(cls, us0):
    return {"in-progress at stimulus": "lid moving at onset, no later blink"}.get(
        cls, cls.replace("<", "< ").replace(">=", "≥ "))


PLATE = dict(fc="white", ec="none", alpha=.82, pad=1.6)


def window_note(win):
    """One line saying where the CR window came from, for under a title."""
    if win["measured"]:
        return ("CR window %.0f–%.0f ms  ·  both edges sit %.0f ms after their own "
                "stimulus, the reflex latency measured in the US-only baseline "
                "(mean − %.1f SD of %d unconditioned onsets)"
                % (win["lo_ms"], win["hi_ms"], win["reflex_ms"],
                   win["reflex"]["k"], win["reflex"]["n"]))
    return ("CR window %.0f–%.0f ms  ·  no US-only baseline to measure the reflex "
            "from, so the protocol's startle cut-off and the bare US onset are used"
            % (win["lo_ms"], win["hi_ms"]))


def cr_band(ax, win, xmax, label_it=True):
    """The window inside which a blink counts as conditioned, drawn on a latency axis.

    Its edges are the point of the whole measure, so they are drawn rather than left to
    a legend: below the lower one nothing has had time to be a response to the CS, above
    the upper one the puff has had time to cause the blink itself.
    """
    lo, hi = win["lo_ms"], win["hi_ms"]
    ax.axhspan(lo, hi, color=CR_C, alpha=.05, lw=0, zorder=0)
    for y in (lo, hi):
        ax.axhline(y, color=CR_C, lw=1.0, ls=(0, (5, 4)), alpha=.65, zorder=2)
    if label_it:
        ax.text(xmax * .012, lo + 7,
                "CR window  ·  %.0f–%.0f ms%s"
                % (lo, hi, "  ·  reflex %.0f ms" % win["reflex_ms"]
                   if win["measured"] else ""),
                color=CR_C, fontsize=9, ha="left", va="bottom", zorder=8, bbox=PLATE)


def trend(ax, xs, ys, col, where="upper right", pad=(-8, -8)):
    """A straight line through the blocks, with the R² that says how straight it is.

    The published figures this lab works from put one on every block panel, and it is
    worth having for the reason they do: a CR rate that climbs is the whole claim, and a
    line with an R² of 0.03 says the climb is not in these ten numbers.  It is drawn
    dashed and thin so it can never be mistaken for the data.

    Two points make a perfect line and no case, so nothing is drawn below three; a
    vertical fit (every block the same x, which cannot happen here but would divide by
    zero if it did) is refused rather than caught afterwards.  Returns R², or None.
    """
    x = np.asarray(xs, float)
    y = np.asarray(ys, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 3 or np.ptp(x) == 0:
        return None
    m, c = np.polyfit(x, y, 1)
    fit = m * x + c
    ss_res = float(np.sum((y - fit) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 1.0
    xx = np.array([x.min(), x.max()])
    ax.plot(xx, m * xx + c, ls=(0, (5, 4)), lw=1.6, color=col, alpha=.85, zorder=4)
    ax.annotate("R² = %.3f" % r2, xy=(1, 1) if "right" in where else (0, 1),
                xycoords="axes fraction", xytext=pad, textcoords="offset points",
                ha="right" if "right" in where else "left", va="top",
                fontsize=9.5, color=col)
    return r2


def groups_present(rows, proto):
    """The trial sets worth plotting on their own, in the order they were run."""
    g = []
    for role, tt, key, sub in (
            ("conditioning", "CS-US", "cond_paired",
             "each dot is one paired trial, joined in the order they were run"),
            ("conditioning", "CS-only", "cond_csonly", "the CS-only probe that ends each block"),
            ("extinction", "CS-only", "ext", "CS-only trials after conditioning"),
            ("baseline_cs", "CS-only", "baseline_cs", "CS alone, before conditioning"),
            ("baseline_us", "US-only", "baseline_us", "US alone, before conditioning")):
        rs = [r for r in rows if r["role"] == role and r["trial_type"] == tt]
        if rs:
            g.append((key, role, tt, sub, sorted(rs, key=lambda r: r["group_index"])))
    return g


def scatter(rows, key, role, sub, title, proto, win, odir, n_blocks, per_block):
    # The US band is where the US actually is - from its onset to its offset - not
    # "from the US onset to the end of the CS".  For the co-terminating delay protocol
    # the two are the same band; for a trace protocol they are not, and drawing the
    # second would put the US on top of a CS that ended long before it.
    des = C.design(proto)
    us0, us1 = des["isi_ms"], des["us_offset_ms"]
    cs_off = des["cs_offset_ms"]
    has_us = (role == "conditioning" and rows[0]["trial_type"] == "CS-US") or role == "baseline_us"
    anchored_us = role == "baseline_us"
    gi = len(rows)
    on = [r["scored_onset_ms"] for r in rows if r["scored_onset_ms"] is not None]
    YMAX = max(500.0, us1 + 130.0, (float(np.percentile(on, 95)) + 90) if on else 500.0)
    YMIN, NOBLINK_Y = -120.0, -100.0
    wide = gi > 14

    fig = plt.figure(figsize=(15.5 if wide else 10.5, 8.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[4.6, 1] if wide else [2.4, 1], wspace=.04)
    ax = fig.add_subplot(gs[0]); hx = fig.add_subplot(gs[1], sharey=ax)
    XMAX = gi + 1.0

    if anchored_us:
        ax.axhline(0, color=US_C, lw=2.4, zorder=2)
        ax.text(XMAX / 2, -46, "US  ·  blue LED  ·  0 ms", color=US_C, fontsize=10.5,
                ha="center", va="center")
    else:
        if has_us:
            ax.axhspan(us0, us1, color=US_C, alpha=.20, lw=0, zorder=1)
            ax.axhline(us0, color=US_C, lw=1.1, zorder=2)
            ax.text(XMAX / 2, (us0 + us1) / 2,
                    "US  ·  blue LED  ·  %.0f-%.0f ms  (%s)" % (us0, us1, des["short"]),
                    color=US_C, fontsize=10.5, ha="center", va="center", zorder=8,
                    bbox=PLATE)
        else:
            ax.axhspan(us0, us1, color=US_C, alpha=.09, lw=0, zorder=1)
            for yv in (us0, us1):
                ax.plot([0, XMAX], [yv, yv], color=US_C, lw=1.2, ls=(0, (6, 4)), alpha=.75, zorder=2)
            ax.text(XMAX / 2, (us0 + us1) / 2, "where the US would have been  ·  none delivered",
                    color=US_C, fontsize=10.5, ha="center", va="center", style="italic",
                    zorder=8, bbox=PLATE)
        hx.axhspan(us0, us1, color=US_C, alpha=.20 if has_us else .09, lw=0, zorder=1)
        ax.axhline(0, color=CS_C, lw=2.4, zorder=2)
        ax.text(XMAX / 2, -46, "CS  ·  yellow LED  ·  0-%.0f ms" % cs_off, color=CS_C,
                fontsize=10.5, ha="center", va="center", zorder=8, bbox=PLATE)
        # When the CS does not end where the US band does, its offset is a real moment in
        # the trial and belongs on the axis - in a trace protocol it is the start of the
        # interval the participant has to bridge.
        if abs(cs_off - us1) > 0.5:
            ax.axhline(cs_off, color=CS_C, lw=1.1, ls=(0, (6, 4)), alpha=.8, zorder=2)
            hx.axhline(cs_off, color=CS_C, lw=1.1, ls=(0, (6, 4)), alpha=.6, zorder=2)
            if des["kind"] == "trace":
                ax.axhspan(cs_off, us0, color=CS_C, alpha=.07, lw=0, zorder=1)
                ax.text(XMAX / 2, (cs_off + us0) / 2,
                        "trace interval  ·  %.0f ms with neither stimulus on" % des["trace_gap_ms"],
                        color=MUT, fontsize=9.5, ha="center", va="center", style="italic")
            else:
                ax.text(XMAX * .012, cs_off + 8, "CS off  ·  %.0f ms" % cs_off, color=CS_C,
                        fontsize=9, ha="left", va="bottom")
    hx.axhline(0, color=US_C if anchored_us else CS_C, lw=2.4, zorder=2)
    # A US-only recording has no CS, so there is no window to anticipate in: those trials
    # are what the window is measured from, and drawing it over them would be circular.
    if not anchored_us:
        cr_band(ax, win, XMAX)
        for y in (win["lo_ms"], win["hi_ms"]):
            hx.axhline(y, color=CR_C, lw=1.0, ls=(0, (5, 4)), alpha=.45, zorder=2)

    blocks = sorted({r["block"] for r in rows if r["block"]}) if key == "cond_paired" else []
    for b in blocks:
        ax.axvline(b * per_block + .5, color=RULE, lw=1, ls=":", zorder=1)
        ax.text((b - .5) * per_block + .5, YMAX - 20, "block %d" % b, ha="center", va="center",
                fontsize=8.5, color=MUT)

    CLIP = YMAX - 14
    lx = [r["group_index"] for r in rows if r["scored_onset_ms"] is not None]
    ly = [min(r["scored_onset_ms"], CLIP) for r in rows if r["scored_onset_ms"] is not None]
    ax.plot(lx, ly, "-", color=LINK, lw=1.2, zorder=3)

    if blocks:
        bx, by = [], []
        for b in blocks:
            g = [r["scored_onset_ms"] for r in rows if r["block"] == b
                 and r["scored_onset_ms"] is not None
                 and not excluded(r["scored_class"], win)]
            if g:
                bx.append((b - .5) * per_block + .5); by.append(float(np.mean(g)))
        ax.plot(bx, by, "-o", color=INK, lw=2.8, ms=9, zorder=6,
                markerfacecolor="white", markeredgewidth=2.2)
        for x_, y_ in zip(bx, by):
            ax.annotate("%.0f" % y_, (x_, y_), textcoords="offset points", xytext=(0, 15),
                        ha="center", fontsize=8.5, color=INK, fontweight="bold", zorder=7,
                        bbox=dict(fc="white", ec="none", alpha=.8, pad=1.2))

    nb = [r for r in rows if r["scored_onset_ms"] is None]
    if nb:
        ax.plot([r["group_index"] for r in nb], [NOBLINK_Y] * len(nb), "x", ms=9, mew=2,
                color=GY, zorder=5)
    for r in rows:
        if r["scored_onset_ms"] is None:
            continue
        c = colour(r["scored_class"])
        flag = r["quality"] != "clean"
        v = r["scored_onset_ms"]
        over = v > CLIP
        yv = min(v, CLIP)
        ax.scatter(r["group_index"], yv, s=86, marker="^" if over else "o",
                   facecolor="none" if flag else c, edgecolor=c,
                   linewidths=2.0 if flag else 1.1, zorder=5)
        if over:
            ax.annotate("%.0f" % v, (r["group_index"], yv), textcoords="offset points",
                        xytext=(0, -14), ha="center", fontsize=8, color=c)
        if r["first_response_obscured"] == "yes" and r["secondary_onset_ms"] is not None:
            ax.scatter(r["group_index"], yv, s=210, marker="o", facecolor="none",
                       edgecolor=c, linewidths=.9, alpha=.55, zorder=4)

    ax.set_xlim(0, XMAX); ax.set_ylim(YMIN, YMAX)
    step = max(1, gi // 45)
    tk = [r["group_index"] for r in rows][::step]
    ax.set_xticks(tk); ax.set_xticklabels(tk, fontsize=7.5 if gi > 40 else 9)
    ax.tick_params(labelsize=8.5)
    ax.set_xlabel(sub, fontsize=11)
    ax.set_ylabel("blink onset, ms from %s onset" % ("US (blue LED)" if anchored_us else "CS (yellow LED)"),
                  fontsize=11)
    ax.set_title("Blink onset per trial", fontsize=14.5, loc="left", pad=14,
                 color=INK, fontweight="semibold")
    ax.grid(axis="y", color=GRID, lw=.8, zorder=0); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    allo = [r["scored_onset_ms"] for r in rows if r["scored_onset_ms"] is not None
            and not excluded(r["scored_class"], win)]
    if allo:
        hx.hist(allo, bins=np.arange(-100, YMAX, 25), orientation="horizontal",
                color=CR_C, alpha=.55, zorder=4)
    hx.set_xlabel("trials", fontsize=10)
    hx.tick_params(labelleft=False, labelsize=8.5)
    hx.grid(axis="x", color=GRID, lw=.8, zorder=0); hx.set_axisbelow(True)
    for s in ("top", "right", "left"):
        hx.spines[s].set_visible(False)
    hx.set_title("distribution", fontsize=10.5, loc="left", color=MUT)

    present = []
    for r in rows:
        if r["scored_class"] and r["scored_class"] not in present:
            present.append(r["scored_class"])
    h = [Line2D([], [], ls="", marker="o", ms=9, mfc=colour(k), mec=colour(k), label=label(k, us0))
         for k in present]
    if blocks:
        h += [Line2D([], [], color=INK, lw=2.8, marker="o", ms=9, mfc="white", mew=2.2,
                     label="block mean (learning curve)")]
    h += [Line2D([], [], color=LINK, lw=1.2, label="trial order")]
    # The two edges of the CR window, named the way the published figures name them, so
    # a reader coming from one of those knows which line is which without the caption.
    if not anchored_us:
        h += [Line2D([], [], color=CR_C, lw=1.0, ls=(0, (5, 4)), alpha=.65,
                     label="CR lower-bound (%.0f ms)" % win["lo_ms"]),
              Line2D([], [], color=CR_C, lw=1.0, ls=(0, (5, 4)), alpha=.65,
                     label="CR upper-bound (%.0f ms)" % win["hi_ms"])]
    if any(r["first_response_obscured"] == "yes" and r["secondary_onset_ms"] for r in rows):
        h += [Line2D([], [], ls="", marker="o", ms=13, mfc="none", mec=MUT, alpha=.7,
                     label="response recovered behind an artefact")]
    if nb:
        h += [Line2D([], [], ls="", marker="x", ms=9, mew=2, color=GY, label="no blink detected")]
    h += [Line2D([], [], ls="", marker="o", ms=9, mfc="none", mec=INK, mew=2, label="hollow = quality flag")]
    ax.legend(handles=h, fontsize=9.5, frameon=False, loc="lower left", ncol=3,
              bbox_to_anchor=(0, -.235), handletextpad=.4, columnspacing=1.6)
    # The title runs to the very edge of the sheet rather than sitting inside the axes,
    # so the figure reads as a page with a masthead instead of a chart with a caption.
    fig.suptitle(title, fontsize=17, y=.988, x=.010, ha="left", va="top",
                 color=INK, fontweight="semibold")
    if not anchored_us:
        fig.text(.010, .944, window_note(win), fontsize=9.5, color=MUT, ha="left", va="top")
    fig.subplots_adjust(left=.062 if wide else .09, right=.985, top=.845, bottom=.20)
    p = os.path.join(odir, "%s_onset_scatter.png" % key)
    fig.savefig(p, dpi=170); plt.close(fig)
    print("wrote " + os.path.basename(p))


def block_rate(rows, cls_prefix, win):
    """Per block: what fraction of the scoreable trials in it were `cls_prefix`.

    Blocks come from the protocol - ebc_protocol numbers them from the paired-trial
    count the study file declares - so nothing here decides where a block begins.  It
    reads the number off the trial and counts.
    """
    out = []
    for b in sorted({r["block"] for r in rows if r["block"]}):
        g = [r for r in rows if r["block"] == b
             and r["scored_class"] is not None and not excluded(r["scored_class"], win)]
        if not g:
            continue
        k = sum(str(r["scored_class"]).startswith(cls_prefix) for r in g)
        out.append((b, 100.0 * k / len(g), len(g), k))
    return out


def block_onset(rows, win=None, every=False):
    """Per block: the mean blink onset and its SD, and how many it is over.

    The SD is the spread of the onsets in that block and nothing else - no SD is shown
    for a block with one onset in it, because one number has no spread, and none is shown
    for a block with none.

    `every` decides WHICH onsets, and the difference between the two answers is the point
    of drawing both.  Over the CRs alone the mean is conditioned on already being inside
    the CR window: it cannot move much, because a block improves by trials CROSSING INTO
    the window, not by the ones already in it starting earlier.  On Marie that mean is
    flat - 214 ms in block 1, 245 in block 10, R² 0.001 - while her CR rate goes 33% to
    86%, which reads as "she learnt but the timing never changed" and is an artefact of
    the selection.  Over every scoreable trial the same blocks run 310 ms to 220 ms, R²
    0.69: the blink moving from after the puff to before it, which is the learning
    itself.  Neither number is wrong; the first answers "how well timed are the CRs" and
    the second "where is the blink", and only the second can show the shift.
    """
    out = []
    for b in sorted({r["block"] for r in rows if r["block"]}):
        o = [r["scored_onset_ms"] for r in rows
             if r["block"] == b and r["scored_onset_ms"] is not None
             and (not excluded(r["scored_class"], win) if every
                  else str(r["scored_class"]).startswith("CR"))]
        if not o:
            continue
        out.append((b, float(np.mean(o)),
                    float(np.std(o, ddof=1)) if len(o) > 1 else None, len(o)))
    return out


def acquisition(rows, proto, win, title, odir, key="cond", probes=None):
    """The learning curve: how the response changes block by block.

    Three things are drawn, and the third is the reason the figure has probes on it at
    all.  The paired CR rate is the acquisition.  The rate of blinks that only follow
    the puff is its mirror - as one rises the other falls.  The CS-only probes are the
    check on both: a probe delivers no puff, so a response on one cannot be a reaction
    to anything but the CS, and a probe rate that tracks the paired rate is what says
    the learning is real and stable rather than an artefact of the puff arriving.

    The lower panel is the same measurement in time rather than in count: the mean onset
    of the CRs scored in each block, with the SD of that block's onsets.  A CR rate that
    climbs while the onsets creep earlier is the response moving to where it does some
    good; a rate that climbs with no change in onset is a different thing and is worth
    seeing separately.
    """
    paired = block_rate(rows, "CR", win)
    if not paired:
        return
    urs = block_rate(rows, "UR", win)
    onsets = block_onset(rows)
    pr = block_rate(probes, "CR", win) if probes else []

    f, (a, b_ax) = plt.subplots(2, 1, figsize=(13.5, 8.4), sharex=True,
                                gridspec_kw=dict(height_ratios=[2.05, 1.0], hspace=.16))

    xs = [p[0] for p in paired]
    cr_r = [p[1] for p in paired]
    a.fill_between(xs, cr_r, color=CR_C, alpha=.16)
    a.plot(xs, cr_r, "-o", color=CR_C, lw=2.8, ms=9,
           label="conditioned response, paired trials (%.0f–%.0f ms)"
                 % (win["lo_ms"], win["hi_ms"]))
    if urs:
        a.plot([u[0] for u in urs], [u[1] for u in urs], "-s", color=AL, lw=2.0, ms=7,
               label="reaction to the puff only (≥ %.0f ms)" % win["hi_ms"])
    if pr:
        # The probe line is BROKEN across blocks that have no scoreable probe, rather
        # than drawn straight through them.  Joining block 2 to block 10 because those
        # are the only two probes that could be scored draws eight blocks of trend that
        # were never measured, and it reads as a decline that nothing in the data says.
        # A gap is a gap.  Each point is also labelled with the number of probes behind
        # it, because one probe is 0% or 100% and nothing in between, and a reader has
        # to be able to see that.
        got = {p[0]: p for p in pr}
        py = [got[x][1] if x in got else float("nan") for x in xs]
        a.plot(xs, py, "--^", color=CS_C, lw=2.2, ms=9,
               label="CS-only probe, no puff delivered  (%d scoreable, %d block(s))"
                     % (sum(p[2] for p in pr), len(pr)))
        for x_, y_, n_, _k in pr:
            a.annotate("n=%d" % n_, (x_, y_), textcoords="offset points",
                       xytext=(0, 11 if y_ < 8 else -15), ha="center", fontsize=8,
                       color=CS_C)
    for x_, y_, n_, _k in paired:
        a.annotate("%.0f%%\nn=%d" % (y_, n_), (x_, y_), textcoords="offset points",
                   xytext=(0, 12), ha="center", fontsize=8.5, color=MUT, linespacing=1.3)
    # The straight line through the blocks, and the R² that says whether the climb the
    # eye reads into ten points is in the numbers at all.
    trend(a, xs, cr_r, CR_C, pad=(-8, -22))
    a.set_ylim(-5, 122); a.set_xlim(min(xs) - .5, max(xs) + .5)
    a.set_ylabel("% of scoreable trials in the block", fontsize=11)
    a.set_title("Acquisition — the blink shifts from reacting to the puff, to anticipating it",
                fontsize=14.5, loc="left", pad=14, color=INK, fontweight="semibold")
    a.grid(axis="y", color=GRID, lw=.8); a.set_axisbelow(True)
    for s in ("top", "right"):
        a.spines[s].set_visible(False)
    a.legend(fontsize=10, frameon=False, loc="upper left", ncol=3,
             bbox_to_anchor=(0, -.02), handletextpad=.5, columnspacing=2.2)

    if onsets:
        ox = [o[0] for o in onsets]
        om = [o[1] for o in onsets]
        oe = [o[2] if o[2] is not None else 0.0 for o in onsets]
        b_ax.errorbar(ox, om, yerr=oe, fmt="-o", color=CR_C, lw=2.2, ms=7.5,
                      ecolor=CR_C, elinewidth=1.4, capsize=4, alpha=.95,
                      label="mean CR onset  ·  the CRs only", zorder=5)
        for x_, y_, sd_, n_ in onsets:
            b_ax.annotate("n=%d" % n_, (x_, y_), textcoords="offset points",
                          xytext=(10, -3.5), ha="left", fontsize=8, color=MUT)
        trend(b_ax, ox, om, CR_C)
        # The same blocks over EVERY scoreable trial.  The line above is conditioned on
        # already being a CR, so it cannot show the shift that makes one: a block gets
        # better by trials crossing into the window, and a mean taken inside the window
        # cannot see them arrive.  This one can, and on a learner it falls while the
        # other stays put.
        allo = block_onset(rows, win, every=True)
        if allo and len(allo) > 2:
            ax_, am = [o[0] for o in allo], [o[1] for o in allo]
            b_ax.plot(ax_, am, "-s", color=MUT, lw=1.6, ms=5.5, alpha=.85, zorder=4,
                      label="mean blink onset  ·  every scoreable trial")
            trend(b_ax, ax_, am, MUT, where="upper left", pad=(8, -8))
        b_ax.axhspan(win["lo_ms"], win["hi_ms"], color=CR_C, alpha=.07, zorder=0)
        lows = [o[1] - (o[2] or 0) for o in onsets] + [o[1] for o in (allo or [])]
        highs = [o[1] + (o[2] or 0) for o in onsets] + [o[1] for o in (allo or [])]
        b_ax.set_ylim(min([win["lo_ms"]] + lows) - 30, max([win["hi_ms"]] + highs) + 30)
        b_ax.legend(fontsize=8.5, frameon=False, loc="lower right", ncol=2,
                    handletextpad=.5, columnspacing=1.6)
    else:
        b_ax.text(.5, .5, "no CR onsets to average", transform=b_ax.transAxes,
                  ha="center", va="center", fontsize=10, color=MUT)
    b_ax.set_ylabel("mean blink onset ± SD  (ms)", fontsize=11)
    b_ax.set_xticks(xs)
    b_ax.set_xlabel("block  (%d paired CS-US trials each)  ·  %s"
                    % (proto["paired_per_block"], C.design(proto)["label"].lower()),
                    fontsize=11)
    b_ax.grid(axis="y", color=GRID, lw=.8); b_ax.set_axisbelow(True)
    for s in ("top", "right"):
        b_ax.spines[s].set_visible(False)

    f.suptitle(title, fontsize=17, y=.988, x=.010, ha="left", va="top",
               color=INK, fontweight="semibold")
    f.text(.010, .944, window_note(win), fontsize=9.5, color=MUT, ha="left", va="top")
    f.subplots_adjust(left=.075, right=.985, top=.855, bottom=.095)
    p = os.path.join(odir, "%s_acquisition.png" % key)
    f.savefig(p, dpi=170); plt.close(f)
    print("wrote " + os.path.basename(p))


def paper(rows, probes, proto, win, title, odir, key="cond"):
    """The same numbers again, drawn the way this lab's published figures draw them.

    Four panels, in the order the reference works through them: what the windows ARE, the
    onset of every trial against them, then the two block summaries with a straight line
    and an R² on each.  It is deliberately plainer than the diagnostic figures - one
    marker, one colour, four named boundaries in a legend - because it is meant to sit
    beside a figure drawn from another system and be read as the same measurement rather
    than as a different one.

    The y axis is time from CS onset, as everywhere else here.  The reference plots the
    same quantity on an absolute axis, with the CS at 600 ms, because its trial window
    opens 600 ms before the CS; adding that offset would make every number on the axis
    disagree with the same number in the workbooks, which is a worse thing than an axis
    that starts somewhere else.  The four boundaries are the ones it names, and they mean
    the same: the CR window's edges each sit one reflex latency after their own stimulus.
    """
    des = C.design(proto)
    us0 = des["isi_ms"]
    lo, hi = win["lo_ms"], win["hi_ms"]
    RED, EDGE = P["ur"], P["ur"]
    onsets = block_onset(rows)
    rates = block_rate(rows, "CR", win)

    f = plt.figure(figsize=(11.6, 10.6))
    # The right margin is wide because panel B's legend hangs in it, the way the
    # reference's does; C and D are narrower for it, which is also how they are drawn
    # there - the per-trial panel is the wide one.
    gs = f.add_gridspec(3, 2, height_ratios=[.38, 1.60, 1.12],
                        hspace=.30, wspace=.26, left=.095, right=.835,
                        top=.900, bottom=.075)
    a_ax = f.add_subplot(gs[0, :])
    b_ax = f.add_subplot(gs[1, :])
    c_ax = f.add_subplot(gs[2, 0])
    d_ax = f.add_subplot(gs[2, 1])

    def letter(ax, ch):
        ax.annotate(ch, xy=(0, 1), xycoords="axes fraction", xytext=(-42, 20),
                    textcoords="offset points", fontsize=13, fontweight="bold",
                    color=INK, va="top", ha="left")

    # ---- A: the windows themselves, on a bare time axis
    hi_x = max(hi + 120, us0 + 200)
    a_ax.set_xlim(-90, hi_x); a_ax.set_ylim(0, 1)
    a_ax.annotate("", xy=(hi_x, .52), xytext=(-90, .52),
                  arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.6))
    a_ax.text(hi_x, .62, "ms", fontsize=9, color=INK, ha="right", va="bottom")
    bands = [(0, lo, "spontaneous / startle", P["cs_soft"]),
             (lo, hi, "conditioned responses: CRs", P["cr_soft"])]
    qhi = des["cs_offset_ms"]
    if qhi > hi + .5:                     # the ?CR band, where this lab scores neither
        bands.append((hi, qhi, "neither: ?CR", P["us_soft"]))
    bands.append((max(hi, qhi), hi_x, "unconditioned responses: URs", P["ur_soft"]))
    span = hi_x + 90.0
    for x0, x1, txt, col in bands:
        a_ax.axvspan(x0, x1, ymin=.44, ymax=.68, color=col, lw=0, zorder=1)
        # A band too narrow to hold its own name is labelled above the axis instead of
        # under it, where the two bound labels already are: the ?CR band is 50 ms wide
        # on the standard window and its name is wider than the band.
        narrow = (min(x1, hi_x) - x0) / span < .13
        a_ax.text((x0 + min(x1, hi_x)) / 2, .78 if narrow else .36, txt,
                  fontsize=7.5 if narrow else 8, color=MUT, ha="center",
                  va="bottom" if narrow else "top", style="italic")
    for x, lab_, col in ((0, "CS", CS_C), (us0, "US", US_C)):
        a_ax.plot([x, x], [.44, .84], color=col, lw=2.0, zorder=3)
        a_ax.text(x, .88, "%s  %.0f" % (lab_, x), fontsize=10, color=col, ha="center",
                  va="bottom", fontweight="semibold")
    for x, col in ((lo, CS_C), (hi, US_C)):
        a_ax.plot([x, x], [.30, .68], color=col, lw=1.2, ls=":", zorder=3)
    a_ax.text(lo, .26, "CR lower-bound  %.0f" % lo, fontsize=8, color=CS_C, ha="center",
              va="top")
    a_ax.text(hi, .26, "CR upper-bound  %.0f" % hi, fontsize=8, color=US_C, ha="center",
              va="top")
    a_ax.set_title(window_note(win).split("  ·  ")[0]
                   + ("   ·   mean reflex delay %.0f ms, both bounds one reflex after "
                      "their own stimulus" % win["reflex_ms"] if win["measured"]
                      else "   ·   no US-only baseline, so the protocol's own cut-offs "
                           "are used"),
                   fontsize=9.5, loc="left", color=MUT, pad=8)
    a_ax.axis("off")
    letter(a_ax, "A")

    # ---- B: every trial
    xs = [r["group_index"] for r in rows if r["scored_onset_ms"] is not None]
    ys = [r["scored_onset_ms"] for r in rows if r["scored_onset_ms"] is not None]
    top = max([hi + 120] + ([float(np.percentile(ys, 97)) + 60] if ys else []))
    bot = min([-60] + ([float(np.percentile(ys, 3)) - 60] if ys else []))
    cy = np.clip(ys, bot, top)
    b_ax.plot(xs, cy, "-", color=LINK, lw=.9, zorder=2)
    b_ax.plot(xs, cy, "D", ms=5.2, mfc="none", mec=EDGE, mew=1.1, zorder=3)
    # A blink far outside the axis is held at the edge and told, rather than drawn
    # somewhere it is not: an unlabelled marker on the frame reads as a value there.
    for x_, y_ in zip(xs, ys):
        if y_ > top:
            b_ax.annotate("%.0f" % y_, (x_, top), textcoords="offset points",
                          xytext=(0, -11), ha="center", fontsize=7, color=EDGE)
    h = []
    for y, col, ls, name in ((0, CS_C, "-", "CS onset"),
                             (us0, US_C, "-", "US onset"),
                             (lo, CS_C, ":", "CR lower-bound"),
                             (hi, US_C, ":", "CR upper-bound")):
        b_ax.axhline(y, color=col, lw=1.5 if ls == "-" else 1.3, ls=ls, zorder=1)
        h.append(Line2D([], [], color=col, lw=1.5, ls=ls, label=name))
    h.append(Line2D([], [], ls="", marker="D", ms=5.2, mfc="none", mec=EDGE, mew=1.1,
                    label="blink onset"))
    b_ax.set_xlim(0, (max(xs) if xs else 1) + 1); b_ax.set_ylim(bot, top)
    step = max(1, len(rows) // 34)
    tk = [r["group_index"] for r in rows][::step]
    b_ax.set_xticks(tk)
    b_ax.set_xticklabels(["Trial %d" % t for t in tk], rotation=90, fontsize=6.8)
    b_ax.set_ylabel("Blink onset (ms from CS)", fontsize=10)
    b_ax.legend(handles=h, fontsize=8, frameon=True, loc="center left",
                bbox_to_anchor=(1.005, .5), handlelength=2.0, borderpad=.6)
    b_ax.tick_params(labelsize=8)
    b_ax.grid(axis="y", color=GRID, lw=.7); b_ax.set_axisbelow(True)
    letter(b_ax, "B")

    # ---- C: mean onset per block, with its spread
    if onsets:
        ox = [o[0] for o in onsets]
        om = [o[1] for o in onsets]
        oe = [o[2] if o[2] is not None else 0.0 for o in onsets]
        c_ax.errorbar(ox, om, yerr=oe, fmt="D", ms=5.2, mfc="none", mec=EDGE, mew=1.1,
                      ecolor=EDGE, elinewidth=1.0, capsize=2.5, ls="none", zorder=3,
                      label="CRs only")
        trend(c_ax, ox, om, RED)
        # see block_onset(): a mean taken inside the CR window cannot show trials
        # arriving in it, which is what a block improving actually consists of
        allo = block_onset(rows, win, every=True)
        if allo and len(allo) > 2:
            ax_, am = [o[0] for o in allo], [o[1] for o in allo]
            c_ax.plot(ax_, am, "-", color=MUT, lw=1.3, alpha=.9, zorder=2,
                      label="every scoreable trial")
            trend(c_ax, ax_, am, MUT, where="upper left", pad=(8, -8))
            c_ax.legend(fontsize=7.5, frameon=False, loc="lower left",
                        handletextpad=.5, borderaxespad=.2)
        c_ax.set_xticks(ox)
        c_ax.set_xticklabels(["block %d" % b for b in ox], rotation=90, fontsize=7.5)
        c_ax.set_ylim(min(0, lo - 120),
                      max(list(om) + [o[1] for o in (allo or [])]) + max(oe) + 120)
    else:
        c_ax.text(.5, .5, "no CR onsets to average", transform=c_ax.transAxes,
                  ha="center", va="center", fontsize=9, color=MUT)
    c_ax.set_title("Mean blink onset", fontsize=10.5, color=INK)
    c_ax.set_ylabel("ms from CS", fontsize=9.5)
    c_ax.tick_params(labelsize=8)
    c_ax.grid(axis="y", color=GRID, lw=.7); c_ax.set_axisbelow(True)
    letter(c_ax, "C")

    # ---- D: how many of them were CRs
    if rates:
        rx = [p[0] for p in rates]
        ry = [p[1] for p in rates]
        d_ax.plot(rx, ry, "D", ms=5.2, mfc=RED, mec=EDGE, mew=1.0, ls="none", zorder=3)
        # left, because an acquisition curve rises to the right and the R² would sit on
        # top of the last blocks - the ones it is a claim about
        trend(d_ax, rx, ry, RED, where="upper left", pad=(8, -8))
        d_ax.set_xticks(rx)
        d_ax.set_xticklabels(["block %d" % b for b in rx], rotation=90, fontsize=7.5)
    d_ax.set_ylim(-5, 105)
    d_ax.set_title("Percentage of CRs", fontsize=10.5, color=INK)
    d_ax.set_ylabel("% of scoreable trials", fontsize=9.5)
    d_ax.tick_params(labelsize=8)
    d_ax.grid(axis="y", color=GRID, lw=.7); d_ax.set_axisbelow(True)
    letter(d_ax, "D")

    for ax in (b_ax, c_ax, d_ax):
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(RULE)

    f.suptitle(title, fontsize=15, y=.985, x=.010, ha="left", va="top",
               color=INK, fontweight="semibold")
    f.text(.010, .944, "the same trials as the other figures, drawn in the published "
                       "style: one marker, the four boundaries named, a straight line "
                       "through the blocks", fontsize=9, color=MUT, ha="left", va="top")
    p = os.path.join(odir, "%s_paper_figure.png" % key)
    f.savefig(p, dpi=200); plt.close(f)
    print("wrote " + os.path.basename(p))


def rasters(rows, traces, key, role, title, proto, win, odir, order):
    des = C.design(proto)
    us0, us1 = des["isi_ms"], des["us_offset_ms"]
    cs_off = des["cs_offset_ms"]
    has_us = rows[0]["trial_type"] == "CS-US"
    anchored_us = role == "baseline_us"
    by_s = {}
    for r in rows:
        by_s.setdefault(r["session"], []).append(r)
    use = [t for t in order if t in by_s]
    if not use:
        return
    heights = [max(len(by_s[t]), 5) for t in use]
    fig = plt.figure(figsize=(15.5, 3.1 * len(use) + 2.2))
    gs = fig.add_gridspec(len(use), 2, width_ratios=[1.25, 1],
                          height_ratios=[h / sum(heights) for h in heights], hspace=.38, wspace=.14)
    for si, tag in enumerate(use):
        rs = by_s[tag]
        TRC = traces[tag]
        t = np.array(TRC[str(rs[0]["session_trial"])]["t"])
        L = len(t)
        Mx = np.vstack([np.array(TRC[str(r["session_trial"])]["C"] + [np.nan] * L)[:L] for r in rs]) * 100
        a0 = fig.add_subplot(gs[si, 0])
        a0.imshow(Mx, aspect="auto", cmap=CMAP, vmin=0, vmax=100,
                  extent=[t[0], t[-1], len(rs) + .5, .5], interpolation="nearest")
        a0.axvline(0, color=US_C if anchored_us else CS_C, lw=2)
        if not anchored_us:
            if has_us:
                a0.axvspan(us0, us1, color=US_C, alpha=.30, lw=0)
            else:
                for yv in (us0, us1):
                    a0.axvline(yv, color=US_C, lw=1.1, ls=(0, (5, 4)), alpha=.8)
            if abs(cs_off - us1) > 0.5:
                a0.axvline(cs_off, color=CS_C, lw=1.1, ls=(0, (5, 4)), alpha=.85)
        for i, r in enumerate(rs):
            if r["scored_onset_ms"] is not None:
                a0.plot(r["scored_onset_ms"], i + 1, "o", ms=4.2,
                        mfc=colour(r["scored_class"]), mec="white", mew=.8)
        a0.set_yticks(range(1, len(rs) + 1, max(1, len(rs) // 12)))
        a0.tick_params(labelsize=7.5)
        a0.set_ylabel("%s\ntrial" % rs[0]["session_name"], fontsize=9.5)
        if si == len(use) - 1:
            a0.set_xlabel("time from stimulus onset, ms", fontsize=10.5)
        if si == 0:
            a0.set_title("Eyelid closure per trial   ·   dot = scored blink onset, "
                         "coloured by what it was scored as",
                         fontsize=12, loc="left", color=INK)
        a1 = fig.add_subplot(gs[si, 1])
        for i in range(len(Mx)):
            a1.plot(t, Mx[i], color=TRACE, lw=.7, alpha=.32)
        a1.plot(t, np.nanmean(Mx, axis=0), color=TRACE_MEAN, lw=2.5)
        a1.axvline(0, color=US_C if anchored_us else CS_C, lw=2)
        if not anchored_us:
            a1.axvspan(us0, us1, color=US_C, alpha=.30 if has_us else .10, lw=0)
            if abs(cs_off - us1) > 0.5:
                a1.axvspan(0, cs_off, color=CS_C, alpha=.10, lw=0)
                a1.axvline(cs_off, color=CS_C, lw=1.1, ls=(0, (5, 4)), alpha=.85)
            if not has_us:
                a1.text(.99, .93, "no US delivered", transform=a1.transAxes, ha="right",
                        fontsize=9, color=MUT, style="italic")
        a1.set_xlim(t[0], t[-1]); a1.set_ylim(-18, 112)
        a1.set_ylabel("% closure", fontsize=9.5); a1.tick_params(labelsize=8)
        if si == len(use) - 1:
            a1.set_xlabel("time from stimulus onset, ms", fontsize=10.5)
        if si == 0:
            a1.set_title("All trials overlaid, mean in black", fontsize=12, loc="left",
                         color=INK)
        for a in (a0, a1):
            for s in ("top", "right"):
                a.spines[s].set_visible(False)
    fig.suptitle(title + " — eyelid closure", fontsize=17, y=.994, x=.010, ha="left",
                 va="top", color=INK, fontweight="semibold")
    fig.subplots_adjust(left=.075, right=.985, top=.915, bottom=.075)
    p = os.path.join(odir, "%s_overview.png" % key)
    fig.savefig(p, dpi=155); plt.close(fig)
    print("wrote " + os.path.basename(p))


def main():
    cfg = C.load(sys.argv[1] if len(sys.argv) > 1 else None)
    wdir, odir = work_dir(cfg), out_dir(cfg)
    with open(os.path.join(wdir, "merged.json"), encoding="utf-8") as fh:
        M = json.load(fh)
    with open(os.path.join(wdir, "merged_rows.json"), encoding="utf-8") as fh:
        ROWS = json.load(fh)
    proto = C.fill(M["protocol"])
    # The window the run was actually scored against, read back rather than recomputed,
    # so a figure can never draw a boundary the numbers were not classified on.  A
    # merged.json from before the window was measured falls back to the protocol.
    win = M.get("cr_window") or C.cr_window(proto)
    order = [r["tag"] for r in cfg["recordings"]]
    study = cfg["study"]
    # the figure names the design it actually was, so a trace figure does not carry a
    # title calling itself delay conditioning
    TITLE = {"cond_paired": "%s — " + C.design(proto)["label"].lower()
                            + "  |  %d paired CS-US trials",
             "cond_csonly": "%s — CS-only probes during conditioning  |  one per block",
             "ext": "%s — extinction  |  CS-only",
             "baseline_cs": "%s — baseline, CS alone",
             "baseline_us": "%s — baseline, US alone"}
    # The CS-only probes are drawn on the paired trials' own block axis, because that
    # comparison - does responding to the CS alone track responding on the paired
    # trials? - is the whole reason the probes are run.  They are picked out here rather
    # than inside the figure so the figure is handed the rows and decides nothing about
    # which recordings they came from.
    PROBES = [r for r in ROWS
              if r["role"] == "conditioning" and r["trial_type"] == "CS-only"]
    for key, role, tt, sub, rows in groups_present(ROWS, proto):
        t = TITLE[key]
        title = (t % (study, len(rows))) if key == "cond_paired" else (t % study)
        scatter(rows, key, role, sub, title, proto, win, odir,
                proto["n_blocks"], proto["paired_per_block"])
        if key == "cond_paired":
            acquisition(rows, proto, win, title, odir, probes=PROBES)
            paper(rows, PROBES, proto, win, title, odir)
        rasters(rows, M["traces"], key, role, title, proto, win, odir, order)


if __name__ == "__main__":
    main()
