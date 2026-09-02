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
C.mpl_font(plt)
# The closure ramp runs from the page to the ink through the US hue, so a raster and a
# scatter drawn side by side are made of the same colours.
CMAP = LinearSegmentedColormap.from_list(
    "lid", [P["surface"], "#DDE4EC", "#93A9C0", "#4A6076", P["ink"]])
TRACE, TRACE_MEAN = "#4A6076", P["ink"]


def colour(cls):
    if cls is None:
        return GY
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
                 and r["scored_class"] != "in-progress at stimulus"]
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
            and r["scored_class"] != "in-progress at stimulus"]
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


def acquisition(rows, proto, win, title, odir, key="cond"):
    blocks = sorted({r["block"] for r in rows if r["block"]})
    xs, cr_r, ur_r, ns = [], [], [], []
    for b in blocks:
        g = [r for r in rows if r["block"] == b
             and r["scored_class"] not in (None, "in-progress at stimulus")]
        if not g:
            continue
        xs.append(b)
        cr_r.append(100 * sum(str(r["scored_class"]).startswith("CR") for r in g) / len(g))
        ur_r.append(100 * sum(str(r["scored_class"]).startswith("UR") for r in g) / len(g))
        ns.append(len(g))
    if not xs:
        return
    f, a = plt.subplots(figsize=(13.5, 5.8))
    a.fill_between(xs, cr_r, color=CR_C, alpha=.16)
    a.plot(xs, cr_r, "-o", color=CR_C, lw=2.8, ms=9,
           label="conditioned response (%.0f–%.0f ms)" % (win["lo_ms"], win["hi_ms"]))
    a.plot(xs, ur_r, "-s", color=AL, lw=2.0, ms=7,
           label="reaction to the puff only (≥ %.0f ms)" % win["hi_ms"])
    for x_, y_, n_ in zip(xs, cr_r, ns):
        a.annotate("%.0f%%\nn=%d" % (y_, n_), (x_, y_), textcoords="offset points",
                   xytext=(0, 12), ha="center", fontsize=8.5, color=MUT, linespacing=1.3)
    a.set_ylim(-5, 122); a.set_xlim(min(xs) - .5, max(xs) + .5)
    a.set_xticks(xs)
    a.set_xlabel("block  (%d paired CS-US trials each)  ·  %s"
                 % (proto["paired_per_block"], C.design(proto)["label"].lower()), fontsize=11)
    a.set_ylabel("% of scoreable trials in the block", fontsize=11)
    a.set_title("Acquisition — the blink shifts from reacting to the puff, to anticipating it",
                fontsize=14.5, loc="left", pad=14, color=INK, fontweight="semibold")
    a.grid(axis="y", color=GRID, lw=.8); a.set_axisbelow(True)
    for s in ("top", "right"):
        a.spines[s].set_visible(False)
    a.legend(fontsize=10, frameon=False, loc="upper left", ncol=2,
             bbox_to_anchor=(0, -.13), handletextpad=.5, columnspacing=2.2)
    f.suptitle(title, fontsize=17, y=.988, x=.010, ha="left", va="top",
               color=INK, fontweight="semibold")
    f.text(.010, .915, window_note(win), fontsize=9.5, color=MUT, ha="left", va="top")
    f.subplots_adjust(left=.075, right=.985, top=.78, bottom=.22)
    p = os.path.join(odir, "%s_acquisition.png" % key)
    f.savefig(p, dpi=170); plt.close(f)
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
    for key, role, tt, sub, rows in groups_present(ROWS, proto):
        t = TITLE[key]
        title = (t % (study, len(rows))) if key == "cond_paired" else (t % study)
        scatter(rows, key, role, sub, title, proto, win, odir,
                proto["n_blocks"], proto["paired_per_block"])
        if key == "cond_paired":
            acquisition(rows, proto, win, title, odir)
        rasters(rows, M["traces"], key, role, title, proto, win, odir, order)


if __name__ == "__main__":
    main()
