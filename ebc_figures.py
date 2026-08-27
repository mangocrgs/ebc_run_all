# --- portable paths -------------------------------------------------------
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ebc_paths import BASE, OUT, WORK          # noqa: E402
os.chdir(WORK)                                  # cache + intermediates live here
# --------------------------------------------------------------------------
import json, sys, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap

CS_C, US_C, AL, INK, MUT = "#B8760F", "#3A67CF", "#B03A32", "#141922", "#59636F"
CR_C, GY, LINK = "#2C4C86", "#93A0AE", "#B9C2CE"
COL = {"CR (100-350ms)": CR_C, "alpha/startle <100ms": CS_C,
       "UR (>=350ms)": AL, "in-progress at CS": GY}
LBL = {"CR (100-350ms)": "CR  (100-350 ms)", "alpha/startle <100ms": "alpha / startle  (<100 ms)",
       "UR (>=350ms)": "UR only  (>=350 ms)", "in-progress at CS": "lid moving at CS, no later blink"}

M = json.load(open("merged.json"))
META = M["meta"]
ROWS = json.load(open("merged_rows.json"))
US0, US1 = M["nominal"]["us_onset_ms"], M["nominal"]["cs_ms"]      # 350 -> 400, co-terminating

KIND = sys.argv[1]          # cond_paired | cond_csonly | ext
TITLE, TAG = sys.argv[2], sys.argv[3]
SEL = {"cond_paired": lambda r: r["block_kind"] == "conditioning" and r["trial_type"] == "CS-US",
       "cond_csonly": lambda r: r["block_kind"] == "conditioning" and r["trial_type"] == "CS-only",
       "ext": lambda r: r["block_kind"] == "extinction"}[KIND]
rows = sorted([r for r in ROWS if SEL(r)], key=lambda r: r["gidx"])
HAS_US = KIND == "cond_paired"
PER_BLOCK = 9 if KIND == "cond_paired" else 1
gi = len(rows)
on = [r["scored_onset_ms"] for r in rows if r["scored_onset_ms"] is not None]
YMAX = max(500.0, (float(np.percentile(on, 95)) + 90) if on else 500.0)
YMIN, NOBLINK_Y = -120.0, -100.0
SUB = {"cond_paired": "each dot is one paired trial, joined in the order they were run",
       "cond_csonly": "the CS-only probe that ends each block of ten",
       "ext": "CS-only probes after conditioning"}[KIND]

fig = plt.figure(figsize=(15.5 if gi > 14 else 10.5, 8.2))
gs = fig.add_gridspec(1, 2, width_ratios=[4.6, 1] if gi > 14 else [2.4, 1], wspace=.04)
ax = fig.add_subplot(gs[0]); hx = fig.add_subplot(gs[1], sharey=ax)
XMAX = gi + 1.0

if HAS_US:
    ax.axhspan(US0, US1, color=US_C, alpha=.20, lw=0, zorder=1)
    ax.axhline(US0, color=US_C, lw=1.1, zorder=2)
    ax.text(XMAX / 2, (US0 + US1) / 2, "US  ·  blue LED  ·  350–400 ms  (co-terminates with CS)",
            color=US_C, fontsize=10.5, ha="center", va="center")
else:
    ax.axhspan(US0, US1, color=US_C, alpha=.09, lw=0, zorder=1)
    for yv in (US0, US1):
        ax.plot([0, XMAX], [yv, yv], color=US_C, lw=1.2, ls=(0, (6, 4)), alpha=.75, zorder=2)
    ax.text(XMAX / 2, (US0 + US1) / 2, "where the US would have been  ·  none delivered",
            color=US_C, fontsize=10.5, ha="center", va="center", style="italic")
hx.axhspan(US0, US1, color=US_C, alpha=.20 if HAS_US else .09, lw=0, zorder=1)
ax.axhline(0, color=CS_C, lw=2.4, zorder=2)
hx.axhline(0, color=CS_C, lw=2.4, zorder=2)
ax.text(XMAX / 2, -46, "CS  ·  yellow LED  ·  0–400 ms", color=CS_C, fontsize=10.5,
        ha="center", va="center")

if KIND == "cond_paired":
    for b in range(1, 11):
        ax.axvline(b * PER_BLOCK + .5, color="#DDE2E9", lw=1, ls=":", zorder=1)
        ax.text((b - .5) * PER_BLOCK + .5, YMAX - 20, f"block {b}", ha="center", va="center",
                fontsize=8.5, color=MUT)

CLIP = YMAX - 14
lx = [r["gidx"] for r in rows if r["scored_onset_ms"] is not None]
ly = [min(r["scored_onset_ms"], CLIP) for r in rows if r["scored_onset_ms"] is not None]
ax.plot(lx, ly, "-", color=LINK, lw=1.2, zorder=3)

if KIND == "cond_paired":
    bx, by, bn = [], [], []
    for b in range(1, 11):
        g = [r["scored_onset_ms"] for r in rows
             if r["block"] == b and r["scored_onset_ms"] is not None
             and r["scored_class"] != "in-progress at CS"]
        if g:
            bx.append((b - .5) * PER_BLOCK + .5); by.append(float(np.mean(g))); bn.append(len(g))
    ax.plot(bx, by, "-o", color=INK, lw=2.8, ms=9, zorder=6,
            markerfacecolor="white", markeredgewidth=2.2)
    for x_, y_ in zip(bx, by):
        ax.annotate(f"{y_:.0f}", (x_, y_), textcoords="offset points", xytext=(0, 15),
                    ha="center", fontsize=8.5, color=INK, fontweight="bold", zorder=7,
                    bbox=dict(fc="white", ec="none", alpha=.8, pad=1.2))

nb = [r for r in rows if r["scored_onset_ms"] is None]
if nb:
    ax.plot([r["gidx"] for r in nb], [NOBLINK_Y] * len(nb), "x", ms=9, mew=2, color=GY, zorder=5)

for r in rows:
    if r["scored_onset_ms"] is None:
        continue
    c = COL[r["scored_class"]]
    flag = r["quality"] != "clean"
    v = r["scored_onset_ms"]
    over = v > CLIP
    yv = min(v, CLIP)
    ax.scatter(r["gidx"], yv, s=86, marker="^" if over else "o",
               facecolor="none" if flag else c, edgecolor=c,
               linewidths=2.0 if flag else 1.1, zorder=5)
    if over:
        ax.annotate(f"{v:.0f}", (r["gidx"], yv), textcoords="offset points", xytext=(0, -14),
                    ha="center", fontsize=8, color=c)
    if r["first_response_obscured"] == "yes" and r["secondary_onset_ms"] is not None:
        ax.scatter(r["gidx"], yv, s=210, marker="o", facecolor="none",
                   edgecolor=c, linewidths=.9, alpha=.55, zorder=4)

ax.set_xlim(0, XMAX); ax.set_ylim(YMIN, YMAX)
step = max(1, gi // 45)
tk = [r["gidx"] for r in rows][::step]
ax.set_xticks(tk); ax.set_xticklabels(tk, fontsize=7.5 if gi > 40 else 9)
ax.tick_params(labelsize=8.5)
ax.set_xlabel({"cond_paired": "paired CS–US trial, in order  (9 per block, 10 blocks)",
               "cond_csonly": "CS-only probe (one per block)",
               "ext": "extinction trial"}[KIND], fontsize=11)
ax.set_ylabel("blink onset, ms from yellow LED (CS) onset", fontsize=11)
ax.set_title(f"Blink onset per trial  ·  {SUB}", fontsize=13, loc="left", pad=26)
ax.grid(axis="y", color="#EDF0F4", lw=.8, zorder=0); ax.set_axisbelow(True)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)

allo = [r["scored_onset_ms"] for r in rows
        if r["scored_onset_ms"] is not None and r["scored_class"] != "in-progress at CS"]
hx.hist(allo, bins=np.arange(-100, YMAX, 25), orientation="horizontal", color=CR_C, alpha=.55, zorder=4)
hx.set_xlabel("trials", fontsize=10)
hx.tick_params(labelleft=False, labelsize=8.5)
hx.grid(axis="x", color="#EDF0F4", lw=.8, zorder=0); hx.set_axisbelow(True)
for s in ("top", "right", "left"):
    hx.spines[s].set_visible(False)
hx.set_title("distribution", fontsize=10, loc="left", color=MUT)

present = [k for k in ["CR (100-350ms)", "alpha/startle <100ms", "UR (>=350ms)", "in-progress at CS"]
           if any(r["scored_class"] == k for r in rows)]
h = [Line2D([], [], ls="", marker="o", ms=9, mfc=COL[k], mec=COL[k], label=LBL[k]) for k in present]
if KIND == "cond_paired":
    h += [Line2D([], [], color=INK, lw=2.8, marker="o", ms=9, mfc="white", mew=2.2,
                 label="block mean (learning curve)")]
h += [Line2D([], [], color=LINK, lw=1.2, label="trial order")]
if any(r["first_response_obscured"] == "yes" and r["secondary_onset_ms"] for r in rows):
    h += [Line2D([], [], ls="", marker="o", ms=13, mfc="none", mec=MUT, alpha=.7,
                 label="response recovered behind an artefact")]
if nb:
    h += [Line2D([], [], ls="", marker="x", ms=9, mew=2, color=GY, label="no blink detected")]
h += [Line2D([], [], ls="", marker="o", ms=9, mfc="none", mec=INK, mew=2, label="hollow = quality flag")]
if any((r["scored_onset_ms"] or 0) > CLIP for r in rows):
    h += [Line2D([], [], ls="", marker="^", ms=9, mfc=MUT, mec=MUT, label="off-scale, value shown")]
ax.legend(handles=h, fontsize=9.5, frameon=False, loc="lower left", ncol=3,
          bbox_to_anchor=(0, -.235), handletextpad=.4, columnspacing=1.6)
fig.suptitle(TITLE, fontsize=13.5, y=.985, x=.012, ha="left")
fig.subplots_adjust(left=.062 if gi > 14 else .09, right=.985, top=.87, bottom=.20)
fig.savefig(os.path.join(OUT, f"{TAG}_onset_scatter.png"), dpi=170)
print("wrote", f"{TAG}_onset_scatter.png")

# ======================= acquisition curve (paired only) =====================
if KIND == "cond_paired":
    xs, cr_r, ur_r, ns = [], [], [], []
    for b in range(1, 11):
        g = [r for r in rows if r["block"] == b and r["scored_class"] not in (None, "in-progress at CS")]
        if not g:
            continue
        xs.append(b)
        cr_r.append(100 * sum(r["scored_class"] == "CR (100-350ms)" for r in g) / len(g))
        ur_r.append(100 * sum(r["scored_class"] == "UR (>=350ms)" for r in g) / len(g))
        ns.append(len(g))
    f3, a3 = plt.subplots(figsize=(13.5, 5.8))
    a3.fill_between(xs, cr_r, color=CR_C, alpha=.16)
    a3.plot(xs, cr_r, "-o", color=CR_C, lw=2.8, ms=9, label="conditioned response (100–350 ms)")
    a3.plot(xs, ur_r, "-s", color=AL, lw=2.0, ms=7, label="reaction to the puff only (≥350 ms)")
    for x_, y_, n_ in zip(xs, cr_r, ns):
        a3.annotate(f"{y_:.0f}%\nn={n_}", (x_, y_), textcoords="offset points", xytext=(0, 12),
                    ha="center", fontsize=8.5, color=MUT, linespacing=1.3)
    a3.set_ylim(-5, 122); a3.set_xlim(.5, 10.5)
    a3.set_xticks(range(1, 11))
    a3.set_xlabel("block  (9 paired CS–US trials each)", fontsize=11)
    a3.set_ylabel("% of scoreable trials in the block", fontsize=11)
    a3.set_title("Acquisition — the blink shifts from reacting to the puff, to anticipating it",
                 fontsize=13, loc="left", pad=14)
    a3.grid(axis="y", color="#EDF0F4", lw=.8); a3.set_axisbelow(True)
    for s in ("top", "right"):
        a3.spines[s].set_visible(False)
    a3.legend(fontsize=10, frameon=False, loc="upper left", ncol=2,
              bbox_to_anchor=(0, -.13), handletextpad=.5, columnspacing=2.2)
    f3.suptitle(TITLE, fontsize=13.5, y=.985, x=.012, ha="left")
    f3.subplots_adjust(left=.075, right=.985, top=.82, bottom=.22)
    f3.savefig(os.path.join(OUT, f"{TAG}_acquisition.png"), dpi=170)
    print("wrote", f"{TAG}_acquisition.png")

# ======================= rasters ============================================
cmap = LinearSegmentedColormap.from_list("lid", ["#FBFCFD", "#DCE3EC", "#8FA3BD", "#4A5F7E", "#1B2432"])
by_s = {}
for r in rows:
    by_s.setdefault(r["session"], []).append(r)
use = [t for t in ["csus1", "csus2", "csus3", "csus4"] if t in by_s]
heights = [max(len(by_s[t]), 5) for t in use]
fig2 = plt.figure(figsize=(15.5, 3.1 * len(use) + 2.2))
gs2 = fig2.add_gridspec(len(use), 2, width_ratios=[1.25, 1],
                        height_ratios=[h / sum(heights) for h in heights], hspace=.38, wspace=.14)
for si, tag in enumerate(use):
    rs = by_s[tag]
    TRC = META[tag]["traces"]
    t = np.array(TRC[str(rs[0]["session_trial"])]["t"])
    Mx = np.vstack([np.array(TRC[str(r["session_trial"])]["C"])[:len(t)] for r in rs]) * 100
    a0 = fig2.add_subplot(gs2[si, 0])
    a0.imshow(Mx, aspect="auto", cmap=cmap, vmin=0, vmax=100,
              extent=[t[0], t[-1], len(rs) + .5, .5], interpolation="nearest")
    a0.axvline(0, color=CS_C, lw=2)
    if HAS_US:
        a0.axvspan(US0, US1, color=US_C, alpha=.30, lw=0)
    else:
        for yv in (US0, US1):
            a0.axvline(yv, color=US_C, lw=1.1, ls=(0, (5, 4)), alpha=.8)
    for i, r in enumerate(rs):
        if r["scored_onset_ms"] is not None:
            a0.plot(r["scored_onset_ms"], i + 1, "o", ms=4.2, mfc=AL, mec="white", mew=.8)
    a0.set_yticks(range(1, len(rs) + 1, max(1, len(rs) // 12)))
    a0.tick_params(labelsize=7.5)
    a0.set_ylabel(f"{rs[0]['session_name']}\ntrial", fontsize=9.5)
    if si == len(use) - 1:
        a0.set_xlabel("time from yellow LED (CS) onset, ms", fontsize=10.5)
    if si == 0:
        a0.set_title("Eyelid closure per trial   ·   red = scored blink onset", fontsize=11.5, loc="left")
    a1 = fig2.add_subplot(gs2[si, 1])
    for i in range(len(Mx)):
        a1.plot(t, Mx[i], color="#4A5F7E", lw=.7, alpha=.32)
    a1.plot(t, Mx.mean(0), color="#1B2432", lw=2.5)
    a1.axvline(0, color=CS_C, lw=2)
    a1.axvspan(US0, US1, color=US_C, alpha=.30 if HAS_US else .10, lw=0)
    if not HAS_US:
        a1.text(.99, .93, "no US delivered", transform=a1.transAxes, ha="right",
                fontsize=9, color=MUT, style="italic")
    a1.set_xlim(t[0], t[-1]); a1.set_ylim(-18, 112)
    a1.set_ylabel("% closure", fontsize=9.5); a1.tick_params(labelsize=8)
    if si == len(use) - 1:
        a1.set_xlabel("time from yellow LED (CS) onset, ms", fontsize=10.5)
    if si == 0:
        a1.set_title("All trials overlaid, mean in black", fontsize=11.5, loc="left")
    for a in (a0, a1):
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
fig2.suptitle(TITLE.split("|")[0].strip() + " — eyelid closure", fontsize=13.5, y=.99, x=.012, ha="left")
fig2.subplots_adjust(left=.075, right=.985, top=.925, bottom=.075)
fig2.savefig(os.path.join(OUT, f"{TAG}_overview.png"), dpi=155)
print("wrote", f"{TAG}_overview.png")
