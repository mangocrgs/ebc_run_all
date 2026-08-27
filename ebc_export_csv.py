"""Write the scored data out as plain CSV, next to the workbooks.

    trials_conditioning_paired.csv   90 paired CS-US trials
    trials_conditioning_CSonly.csv   10 CS-only probes (one per block)
    trials_extinction_CSUS4.csv       3 CS-only trials from CSUS 4
    stimulus_events.csv              every accepted CS and US event
    closure_traces_all.csv           full eyelid traces, long format
"""
# --- portable paths -------------------------------------------------------
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ebc_paths import BASE, OUT, WORK          # noqa: E402
os.chdir(WORK)
# --------------------------------------------------------------------------
import csv
import json

rows = json.load(open("merged_rows.json"))
M = json.load(open("merged.json"))
META = M["meta"]

SETS = [("trials_conditioning_paired.csv",
         lambda r: r["block_kind"] == "conditioning" and r["trial_type"] == "CS-US"),
        ("trials_conditioning_CSonly.csv",
         lambda r: r["block_kind"] == "conditioning" and r["trial_type"] == "CS-only"),
        ("trials_extinction_CSUS4.csv",
         lambda r: r["block_kind"] == "extinction")]

for name, sel in SETS:
    rs = [r for r in rows if sel(r)]
    if not rs:
        continue
    with open(os.path.join(OUT, name), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rs[0].keys()))
        w.writeheader()
        w.writerows(rs)
    print(f"{name:<34} {len(rs):>4} trials")

# every accepted stimulus event, in recording order
keep = {(r["session"], round(r["cs_onset_video_s"] * META[r["session"]]["fps"])) for r in rows}
with open(os.path.join(OUT, "stimulus_events.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["session", "event", "onset_frame", "onset_s", "duration_ms",
                "paired_with", "cs_us_interval_ms"])
    n = 0
    for tag in ["csus1", "csus2", "csus3", "csus4"]:
        if tag not in META:
            continue
        fps = META[tag]["fps"]
        name = META[tag]["rows"][0]["session_name"]
        used = {c for a, b, c in META[tag]["cs_events"] if c >= 0}
        ev = [(a, "CS (yellow LED)", a, b, c) for a, b, c in META[tag]["cs_events"]
              if any((tag, a + d) in keep for d in (-1, 0, 1))]
        ev += [(a, "US (blue LED)", a, b, -2 if a in used else -1) for a, b in META[tag]["us_events"]]
        for _, kind, a, b, c in sorted(ev):
            pair = ("US" if c >= 0 else "unpaired") if kind.startswith("CS") else \
                   ("CS" if c == -2 else "unpaired")
            w.writerow([name, kind, a, round(a / fps, 4), round((b - a + 1) / fps * 1000, 1), pair,
                        round((c - a) / fps * 1000, 1) if (kind.startswith("CS") and c >= 0) else ""])
            n += 1
print(f"{'stimulus_events.csv':<34} {n:>4} events")

with open(os.path.join(OUT, "closure_traces_all.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["block_kind", "trial_type", "block", "session", "trial",
                "time_ms_from_CS", "pct_closure"])
    for r in rows:
        T = META[r["session"]]["traces"][str(r["session_trial"])]
        for t_, c_ in zip(T["t"], T["C"]):
            w.writerow([r["block_kind"], r["trial_type"], r["block"], r["session_name"],
                        r["session_trial"], round(t_, 2), round(c_ * 100, 2)])
print(f"{'closure_traces_all.csv':<34} {len(rows):>4} trials")
