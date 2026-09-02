"""Write the scored data out as plain CSV, next to the workbooks.

    python ebc_export_csv.py <config.json>

    trials_<role>_<type>.csv        one row per trial
    trials_to_score_by_hand.csv    the trials the scorer will not stand behind
    stimulus_events.csv            every CS and US pulse read from the LEDs, accepted or not
    closure_traces_all.csv         full eyelid traces, long format
"""
import os
import sys
import csv
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ebc_config as C
from ebc_paths import work_dir, out_dir


def main():
    cfg = C.load(sys.argv[1] if len(sys.argv) > 1 else None)
    wdir, odir = work_dir(cfg), out_dir(cfg)
    with open(os.path.join(wdir, "merged.json"), encoding="utf-8") as fh:
        M = json.load(fh)
    with open(os.path.join(wdir, "merged_rows.json"), encoding="utf-8") as fh:
        rows = json.load(fh)

    seen = []
    for r in rows:
        k = (r["role"], r["trial_type"])
        if k not in seen:
            seen.append(k)
    for role, tt in seen:
        rs = [r for r in rows if r["role"] == role and r["trial_type"] == tt]
        name = "trials_%s_%s.csv" % (role, tt.replace("-", ""))
        with open(os.path.join(odir, name), "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rs[0].keys()))
            w.writeheader()
            w.writerows(rs)
        print("%-40s %4d trials" % (name, len(rs)))

    # The worklist for whoever scores the doubtful trials by hand: where each one is in
    # its own recording, and what the scorer could not see past.  Written even when it is
    # empty, so "no file" never has to be told apart from "not run yet".
    MR = (M.get("manual_review") or {}).get("trials", [])
    name = "trials_to_score_by_hand.csv"
    cols = ["session_name", "role", "trial_type", "session_trial", "block",
            "global_trial", "at", "at_s", "session_clock_s", "scored_class",
            "scored_onset_ms", "because"]
    with open(os.path.join(odir, name), "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(MR)
    print("%-40s %4d trials" % (name, len(MR)))

    n = 0
    with open(os.path.join(odir, "stimulus_events.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["session", "session_name", "role", "stimulus", "onset_frame", "onset_s",
                    "duration_ms", "accepted", "rejected_because"])
        for rec in cfg["recordings"]:
            p = os.path.join(wdir, rec["tag"] + "_stim.json")
            if not os.path.exists(p):
                continue
            with open(p, encoding="utf-8") as fh:
                S = json.load(fh)
            ev = []
            for key, nm in (("yellow", "CS (yellow LED)"), ("blue", "US (blue LED)")):
                for e in S["events"].get(key, []):
                    ev.append((e["t"], nm, e))
            for t, nm, e in sorted(ev):
                w.writerow([rec["tag"], rec["label"], rec["role"], nm, e["frame"],
                            round(e["t"], 4), e["dur_ms"], "yes" if e["ok"] else "no",
                            e.get("reason", "") or ("" if e["ok"] else "duration off-spec")])
                n += 1
    print("%-40s %4d events" % ("stimulus_events.csv", n))

    with open(os.path.join(odir, "closure_traces_all.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["role", "trial_type", "block", "session", "session_name", "trial",
                    "time_ms_from_onset", "pct_closure"])
        for r in rows:
            T = M["traces"][r["session"]][str(r["session_trial"])]
            for t_, c_ in zip(T["t"], T["C"]):
                w.writerow([r["role"], r["trial_type"], r["block"], r["session"],
                            r["session_name"], r["session_trial"], t_, round(c_ * 100, 2)])
    print("%-40s %4d trials" % ("closure_traces_all.csv", len(rows)))


if __name__ == "__main__":
    main()
