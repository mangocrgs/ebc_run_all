"""Run the whole eyeblink-conditioning analysis, end to end.

    python ebc_run_all.py                 # everything (skips videos already processed)
    python ebc_run_all.py --force         # re-process the videos from scratch
    python ebc_run_all.py --score-only    # skip the videos, just re-score and rebuild

Processing a 4 GB / 531 s recording takes roughly 12 minutes; the three conditioning
chapters plus CSUS 4 take about 40 minutes in total. Results are cached in
analysis_CSUS/_work, so --score-only reruns everything downstream in seconds.
"""
import os
import subprocess
import sys

from ebc_paths import BASE, OUT, WORK, video

# tag -> (file name, label).  Add a recording here and it joins the pipeline.
RECORDINGS = [
    ("csus1", "CSUS 1.MP4", "CSUS 1"),
    ("csus2", "CSUS 2.MP4", "CSUS 2"),
    ("csus3", "CSUS3.MP4", "CSUS 3"),
    ("csus4", "CSUS 4.MP4", "CSUS 4"),
]

FIGURES = [
    ("cond_paired", "Marie — delay eyeblink conditioning  |  90 paired CS–US trials, 10 blocks", "cond"),
    ("cond_csonly", "Marie — CS-only probes during conditioning  |  one per block", "csonly"),
    ("ext", "Marie — extinction  |  CSUS 4, CS-only", "ext"),
]

PY = sys.executable
force = "--force" in sys.argv
score_only = "--score-only" in sys.argv


def run(script, *args):
    cmd = [PY, os.path.join(BASE, script), *[str(a) for a in args]]
    print(f"\n>>> {script} {' '.join(str(a) for a in args)}", flush=True)
    r = subprocess.run(cmd)
    if r.returncode != 0:
        sys.exit(f"!! {script} failed (exit {r.returncode})")


if not score_only:
    for tag, fname, _ in RECORDINGS:
        path = video(fname)
        if not os.path.exists(path):
            print(f"-- {fname} not found, skipping")
            continue
        done = os.path.join(WORK, f"{tag}_result.json")
        if os.path.exists(done) and not force:
            print(f"-- {tag} already processed (delete {tag}_result.json to redo)")
            continue
        if force:
            for suf in ("_blue.npz", "_yellow.npz", "_facebox.json", "_traces.json", "_result.json"):
                p = os.path.join(WORK, tag + suf)
                if os.path.exists(p):
                    os.remove(p)
        run("ebc_pipeline.py", path, tag)

run("ebc_score.py")
for kind, title, tag in FIGURES:
    run("ebc_figures.py", kind, title, tag)
run("ebc_export_csv.py")
run("ebc_workbooks.py")

print(f"\nDone. Workbooks, figures and CSVs are in:\n  {OUT}")
