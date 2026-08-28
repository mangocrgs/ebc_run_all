"""Run the whole eyeblink-conditioning analysis for one participant, end to end.

    python ebc_run_all.py --config studies/thomas.json
    python ebc_run_all.py --videos "D:/EBC/Video/Alice"      # no config: discover by name
    python ebc_run_all.py --config studies/thomas.json --from score
    python ebc_run_all.py --config studies/thomas.json --force

Stages, in order:

    locate      find the stimulator box in every recording          (one pass per video)
    stimulus    read both LEDs at full rate and detect every pulse   (one pass per video)
    triage      judge each CS channel; anchor on the US, or exclude, where it failed
    protocol    turn pulses into trials and check the block structure
    eyes        track the eyelids in a window around every trial
    score       one pooled closure scale, blink metrics, response classes
    report      figures, CSVs, Excel workbooks and the LED check pages

Everything is cached in <out>/_work, so a stage is skipped when its output is already
there.  --force redoes the video work; --from <stage> restarts from one point.
"""
import os
import sys
import time
import argparse
import subprocess

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ebc_config as C
from ebc_paths import BASE, work_dir, out_dir

PY = sys.executable
STAGES = ["locate", "stimulus", "protocol", "eyes", "score", "report"]


def run(script, *args, **kw):
    cmd = [PY, os.path.join(BASE, script)] + [str(a) for a in args]
    print("\n>>> %s %s" % (script, " ".join(str(a) for a in args)), flush=True)
    r = subprocess.run(cmd)
    if r.returncode != 0 and not kw.get("allow_fail"):
        sys.exit("!! %s failed (exit %d)" % (script, r.returncode))
    return r.returncode


def run_parallel(script, cfg_path, tags, jobs, log_dir):
    """Per-recording stages are independent; run a few at a time."""
    queue = list(tags)
    live = []
    while queue or live:
        while queue and len(live) < jobs:
            tag = queue.pop(0)
            lf = open(os.path.join(log_dir, "%s_%s.log" % (tag, script.split(".")[0])), "w")
            p = subprocess.Popen([PY, os.path.join(BASE, script), cfg_path, tag],
                                 stdout=lf, stderr=subprocess.STDOUT)
            live.append((tag, p, lf))
            print("    started %-14s %s" % (tag, script), flush=True)
        time.sleep(2.0)
        for item in list(live):
            tag, p, lf = item
            if p.poll() is None:
                continue
            lf.close()
            live.remove(item)
            path = os.path.join(log_dir, "%s_%s.log" % (tag, script.split(".")[0]))
            txt = open(path, encoding="utf-8", errors="ignore").read()
            for line in txt.splitlines():
                if line.strip():
                    print("    " + line, flush=True)
            if p.returncode != 0:
                sys.exit("!! %s %s failed (exit %d) - see %s" % (script, tag, p.returncode, path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config")
    ap.add_argument("--videos", help="folder of recordings; roles are guessed from the names")
    ap.add_argument("--study", help="participant / study name used in the output file names")
    ap.add_argument("--from", dest="from_stage", choices=STAGES, default="locate")
    ap.add_argument("--only", choices=STAGES)
    ap.add_argument("--force", action="store_true", help="redo the video passes from scratch")
    ap.add_argument("--jobs", type=int, default=3,
                    help="recordings to process at once (decoding is the bottleneck)")
    a = ap.parse_args()

    cfg = C.load(a.config, video_dir=a.videos, study=a.study)
    wdir, odir = work_dir(cfg), out_dir(cfg)
    cfg_path = a.config
    if not cfg_path:                       # discovery mode: freeze what was found
        cfg_path = os.path.join(wdir, "run_config.json")
        import json
        with open(cfg_path, "w", encoding="utf-8") as fh:
            json.dump({k: v for k, v in cfg.items() if k != "recordings"} |
                      {"recordings": [{k: v for k, v in r.items() if k != "path"}
                                      for r in cfg["recordings"]]}, fh, indent=1)
        print("wrote " + cfg_path)

    present = [r for r in cfg["recordings"] if os.path.exists(r["path"])]
    missing = [r["file"] for r in cfg["recordings"] if not os.path.exists(r["path"])]
    print("study %s   %d recording(s)   out: %s" % (cfg["study"], len(present), odir))
    for r in present:
        print("   %-14s %-13s %s" % (r["tag"], r["role"], r["file"]))
    if missing:
        print("   missing, skipped: " + ", ".join(missing))
    if not present:
        sys.exit("no recordings found in " + cfg["video_dir"])

    logs = os.path.join(wdir, "logs")
    os.makedirs(logs, exist_ok=True)
    tags = [r["tag"] for r in present]

    if a.force:
        for t in tags:
            for suf in ("_survey.npz", "_led.npz", "_stim.json", "_facebox.json", "_traces.json"):
                p = os.path.join(wdir, t + suf)
                if os.path.exists(p):
                    os.remove(p)

    todo = STAGES[STAGES.index(a.from_stage):] if not a.only else [a.only]

    if "locate" in todo:
        run("ebc_locate.py", cfg_path)
    if "stimulus" in todo:
        print("\n>>> ebc_stimulus.py  (%d recordings, %d at a time)" % (len(tags), a.jobs))
        run_parallel("ebc_stimulus.py", cfg_path, tags, a.jobs, logs)
    # Between reading the LEDs and building trials, decide what each recording can
    # actually support.  Everything downstream runs on the effective config this writes.
    eff = os.path.join(wdir, "effective_config.json")
    if "protocol" in todo:
        run("ebc_triage.py", cfg_path)
    if os.path.exists(eff):
        cfg_path = eff
        # triage may have dropped a recording it cannot score; nothing downstream should
        # still be asked to process it
        cfg = C.load(cfg_path)
        kept = {r["tag"] for r in cfg["recordings"]}
        dropped = [t for t in tags if t not in kept]
        if dropped:
            print("   left out by triage: " + ", ".join(dropped))
        tags = [t for t in tags if t in kept]

    if "protocol" in todo:
        run("ebc_protocol.py", cfg_path)
    if "eyes" in todo:
        print("\n>>> ebc_eyes.py  (%d recordings, %d at a time)" % (len(tags), a.jobs))
        run_parallel("ebc_eyes.py", cfg_path, tags, a.jobs, logs)
    if "score" in todo:
        run("ebc_score.py", cfg_path)
    if "report" in todo:
        run("ebc_figures.py", cfg_path)
        run("ebc_export_csv.py", cfg_path)
        run("ebc_workbooks.py", cfg_path)
        run("ebc_qc.py", cfg_path, "leds")

    print("\nDone. Workbooks, figures, CSVs and the LED check pages are in:\n  %s" % odir)


if __name__ == "__main__":
    main()
