"""Where the intermediates and the results of a study live.

Nothing is written next to the scripts, so one checkout can process any number of
participants:  <out_dir>/ holds the workbooks, figures and CSVs, <out_dir>/_work/ the
cache (safe to delete, costs a re-run).
"""
import os
import sys

# Where the pipeline's own files are: the scripts, the page, the logo.  Packaged into an
# .exe they are unpacked somewhere temporary instead, and _MEIPASS is where.  Nothing a
# run produces is written here - that all goes to out_dir, next to the recordings - so
# the app is happy installed read-only.
BASE = getattr(sys, "_MEIPASS", None) or os.path.dirname(os.path.abspath(__file__))


def out_dir(cfg):
    os.makedirs(cfg["out_dir"], exist_ok=True)
    return cfg["out_dir"]


def work_dir(cfg):
    w = os.path.join(cfg["out_dir"], "_work")
    os.makedirs(w, exist_ok=True)
    return w
