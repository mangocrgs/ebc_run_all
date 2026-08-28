"""Where the intermediates and the results of a study live.

Nothing is written next to the scripts, so one checkout can process any number of
participants:  <out_dir>/ holds the workbooks, figures and CSVs, <out_dir>/_work/ the
cache (safe to delete, costs a re-run).
"""
import os

BASE = os.path.dirname(os.path.abspath(__file__))


def out_dir(cfg):
    os.makedirs(cfg["out_dir"], exist_ok=True)
    return cfg["out_dir"]


def work_dir(cfg):
    w = os.path.join(cfg["out_dir"], "_work")
    os.makedirs(w, exist_ok=True)
    return w
