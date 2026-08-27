"""Shared paths for the eyeblink-conditioning pipeline.

Everything is relative to this file, so the whole folder can be moved or copied
to another machine and still run.

  <this folder>/                     the .MP4 recordings live here
  <this folder>/analysis_CSUS/       final workbooks, figures and CSVs
  <this folder>/analysis_CSUS/_work/ intermediate cache (safe to delete)
"""
import os

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "analysis_CSUS")
WORK = os.path.join(OUT, "_work")
os.makedirs(WORK, exist_ok=True)


def video(name):
    """Absolute path to a recording sitting next to these scripts."""
    return os.path.join(BASE, name)
