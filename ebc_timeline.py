"""Put the recordings back in the order they were made, and say what is wrong with them.

    python ebc_timeline.py <config.json>

This runs before anything decodes a frame.  It reads what the camera wrote into each
file (ebc_media.py), and from that answers the questions the file names cannot:

  * **In what order were these recorded?**  Not the order they sort in.  `CSUS fin`
    sorts as `CSUS 1` on a numeric key, and it is in fact the second chapter of the
    *extinction* take - recorded after extinction, not after `CSUS 3`.  The conditioning
    chapters are concatenated onto one clock downstream, so getting this wrong shifts
    every block boundary after it.
  * **Which files are one recording?**  A camera splits at 4 GB.  `csus1 - failed`,
    `csus 2` and `csus 3` are chapters 1, 2 and 3 of one unbroken 25-minute take, not
    three sessions - and the first of them is not "failed" in any sense the recording
    knows about.
  * **Which files are the same clip twice?**  A re-encode under a new name doubles every
    trial in it if both are ticked.
  * **What will stall or mislead the run?**  A cloud-only file that has not been
    downloaded, an empty file, a damaged one, a folder holding two sessions, a camera
    whose frame rate changed half way through the participant.

Nothing is renamed and nothing is silently dropped.  What the stage decides is written
into `<out>/_work/ordered_config.json`, which the rest of the pipeline runs on, and every
exclusion carries the reason it was excluded, in the report and in the study file.

A recording listed by hand in a study file is treated as a deliberate choice: it is
warned about, never dropped.  Only what folder discovery guessed at can be dropped.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ebc_config as C
import ebc_media as M
from ebc_paths import work_dir

BANNER = "=" * 78

# recordings this far apart in one folder are probably not one session
SESSION_SPAN_H = 8.0


def _fmt_clock(sec):
    if sec is None:
        return "-"
    return "%d:%02d:%02d" % (int(sec) // 3600, int(sec) % 3600 // 60, int(sec) % 60)


def examine(cfg, meta, rows, dups):
    """One verdict per recording: keep it, or leave it out, and why."""
    by_file = {r["file"]: r for r in rows}
    copy_of = {}
    for d in dups:
        for c in d["copies"]:
            copy_of[c] = d

    out = []
    for rec in cfg["recordings"]:
        fn = rec["file"]
        m = meta.get(fn) or {}
        row = by_file.get(fn, {})
        guessed = rec.get("from") == "folder scan"
        notes, drop = [], None

        if not os.path.exists(rec["path"]):
            drop = "the file is not in the folder"
        elif m.get("error"):
            drop = m["error"]
        elif fn in copy_of:
            drop = ("the same clip as %s, under another name (%s).  Scoring both would "
                    "count every trial in it twice"
                    % (copy_of[fn]["keep"], copy_of[fn]["why"]))
        else:
            why = M.flagged(os.path.splitext(fn)[0])
            if why and guessed:
                drop = why + ", so folder discovery left it out.  Add it to the study " \
                             "file if you want it analysed"
            elif why:
                notes.append(why + ", but it is listed in the study file, so it is kept")
            d = M.derived(meta, fn)
            if d:
                notes.append(d)
                if guessed and not drop:
                    drop = d
        if m.get("offline"):
            notes.append("this file is in the cloud and has not been downloaded - the run "
                         "will stall on it until OneDrive fetches it.  Right-click it and "
                         "choose 'Always keep on this device' first")
        if row.get("n_chapters", 1) > 1:
            notes.append("chapter %d of %d of one continuous recording"
                         % (row["chapter"], row["n_chapters"]))

        out.append(dict(rec=rec, meta=m, row=row, notes=notes, drop=drop))
    return out


def reorder(kept, rows):
    """Renumber `order` per role by when each recording was actually made."""
    rank = {r["file"]: r["rank"] for r in rows}
    kept.sort(key=lambda e: rank.get(e["rec"]["file"], 10 ** 6))
    seen = {r: 0 for r in C.ROLES}
    changed = []
    for e in kept:
        rec, row = e["rec"], e["row"]
        seen[rec["role"]] += 1
        was = rec.get("order")
        rec["order"] = seen[rec["role"]]
        rec["recorded_at"] = (e["meta"].get("created") or "")[:19] or None
        rec["elapsed_s"] = row.get("elapsed_s")
        rec["chapter"] = row.get("chapter")
        rec["n_chapters"] = row.get("n_chapters")
        rec["continues_previous"] = row.get("continues_previous")
        rec["order_source"] = row.get("order_source")
        if was is not None and was != rec["order"]:
            changed.append((rec["file"], rec["role"], was, rec["order"]))
    return changed


def crosschecks(kept, rows):
    """What is odd about this set of recordings taken as a whole."""
    out = []
    fps = {round(e["meta"].get("fps") or 0, 2) for e in kept if e["meta"].get("fps")}
    if len(fps) > 1:
        out.append("the recordings are not all at the same frame rate (%s) - either the "
                   "camera setting changed mid-session, or this folder holds recordings "
                   "from more than one session"
                   % ", ".join("%.2f fps" % f for f in sorted(fps)))
    size = {(e["meta"].get("width"), e["meta"].get("height")) for e in kept
            if e["meta"].get("width")}
    if len(size) > 1:
        out.append("the recordings are not all the same frame size (%s) - the stimulator "
                   "box is located in pixels, so a study-level box position cannot be "
                   "shared across them"
                   % ", ".join("%dx%d" % s for s in sorted(size)))
    rot = {e["meta"].get("rotation", 0) for e in kept}
    if len(rot) > 1:
        out.append("some recordings carry a rotation flag and others do not (%s degrees) - "
                   "check the LED check pages, the frames may not be the same way up"
                   % ", ".join(str(r) for r in sorted(rot)))
    dated = [e for e in kept if e["meta"].get("created_epoch")]
    if len(dated) > 1:
        span = (max(e["meta"]["created_epoch"] for e in dated)
                - min(e["meta"]["created_epoch"] for e in dated)) / 3600.0
        if span > SESSION_SPAN_H:
            out.append("these recordings span %.1f hours (%s to %s) - one participant's "
                       "session does not, so this folder probably holds more than one "
                       % (span, min(e["meta"]["created"] for e in dated)[:16],
                          max(e["meta"]["created"] for e in dated)[:16])
                       + "session, or a stray file from another day")
    if not dated:
        out.append("no recording carries a camera date, so the order below is the order "
                   "of the file dates on disk, which a copy or a sync can change")
    return out


def takes(kept, dropped, rows):
    """The takes, in order, with what each chapter was labelled as.

    A chapter that is not being analysed still appears here.  A take whose first
    chapters were left out is not the same thing as a take that started when its first
    analysed chapter did, and the gap has to be visible.
    """
    role_of = {e["rec"]["file"]: (e["rec"]["role"], False) for e in kept}
    role_of.update({e["rec"]["file"]: (e["rec"]["role"], True) for e in dropped})
    out, seen = [], []
    for r in rows:
        if r["file"] not in role_of:
            continue
        key = tuple(r["take"])
        if key not in seen:
            seen.append(key)
            out.append([])
        out[seen.index(key)].append(dict(file=r["file"], chapter=r["chapter"],
                                         role=role_of[r["file"]][0],
                                         left_out=role_of[r["file"]][1],
                                         elapsed_s=r.get("elapsed_s"),
                                         duration_s=r.get("duration_s")))
    return out


def missing_chapters(kept, groups):
    """Note, per recording, when the chapters before it in its take are not being read."""
    notes = 0
    for g in groups:
        gone = [c for c in g if c["left_out"]]
        if not gone:
            continue
        for e in kept:
            here = next((c for c in g if c["file"] == e["rec"]["file"]), None)
            if not here:
                continue
            before = [c for c in gone if c["chapter"] < here["chapter"]]
            if before:
                e["notes"].append(
                    "the earlier chapter(s) of this continuous recording are not being "
                    "analysed (%s), so what is scored starts %s into the take"
                    % (", ".join(c["file"] for c in before),
                       _fmt_clock(sum(c["duration_s"] or 0 for c in before))))
                notes += 1
    return notes


def run(cfg, cfg_path):
    wdir = work_dir(cfg)
    files = [r["file"] for r in cfg["recordings"]]
    paths = [r["path"] for r in cfg["recordings"]]
    print("reading what the camera wrote into %d file(s)..." % len(paths), flush=True)
    t0 = time.time()
    meta = M.load(paths, cache=os.path.join(wdir, "media.json"))
    rows = M.timeline(meta, files)
    dups = M.duplicates(meta, files)
    print("   %.1fs" % (time.time() - t0))

    ex = examine(cfg, meta, rows, dups)
    kept = [e for e in ex if not e["drop"]]
    dropped = [e for e in ex if e["drop"]]
    if not kept:                       # never leave a run with nothing to do
        print("\n!! every recording would have been left out; keeping them all instead")
        for e in dropped:
            e["notes"].append("would have been left out: " + e["drop"])
            e["drop"] = None
        kept, dropped = ex, []

    changed = reorder(kept, rows)
    groups = takes(kept, dropped, rows)
    missing_chapters(kept, groups)
    checks = crosschecks(kept, rows)

    # ---- report ---------------------------------------------------------------
    print("\n%-28s %-13s %5s %9s %8s  %s"
          % ("recording", "role", "order", "recorded", "start", "as recorded"))
    for e in kept:
        rec, row = e["rec"], e["row"]
        print("%-28s %-13s %5d %9s %8s  %s"
              % (rec["file"][:28], rec["role"], rec["order"],
                 (rec.get("recorded_at") or "-")[11:19],
                 _fmt_clock(rec.get("elapsed_s")),
                 "chapter %d/%d%s" % (row.get("chapter", 1), row.get("n_chapters", 1),
                                      ", continues the previous" if row.get("continues_previous")
                                      else "") if row.get("n_chapters", 1) > 1 else ""))

    if any(len(g) > 1 for g in groups):
        print("\nContinuous recordings (one take, split by the camera at 4 GB):")
        for g in groups:
            if len(g) < 2:
                continue
            print("  " + "  ->  ".join(
                "%s [%s]" % (c["file"], "LEFT OUT" if c["left_out"] else c["role"]) for c in g))
            roles = {c["role"] for c in g if not c["left_out"]}
            if len(roles) > 1:
                print("     these chapters were labelled with different roles.  That is "
                      "normal when the camera ran through the end of one condition into")
                print("     the next, but it is worth a look: they are one unbroken "
                      "recording, so the labels are a judgement, not a fact.")

    if changed:
        print("\n" + BANNER)
        print("ORDER CORRECTED - the file names did not match the recording times.")
        print(BANNER)
        for fn, role, was, now in changed:
            print("  %-28s %-13s was #%d by name, is #%d by the clock" % (fn, role, was, now))
        print("""
  The conditioning chapters are laid end to end on one clock, so their order sets
  every block boundary.  The order used from here on is the one the camera recorded,
  not the one the names imply.""")
        print(BANNER)

    if dropped:
        print("\n" + BANNER)
        print("LEFT OUT - %d file(s) will not be analysed." % len(dropped))
        print(BANNER)
        for e in dropped:
            print("\n  %s" % e["rec"]["file"])
            print("      %s" % e["drop"])
        print(BANNER)

    notes = [(e["rec"]["file"], n) for e in kept for n in e["notes"]
             if not n.startswith("chapter ")]
    if notes:
        print("\n" + BANNER)
        print("WORTH KNOWING")
        print(BANNER)
        for fn, n in notes:
            print("  %-28s %s" % (fn[:28], n))
        print(BANNER)

    if checks:
        print("\n" + BANNER)
        print("THIS SET OF RECORDINGS")
        print(BANNER)
        for c in checks:
            print("  - " + c)
        print(BANNER)

    unknown = [u for u in C.unrecognised(cfg["video_dir"]) if u not in files]
    if unknown:
        meta.update(M.load([os.path.join(cfg["video_dir"], u) for u in unknown],
                           cache=os.path.join(wdir, "media.json")))
        print("\nin the folder but not analysed - no role can be guessed from these names:")
        for u in unknown:
            print("   %-28s %s" % (u, (meta.get(u, {}).get("created") or "")[:19]))
        print("   if one of them is a session, tick it in the app or add it to the study file")

    # ---- what the rest of the pipeline will run on ------------------------------
    out = dict(cfg)
    out["recordings"] = [{k: v for k, v in e["rec"].items() if k != "path"} for e in kept]
    out["excluded"] = out.get("excluded", []) + [
        dict({k: v for k, v in e["rec"].items() if k != "path"},
             include=False, excluded_because=e["drop"]) for e in dropped]
    ordered = os.path.join(wdir, "ordered_config.json")
    with open(ordered, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    with open(os.path.join(wdir, "timeline.json"), "w", encoding="utf-8") as fh:
        json.dump(dict(study=cfg["study"], source_config=cfg_path,
                       recordings=[dict(e["row"], tag=e["rec"]["tag"], file=e["rec"]["file"],
                                        role=e["rec"]["role"], order=e["rec"]["order"],
                                        notes=e["notes"]) for e in kept],
                       excluded=[dict(file=e["rec"]["file"], reason=e["drop"]) for e in dropped],
                       order_changed=[dict(file=f, role=r, was=w, now=n)
                                      for f, r, w, n in changed],
                       takes=groups, checks=checks, duplicates=dups,
                       unrecognised=unknown), fh, indent=1)
    print("\nwrote " + ordered)
    return ordered


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else None
    cfg = C.load(path)
    run(cfg, path)


if __name__ == "__main__":
    main()
