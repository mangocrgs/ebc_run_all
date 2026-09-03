"""What the video files themselves say about when they were recorded, and in what order.

    python ebc_media.py <folder>        print the timeline of a folder

The file names are what a person typed at the end of a long session, so they are the
least reliable thing in the folder.  The recordings themselves carry three facts that
were written by the camera and cannot be mistyped:

  creation_time   when the camera started this *recording*.  Every chapter of one
                  continuous recording carries the same value, so it names the take.
  timecode        the camera's own clock at the first frame (tmcd track).  It is
                  monotonic within a take and separates its chapters, which
                  creation_time cannot.  It resets when the camera loses power, so it
                  orders chapters, never takes.
  duration        with the timecode, this says whether two files are contiguous: a
                  GoPro splits at 4 GB, and chapter n+1 starts where chapter n ended,
                  to within the rounding of the timecode seconds field.

From those three, this module recovers what actually happened:

  * which files are chapters of one continuous recording, and in which order;
  * the order the takes were run in, whatever the names say;
  * which files are copies, transcodes or re-exports of the same clip.

Everything else in the pipeline consumes that instead of trusting `CSUS 3` to come
after `CSUS 2`.  Real examples from this study that the names got wrong:

  Carole   `CSUS fin` is chapter 2 of the take whose chapter 1 is `extinction` - it
           was recorded *after* extinction, not after `CSUS 3`.
  Marie    same shape: `CSUS 4` is the tail of the `extinction` take.
  Marie    `CSUS 2 test.mp4` is a re-encode of `CSUS 2.MP4` - same clip, ticking both
           doubles every trial in it.
  Charles  `csus1 - failed`, `csus 2` and `csus 3` are chapters 1, 2 and 3 of one
           unbroken 25-minute recording, so "csus1" and "csus2" are not two sessions.
"""
import json
import os
import re
import subprocess
import sys

# two chapters of one recording meet within a second; the timecode seconds field is
# truncated, so half a second of slack in each direction is normal
CHAPTER_GAP_S = 2.0

# names people give to a recording they do not want analysed, in English and French
FLAGGED = [
    (r"\bfail(ed|ure)?\b|\bfoire|rat[ée]e?\b|\b[ée]chec\b", "the name says it failed"),
    (r"\btests?\b|\bessai\b|\btrial run\b", "the name says it is a test"),
    (r"raccourci|shortcut|extrait|\btrim(med)?\b|\bcut\b|\bcourt\b", "the name says it is a cut-down copy"),
    (r"\bcopy\b|\bcopie\b|\bbis\b|\(\d+\)\s*$|\bold\b|\bbak\b|\bancien", "the name says it is a copy"),
]

# GoPro puts the chapter first and the recording second:  GX 01 2912 .MP4
RE_GOPRO = re.compile(r"^(G[XHLP])(\d{2})(\d{4})$", re.I)


def _ffprobe(path):
    q = ["ffprobe", "-v", "error", "-of", "json", "-show_entries",
         "format=duration,size:format_tags=creation_time:"
         "stream=index,codec_type,width,height,r_frame_rate,nb_frames:"
         "stream_tags=timecode:stream_side_data=rotation", path]
    r = subprocess.run(q, capture_output=True, text=True)
    try:
        return json.loads(r.stdout or "{}")
    except ValueError:
        return {}


def _epoch(iso):
    """ISO-8601 UTC as a number of seconds, without dragging in a timezone database."""
    if not iso:
        return None
    import calendar
    import time as _t
    try:
        return calendar.timegm(_t.strptime(iso[:19], "%Y-%m-%dT%H:%M:%S"))
    except ValueError:
        return None


def _timecode_s(tc, fps):
    """HH:MM:SS:FF on the camera clock, in seconds."""
    if not tc:
        return None
    p = tc.split(":")
    if len(p) < 3 or not all(x.strip().isdigit() for x in p[:3]):
        return None
    s = int(p[0]) * 3600 + int(p[1]) * 60 + int(p[2])
    if len(p) > 3 and p[3].isdigit() and fps:
        s += int(p[3]) / fps
    return float(s)


def offline(path):
    """True when Windows is holding the file in the cloud and has not downloaded it.

    A OneDrive placeholder opens fine and then stalls for as long as the download takes,
    which on a 4 GB recording looks exactly like a hung pipeline.  Better to say so.
    """
    try:
        att = os.stat(path).st_file_attributes           # Windows only
    except (AttributeError, OSError):
        return False
    # FILE_ATTRIBUTE_OFFLINE | _RECALL_ON_OPEN | _RECALL_ON_DATA_ACCESS
    return bool(att & (0x1000 | 0x40000 | 0x400000))


def probe_file(path):
    """Everything the pipeline wants to know about one file before it decodes it."""
    st = os.stat(path)
    m = dict(file=os.path.basename(path), size=st.st_size, mtime=st.st_mtime,
             offline=offline(path), duration_s=None, fps=None, n_frames=None,
             width=None, height=None, rotation=0, created=None, created_epoch=None,
             timecode=None, timecode_s=None, video=False, error=None)
    if st.st_size == 0:
        m["error"] = "the file is empty (0 bytes)"
        return m
    j = _ffprobe(path)
    fmt = j.get("format") or {}
    streams = j.get("streams") or []
    vid = next((s for s in streams if s.get("codec_type") == "video"), None)
    if not vid:
        m["error"] = "no video stream - not a recording, or the file is damaged"
        return m
    m["video"] = True
    try:
        m["duration_s"] = round(float(fmt.get("duration")), 3)
    except (TypeError, ValueError):
        pass
    num, _, den = (vid.get("r_frame_rate") or "0/1").partition("/")
    try:
        m["fps"] = round(float(num) / float(den or 1), 4) or None
    except (ValueError, ZeroDivisionError):
        pass
    if str(vid.get("nb_frames", "")).isdigit():
        m["n_frames"] = int(vid["nb_frames"])
    m["width"], m["height"] = vid.get("width"), vid.get("height")
    for sd in vid.get("side_data_list") or []:
        if "rotation" in sd:
            m["rotation"] = int(sd["rotation"]) % 360
    m["created"] = (fmt.get("tags") or {}).get("creation_time")
    m["created_epoch"] = _epoch(m["created"])
    for s in streams:
        tc = (s.get("tags") or {}).get("timecode")
        if tc:
            m["timecode"] = tc
            m["timecode_s"] = _timecode_s(tc, m["fps"])
            break
    if m["duration_s"] is None and m["n_frames"] and m["fps"]:
        m["duration_s"] = round(m["n_frames"] / m["fps"], 3)
    return m


def load(paths, cache=None):
    """probe_file over a list of paths, cached on (size, mtime) so a re-run is instant."""
    old = {}
    if cache and os.path.exists(cache):
        try:
            with open(cache, encoding="utf-8") as fh:
                old = json.load(fh)
        except ValueError:
            old = {}
    out = {}
    for p in paths:
        key = os.path.basename(p)
        try:
            st = os.stat(p)
        except OSError as e:
            out[key] = dict(file=key, error=str(e), video=False, size=0, mtime=0,
                            offline=False, duration_s=None, fps=None)
            continue
        prev = old.get(key)
        if prev and prev.get("size") == st.st_size and abs(prev.get("mtime", 0) - st.st_mtime) < 1:
            out[key] = prev
        else:
            out[key] = probe_file(p)
    if cache:
        # merge, never replace: load() is called more than once per run (the recordings,
        # then the files whose names imply no role), and writing only what this call
        # asked about threw away the entry for every real recording - so the next run
        # re-probed the whole folder for nothing.
        with open(cache, "w", encoding="utf-8") as fh:
            json.dump(dict(old, **out), fh, indent=1)
    return out


# ------------------------------------------------------------------ takes and chapters
def flagged(stem):
    """What the person who named this file was trying to tell us, if anything."""
    for pat, why in FLAGGED:
        if re.search(pat, stem, re.I):
            return why
    return None


def gopro_parts(stem):
    """(recording number, chapter number) for an untouched GoPro name, else None."""
    m = RE_GOPRO.match(stem.strip())
    return (m.group(3), int(m.group(2))) if m else None


def _take_key(m, name):
    """What identifies the take a file belongs to.

    creation_time is the camera's own name for the recording and every chapter shares
    it.  An untouched GoPro name carries the recording number, which survives a copy
    that lost its metadata.  Failing both, the file is its own take.
    """
    if m.get("created_epoch") is not None:
        return ("t", m["created_epoch"])
    g = gopro_parts(os.path.splitext(name)[0])
    if g:
        return ("g", g[0])
    return ("f", name)


def timeline(meta, names=None):
    """Group files into takes, order the chapters inside each, and order the takes.

    Returns one row per file, in the order it was recorded, each carrying its take,
    its chapter number within that take, whether it runs on continuously from the
    previous chapter, and how confident that placement is.
    """
    names = list(names if names is not None else meta.keys())
    takes = {}
    for n in names:
        m = meta.get(n) or {}
        takes.setdefault(_take_key(m, n), []).append(n)

    rows = []
    for key, group in takes.items():
        def chap_key(n):
            m = meta.get(n) or {}
            g = gopro_parts(os.path.splitext(n)[0])
            # inside a take the camera clock is the truth; a GoPro chapter number is
            # the same fact under a different name; a file date is a guess
            return (0, m["timecode_s"]) if m.get("timecode_s") is not None else \
                   ((1, g[1]) if g else (2, (m.get("mtime") or 0)))
        group.sort(key=chap_key)
        prev_end = None
        for i, n in enumerate(group, 1):
            m = meta.get(n) or {}
            tc, dur = m.get("timecode_s"), m.get("duration_s")
            cont = None
            if tc is not None and prev_end is not None:
                cont = abs(tc - prev_end) <= CHAPTER_GAP_S
            rows.append(dict(file=n, take=list(key), chapter=i, n_chapters=len(group),
                             continues_previous=cont,
                             take_start=(m.get("created_epoch") if key[0] == "t"
                                         else m.get("mtime")),
                             timecode_s=tc, duration_s=dur,
                             order_source=("camera clock" if tc is not None else
                                           ("GoPro chapter number" if gopro_parts(os.path.splitext(n)[0])
                                            else "file date")),
                             dated=key[0] == "t"))
            if tc is not None and dur is not None:
                prev_end = tc + dur
            else:
                prev_end = None

    # a file the camera dated is placed on the session clock; one that carries no camera
    # metadata at all cannot be, so it goes at the end, ordered among its own kind by
    # file date and marked as unplaceable rather than silently slotted in somewhere
    rows.sort(key=lambda r: (not r["dated"], r["take_start"] or 0, r["chapter"]))
    # Elapsed seconds since the first recording of the session, so a stage that
    # concatenates chapters puts the real gap between them instead of butting them up.
    # Between takes only creation_time is comparable; inside a take the timecode gives
    # the exact start of each chapter, so the two are combined.
    dated = [r for r in rows if r["dated"]]
    t0 = dated[0]["take_start"] if dated else None
    for i, r in enumerate(rows, 1):
        r["rank"] = i
        if t0 is None or not r["dated"] or r["take_start"] is None:
            r["elapsed_s"] = None
            continue
        first_tc = next((q["timecode_s"] for q in rows
                         if q["take"] == r["take"] and q["chapter"] == 1), None)
        into = 0.0
        if first_tc is not None and r["timecode_s"] is not None:
            into = max(0.0, r["timecode_s"] - first_tc)
        r["elapsed_s"] = round(r["take_start"] - t0 + into, 3)
    return rows


def duplicates(meta, names=None):
    """Files that are the same clip twice, under two names.

    The camera stamps each clip with a recording time and a start timecode; two files
    carrying both the same are the same clip, however they were renamed.  The larger
    file is the one to keep - a smaller one of the same clip has been re-encoded, and
    re-encoding is what flattens the LED contrast the whole pipeline depends on.

    Nothing weaker is used.  Duration was tried and does not work here: every full
    chapter is exactly 531.541 s, so `CSUS 1`, `CSUS 2` and a transcode of either are
    all the same length.  Neither does comparing the pictures - the participant sits
    still in front of a fixed camera, so two different chapters look more alike than a
    clip and its own washed-out re-encode.  A file that has lost its camera metadata is
    reported as derived (below) instead of being matched to a guessed original.
    """
    names = list(names if names is not None else meta.keys())
    groups = {}
    for n in names:
        m = meta.get(n) or {}
        if not m.get("video") or m.get("created_epoch") is None or m.get("timecode_s") is None:
            continue
        groups.setdefault((m["created_epoch"], round(m["timecode_s"], 1)), []).append(n)
    out = []
    for g in groups.values():
        if len(g) < 2:
            continue
        g.sort(key=lambda n: -(meta[n].get("size") or 0))
        out.append(dict(keep=g[0], copies=g[1:],
                        why="same recording time and camera timecode - one clip, two names"))
    return out


def derived(meta, name):
    """Why this file cannot be an untouched camera original, or None.

    Every recording off this rig carries a creation time and a timecode track.  A file
    with neither has been through something else - an export, a trim, a re-encode - and
    two things follow: it cannot be placed on the session clock, and its LED contrast
    may have been flattened by the re-encode.  Both are worth saying out loud.
    """
    m = meta.get(name) or {}
    if not m.get("video"):
        return None
    if m.get("created_epoch") is None and m.get("timecode_s") is None:
        return ("no camera metadata (no recording time, no timecode) - an export or a "
                "re-encode rather than the original, and it cannot be placed in the session")
    return None


def describe_take(rows, take):
    part = [r for r in rows if r["take"] == take]
    return "%d chapter(s): %s" % (len(part), " -> ".join(r["file"] for r in part))


def main():
    folder = sys.argv[1] if len(sys.argv) > 1 else "."
    ext = (".mp4", ".mov", ".mkv", ".avi", ".m4v", ".mts")
    names = [f for f in sorted(os.listdir(folder))
             if f.lower().endswith(ext) and not f.startswith("~")]
    meta = load([os.path.join(folder, f) for f in names])
    rows = timeline(meta, names)
    print("%-34s %5s %9s %9s  %s" % ("file", "chap", "start (s)", "dur (s)", "recorded"))
    last = None
    for r in rows:
        m = meta[r["file"]]
        if last is not None and r["take"] != last:
            print("   %s" % ("-" * 60))
        last = r["take"]
        print("%-34s %2d/%-2d %9s %9s  %s%s"
              % (r["file"], r["chapter"], r["n_chapters"],
                 "-" if r["elapsed_s"] is None else "%.0f" % r["elapsed_s"],
                 "-" if r["duration_s"] is None else "%.1f" % r["duration_s"],
                 (m.get("created") or "date unknown")[:19],
                 "  continues the previous chapter" if r["continues_previous"] else ""))
    for d in duplicates(meta, names):
        print("\nsame clip twice: %s\n  keep %s, the other(s) are copies: %s"
              % (d["why"], d["keep"], ", ".join(d["copies"])))
    for n in names:
        for why in (flagged(os.path.splitext(n)[0]), derived(meta, n)):
            if why:
                print("\n%s: %s" % (n, why))


if __name__ == "__main__":
    main()
