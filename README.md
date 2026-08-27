# Eyeblink-conditioning video scoring

Scores delay eyeblink conditioning directly from high-speed video: it finds both
stimulus LEDs in the pixels, aligns every trial to the CS to within one frame, and
measures eyelid aperture with facial landmarks.

Nothing about the timing is assumed — the CS–US interval, the trial structure and the
block boundaries are all recovered from the recording and then checked against the
protocol.

## Run it

```
python ebc_run_all.py                # everything; skips videos already processed
python ebc_run_all.py --force        # re-process the videos from scratch
python ebc_run_all.py --score-only   # skip the videos, just re-score and rebuild
```

A 4 GB / 531 s recording takes ~12 min. All four take ~40 min. Intermediates are cached
in `analysis_CSUS/_work`, so `--score-only` reruns everything downstream in seconds —
that is the flag to use when changing a scoring rule or a figure.

Requires `ffmpeg` on PATH, and `opencv-python mediapipe numpy scipy matplotlib openpyxl pillow`.

## The protocol it expects

| | |
|---|---|
| Block | 9 paired CS–US trials, then 1 CS-only trial |
| Conditioning | 10 blocks |
| CS | yellow LED, 400 ms |
| US | blue LED, 50 ms |
| Overlap | the two co-terminate, so US onset is 350 ms after CS onset |
| Extinction | CS-only trials afterwards (here, the separate CSUS 4 recording) |

Change these in `ebc_score.py` (`CS_DUR`, `US_ONSET`, `US_DUR`) if the protocol differs.
Recordings are listed in `RECORDINGS` in `ebc_run_all.py`; add a line and it joins the run.

## Files

| Script | What it does |
|---|---|
| `ebc_paths.py` | Where everything lives. Paths are relative to this folder, so it can be moved or copied to another machine. |
| `ebc_pipeline.py` | Per-recording. Finds the blue LED and the stimulator box, then the yellow LED, then tracks eyelids in a window around every CS. Writes `<tag>_result.json`. |
| `ebc_score.py` | All recordings together. Filters detections, recovers the block structure, and scores every trial on one pooled closure scale. Writes `merged.json` / `merged_rows.json`. |
| `ebc_figures.py` | `cond_paired` / `cond_csonly` / `ext` — scatter, acquisition curve, rasters. |
| `ebc_export_csv.py` | Trials, stimulus events and full traces as CSV. |
| `ebc_workbooks.py` | The two Excel workbooks. |
| `ebc_qc.py` | `python ebc_qc.py <tag> "<video>" <trial> [...]` — renders an eye filmstrip for a trial with the measured closure printed on each frame. Use it whenever a number looks wrong. |
| `ebc_progress.py` | One-line progress readout while a long run is going. |

## How a trial is scored

1. **Stimulus detection.** Each LED is found as a *transient* against a running
   background, so a static coloured object in the room cannot trigger it, and the two
   LEDs are detected independently rather than one being anchored to the other.
2. **Detection filter.** A genuine CS lasts ~400 ms. Detections outside 330–470 ms, or
   within 6 s of the previous one, are LED flicker and are discarded. Blue transients
   that do not match the US pulse duration go the same way. The check that this is right:
   afterwards the sequence comes out as exactly nine paired trials before each CS-only
   trial, ten times over, with the probes 150.0–150.2 s apart.
3. **Alignment.** Each trial window is cut with the LED *and* the face inside the same
   crop, so the CS onset is re-detected inside every window rather than trusted from the
   seek. Good to one frame — 8.34 ms at 119.88 fps.
4. **Eyelid measure.** MediaPipe FaceMesh (478 landmarks, iris refinement) on a
   2×-upscaled face crop. Eye aspect ratio per eye, averaged. EAR is normalised by eye
   width, so it survives head movement and camera distance.
5. **Closure scale.** 0% = a blink-robust open-eye reference (85th percentile of EAR in
   the window, which survives a baseline blink where a median does not). 100% = a
   full-closure reference **pooled across every recording**, so short blocks stay
   comparable. Smoothed with a 5-frame Savitzky–Golay filter (42 ms).
6. **Blink criterion.** Five robust SDs above the trial's own pre-CS baseline, floor 15%
   closure, then walked back along the falling edge to the true onset. A separate blink
   must re-reach 40% closure after first returning below 20%.
7. **Second look.** If the first event is an alpha blink (<100 ms) or the lid was already
   moving at CS onset, the window is searched for a *later* blink — a real CR or UR may
   sit behind the artefact. Where one is found it becomes the scored response.

**Analyse the `SCORED onset` / `SCORED class` columns.** They already apply the second-look
rule. `Raw first blink` keeps the unmodified first event for transparency.

Classes: `alpha/startle` <100 ms · `CR` 100–350 ms (began before the US) · `UR only`
≥350 ms · `in-progress at CS` (untimeable, excluded from summaries).

CS-only trials are scored but kept out of the session and block summaries and out of the
main scatter — a trial with no US is a different measurement.

## Known limits

- Landmark EAR is a good proxy for lid aperture but it is **not** EMG or a magnetic search
  coil. The timings are the reliable quantity; treat absolute closure percentages as
  relative.
- Trials carrying a quality flag are kept in the tables so nothing is silently dropped.
  Filter `Quality flag = clean` for the strictest subset.
- The pipeline assumes one face, roughly frontal, and a stimulator box visible somewhere
  in frame. If the camera is moved mid-recording, split the file first.

## Output

Everything lands in `analysis_CSUS/`:

- `EBC_Marie_CSUS.xlsx` — conditioning (90 paired + 10 CS-only)
- `EBC_Marie_CSUS4.xlsx` — extinction
- `cond_*` / `csonly_*` / `ext_*` PNG figures
- `trials_*.csv`, `stimulus_events.csv`, `closure_traces_all.csv`
- `_work/` — cache, safe to delete (costs a full re-run)
