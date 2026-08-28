# Eyeblink-conditioning video scoring

Scores delay eyeblink conditioning directly from high-speed video: it finds both stimulus
LEDs in the pixels, aligns every trial to the CS to within one frame, and measures eyelid
aperture with facial landmarks.

Nothing about the timing is assumed — the CS duration, the CS–US interval, the trial
structure and the block boundaries are all recovered from the recording and then *checked*
against the protocol. The protocol is a test, never an instruction: no trial is renumbered
to make the structure come out right, and the report says exactly where recording and
protocol disagree.

One participant per run, driven by a small JSON file, so the same checkout processes any
number of participants.

## Run it

```
python ebc_run_all.py --config studies/thomas.json     # everything
python ebc_run_all.py --videos "D:/EBC/Video/Alice"    # no config: roles from file names
python ebc_run_all.py --config studies/thomas.json --from score   # re-score, rebuild
python ebc_run_all.py --config studies/thomas.json --force        # redo the video passes
```

Requires `ffmpeg` and `ffprobe` on PATH, and
`opencv-python mediapipe numpy scipy matplotlib openpyxl pillow`.

Two passes are made over each recording (a survey and a full-rate read of a small window),
plus a short seek per trial for the eyelids. Intermediates are cached in `<out>/_work`, so
`--from score` reruns everything downstream in seconds — that is the flag to use when
changing a scoring rule or a figure. `--jobs N` sets how many recordings decode at once.

## The study file

```json
{
  "study": "Thomas",
  "video_dir": "C:/.../Video/Thomas",
  "protocol": {
    "cs_ms": 400.0, "us_onset_ms": 350.0, "us_dur_ms": 50.0,
    "paired_per_block": 9, "cs_only_per_block": 1, "n_blocks": 10,
    "min_iti_s": 5.0, "cs_tol": 0.35, "us_tol": 0.60
  },
  "recordings": [
    {"tag": "csus1", "file": "CSUS 1.MP4", "label": "CSUS 1", "role": "conditioning", "order": 1},
    {"tag": "extinction", "file": "extinction.MP4", "label": "Extinction 1", "role": "extinction", "order": 1},
    {"tag": "csonly", "file": "CS ONLY.MP4", "label": "CS-only baseline", "role": "baseline_cs"},
    {"tag": "usonly", "file": "US ONLY.MP4", "label": "US-only baseline", "role": "baseline_us"}
  ]
}
```

A recording's **role** decides what is expected of it and how it is scored:

| role | what it is | how it is treated |
|---|---|---|
| `conditioning` | paired CS–US trials plus the CS-only probe that closes each block | chapters of one session; they concatenate onto one clock and carry the block structure |
| `extinction` | CS alone after conditioning | scored against the *learned* US window; no US is delivered |
| `baseline_cs` | CS alone, outside conditioning | gives the false-positive rate for the CR window |
| `baseline_us` | US alone | no CS exists, so every window is anchored on the US instead |

Omit `recordings` and the folder is scanned: `CSUS *`, `extinction*`, `CS ONLY`, `US ONLY`
map onto the four roles. Add `"led_yellow": [x, y]` to a recording to pin the CS LED by
hand if the automatic search ever picks the wrong spot.

### When the CS LED cannot be read

`"anchor": "us"` on a recording builds its trials from the US (blue) LED instead of the CS
(yellow) one, and infers the CS onset as `us_onset - us_onset_ms`:

```json
{"tag": "csus2", "file": "CSUS 2.MP4", "role": "conditioning", "order": 2, "anchor": "us"}
```

Reach for it when `ebc_qc.py leds` reports weak contrast and a pulse count far below the
rejected count - the signature of a CS window whose baseline has drifted up into the
threshold, so the detector is triggering on noise. A wooden or warm-coloured stimulator box
under changing room light does exactly this to the yellow channel, while the blue US LED,
having nothing in the scene to compete with, stays clean.

Because a US is only ever delivered inside a CS, every accepted US marks a paired trial.
`ebc_eyes.py` then aligns the window on the blue LED and steps back by the protocol lag, so
alignment is as good as the US pulse - in practice exact to the frame.

Two things are given up, and both are visible in the output rather than assumed away:

- **CS-only probes are invisible**, since a probe delivers no US. Blocks can no longer be
  closed by their probe, so the block structure is only recovered as far as the last
  CS-anchored probe. Prefer mixed anchoring - leave the recordings whose CS LED is clean on
  the default `"cs"` - so the probes that do exist are still found.
- **The CS onset is inferred, not measured**, and `cs_duration_ms` is empty. Every trial
  carries a **CS timing source** column saying which it was.

Validated on a recording where both LEDs were clean: inferred CS onset matched the measured
one to a median of +0.3 ms (SD 1.5 ms, max 8.0 ms - under one frame at 119.88 fps), and the
scored response class agreed on 29/31 trials.

## How the stimuli are found

This is the part that has to be right, because everything downstream is measured from it.

1. **The CS LED is found by what it does, not by how it looks.** A survey pass keeps, for
   every 6×6 block of a 480×270 frame, the strongest yellow value per sampled frame — so
   every block has a time course. Blocks are then ranked on the two things that make an LED
   an LED: a large gap between a resting level and a lit level, and pulses that all last the
   protocol's duration. A table lamp is bright, a painting is yellow, a wooden mask is both;
   none of them switch. In a sunlit room full of warm-coloured objects the true LED wins by
   a factor of two to three over the runner-up.
2. **The US LED is never searched for across the frame.** At 50 ms it is one or two frames in
   a subsampled survey, indistinguishable from sensor noise or someone walking past — a
   whole-frame search finds noise, reliably. Instead its window is pinned beside the CS LED,
   where it physically is on the same panel. That is also what keeps the read window small.
3. **The box position is decided for the whole participant at once.** Short clips with two
   CS presentations cannot locate anything on their own, and a US-only recording has no CS
   to find; both take the position from the recordings that *were* confident. Where the
   local estimate and the study consensus disagree, the read window spans both.
4. **Timing comes from a full-rate, full-resolution read** of that window — never from the
   survey. Onsets are good to one frame: 8.34 ms at 119.88 fps.
5. **The threshold is derived per recording, not fixed.** An amber lens is yellow even when
   dark, so the resting level is nowhere near zero and no constant transfers between
   recordings. What transfers is the switch, so the threshold is placed between the two
   modes of the signal, with a Schmitt trigger so a flickering edge cannot split one pulse
   into two.
6. **Every pulse is kept, accepted or rejected, with the reason.** A pulse outside the
   duration tolerance, or closer than `min_iti_s` to the previous accepted one, is LED
   flicker rather than a stimulus — but it still appears in `stimulus_events.csv`. Nothing
   is dropped silently.
7. **The pipeline says when it does not trust itself.** Weak contrast, accepted pulses only
   a fraction of a second apart, or a "lit pixel" that wanders across the frame instead of
   staying a point source all raise a warning that is carried into the QC page and the JSON.

**Look at `qc_leds_<tag>.png` first on a new participant.** It shows the marker on the LED,
the window that was read, the whole signal with its thresholds, and a tick for every pulse
accepted or rejected. If that page is right, the numbers are right.

## How a trial is scored

1. **Alignment.** Each trial window is cut with the anchoring LED *and* the face inside the
   same crop, so the onset is re-detected inside the window rather than trusted from the
   seek. The residual error is reported per trial.
2. **Eyelid measure.** MediaPipe FaceMesh (478 landmarks, iris refinement) on a 2× upscaled
   face crop; eye aspect ratio per eye, averaged. EAR is normalised by eye width, so it
   survives head movement and changes of camera distance.
3. **Closure scale.** 0% = a blink-robust open-eye reference (85th percentile of EAR in the
   window, which survives a baseline blink where a median does not). 100% = a full-closure
   reference **pooled across every recording of the participant**, so a two-minute clip and
   a nine-minute chapter sit on the same axis. Smoothed with a 5-frame Savitzky–Golay filter.
4. **Blink criterion.** Five robust SDs above the trial's own pre-CS baseline, floor 15%
   closure, then walked back along the rising edge to the true onset. A separate blink must
   re-reach 40% closure after first returning below 20%.
5. **Second look.** If the first event is an alpha blink (<100 ms) or the lid was already
   moving at onset, the window is searched for a *later* blink — a real CR or UR may sit
   behind the artefact. Where one is found it becomes the scored response.

**Analyse the `scored_onset_ms` / `scored_class` columns.** They already apply the
second-look rule; `blink_onset_ms` keeps the unmodified first event for transparency.

Classes: `alpha/startle` <100 ms · `CR` 100 ms–US onset (began before the puff) ·
`UR` at or after US onset · `in-progress at stimulus` (untimeable, excluded from summaries).
US-only trials are anchored on the puff, so their latencies are measured from it and every
response there is by definition unconditioned.

## Files

| Script | What it does |
|---|---|
| `ebc_config.py` | The study file: recordings, roles, protocol. Discovers a folder when there is no config. |
| `ebc_paths.py` | Where the outputs and the cache live, per study. |
| `ebc_video.py` | ffmpeg/ffprobe helpers. |
| `ebc_signal.py` | Bimodal threshold + Schmitt trigger: a 1-D LED signal to a list of pulses. |
| `ebc_locate.py` | Study-level: where the stimulator box is in each recording, with a consensus for the clips that cannot tell. |
| `ebc_stimulus.py` | Per recording: survey pass, then the full-rate read. Writes `<tag>_stim.json`. |
| `ebc_protocol.py` | Pulses to trials; pairs CS with US; recovers the blocks and checks them against the protocol. |
| `ebc_eyes.py` | Per recording: eyelid tracking in a window around every trial. |
| `ebc_score.py` | One pooled closure scale, blink metrics, response classes. |
| `ebc_figures.py` | Onset scatter, acquisition curve, closure rasters — one set per trial group. |
| `ebc_export_csv.py` | Trials, stimulus events and full traces as CSV. |
| `ebc_workbooks.py` | One Excel workbook per role, each with its own read-me. |
| `ebc_qc.py` | `leds` — the LED check page per recording. `trial <tag> <n>` — an eye filmstrip with the measured closure printed on each frame. |
| `ebc_run_all.py` | The driver. |

## Known limits

- Landmark EAR is a good proxy for lid aperture but it is **not** EMG or a magnetic search
  coil. The timings are the reliable quantity; treat absolute closure percentages as
  relative.
- Trials carrying a quality flag are kept in the tables so nothing is silently dropped.
  Filter `quality = clean` for the strictest subset.
- One face, roughly frontal, and the stimulator box visible somewhere in frame. A camera
  re-aimed *between* recordings is handled; a camera re-aimed *within* one is detected and
  flagged, and the LED window is widened to cover both positions, but a very large move may
  still need `led_yellow` set by hand.
- Trials whose window runs off the end of a recording are marked `truncated`.
- A recording set to `"anchor": "us"` cannot contribute CS-only probes, so a study anchored
  that way throughout recovers its paired trials but not its block boundaries.

## Output

Everything lands in `<video_dir>/analysis_EBC/`:

- `EBC_<study>_conditioning.xlsx`, `_extinction.xlsx`, `_baseline_cs.xlsx`, `_baseline_us.xlsx`
- `qc_leds_<tag>.png` — the stimulus-detection check, one per recording
- `cond_*`, `ext_*`, `baseline_*` PNG figures
- `trials_*.csv`, `stimulus_events.csv`, `closure_traces_all.csv`
- `_work/` — cache, safe to delete (costs a full re-run)
