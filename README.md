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

## Point and click

Not everyone wants a terminal.  Double-click the **EBC Analyzer** shortcut on the Desktop
— or, on a machine with Python, **`EBC Analyzer.bat`** or `python ebc_app.py` — and the
app opens **in its own window**: pick a folder, tick the videos, confirm what each one is,
press Run.  Progress streams per recording; the workbooks, figures and CSVs appear as
download links when it finishes.

It is a desktop window, not a browser tab — no address bar, no tabs, nothing to navigate
away from.  Underneath, the window is drawn by the Edge WebView2 runtime, which ships with
Edge and is on every current Windows machine; the interface it shows is served by a server
bound to 127.0.0.1 that only this computer can reach.  If WebView2 is somehow absent the
app falls back to opening a browser rather than failing.

It is only a front end - it writes a study file from what you ticked and hands it to
`ebc_run_all.py`.  The study file is left in the output folder as `<study>.json`, so
anything done in the window can be repeated, tweaked or scripted from the command line.
Nothing is uploaded, and the recordings are read where they sit.

## The app

The whole thing packages into one installer you can send to anybody:

```
packaging\build.bat        ->  packaging\Setup EBC Analyzer 1.0.exe   (~240 MB)
```

**The machine it lands on needs nothing.** Python, OpenCV, MediaPipe, SciPy, matplotlib
and ffmpeg all travel inside it — that is what the ~880 MB installed is. No `pip install`,
no ffmpeg on PATH, and no administrator: it installs into the user's own AppData, so a
locked-down lab machine is fine. Verified by running the eyelid stage on a PATH stripped
to `C:\Windows\system32`, with neither Python nor ffmpeg present.

Windows will say *"Windows protected your PC"* the first time, because the installer is
not signed with a paid code-signing certificate. **More info → Run anyway.** Buying a
certificate is the only thing that removes that.

The app is staged in `build/dist/` (git-ignored), deliberately not the folder the
installer installs into, so a rebuild never overwrites the copy someone has installed.

Everything lives under one folder:

```
C:\Users\marga\EBC Analyzer\
    ebc_*.py, ebc_app_ui.html    the pipeline and the interface   (in git)
    assets/                      the lab mark, cut to three shapes (in git)
    packaging/                   spec, installer script, build.bat (in git)
    studies/                     one JSON per participant          (in git)
    results/                     workbooks, figures, CSVs, caches  (not in git)
    app/                         the installed app                 (not in git)
    build/                       PyInstaller staging               (not in git)
```

`results/` and `app/` are deliberately untracked: they are hundreds of megabytes of
output and binaries, regenerated from the recordings and from the source respectively.
The **recordings themselves stay where they are**, in OneDrive under `EBC/Video/<name>/` —
they are the raw data, they are backed up there, and nothing here copies them.

Building needs `pyinstaller` and `pywebview` on top of the packages below, `winget install
JRSoftware.InnoSetup`, and an ffmpeg on PATH — that last one is the ffmpeg that gets
packaged.

**One thing to know if you change the pipeline.** Nothing here *imports* `ebc_eyes`;
something *runs* it, as its own process, because three recordings decode at once. Inside
an `.exe` there is no Python to hand a script to, so the app re-launches itself instead —
`EBC Analyzer.exe --stage ebc_eyes.py <config> <tag>` — and `ebc_launch.py` runs the
script from the copy carried inside the executable. So **a new stage script has to be
added to `STAGES` in `packaging/ebc_analyzer.spec`**, or it will work from source and be
missing from the app. Anything it imports that is not already there belongs in
`hiddenimports` for the same reason: no import statement in the frozen graph points at it.

**And one trap worth knowing.** Packaged, `import mediapipe` closes the process's stderr
handle — `GetFileType` on it goes from *pipe* to *unknown* — after which any `Popen` that
tries to hand that handle to a child dies with `[WinError 50] The request is not
supported`. The eyelid stage imports mediapipe and then reads video, so it hit exactly
that, in the packaged build only. `ebc_video.py` therefore gives every ffmpeg child its
own stdio instead of inheriting ours. Anything else that spawns a process after mediapipe
is loaded has to do the same.

The version in `ebc_config.py` (`VERSION`) is stamped on the page, on the console banner
and on every workbook cover. Raise it whenever the scoring changes, so a number in a paper
can be traced back to what produced it.

## Run it

```
python ebc_run_all.py --config studies/thomas.json     # everything
python ebc_run_all.py --videos "D:/EBC/Video/Alice"    # no config: roles from file names
python ebc_run_all.py --config studies/thomas.json --from score   # re-score, rebuild
python ebc_run_all.py --config studies/thomas.json --force        # redo the video passes
```

Requires `ffmpeg` and `ffprobe` on PATH, and
`opencv-python mediapipe numpy scipy matplotlib openpyxl pillow pywebview`.
(`pywebview` is only needed for the app window; without it the pipeline still runs and the
app falls back to a browser.)

Two passes are made over each recording (a survey and a full-rate read of a small window),
plus a short seek per trial for the eyelids. Intermediates are cached in `<out>/_work`, so
`--from score` reruns everything downstream in seconds — that is the flag to use when
changing a scoring rule or a figure. `--jobs N` sets how many recordings decode at once.

## What changes from one participant to the next

**The protocol never does.**  Every session is the same experiment:

| | | |
|---|---|---|
| CS | 400 ms | yellow LED |
| US | 50 ms, co-terminating with the CS | blue LED |
| CS-US interval | 350 ms | US onset, measured from CS onset |
| Structure | 9 paired + 1 CS-only probe, x 10 blocks | 100 trials |

Those are the defaults in `ebc_config.DEFAULT_PROTOCOL`, so **a study file does not need a
`protocol` block at all**.  Write one only to record a genuine deviation, and if you find
yourself editing the numbers to make the trial count come out right, stop: the protocol is
the test, and a disagreement is a finding about the recording, not a parameter to tune.

**What does change is where things are in the frame** - the participant sits differently,
the camera is re-aimed, the stimulator box moves, and the room light is not the same on a
grey afternoon as on a bright morning.  All of that is found per recording:

- the **face** is located by sampling frames across the recording and taking the union of
  the landmark boxes, so a participant who shifts in their seat is still covered;
- the **stimulator box** is found per recording, with a study-level consensus for the clips
  too short or too dim to tell on their own;
- a camera re-aimed **between** recordings is handled; one re-aimed **within** a recording
  is detected, flagged, and the LED window widened to cover both positions.

The one thing that genuinely defeats the automatic search is a **CS LED with no contrast
left**, which is a lighting problem rather than a geometry one.  See *When the CS LED
cannot be read*.

## Running a new participant

1. Put the recordings in one folder per participant, named so the roles are obvious:
   `CSUS 1..n`, `extinction`, `CS ONLY`, `US ONLY`.
2. Run it - no study file needed to start:

   ```
   python ebc_run_all.py --videos "D:/EBC/Video/Alice"
   ```

   Roles are guessed from the names and the study file that was inferred is written to
   `<out>/_work/run_config.json`.  Keep it as `studies/alice.json` once you are happy.
3. **Open every `qc_leds_<tag>.png` before you read a single number.**  This is the step
   that matters.  It shows the located box, both LED traces, and every pulse the detector
   accepted or rejected.  A run can finish cleanly, report confidently, and still be built
   on a CS channel that was reading noise.
4. Read the protocol check.  `93 found / 100 expected` is a result; `41 found / 100` is a
   detection failure to be diagnosed, not a participant who missed trials.
5. Fix what the check page shows, re-run the affected stage, and only then score.

### Reading the LED check page

| What you see | What it means | What to do |
|---|---|---|
| `n/n pulses accepted`, ITI steady near 13 s | healthy | nothing |
| **accepted far below rejected** (`6/464`, `1/1058`) | the threshold is sitting in the noise: hundreds of spurious pulses are found and then thrown out on duration | the CS channel is unusable - see below |
| `!! weak contrast (30)` | lit minus rest has collapsed | check the trace before trusting any trial from this recording |
| resting level climbing across recordings | the room light warmed or dimmed and the box itself is now as "yellow" as the LED | US-anchor the affected recordings |
| blue clean, yellow noisy | the usual case: nothing else in the scene is blue, while a wooden or warm-coloured box competes with the yellow LED | US-anchor |

The `rest` and `lit` numbers on the page are the whole diagnosis.  A yellow LED lighting to
203 above a resting 130 is comfortable; lighting to 192 above a resting 162 is not, and the
detector will fire on noise long before it fails outright.

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

**This is handled for you.**  `anchor` defaults to `"auto"`, and after the LEDs have been
read `ebc_triage.py` looks at what each channel actually did and decides:

| what it finds | what it does |
|---|---|
| CS channel healthy | nothing - trials come from the CS, as normal |
| CS unreadable, US clean, recording has a US | trials from the US, CS onset inferred, **and a warning** |
| CS unreadable, recording delivers no US (extinction, CS-only) | **excluded, with an explanation** - the rest of the study still runs |

Writing `"anchor": "cs"` or `"anchor": "us"` explicitly overrides the decision, and is
obeyed even when the evidence disagrees (it says so, and carries on).

The tests are three, and a recording needs two of them to be judged unreadable - or one
overwhelming one:

- **acceptance ratio** - how many detected pulses survive the duration filter. A healthy
  channel keeps 80-100%; an unreadable one keeps under 2%, because the threshold is sitting
  inside the noise and the "pulses" were never pulses.
- **contrast** - lit level minus resting level, below 55.
- **CS against US** - a US only ever fires inside a CS, so far fewer CS pulses than clean US
  pulses means the CS channel, not the participant, is at fault.

These are calibrated on every recording processed so far, where the two populations do not
come close to overlapping:

| | contrast | pulses kept |
|---|---|---|
| healthy (11 recordings) | 73 - 150 | 80 - 100% |
| unreadable (2 recordings) | 30 - 43 | 0.1 - 1.3% |

A wooden or warm-coloured stimulator box under changing room light is what produces the
second row: the box itself is as "yellow" as the LED, so as the light warms the resting
level climbs toward the lit level and the margin disappears.  The blue US LED has nothing
in the scene to compete with and stays clean, which is why it can be used as the fallback.

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
| `ebc_triage.py` | Judges each CS channel from what the LEDs did; anchors on the US or excludes the recording, and says why. |
| `ebc_protocol.py` | Pulses to trials; pairs CS with US; recovers the blocks and checks them against the protocol. |
| `ebc_eyes.py` | Per recording: eyelid tracking in a window around every trial. |
| `ebc_score.py` | One pooled closure scale, blink metrics, response classes. |
| `ebc_figures.py` | Onset scatter, acquisition curve, closure rasters — one set per trial group. |
| `ebc_export_csv.py` | Trials, stimulus events and full traces as CSV. |
| `ebc_workbooks.py` | One Excel workbook per role, each with its own read-me. |
| `ebc_qc.py` | `leds` — the LED check page per recording. `trial <tag> <n>` — an eye filmstrip with the measured closure printed on each frame. |
| `ebc_run_all.py` | The driver. |
| `ebc_app.py`, `ebc_app_ui.html` | The window and what it shows. Writes a study file and runs the driver on it. |
| `ebc_launch.py` | The one door in. Decides whether a launch is the app, the folder dialog or one pipeline stage — the only file that knows it might be running from an `.exe`. |
| `assets/` | The lab mark, cut from the source logo: `logo_mark.png` (title bar), `logo_full.png` (colophon and workbook covers), `ebc.ico` (the app icon). |
| `packaging/` | `make_assets.py` cuts those three from the source logo; `ebc_analyzer.spec` builds the app, `ebc_analyzer.iss` packs it into an installer, `build.bat` does all of it. |

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
- A block is closed by its CS-only probe where one was recovered, and by the count of
  paired trials where none was. The protocol check reports how many boundaries came from
  each, and every trial carries `block_closed_by`. A counted boundary is an assumption
  from the protocol; a probe is an observation. Treat block-wise results from a US-anchored
  study accordingly.

## What the recordings have looked like so far

Two participants, both on the same protocol and the same rig, to calibrate expectations:

| | Thomas | Carole |
|---|---|---|
| Conditioning trials recovered | 85 / 100 | 93 / 100 |
| Paired CS-US | 77 | 90 / 90 |
| CS-only probes | 8 | 3 (two recordings US-anchored) |
| CS LED readable | all recordings | CSUS 1 only |
| **CR rate** | **10%** | **48%** |
| CR onset | 198 ms | 303 ms |

The spread in CR rate between two people on an identical protocol is the point: 10% and 48%
are both real results, and a pipeline that quietly discarded two thirds of Carole's trials
would have reported 36% for her instead of 48% and hidden the acquisition curve
(32% - 57% - 56% across her three sessions) entirely.

Practical expectations: face tracking runs at 100% on a cooperative, frontally seated
participant; alignment error is 0.0 ms when a clean LED marks the trial; a 4 GB / 531 s
recording takes roughly 12 minutes for the LED pass and about 4 more for the eyelids.

## Output

Everything lands in `<video_dir>/analysis_EBC/`:

- `EBC_<study>_conditioning.xlsx`, `_extinction.xlsx`, `_baseline_cs.xlsx`, `_baseline_us.xlsx`
- `qc_leds_<tag>.png` — the stimulus-detection check, one per recording
- `cond_*`, `ext_*`, `baseline_*` PNG figures
- `trials_*.csv`, `stimulus_events.csv`, `closure_traces_all.csv`
- `_work/` — cache, safe to delete (costs a full re-run)
