# Eyeblink-conditioning video scoring

Scores eyeblink conditioning directly from high-speed video: it finds both stimulus LEDs
in the pixels, aligns every trial to the CS to within one frame, and measures eyelid
aperture with facial landmarks. Delay and trace designs are both handled — where the US
sits relative to the CS is a number in the study file, not an assumption in the code.

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
press Run.  Progress streams per recording; the workbooks, figures and CSVs are listed
when it finishes, and a click on any of them opens the output folder with that file
selected.

It is a desktop window, not a browser tab — no address bar, no tabs, nothing to navigate
away from.  Underneath, the window is drawn by the Edge WebView2 runtime, which ships with
Edge and is on every current Windows machine; the interface it shows is served by a server
bound to 127.0.0.1 that only this computer can reach.  If WebView2 is somehow absent the
app falls back to opening a browser rather than failing.

**Where the results end up.** The output folder is set per study and is often nowhere
near the recordings, so when a run finishes the readable results - the workbooks, the
figures, the LED check pages and the CSVs - are **copied back into an `EBC results`
folder beside the recordings**. Copied, not moved: `<out>/` stays the authoritative copy
and keeps the caches. Nobody should have to go hunting under `C:\Users` for their own
session.

It is only a front end - it writes a study file from what you ticked and hands it to
`ebc_run_all.py`.  The study file is left in the output folder as `<study>.json`, so
anything done in the window can be repeated, tweaked or scripted from the command line.
Nothing is uploaded, and the recordings are read where they sit.

## The app

The whole thing packages into one installer you can send to anybody:

```
packaging\build.bat        ->  packaging\Setup EBC Analyzer 1.1.exe   (~240 MB)
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
    assets/                      the lab mark and the two crests   (in git)
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

**The protocol usually does not.**  The session the app opens on is:

| | | |
|---|---|---|
| CS | 400 ms | yellow LED |
| US | 50 ms, co-terminating with the CS | blue LED |
| CS-US interval | 350 ms | US onset, measured from CS onset |
| Structure | 9 paired + 1 CS-only probe, x 10 blocks | 100 trials |

Those are the defaults in `ebc_config.DEFAULT_PROTOCOL`, so **a study file does not need a
`protocol` block at all**.  They are a starting point and not a house standard — which
design is standard is a fact about a lab, so nothing in the app or in its wording calls
one of them the right one.  If you find yourself editing the numbers to make the trial
count come out right, stop: the protocol is the test, and a disagreement is a finding
about the recording, not a parameter to tune.

### A different protocol

Change the numbers when the session really was run differently — the app is not built
around one design.  Nothing requires the US to fall inside the CS.  Put `us_onset_ms` at
or beyond `cs_ms` and the study is a **trace** protocol, with a silent interval between CS
offset and US onset, and the whole pipeline follows:

| what follows the protocol | how |
|---|---|
| pairing a US to its CS | the window runs to the end of the stimulus pair, not to the end of the CS, capped at half the minimum ITI so it can never reach the next trial |
| the trial window | `ebc_config.window()` — a trace design tracks further past CS onset, so the US and the response to it are inside the window |
| the CR window | both of its edges sit one measured reflex latency after their own stimulus, so a trace protocol's window moves with the US wherever the protocol puts it |
| block recovery | `paired_per_block`, `cs_only_per_block` (0 is allowed — then the count closes each block) and `n_blocks` |
| figures | the US band is drawn where the US is; a trace interval is shaded and labelled, and the CS offset gets its own marker |
| workbooks | the read-me names the design and states the interval; a trace study gains a *closure mid-gap* column |

`ebc_config.design()` is the one place that decides what a set of numbers means, and
`check_protocol()` the one place that decides whether it can be analysed at all.  A
protocol that is merely unusual is described and left alone; one that is arithmetically
impossible — a trial longer than half the gap between trials, a startle cut-off past the
US — is refused before a run starts, in words, with the numbers named.

The app's step 2 shows the protocol drawn on a time axis as you type it, and offers the
two presets in `ebc_config.PRESETS` — **Delay** and **Trace**, named for the designs and
nothing else.  A preset is only a set of numbers; choosing one is the same as typing them,
and nothing downstream knows which was used.  **Reset every number** puts the whole panel,
detection tolerances included, back to `DEFAULT_PROTOCOL`.  The last protocol actually run
comes back on the next launch, so a lab that has settled on a design does not retype it
for every participant.

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
    "min_iti_s": 5.0, "cs_tol": 0.35, "us_tol": 0.60,
    "alpha_ms": 100.0, "pre_ms": 300.0, "post_ms": 0.0
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
5. **The CR window, measured.** Neither the eye nor the brainstem responds instantly, so
   the window a CR is counted in is not the bare interval between the two stimuli. The
   **US-only baseline is scored first**, and the reflex latency is measured from it: the
   mean of the unconditioned blink onsets minus 1.5 SD is the soonest a stimulus can have
   caused a blink. Both edges of the CR window then sit that far after *their own*
   stimulus:

   ```
   CR window  =  [ CS onset + reflex ,  US onset + reflex ]
   ```

   A blink before the lower edge began too soon after the CS for the CS to have caused it;
   one after the upper edge began late enough that the puff could have. In between, the
   blink was already under way before the puff could have driven it — which is what a
   conditioned response is. `ebc_config.cr_window()` is the one place this is decided, and
   it writes the class labels every other module matches on.

   Only trials the scorer stands behind feed the measurement, and a mean and an SD have no
   defence against one bad trial. Two filters, both of which name what they set aside:

   - a US-only trial with the lid already moving at the puff, or whose response was
     recovered behind an artefact, is left out — its "onset" is not the reflex;
   - of what is left, onsets more than **3 robust SDs** from the median of that baseline
     are set aside before the mean and the SD are taken (median and MAD, which an outlier
     cannot move; applied only once there are 5 or more onsets).

   Both matter in practice. In Thomas's 35-trial US-only baseline a single onset of 826 ms
   — a spontaneous blink scored long after the reflex had been missed — took the SD from
   11 ms to 152 and drove `mean − 1.5 SD` to −129 ms, i.e. no window at all. With the
   outlier set aside the same baseline gives 70 ± 11 ms over 21 trials and a 54 ms reflex.

   With **no US-only recording** in the study there is nothing to measure, and the window
   falls back to the protocol's `alpha_ms` startle cut-off and the bare US onset — which is
   how this app scored before, so nothing already analysed moves. The fallback is also what
   happens if the measurement comes out impossible (`mean − 1.5 SD` at or before the puff),
   and the reason is printed, put on the results card and written into every read-me.
6. **Second look.** If the first event began too soon to be a response to anything — before
   the CR window's lower edge — or the lid was already moving at onset, the window is
   searched for a *later* blink, because a real CR or UR may sit behind the artefact. Where
   one is found it becomes the scored response.

**Analyse the `scored_onset_ms` / `scored_class` columns.** They already apply the
second-look rule; `blink_onset_ms` keeps the unmodified first event for transparency.

Classes, with `reflex` the measured latency and `us` the US onset: `alpha/startle`
< reflex · `CR` reflex–(us + reflex) · `UR` at or after us + reflex ·
`in-progress at stimulus` (untimeable, excluded from summaries). The class *names* carry
their own boundaries — `CR (43-393ms)` — so a column always says which window produced it.
US-only trials are anchored on the puff, so their latencies are measured from it and every
response there is by definition unconditioned; they are never classified against the window
they are used to build.

The window this run used, the trials that measured it and the ones left out are written to
`merged.json`, printed on the console, reported on the app's results card, drawn on the
figures, and stated on every workbook's read-me — the US-only workbook additionally lists
the individual onsets that went into it.

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
| `assets/` | The lab mark, cut from the source logo: `logo_mark.png` (title bar), `logo_full.png` (credits bar and workbook covers), `ebc.ico` (the app icon). Beside them the two affiliations that close the page: `logo_upcite.png` and `logo_cnrs.svg`, both the white versions for a dark ground — the official marks with nothing redrawn, the CNRS field knocked out of white and the Université Paris Cité lockup reversed. Replacing either is a matter of dropping a new file in under the same name. |
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
| US-only trials measuring the reflex | 21 of 35 | 4 of 5 |
| Reflex latency | 70 ± 11 ms → **54 ms** | 75 ± 22 ms → **43 ms** |
| CR window | 54–404 ms | 43–393 ms |
| **CR rate** | **29%** | **61%** |
| CR onset | 328 ms | 312 ms |

The spread in CR rate between two people on an identical protocol is the point: 29% and 61%
are both real results, and a pipeline that quietly discarded two thirds of Carole's trials
would have hidden her acquisition curve entirely.

Both rates are higher than the 10% and 48% an earlier build reported, and the reason is the
measured window rather than any change to how a blink is found. Carole's reflex is 43 ms, so
eleven blinks that began between 350 and 393 ms — after the puff, but sooner than the puff
could have caused anything — are conditioned responses, not reactions to it. Thomas's reflex
is 54 ms and the same correction moves fourteen of his trials. The two participants also
arrive at nearly the same reflex latency (43 and 54 ms) from independent baselines recorded
months apart, which is the sanity check the measurement had to pass.

Practical expectations: face tracking runs at 100% on a cooperative, frontally seated
participant; alignment error is 0.0 ms when a clean LED marks the trial; a 4 GB / 531 s
recording takes roughly 12 minutes for the LED pass and about 4 more for the eyelids.

## Output

Everything lands in `<video_dir>/analysis_EBC/`:

- `EBC_<study>_conditioning.xlsx`, `_extinction.xlsx`, `_baseline_cs.xlsx`, `_baseline_us.xlsx`
- `qc_leds_<tag>.png` — the stimulus-detection check, one per recording
- `cond_*`, `ext_*`, `baseline_*` PNG figures
- `trials_*.csv`, `stimulus_events.csv`, `closure_traces_all.csv`
- `trials_to_score_by_hand.csv` — the trials the scorer will not stand behind
- `_work/` — cache, safe to delete (costs a full re-run)

The app cannot open any of these itself: it draws its window with WebView2, which has no
downloads and no file viewer. The results list says so, and a click on any entry opens the
output folder with that file selected, which is the one thing the app *can* do.

### Trials to score by hand

A trial is flagged when something measured in that trial defeats the scorer, not when the
recording as a whole is doubtful — triage already reports the latter, separately. The
rules, all in `ebc_score.manual_reasons()`:

| flagged when | why it cannot be left to the scorer |
|---|---|
| the lid was already closing at CS onset, and no later blink was found | there is no onset to time |
| the only movement in the window began before the CR window opens, with nothing behind it | the response, if there was one, is under the artefact |
| the trial window runs past the end of the recording | the response may be cut off |
| the face was tracked in under 80% of the window | the trace has holes where the blink would be |
| the CS onset re-found inside the window is more than 25 ms from the seek | time zero is uncertain |

Each one is listed with **where it is in its own recording**, as `m:ss.mmm`, so it can be
scrubbed to directly. The paired CS–US conditioning trials are called out first, because
those are the measurement; probes and baselines follow.

Every workbook carries a **Score by hand** sheet — the first sheet after the cover — with
one row per flagged trial: the recording, the time to open it at, what the scorer put down,
and what it could not see past. The count also appears on the cover, on the session, block
and trial summaries, and against the trial itself in the trial table, so "which ones do I
have to look at?" is answered by opening the workbook rather than by filtering it.

Nothing is dropped: every flagged trial stays in the workbooks and the CSVs with **Score
by hand** set and the reason beside it. They are excluded from the CR percentages for the
same reason they are flagged — a trial that cannot be timed cannot be counted either way.
