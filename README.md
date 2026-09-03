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

Not everyone wants a terminal.  Double-click **`EBC Analyzer.bat`** (or run
`python ebc_app.py`) and a page opens in your browser: pick a folder, tick the videos,
confirm what each one is, press Run.  Progress streams per recording; the workbooks,
figures and CSVs appear as download links when it finishes.

The folder is listed in **recording order** - the order the camera says, not the order the
names imply - with the recorded time, the chapter number and any warning against each file.
What the metadata says is a copy, a test or a failed take is left unticked, and a file whose
name implies no role at all arrives with an empty **- choose a role -** box: it cannot be
ticked, and the run is refused, until someone says what it is.  Nothing is guessed on your
behalf.

It is only a front end - it writes a study file from what you ticked and hands it to
`ebc_run_all.py`.  The study file is left in the output folder as `<study>.json`, so
anything done in the browser can be repeated, tweaked or scripted from the command line.
Nothing is uploaded: the server listens on 127.0.0.1 only and reads the videos in place.

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

## What the file names are not allowed to decide

The names are typed by whoever emptied the SD card at the end of a long session, so they
are the least reliable thing in the folder.  Before anything is decoded, `ebc_timeline.py`
reads what the *camera* wrote into each file - the recording time, the timecode track and
the duration - and recovers what actually happened.  Three facts from real sessions in
this study, all of which the names got wrong:

| What the names say | What the camera says |
|---|---|
| Carole: `CSUS 1`, `CSUS 2`, `CSUS 3`, `CSUS fin` | `CSUS fin` is **chapter 2 of the extinction take** - recorded after extinction, not after `CSUS 3`.  On a numeric sort it landed *second*. |
| Marie: `CSUS 4` closes conditioning | same shape: it is the tail of the `extinction` take |
| Charles: `csus1 - failed`, `csus 2`, `csus 3` | chapters 1, 2 and 3 of **one unbroken 25-minute recording**.  Not three sessions, and chapter 1 is not "failed" in any sense the recording knows about |

So the order of the recordings, and which files are one continuous take, come from the
camera clock and never from the names:

* **creation_time** names the take - every chapter of one recording carries the same value.
* **the timecode track** is the camera's own clock at the first frame.  It is monotonic
  within a take and separates its chapters, which creation_time cannot; it resets when
  the camera loses power, so it orders chapters, never takes.
* **timecode + duration** say whether two files are contiguous: chapter n+1 starts where
  chapter n ended, to within the rounding of the timecode seconds field.

This matters because the conditioning chapters are laid on one clock and that clock sets
every block boundary.  It is now the real clock: a session recorded in two takes with ten
minutes of extinction between them no longer has the gap silently closed.

### The human errors it is built for

| What happens | What the pipeline does |
|---|---|
| a name says `failed`, `test`, `raccourci`, `copie`, `(2)` | left out of a folder scan, with the reason.  **Listed by hand in a study file it is kept** - an explicit list is a deliberate choice |
| the same clip under two names (same recording time and timecode) | one is left out; scoring both would count every trial in it twice |
| a re-encoded copy (no camera metadata at all) | reported as derived, and left out of a folder scan.  Re-encoding is what flattens the LED contrast the whole pipeline depends on |
| chapters of one take, renamed as separate sessions | recognised as one take, ordered by chapter, and if an earlier chapter is not being analysed the report says how far into the take the scored part starts |
| a chapter labelled with a different role than its neighbours | reported: one unbroken recording, so the labels are a judgement, not a fact |
| two files whose names differ only in spacing (`CSUS 1`, `CSUS1`) | the second is renamed instead of stopping the run |
| a file in OneDrive that has not been downloaded | said before the run stalls on it for twenty minutes |
| empty, damaged, or not a video | left out with the reason |
| the frame rate, frame size or rotation changes mid-participant | reported - the box is located in pixels, so a shared position cannot cross that |
| recordings in one folder spanning more than eight hours | reported: probably two sessions, or a stray file from another day |
| a recording whose name says nothing (`GX012908.MP4`) | never given a role by guesswork.  A folder scan lists it, with its date, as in the folder but not analysed; the app shows it with an empty **&mdash; choose a role &mdash;** box and will not let it be ticked, or run, until a person picks one |
| a recording that does not behave like its role - a `US` in an extinction file, no `US` at all in a conditioning file | reported by `ebc_triage.py`, and **nothing is renamed or re-scored on the strength of it**.  Fix the role and re-run |

The blue channel is now read for *every* role, including the ones the protocol says
deliver no US.  It costs one wider crop and it is the only way to notice that a file
labelled `extinction` is full of puffs.  It changes no trial: those roles still get
CS-only trials whatever the blue channel saw.

## Which side of the panel the LEDs are on

Nothing assumes it.  The US LED is looked for in a window **centred** on the CS LED and
symmetric - as far left as right, as far up as down - because which way round the two sit
depends on how the box was turned and where the camera stood, not on the protocol.  The
side is then *measured* (`us_offset` in `<tag>_stim.json`), carried to the other
recordings of the participant so their windows are aimed rather than blind, and checked
across them: if it changes half way through a participant, something moved between
recordings and every inherited position across that boundary is suspect, which is said.

Two things make a window that wide safe:

* **Every pulse must come from the same place.**  An LED does not move.  Once a handful
  of pulses agree on a spot, a "pulse" 200 px away is a reflection, a phone screen or a
  white sleeve catching the sun - and it is rejected as one, with the reason kept.  This,
  rather than a tight window, is what makes the search side-agnostic.
* **Window sizes are in pixels of a 1920-wide frame and scaled** to whatever the
  recording is, so a 2.7K or 4K camera does not read a window a third of the intended size.

The stimulator box moving *between* recordings is handled too: confident positions are
clustered rather than averaged, and a recording that cannot locate the box itself
inherits the position of the recording nearest it **in time**.  A median over a session
where the box moved sits where the box has never been, and every recording inheriting it
inherits a wrong answer.

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
hand if the automatic search ever picks the wrong spot, and `"us_offset": [dx, dy]` to say
where the US LED sits relative to it.  `"include": false` leaves a listed recording out
of the run without deleting the line.

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
   whole-frame search finds noise, reliably. Instead it is looked for in a symmetric window
   around the CS LED, since the two are centimetres apart on the same panel; which side it
   is on is measured, not assumed (see *Which side of the panel the LEDs are on*).
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
   duration tolerance, closer than `min_iti_s` to the previous accepted one, or lit
   somewhere else in the window than the LED, is flicker or a reflection rather than a
   stimulus — but it still appears in `stimulus_events.csv`. Nothing is dropped silently.
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

Classes: `alpha/startle` <100 ms · `CR` 100 ms to **US onset + the reflex latency** ·
`UR` after that · `spontaneous blink` >800 ms (not time-locked to anything in the trial)
· `in-progress at stimulus` (untimeable).  The last two are excluded from the rate rather
than counted as either kind of response.  A trial that delivered **no** US is never judged
against the US at all: its classes are `alpha/startle` and `CR (no US on this trial)`.

**The CR/UR line is not the puff.**  A puff-driven response cannot begin at the instant of
the puff - the reflex takes time - so the line sits at `US onset + reflex latency`, and a
response in between began too early for the puff to have caused it.  That is a CR.  With
Carole's measured 67 ms the old line at 350 ms was putting 21 of her 85 scoreable trials
on the wrong side; Thomas is the proof, with 47 of his 87 trials landing in one bin at
400-425 ms, which is 350 + 67.

The latency is **measured per participant** from their own US-only baseline, where trials
are anchored on the puff and the onsets therefore *are* reflex latencies - but only those
inside a 20-250 ms window, because a spontaneous blink 900 ms after a puff is not a reflex
and letting it into the median moves the CR/UR line for every conditioning trial in the
study.  Fewer than three usable puffs and the run says so loudly and falls back to a
default: the line is then resting on an assumption, and a participant with no proper
US-only baseline needs one collected before their CR rate means anything.

> **The CR/UR line is knife-edge, and the CR rate inherits that.**  The split is made at
> exactly the CS-US interval, but onsets are quantised to one video frame (8.34 ms at
> 119.88 fps) and they pile up on the boundary: in Carole's 85 scoreable paired trials,
> **12 sit within one frame of it** - 4 at 342.0 ms scored CR, 6 at 350.4 ms and 2 at
> 358.7 ms scored UR.  Shifting the eyelid crop by a few pixels moved eight of them across
> the line and the headline rate with them, 48% to 40%, on identical trials.  Neither
> number is more correct than the other.  Before quoting a CR percentage, look at how many
> trials sit in that band (`scored_onset_ms` in the trial CSVs) and say so; a response
> beginning 0.4 ms after the puff is not meaningfully different from one beginning 8 ms
> before it.

## Files

| Script | What it does |
|---|---|
| `ebc_config.py` | The study file: recordings, roles, protocol. Discovers a folder when there is no config. |
| `ebc_media.py` | What the camera wrote into each file: recording time, timecode, takes and chapters, copies. `python ebc_media.py <folder>` prints a folder's timeline. |
| `ebc_timeline.py` | Puts the recordings in the order they were made, leaves out copies and files that say they failed, and reports everything odd about the set. Writes `ordered_config.json`. |
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
| `ebc_app.py`, `ebc_app_ui.html` | The browser front end. Writes a study file and runs the driver on it. |

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
- **The CR rate is only as stable as the CR/UR boundary** - see the note under *How a
  trial is scored*. Report the number of trials within one frame of the CS-US interval
  alongside any CR percentage.
- A pulse is rejected when it lights up away from where the rest of that LED's pulses did.
  That is what makes the wide, side-agnostic US window safe, but a genuine pulse whose
  brightest pixel is momentarily stolen by a reflection can be rejected with it: watch for
  a `lit N px from where the ... LED is` reason on a pulse whose duration matches the
  stimulus exactly.
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
