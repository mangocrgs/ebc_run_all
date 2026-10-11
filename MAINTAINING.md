# Maintaining EBC Analyzer without the person who wrote it

This file is for whoever has to change the app next, with or without a programmer at
hand. It says where each kind of change lives, how to check a change did not break
anything, and how to get the change into the installed app. The README is the manual for
*what* the pipeline does; this is the manual for *touching* it.

Everything is plain source code in this folder. The installed app (`app/`, the Desktop
shortcut) is only a frozen copy of these files. Every change starts in the source here and
ends with a rebuild. The source is also on GitHub at `mangocrgs/ebc_run_all`, so a copy of
this folder can always be recovered with `git clone`.

## 1. Which file to open

| You want to change | Open this file |
|---|---|
| How a trial is classified (CR, ?CR, UR, startle), blink metrics, the CR window | `ebc_score.py` |
| What counts as a trial, how CS and US are paired, block boundaries, protocol checks | `ebc_protocol.py` |
| How the stimulus LEDs are found and read | `ebc_stimulus.py`, `ebc_locate.py`, `ebc_signal.py` |
| Whether a recording's CS LED is trusted, and what happens when it is not | `ebc_triage.py` |
| Eyelid measurement (face landmarks, EAR, the crop) | `ebc_eyes.py` |
| Recording order from the camera clock, take/chapter metadata | `ebc_media.py`, `ebc_timeline.py` |
| The figures (PNG) | `ebc_figures.py` |
| The Excel workbooks: sheets, read-me text, Excel charts | `ebc_workbooks.py` |
| The CSV exports | `ebc_export_csv.py` |
| The four video clips | `ebc_clips.py` |
| The LED check pages and the trial filmstrip | `ebc_qc.py` |
| What the window *looks like*: layout, text, buttons, colours | `ebc_app_ui.html` |
| What the window *does* behind the buttons: name and role checks, renaming, running | `ebc_app.py` |
| Default protocol numbers, the shared colour palette, the version number | `ebc_config.py` |
| The order of the stages, which ones may fail without stopping the run | `ebc_run_all.py` |
| How a study file (`studies/*.json`) is read | `ebc_config.py` |

Every `.py` file opens with a docstring saying what it does and how to run it on its own.
The README has a *Files* table and a section *How a trial is scored*. Read those before
anything else; they are the map, and most scientific changes land in `ebc_score.py`.

## 2. The loop for any change

Do these in order, every time. Skipping step 2 is how a bug ships.

1. **Edit** the one file, in any text editor (Notepad++, VS Code, even Notepad).

2. **Run the tests.** There are over a hundred, and they catch most mistakes, including
   several that are invisible until a workbook is opened in Excel.

   ```
   cd "C:\Users\marga\EBC Analyzer"
   py -3 test_ebc.py
   ```

   It prints one line per test and ends with `118 passed, 0 failed` (the count grows as
   tests are added). A failure prints the name of the test and the line it stopped on;
   the message is written to say what went wrong. The file has its own runner, so nothing
   needs installing; `py -3 -m pytest test_ebc.py -q` works too if pytest is present.

3. **Re-run one participant from the cache.** Seconds, no video decoding:

   ```
   py -3 ebc_run_all.py --config studies/thomas.json --from score
   ```

   `--from score` is for anything in scoring, figures, workbooks, clips or CSVs.
   `--from report` is enough for figures, workbooks, clips and CSVs alone.
   Leave off `--from` only if you changed how the LEDs or the eyelids are read; that is a
   full run, around an hour per participant.

   Then open the workbook that came out in `results/Thomas/` and look at it.

4. **Commit**, so there is always a known-good state to go back to:

   ```
   git add -A
   git commit -m "One line saying what changed and why"
   git push
   ```

   To see what you changed before committing: `git diff`. To throw a change away:
   `git checkout -- the_file.py`. To go back to the last shipped version of everything:
   `git log --oneline` to find it, then `git checkout <hash> -- .`.

5. **Rebuild the app** only when you want the Desktop shortcut to see the change
   (section 4). Until then the shortcut runs the old frozen copy, and that is fine for
   testing in the source tree.

Running from the source tree without rebuilding: double-click `EBC Analyzer.bat`, or
`py -3 ebc_app.py`. That is the same window, reading the files in this folder.

## 3. Working with an AI assistant on the web

This works well, and it is how the app was written. The rules that make it work:

- **Paste the whole file**, not a fragment. All of them fit in one message (the largest,
  `ebc_workbooks.py`, is about 100 KB). The assistant needs to see the helpers a function
  calls, or it will invent ones that do not exist.
- **For scoring questions, also paste the README section *How a trial is scored***, so it
  knows the window definitions and the pooled closure scale.
- **Ask for the complete rewritten function**, start to end, so you can select the old one
  in the editor and replace it in one go. A "change line 412 to ..." answer is easy to
  apply wrongly.
- **Paste the test output back.** If step 2 fails after a change, copy the whole failure
  text to the assistant. It usually names the line and the cause.
- **Paste `test_ebc.py` when the change is a rule.** Ask the assistant to add a test for
  the new behaviour. A rule with a test survives the next change; one without may not.
- **Say what the file is for.** One sentence from the table above is enough context.

Three traps to tell the assistant about every time, because they are not visible from one
file alone:

- **The page and the Python palette must stay identical.** The colours in `:root` at the
  top of `ebc_app_ui.html` and `PALETTE` in `ebc_config.py` are the same table twice, and a
  test fails if they differ. Change one, change both.
- **Chart colours in the workbooks must be six hex digits** (`RRGGBB`). Cell fills take
  eight (`FFRRGGBB`), charts do not, and an eight-digit chart colour makes Excel refuse the
  whole workbook, not just the chart. Use `C.dml()` for charts and `C.xl()` for cells; a
  test forbids `C.xl(` inside chart properties.
- **A new pipeline stage must be listed in `packaging/ebc_analyzer.spec`**, or the frozen
  app will not carry it. The spec checks this and refuses to build if one is missing, so
  the failure is loud, but it happens at build time, not at test time.

Two things the assistant cannot know unless told:

- Onsets are quantised to one video frame (8.34 ms at 119.88 fps). About one paired trial
  in eight sits within a frame of the CR/UR boundary. Any change to the window moves those
  trials; a CR rate changing by a few points after a crop or alignment change is noise,
  not a regression. See the README section on the ?CR band.
- The protocol is a *test*, never an instruction: the code must not renumber or drop a
  trial to make the recording match the study file. An assistant will sometimes propose
  exactly that as a "fix". Decline it.

## 4. Rebuilding the installed app

Needed once on the build machine: Python 3.12, `ffmpeg` on PATH, Inno Setup
(`winget install JRSoftware.InnoSetup`), and
`py -3 -m pip install pyinstaller opencv-python mediapipe numpy scipy matplotlib openpyxl pillow pywebview`.

1. **Bump the version in two places**: `VERSION = "1.7"` near the top of `ebc_config.py`
   (the app shows it) and `#define AppVersion "1.7"` near the top of
   `packaging/ebc_analyzer.iss` (the installer's name and the version Windows shows).

2. **Build.** Double-click `packaging\build.bat`. It cuts the logo assets, runs PyInstaller
   (a few minutes), then packs the installer (several minutes, it is compressing about
   900 MB). It ends with a `pause`, so run it by double-clicking, not from a script.
   The same three commands by hand, for when `build.bat` is unsuitable:

   ```
   py -3 packaging\make_assets.py
   py -3 -m PyInstaller --clean --noconfirm --distpath build\dist --workpath build\work packaging\ebc_analyzer.spec
   "%LOCALAPPDATA%\Programs\Inno Setup 6\ISCC.exe" packaging\ebc_analyzer.iss
   ```

3. **Install** by running the new `packaging\Setup EBC Analyzer <version>.exe`. It
   upgrades the existing install in place. Windows will say "Windows protected your PC"
   because the installer is unsigned: More info, Run anyway.

4. **Check the installed copy, not the source tree.** The quickest proof that the frozen
   app carries the change is to run one stage through it:

   ```
   "app\EBC Analyzer.exe" --stage ebc_run_all.py --config studies/thomas.json --from report
   ```

**Never patch files inside `app\_internal\` by hand.** The stage scripts there are loose
`.py` files, but the modules they import are frozen inside the exe, so a hand-patched file
runs new code against old helpers. Rebuild instead.

## 5. Learning enough to follow the code

Nothing here needs code written from scratch; it needs code *read*. A short introduction to
Python covering functions, lists and dictionaries, about ten hours, is enough to follow
every file in this folder. Then read `ebc_score.py` from the top with the README section
*How a trial is scored* open beside it. Each function is a paragraph of the method.

Three habits that matter more than any course:

- Change one thing, test, run, look at the output, commit. Then the next thing.
- When a number changes and you do not know why, do not adjust the code until you do.
  `compare_manual.py` compares the analyser to hand scoring; the README section *Trials to
  score by hand* says how to add a participant to it.
- When in doubt, `git diff` shows exactly what is different from the last version that
  worked.

## 6. Where things are

| | |
|---|---|
| Source and tests | this folder |
| Study files, one per participant | `studies/` |
| Outputs per participant (not in git, regenerated by a run) | `results/<name>/` |
| Cache of the video passes (what makes `--from score` fast) | `results/<name>/_work/` |
| The installed app the Desktop shortcut points at | `app/` |
| Installers, one per version | `packaging/` |
| The GitHub copy | `https://github.com/mangocrgs/ebc_run_all` |
