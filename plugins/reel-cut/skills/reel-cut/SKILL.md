---
name: reel-cut
description: Use when someone wants a raw talking-head video chopped into a tight cut with no dead space, says "cut this up", "chop this video", "remove the pauses", "cut out the dead air", "tighten this clip" or "rough cut this", or hands over a raw .mov/.mp4 take full of pauses, stumbles and retakes. Output is the trimmed clip only, no captions or graphics, ready for CapCut or any editor.
argument-hint: "<raw video path>"
---

# Reel Cut (no dead space)

Turns a raw talking-head take into a clean cut. Earlier takes, restarts, fillers and dead air go, and every join is closed to about 0.06s of quiet. No styling: the output is the file you finish in your editor.

Target: **true silence under 0.1s anywhere in the cut, and no word clipped.**

## Files (relative to this skill's folder)

| Path | What |
|---|---|
| `install.sh` | One-time setup: a Python 3.12 environment with WhisperX at `~/.local/share/reel-cut-venv` |
| `cut.sh <video> [flags]` | Runs the three steps below in one go |
| `scripts/transcribe.py` | WhisperX transcript with word-level timestamps |
| `scripts/roughcut.py` | The cutter. Drops retakes, restarts and fillers, closes gaps, then measures the real audio at every join and trims it without crossing a word's edge |
| `scripts/check_cut.py` | Dead-air check on a finished cut. Exits 1 if any silence is over 0.1s |

`PY=~/.local/share/reel-cut-venv/bin/python`. If it doesn't exist, run `bash install.sh` from this skill's folder first (needs ffmpeg, and uv or Python 3.12). Run every `cut.sh` and `scripts/` command from this skill's folder too.

## Workflow

### 1. Probe
```bash
ffprobe -v error -show_entries stream=codec_type,width,height,color_transfer -show_entries format=duration -of compact "<video>"
ffmpeg -v error -i "<video>" -vf "select='gt(scene,0.25)',showinfo" -an -f null - 2>&1 | grep -o "pts_time:[0-9.]*"
```
HDR iPhone footage (`arib-std-b67`) is tone-mapped automatically on a Mac. Scene-change times mean the file is several clips stitched together; if a cut lands across a stitch, use `--cut A-B` around it.

### 2. Dry run, then read the report
```bash
bash cut.sh "<video>" --no-render
```
This transcribes in 8-second chunks (in 30s chunks Whisper skips a retake it has just heard) and writes `<video>-cut/roughcut/report.md` without rendering.

**Check every row of the Removed table against the transcript.** The retake finder matches shared five-word runs, so it gets two things wrong:
- A *different* line that reuses a phrase gets cut as a "retake". Put it back with `--keep A-B` (source seconds).
- Two alternate endings: the later one wins. Swap with `--keep` / `--cut`.

Also read the kept script for missing words, like a line that suddenly starts mid-sentence. Whisper sometimes never hears a short phrase, so the cutter treats it as a gap. Transcribe a few seconds of the source around it to get the times, then put it back with `--add "The first@19.47-19.87"` (words spread evenly over that range). Re-run with `--no-render` until the kept script reads right.

### 3. Render and check
```bash
bash cut.sh "<video>" [--keep ..] [--cut ..] [--add ..]
```
The last step prints the dead-air check. "silence" must stay under 0.1s. "soft" runs higher by design: that's word tails like "-ng" and soft onsets like "Cl", and trimming into them clips the word.

### 4. Verify no word is clipped (never skip)
```bash
$PY scripts/transcribe.py "<video>-cut/roughcut/roughcut.mp4" --out "<video>-cut/verify" --chunk 8
```
Every line should appear once, in order. A wrong word sitting right on a join ("Lord" for "Claude", "improved" for "improving") means a clipped word. Garbles at 8-second chunk edges are Whisper, not the cut: re-transcribe 3-4 seconds around the spot to settle it.

### 5. Hand over
Give the user `<video>-cut/roughcut/roughcut.mp4`, plus a short table: original vs cut length, how many pieces, what was removed as retakes, any `--keep`/`--cut`/`--add` used, and the dead-air numbers.

## Output
`<video>-cut/roughcut/`: `roughcut.mp4` (source resolution, H.264, no overlays), `edl.json` (kept source ranges), `words.json` (kept words re-timed to the cut, ready for captions), `report.md`.

## Tuning

| Flag | Default | What |
|---|---|---|
| `--join` | 0.06 | Quiet left at each join, measured on the audio |
| `--squeeze` | 0.12 | Mid-sentence pauses longer than this are cut down too; 0 keeps natural pauses |
| `--gap` / `--sentence-gap` | 0.06 / 0.08 | First-pass word grouping |
| `--keep-first` | off | Keep the first take of a repeated line instead of the last |

## Notes
- If the cut still feels slow, the delivery is usually the problem, not the gaps. Try a 1.05-1.1x speed-up in your editor rather than cutting into words.
- Don't judge dead air at -34 dB and trim to it: soft consonants live there. True silence is around -45 dB.
- The footage stays in order. Takes are never reordered.
