# Builds

Things I've built with Claude for my own work, packaged so you can use them too.

Each build is a Claude Code plugin. Add this repo once, then install whichever builds you want.

```
/plugin marketplace add alexsidhu1/builds
/plugin install reel-cut@builds
```

Run those two lines inside [Claude Code](https://code.claude.com). Updates arrive when you run `/plugin marketplace update builds`.

## The builds

| Build | What it does | Run it |
|---|---|---|
| [reel-cut](plugins/reel-cut/skills/reel-cut/SKILL.md) | Chops a raw talking-head video into a tight cut with no dead space. Retakes, fillers and pauses come out, every join is closed to under 0.1s of silence, and no word gets clipped. | `/reel-cut:reel-cut path/to/video.mp4`, or just ask Claude to "cut this video up" |

## reel-cut

I record my reels in one long take, stumbles and all. Cutting the dead air by hand took longer than filming. Now Claude does it.

**What it does**
1. Transcribes the take with word-level timestamps (WhisperX, runs locally and free).
2. Finds retakes, restarts and filler words, and keeps your last, cleanest take of each line.
3. Closes every gap, then measures the actual audio at each join and trims it to about 0.06s of silence, without cutting into the edge of a word.
4. Checks the finished cut for any silence over 0.1s, and re-transcribes it to make sure nothing got clipped.

**What you get:** the trimmed video at full resolution with no captions or graphics, ready for CapCut or any editor, plus a report listing every line it removed so you can put anything back.

**A real run:** a 92.5s take came out at 72.9s in 46 pieces. The longest silence left anywhere was 0.085s.

**One-time setup** (Mac, Windows or Linux):
- [ffmpeg](https://ffmpeg.org) (`brew install ffmpeg` on a Mac)
- [uv](https://docs.astral.sh/uv/) or Python 3.12

Then ask Claude to run the skill's `install.sh`, or run it yourself. The first transcription downloads the speech models (about 1.5 GB). After that it runs offline.

**Want it without Claude Code?** After setup, from the skill folder:

```bash
bash cut.sh my-video.mp4 --no-render   # dry run: read the report first
bash cut.sh my-video.mp4               # render the cut
```

**Honest limits**
- It decides what's a retake by matching repeated phrases. Two different lines that share five words in a row can be mistaken for a retake, so always read the report before you post. Put a line back with `--keep`.
- Whisper occasionally misses a short phrase entirely, and the cutter treats it as silence. The skill shows you how to put it back with `--add`.
- iPhone HDR footage is converted cleanly on a Mac. On Windows or Linux, record in SDR for the best colour.

## Licence

MIT. Use it, change it, ship it.

Built by [Alex Sidhu](https://alexsidhu.com), Whitehorse AI.
