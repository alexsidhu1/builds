#!/usr/bin/env python3
"""WhisperX transcription with forced-aligned word timings.

Run with the reel-cut venv (WhisperX needs Python 3.12; install.sh sets it up):
  ~/.local/share/reel-cut-venv/bin/python transcribe.py <media> [--out DIR] [--model M]
      [--language en] [--chunk 30] [--diarize] [--min-speakers N] [--max-speakers N] [--gap 0.15]

Writes <stem>.words.json, <stem>.srt and <stem>.txt to --out (default: next to the media).
words.json shape:
  {"source", "model", "language", "duration",
   "words": [{"word", "start", "end", "score", "speaker"?, "interpolated"?}],
   "segments": [{"start", "end", "text", "speaker"?}],
   "pauses": [{"after_word_index", "start", "end", "length"}]}
"""
import argparse
import json
import os
import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def fmt_srt(t):
    ms = int(round(t * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def fill_missing(words):
    """Alignment skips tokens with characters outside the wav2vec2 vocab (digits, $, %).
    Give them times by spreading the gap between their timed neighbours, and flag them."""
    n = len(words)
    i = 0
    while i < n:
        if words[i].get("start") is not None:
            i += 1
            continue
        j = i
        while j < n and words[j].get("start") is None:
            j += 1
        left = words[i - 1]["end"] if i > 0 else (words[j]["start"] if j < n else 0.0)
        right = words[j]["start"] if j < n else left + 0.3 * (j - i)
        step = max(right - left, 0.0) / (j - i)
        for k in range(i, j):
            words[k]["start"] = round(left + step * (k - i), 3)
            words[k]["end"] = round(left + step * (k - i + 1), 3)
            words[k]["interpolated"] = True
        i = j
    return words


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("media")
    ap.add_argument("--out")
    ap.add_argument("--model", default="large-v3-turbo")
    ap.add_argument("--language", default="en")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--chunk", type=int, default=30,
                    help="max seconds per VAD chunk; use ~8 for raw takes so retakes aren't merged away")
    ap.add_argument("--diarize", action="store_true")
    ap.add_argument("--min-speakers", type=int)
    ap.add_argument("--max-speakers", type=int)
    ap.add_argument("--gap", type=float, default=0.15, help="report pauses longer than this (s)")
    a = ap.parse_args()

    import whisperx

    src = Path(a.media).expanduser().resolve()
    out = Path(a.out).expanduser() if a.out else src.parent
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    audio = whisperx.load_audio(str(src))
    model = whisperx.load_model(a.model, "cpu", compute_type="int8", language=a.language)
    result = model.transcribe(audio, batch_size=a.batch_size, language=a.language, chunk_size=a.chunk)
    del model

    align_model, meta = whisperx.load_align_model(language_code=a.language, device="cpu")
    result = whisperx.align(result["segments"], align_model, meta, audio, "cpu",
                            return_char_alignments=False)
    del align_model

    if a.diarize:
        token = os.environ.get("HF_TOKEN")
        if not token:
            sys.exit("--diarize needs HF_TOKEN set (see SKILL.md, Speaker labels).")
        from whisperx.diarize import DiarizationPipeline
        diar = DiarizationPipeline(token=token, device="cpu")
        kw = {k: v for k, v in (("min_speakers", a.min_speakers), ("max_speakers", a.max_speakers)) if v}
        result = whisperx.assign_word_speakers(diar(audio, **kw), result)

    words = []
    for seg in result["segments"]:
        for w in seg.get("words", []):
            item = {"word": w["word"].strip(), "start": w.get("start"), "end": w.get("end"),
                    "score": round(w["score"], 3) if w.get("score") is not None else None}
            if "speaker" in w:
                item["speaker"] = w["speaker"]
            words.append(item)
    words = fill_missing(words)
    for w in words:
        w["start"], w["end"] = round(w["start"], 3), round(w["end"], 3)

    pauses = []
    for i in range(len(words) - 1):
        g = words[i + 1]["start"] - words[i]["end"]
        if g > a.gap:
            pauses.append({"after_word_index": i, "start": words[i]["end"],
                           "end": words[i + 1]["start"], "length": round(g, 3)})

    segments = [{"start": round(s["start"], 3), "end": round(s["end"], 3), "text": s["text"].strip(),
                 **({"speaker": s["speaker"]} if "speaker" in s else {})} for s in result["segments"]]
    duration = round(len(audio) / 16000, 3)
    data = {"source": str(src), "model": a.model, "language": a.language, "duration": duration,
            "words": words, "segments": segments, "pauses": pauses}

    stem = out / src.stem
    Path(f"{stem}.words.json").write_text(json.dumps(data, indent=1))
    Path(f"{stem}.txt").write_text("\n".join(s["text"] for s in segments) + "\n")
    srt = []
    for i, s in enumerate(segments, 1):
        who = f"[{s['speaker']}] " if "speaker" in s else ""
        srt.append(f"{i}\n{fmt_srt(s['start'])} --> {fmt_srt(s['end'])}\n{who}{s['text']}\n")
    Path(f"{stem}.srt").write_text("\n".join(srt))

    flagged = sum(1 for w in words if w.get("interpolated"))
    print(f"ok {stem}.words.json  words={len(words)} pauses>{a.gap}s={len(pauses)} "
          f"interpolated={flagged} audio={duration}s took={time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
