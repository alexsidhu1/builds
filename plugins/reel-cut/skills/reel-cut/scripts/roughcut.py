#!/usr/bin/env python3
"""Auto rough cut for a talking-head clip: drop retakes, false starts, stutters, fillers
and dead air, then trim every join (and any mid-sentence pause over 0.12s) to ~0.06s of real quiet
without crossing a transcribed word's edge.

Needs a WhisperX words.json for the same media (scripts/transcribe.py --chunk 8). Run with the reel-cut venv
(set up by install.sh):
  ~/.local/share/reel-cut-venv/bin/python roughcut.py <media> <words.json> [--out DIR]
      [--gap 0.06] [--sentence-gap 0.08] [--join 0.06] [--squeeze 0.12] [--keep A-B] [--cut A-B] [--add 'WORDS@A-B'] [--keep-first] [--no-render]

Writes to --out (default: <media dir>/roughcut):
  roughcut.mp4         hard-cut video at source resolution, no overlays (HDR is tone-mapped first on a Mac)
  edl.json             kept source ranges [[in, out], ...]
  words.json           kept words re-timed to the cut (feed straight to captions)
  report.md            every removal with its reason, for a quick human check
"""
import argparse
import json
import re
import shutil
import subprocess
import sys
import warnings
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")   # torch/torchcodec print noisy, harmless import warnings

SR = 16000
HOP = 0.01
FILLERS = {"um", "umm", "uh", "uhh", "uhm", "erm", "er", "ah", "ahh", "hmm", "mm", "mhm"}
HEAD = 0.04       # lead-in kept before a word onset
MIN_TAIL = 0.03   # always keep this much after a word ends


def norm(w):
    return re.sub(r"[^a-z0-9']", "", w.lower())


def load_db(media):
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", str(media), "-vn", "-ac", "1", "-ar", str(SR),
                          "-f", "f32le", "-"], capture_output=True, check=True).stdout
    x = np.frombuffer(raw, dtype=np.float32)
    n = int(HOP * SR)
    frames = x[: len(x) // n * n].reshape(-1, n)
    return 20 * np.log10(np.sqrt((frames ** 2).mean(axis=1)) + 1e-9), len(x) / SR, x


def refine(words, db, thr):
    """Snap each word's edges to where the audio really is above the speech threshold."""
    nf = len(db)
    f = lambda t: min(max(int(t / HOP), 0), nf - 1)
    for i, w in enumerate(words):
        prev_end = words[i - 1]["end"] if i else 0.0
        nxt = words[i + 1]["start"] if i + 1 < len(words) else len(db) * HOP
        s = f(max(w["start"] - 0.12, prev_end + 0.01))
        while s < f(w["start"] + 0.08) and db[s] < thr:
            s += 1
        e = f(w["end"])
        if db[e] >= thr:   # aligner ended early: extend while still voiced
            while e < f(min(w["end"] + 0.15, nxt - 0.02)) and db[e + 1] >= thr:
                e += 1
        else:              # aligner ended late: pull back to the last voiced frame
            while e > f(w["start"] + 0.05) and db[e] < thr:
                e -= 1
        w["s"], w["e"] = round(s * HOP, 3), round((e + 1) * HOP, 3)
        if w["e"] <= w["s"]:
            w["s"], w["e"] = w["start"], w["end"]
    return words


def voiced_gaps(words, db, thr, min_voiced=0.4):
    """Gaps between transcribed words that still hold speech-level sound."""
    out = []
    edges = [(0.0, words[0]["s"])] + [(words[i]["e"], words[i + 1]["s"]) for i in range(len(words) - 1)]
    edges.append((words[-1]["e"], len(db) * HOP))
    for g0, g1 in edges:
        if g1 - g0 > 0.35:
            voiced = (db[int(g0 / HOP):int(g1 / HOP)] >= thr).sum() * HOP
            if voiced >= min_voiced:
                out.append((g0, g1))
    return out


def repair(words, x, db, thr, language, model_name):
    """Whisper skips text it has just heard, so a clean retake right after a stumble often goes
    untranscribed. Re-transcribe each voiced gap on its own and add the words it finds."""
    gaps = voiced_gaps(words, db, thr, min_voiced=0.25)   # a lone "The first" is ~0.3s
    if not gaps:
        return words, 0
    import whisperx
    model = whisperx.load_model(model_name, "cpu", compute_type="int8", language=language)
    align_model, meta = whisperx.load_align_model(language_code=language, device="cpu")
    added = []
    for g0, g1 in gaps:
        a0, a1 = max(0.0, g0 - 0.3), min(len(x) / SR, g1 + 0.3)
        clip = x[int(a0 * SR):int(a1 * SR)].astype(np.float32)
        res = model.transcribe(clip, batch_size=4, language=language, chunk_size=8)
        if not res["segments"]:
            continue
        res = whisperx.align(res["segments"], align_model, meta, clip, "cpu", return_char_alignments=False)
        for seg in res["segments"]:
            for w in seg.get("words", []):
                if w.get("start") is None:
                    continue
                st, en = w["start"] + a0, w["end"] + a0
                if g0 - 0.02 <= st and en <= g1 + 0.02:
                    added.append({"word": w["word"].strip(), "start": round(st, 3), "end": round(en, 3),
                                  "score": round(w.get("score") or 0, 3), "repaired": True})
    merged = sorted(words + added, key=lambda w: w["start"])
    return merged, len(added)


def sentences(words, split=1.0):
    """Retakes are judged per sentence: break after . ? ! (or ...) and at any pause over 1s."""
    out, cur = [], [0]
    for i in range(1, len(words)):
        gap = words[i]["s"] - words[i - 1]["e"]
        if gap > split or words[i - 1]["word"].rstrip('"\')').endswith((".", "?", "!")):
            out.append(cur)
            cur = []
        cur.append(i)
    out.append(cur)
    return out


def ngrams(t, n=5):
    return {tuple(t[k:k + n]) for k in range(len(t) - n + 1)}


def find_retakes(words, phr, drop, keep_first):
    """Phrase-level: an earlier phrase is a retake of a later one (within ~25s) when they are
    mostly the same words, open with the same 5 words, or share any 5-word run. Keep the
    later take by default (usually the clean one). Lists like "one for research. One for
    writing." stay, because they differ in most of their words."""
    toks = [[norm(words[i]["word"]) for i in p] for p in phr]
    toks = [[t for t in ts if t] for ts in toks]
    for a in range(len(phr)):
        ta = toks[a]
        if len(ta) < 2:
            continue
        for b in range(a + 1, min(a + 6, len(phr))):
            if words[phr[b][0]]["s"] - words[phr[a][-1]]["e"] > 25:
                break
            tb = toks[b]
            ratio = SequenceMatcher(None, ta, tb[: len(ta) + 2]).ratio()
            if (ratio >= 0.7 and len(ta) >= 3) or (len(ta) >= 5 and ta[:5] == tb[:5]) \
                    or (ngrams(ta) & ngrams(tb)):
                loser = phr[b] if keep_first else phr[a]
                for i in loser:
                    drop.setdefault(i, f"retake ({'later' if keep_first else 'earlier'} take of: "
                                       f"\"{' '.join(words[j]['word'] for j in phr[b])[:70]}\")")
                break


def find_restarts(words, drop):
    """Within a line: "we should, we should really" or "stop tre- stop treating". The first
    attempt is dropped when the words right after it repeat it (the last word may be cut off)."""
    t = [norm(w["word"]) for w in words]
    i = 0
    while i < len(words):
        hit = False
        for m in range(i + 1, min(i + 9, len(words))):
            if words[m]["s"] - words[i]["s"] > 6 or not t[i] or t[m] != t[i]:
                continue
            A, B = t[i:m], t[m:m + (m - i)]
            if len(B) < len(A):
                continue
            same = all(A[k] == B[k] for k in range(len(A) - 1))
            last = A[-1] == B[-1] or (len(A[-1]) >= 2 and B[-1].startswith(A[-1]))
            if same and last:
                for k in range(i, m):
                    drop.setdefault(k, f"restart (\"{' '.join(w['word'] for w in words[i:m])}\")")
                i, hit = m, True
                break
        if not hit:
            i += 1


def suspects(words, db, thr, keep, drop):
    """Things the cutter can't judge by itself: words the aligner stretched over a long span
    (a hidden retake or mumble may be inside) and voiced audio in cut gaps (untranscribed speech)."""
    out = []
    for i, w in enumerate(words):
        if i in keep and w["e"] - w["s"] > 1.2:
            out.append((w["s"], w["e"], f"long word \"{w['word']}\" ({w['e'] - w['s']:.1f}s): hidden retake or mumble?"))
    ks = sorted(keep)
    for a, b in zip(ks, ks[1:]):
        g0, g1 = words[a]["e"], words[b]["s"]
        if g1 - g0 > 0.35:
            v = db[int(g0 / HOP):int(g1 / HOP)] >= thr
            for k in range(a + 1, b):   # audio of words we chose to drop isn't a surprise
                if k in drop:
                    v[max(0, int((words[k]["s"] - g0) / HOP)):max(0, int((words[k]["e"] - g0) / HOP))] = False
            voiced = v.sum() * HOP
            if voiced > 0.4:
                out.append((g0, g1, f"{voiced:.1f}s of untranscribed sound cut (filler, laugh or a missed word?)"))
    return out


def build_edl(words, keep, gap, sgap, dur):
    """Join kept words into source ranges, closing every gap to the target pause."""
    edl = []
    ks = [i for i in range(len(words)) if i in keep]
    for n, i in enumerate(ks):
        w = words[i]
        start = max(0.0, w["s"] - HEAD)
        if edl and ks[n - 1] == i - 1 and w["s"] - words[i - 1]["e"] <= gap:
            edl[-1][1] = w["e"]
            continue
        if edl:
            p = words[ks[n - 1]]
            want = sgap if p["word"][-1:] in ".?!," else gap
            tail = max(MIN_TAIL, want - HEAD)
            edl[-1][1] = min(edl[-1][1] + tail, start)
        edl.append([start, w["e"]])
    if edl:
        edl[-1][1] = min(edl[-1][1] + 0.25, dur)
    return [[round(a, 3), round(b, 3)] for a, b in edl if b - a > 0.05]


def split_pauses(edl, x, loud, words, longest):
    """Split pieces at mid-sentence pauses longer than `longest` so tighten() closes them too.
    A pause holding any part of a transcribed word is left alone (soft words read as quiet)."""
    n = int(0.005 * SR)
    db5 = 20 * np.log10(np.sqrt((x[: len(x) // n * n].reshape(-1, n) ** 2).mean(axis=1)) + 1e-9)
    out = []
    for a, b in edl:
        cur, i, end = a, int(a / 0.005) + 10, int(b / 0.005) - 10
        while i < end:
            if db5[i] < loud:
                j = i
                while j < end and db5[j] < loud:
                    j += 1
                q0, q1 = i * 0.005, j * 0.005
                inside = lambda w: min(w["e"], q1) - max(w["s"], q0) > 0.5 * (w["e"] - w["s"])
                if q1 - q0 > longest and not any(inside(w) for w in words):
                    mid = round((q0 + q1) / 2, 3)
                    out.append([cur, mid])
                    cur = mid
                i = j
            else:
                i += 1
        out.append([cur, b])
    return out


def tighten(edl, x, loud, join, words, max_trim=0.25):
    """Measure the real quiet on both sides of every join (word tails ring on past the aligner's
    edges) and trim it so each join holds about `join` seconds of quiet: 40% before the cut, 60%
    lead-in before the next word. Soft consonants ("Cl", "-ng", stop closures) read as quiet, so a
    trim never crosses a transcribed word's edge (raw or refined, whichever is wider)."""
    n = int(0.005 * SR)
    db5 = 20 * np.log10(np.sqrt((x[: len(x) // n * n].reshape(-1, n) ** 2).mean(axis=1)) + 1e-9)
    f = lambda t: min(max(int(round(t / 0.005)), 0), len(db5) - 1)
    w0 = lambda w: min(w["s"], w["start"])
    w1 = lambda w: max(w["e"], w["end"])
    edl = [list(p) for p in edl]
    for k in range(len(edl) - 1):
        a, b = edl[k]
        c, d = edl[k + 1]
        last = [w1(w) for w in words if a - 0.3 <= w["s"] < b]
        first = [w0(w) for w in words if c - 0.3 <= w["s"] < d]
        e = f(b)
        # a lone 5ms blip (click, consonant release) doesn't count as sound
        while e > f(max(a, b - max_trim)) and (db5[e - 1] < loud or db5[e - 2] < loud):
            e -= 1
        s = f(c)
        while s < f(min(d, c + max_trim)) and (db5[s] < loud or db5[s + 1] < loud):
            s += 1
        end = max(e * 0.005 + join * 0.4, max(last) + 0.01 if last else a + 0.05)
        start = min(s * 0.005 - join * 0.6, min(first) - 0.02 if first else d - 0.05)
        edl[k][1] = round(min(b, end), 3)
        edl[k + 1][0] = round(max(c, start), 3)
    return edl


def render(media, edl, out_mp4):
    probe = lambda k: subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                      f"stream={k}", "-of", "default=nw=1:nk=1", str(media)],
                                     capture_output=True, text=True).stdout.split("\n")[0].strip()
    src = Path(media)
    mac = sys.platform == "darwin" and shutil.which("avconvert")
    if probe("color_transfer") in ("arib-std-b67", "smpte2084"):
        if mac:   # iPhone HDR: Apple's own converter gives the cleanest SDR
            sdr = out_mp4.parent / "source-sdr.mov"
            if not sdr.exists():
                print(".. HDR source, tone-mapping with avconvert first")
                subprocess.run(["avconvert", "-s", str(src), "-p", "Preset3840x2160", "-o", str(sdr), "--replace"],
                               check=True, capture_output=True)
            src = sdr
        else:
            print(".. HDR source: no avconvert here, so colours may look washed out. "
                  "Record in SDR (iPhone: Settings > Camera > Formats, HDR Video off) for best results.")
    encoders = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"], capture_output=True, text=True).stdout
    venc = (["-c:v", "h264_videotoolbox", "-q:v", "80"] if "h264_videotoolbox" in encoders
            else ["-c:v", "libx264", "-crf", "16", "-preset", "medium"])
    fc, labels = [], []
    for k, (a, b) in enumerate(edl):
        d = b - a
        fc.append(f"[0:v]trim=start={a}:end={b},setpts=PTS-STARTPTS[v{k}];"
                  f"[0:a]atrim=start={a}:end={b},asetpts=PTS-STARTPTS,"
                  f"afade=t=in:st=0:d=0.012,afade=t=out:st={max(d - 0.012, 0):.3f}:d=0.012[a{k}];")
        labels.append(f"[v{k}][a{k}]")
    # iPhone footage is often full-range (yuvj420p); players tint it unless converted to limited range
    fc.append("".join(labels) + f"concat=n={len(edl)}:v=1:a=1[vc][a];"
              "[vc]scale=in_range=auto:out_range=tv,format=yuv420p[v]")
    script = out_mp4.parent / "filter.txt"
    script.write_text("\n".join(fc))
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-filter_complex_script", str(script),
                    "-map", "[v]", "-map", "[a]", *venc,
                    "-pix_fmt", "yuv420p", "-color_range", "tv", "-color_primaries", "bt709", "-color_trc", "bt709",
                    "-colorspace", "bt709", "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart",
                    str(out_mp4)], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("media")
    ap.add_argument("words")
    ap.add_argument("--out")
    ap.add_argument("--gap", type=float, default=0.06, help="pause kept mid-sentence (s)")
    ap.add_argument("--sentence-gap", type=float, default=0.08, help="pause kept after . ? ! , (s)")
    ap.add_argument("--join", type=float, default=0.06,
                    help="total quiet left at each cut, measured on the audio (s); 0 disables")
    ap.add_argument("--squeeze", type=float, default=0.12,
                    help="also cut mid-sentence pauses longer than this down to --join (s); 0 disables")
    ap.add_argument("--keep-first", action="store_true", help="keep the first take instead of the last")
    ap.add_argument("--keep", action="append", default=[], metavar="A-B",
                    help="force-keep words starting in this source range (repeatable), e.g. 186-192")
    ap.add_argument("--cut", action="append", default=[], metavar="A-B",
                    help="force-cut words starting in this source range (repeatable), e.g. 194-208")
    ap.add_argument("--add", action="append", default=[], metavar="'WORDS@A-B'",
                    help="put back words Whisper missed, spread over source range A-B, e.g. 'The first@19.47-19.87'")
    ap.add_argument("--no-render", action="store_true")
    ap.add_argument("--no-repair", action="store_true", help="skip re-transcribing voiced gaps")
    ap.add_argument("--language", default="en")
    ap.add_argument("--model", default="large-v3-turbo")
    a = ap.parse_args()

    media = Path(a.media).expanduser().resolve()
    out = Path(a.out).expanduser() if a.out else media.parent / "roughcut"
    out.mkdir(parents=True, exist_ok=True)
    words = [dict(w) for w in json.loads(Path(a.words).read_text())["words"]]
    for spec in a.add:
        text, r = spec.rsplit("@", 1)
        t0, t1 = (float(v) for v in r.split("-"))
        toks = text.split()
        step = (t1 - t0) / len(toks)
        words += [{"word": tok, "start": round(t0 + k * step, 3), "end": round(t0 + (k + 1) * step - 0.02, 3)}
                  for k, tok in enumerate(toks)]
    words.sort(key=lambda w: w["start"])
    if not words:
        sys.exit("no words in transcript")

    db, dur, x = load_db(media)
    x_audio = x
    voiced = db[db > -60]
    thr = max(float(np.percentile(voiced, 90)) - 28 if len(voiced) else -45, -50)
    words = refine(words, db, thr)
    repaired = 0
    if not a.no_repair:
        words, repaired = repair(words, x, db, thr, a.language, a.model)
        if repaired:
            words = refine([{k: v for k, v in w.items() if k not in ("s", "e")} for w in words], db, thr)

    drop = {}
    for i, w in enumerate(words):
        if norm(w["word"]) in FILLERS:
            drop[i] = "filler word"
    phr = sentences(words)
    find_retakes(words, phr, drop, a.keep_first)
    find_restarts(words, drop)
    rng = lambda r: tuple(float(v) for v in r.split("-"))
    for r in a.keep:
        x, y = rng(r)
        for i, w in enumerate(words):
            if x <= w["s"] <= y:
                drop.pop(i, None)
    for r in a.cut:
        x, y = rng(r)
        for i, w in enumerate(words):
            if x <= w["s"] <= y:
                drop[i] = "cut by hand (--cut)"
    keep = {i for i in range(len(words)) if i not in drop}

    edl = build_edl(words, keep, a.gap, a.sentence_gap, dur)
    if a.join:
        loud = float(np.percentile(voiced, 90)) - 16
        if a.squeeze:
            edl = split_pauses(edl, x_audio, loud, [words[i] for i in keep], a.squeeze)
        edl = tighten(edl, x_audio, loud, a.join, [words[i] for i in sorted(keep)])
    kept_dur = sum(b - x for x, b in edl)

    # re-time kept words onto the cut timeline
    cut_words, offset, placed = [], 0.0, set()
    for x, b in edl:
        for i in sorted(keep):
            w = words[i]
            # join tightening can shave a word edge, and a pause split can land inside one
            if i not in placed and x - 0.3 <= w["s"] < b and w["e"] > x:
                placed.add(i)
                cut_words.append({"text": w["word"], "start": round(max(w["s"], x) - x + offset, 3),
                                  "end": round(min(w["e"], b) - x + offset, 3)})
        offset += b - x
    (out / "edl.json").write_text(json.dumps(edl))
    (out / "words.json").write_text(json.dumps(cut_words))

    flags = suspects(words, db, thr, keep, drop)
    removed_words = len(drop)
    lines = [f"# Rough cut report: {media.name}", "",
             f"- Source {dur:.1f}s -> cut {kept_dur:.1f}s ({100 * kept_dur / dur:.0f}%), {len(edl)} pieces",
             f"- Repair pass recovered {repaired} untranscribed words from voiced gaps",
             f"- Speech threshold {thr:.1f} dB, gaps closed to {a.gap}s ({a.sentence_gap}s after punctuation)",
             f"- {removed_words} words removed ({sum(1 for r in drop.values() if r.startswith('retake'))} in retakes, "
             f"{sum(1 for r in drop.values() if r.startswith('restart'))} in restarts, "
             f"{sum(1 for r in drop.values() if r == 'filler word')} fillers)", "",
             "## Removed (check these by ear)", "", "| Source time | Words | Why |", "|---|---|---|"]
    i = 0
    while i < len(words):
        if i in drop:
            j, why = i, drop[i]
            while j + 1 < len(words) and drop.get(j + 1) == why:
                j += 1
            lines.append(f"| {words[i]['s']:.2f}-{words[j]['e']:.2f} | "
                         f"{' '.join(w['word'] for w in words[i:j + 1])} | {why} |")
            i = j + 1
        else:
            i += 1
    if flags:
        lines += ["", "## Check by ear", "", "| Source time | What |", "|---|---|"]
        lines += [f"| {x:.2f}-{y:.2f} | {why} |" for x, y, why in flags]
    lines += ["", "## Kept script", "", " ".join(w["text"] for w in cut_words)]
    (out / "report.md").write_text("\n".join(lines) + "\n")

    if not a.no_render:
        render(media, edl, out / "roughcut.mp4")
    print(f"ok {out}  {dur:.1f}s -> {kept_dur:.1f}s  pieces={len(edl)}  removed_words={removed_words}  flags={len(flags)}")


if __name__ == "__main__":
    main()
