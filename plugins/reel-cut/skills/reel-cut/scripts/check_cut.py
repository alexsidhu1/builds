#!/usr/bin/env python3
"""Measure dead air in a rough cut: the quiet at every join and the longest quiet anywhere.

  ~/.local/share/reel-cut-venv/bin/python check_cut.py <roughcut dir> [--max 0.1]

Reads <dir>/roughcut.mp4 and <dir>/edl.json. Reports two levels:
  silence  below -45 dB: true dead air. This is the number that must stay under --max.
  soft     below -34 dB: also counts soft word tails ("-ng", "Cl" onsets). Joins here run
           higher by design, because trimming into them clips words.
Exits 1 if any join or pause holds more true silence than --max.
"""
import argparse
import json
import subprocess
import sys

import numpy as np

STEP = 0.005


def runs(db, th):
    out, i = [], 0
    while i < len(db):
        if db[i] < th:
            j = i
            while j < len(db) and db[j] < th:
                j += 1
            out.append((i * STEP, j * STEP))
            i = j
        else:
            i += 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--max", type=float, default=0.1)
    a = ap.parse_args()
    edl = json.load(open(f"{a.dir}/edl.json"))
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", f"{a.dir}/roughcut.mp4", "-vn", "-ac", "1",
                          "-ar", "16000", "-f", "f32le", "-"], capture_output=True, check=True).stdout
    x = np.frombuffer(raw, np.float32)
    n = int(16000 * STEP)
    db = 20 * np.log10(np.sqrt((x[: len(x) // n * n].reshape(-1, n) ** 2).mean(axis=1)) + 1e-9)
    joins = np.cumsum([b - s for s, b in edl])[:-1]
    end = len(db) * STEP - 0.4          # the tail after the last word isn't a gap
    bad = False
    for name, th in (("silence", -45), ("soft", -34)):
        rs = [(s, e) for s, e in runs(db, th) if s > 0.05 and e < end]
        at = [max([e - s for s, e in rs if s - 0.01 <= j <= e + 0.01] or [0]) for j in joins]
        worst = max(rs, key=lambda r: r[1] - r[0], default=(0, 0))
        print(f"{name:8s} (<{th} dB): joins max {max(at, default=0):.3f}s mean {np.mean(at) if at else 0:.3f}s, "
              f"longest anywhere {worst[1] - worst[0]:.3f}s at {worst[0]:.2f}s")
        if name == "silence":
            for s, e in rs:
                if e - s > a.max:
                    bad = True
                    print(f"  over {a.max}s: {s:.2f}-{e:.2f} ({e - s:.3f}s)")
    print(f"{len(edl)} pieces, {sum(b - s for s, b in edl):.1f}s")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
