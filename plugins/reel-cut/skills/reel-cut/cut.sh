#!/usr/bin/env bash
# One command: transcribe, cut, check.
#   bash cut.sh <video> [extra roughcut.py flags, e.g. --keep 36-44 --no-render]
# Output lands in <video folder>/<video name>-cut/: roughcut.mp4, report.md, edl.json, words.json.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
PY="${REEL_CUT_PY:-${REEL_CUT_VENV:-$HOME/.local/share/reel-cut-venv}/bin/python}"
[ -x "$PY" ] || { echo "Run install.sh first (no Python found at $PY)."; exit 1; }

VIDEO="$1"; shift
NAME="$(basename "${VIDEO%.*}")"
OUT="$(cd "$(dirname "$VIDEO")" && pwd)/${NAME}-cut"
mkdir -p "$OUT"

echo "1/3 Transcribing (8s chunks so retakes aren't skipped)..."
[ -f "$OUT/$NAME.words.json" ] || "$PY" "$HERE/scripts/transcribe.py" "$VIDEO" --out "$OUT" --chunk 8

echo "2/3 Cutting..."
"$PY" "$HERE/scripts/roughcut.py" "$VIDEO" "$OUT/$NAME.words.json" --out "$OUT/roughcut" ${@+"$@"}

if [ -f "$OUT/roughcut/roughcut.mp4" ]; then
  echo "3/3 Checking for dead air..."
  "$PY" "$HERE/scripts/check_cut.py" "$OUT/roughcut" || echo "Some silence is still over 0.1s. See the lines above."
fi
echo
echo "Read $OUT/roughcut/report.md before you use the cut: it lists every line it removed."
