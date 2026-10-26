#!/usr/bin/env bash
# One-time setup for reel-cut: a Python 3.12 environment with WhisperX, plus a check for ffmpeg.
# Safe to re-run. Override the location with REEL_CUT_VENV=/some/path.
set -euo pipefail

VENV="${REEL_CUT_VENV:-$HOME/.local/share/reel-cut-venv}"

if ! command -v ffmpeg >/dev/null 2>&1; then
  echo "ffmpeg is missing. Install it first:"
  echo "  Mac:     brew install ffmpeg"
  echo "  Windows: winget install ffmpeg"
  echo "  Linux:   sudo apt install ffmpeg"
  exit 1
fi

if command -v uv >/dev/null 2>&1; then
  uv venv --python 3.12 "$VENV"
  VIRTUAL_ENV="$VENV" uv pip install whisperx
elif command -v python3.12 >/dev/null 2>&1; then
  python3.12 -m venv "$VENV"
  "$VENV/bin/pip" install --upgrade pip
  "$VENV/bin/pip" install whisperx
else
  echo "WhisperX needs Python 3.12. Easiest fix: install uv (https://docs.astral.sh/uv/), then re-run this."
  exit 1
fi

"$VENV/bin/python" -c "import whisperx, numpy" && echo "reel-cut is ready: $VENV"
echo "The first transcription downloads the speech models (about 1.5 GB). After that it runs offline."
