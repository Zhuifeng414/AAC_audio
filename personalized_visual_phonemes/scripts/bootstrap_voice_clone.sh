#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VOICE_DIR="$ROOT_DIR/third_party/Real-Time-Voice-Cloning"
RUNTIME_DIR="$ROOT_DIR/voice_clone_runtime"
WORKER="$ROOT_DIR/src/voice_clone_worker.py"

env -u LD_LIBRARY_PATH \
  -u CONDA_PREFIX \
  -u CONDA_DEFAULT_ENV \
  -u CONDA_SHLVL \
  -u CONDA_PROMPT_MODIFIER \
  -u CONDA_PYTHON_EXE \
  -u _CE_CONDA \
  -u _CE_M \
  -u VIRTUAL_ENV \
  -u PYTHONHOME \
  -u PYTHONPATH \
  uv run \
  --project "$RUNTIME_DIR" \
  --python 3.9 \
  --extra cpu \
  python "$WORKER" \
  --project-root "$ROOT_DIR" \
  --reference "$VOICE_DIR/samples/1320_00000.mp3" \
  --text "The system is ready." \
  --output "$ROOT_DIR/.voice_clone_smoke.wav"
