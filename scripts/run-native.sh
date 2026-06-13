#!/usr/bin/env bash
# Native launcher for Apple Silicon (Metal/MPS) — NOT Docker. Docker Desktop on
# macOS can't reach the GPU, so the Mac path runs the same app.py in a venv.
# Binds to 0.0.0.0 so Cortex on another host (the Windows laptop) can reach it;
# point RemoteSpeechToText's BaseAddress at http://<mac-lan-ip>:5300.
set -euo pipefail
cd "$(dirname "$0")/.."

python3 -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements-mac.txt

# First run downloads weights to the HF cache; subsequent runs are offline.
python scripts/prefetch_models.py

exec uvicorn app:app --host 0.0.0.0 --port "${PORT:-5300}"
