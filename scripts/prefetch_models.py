"""Download Whisper weights at BUILD time so the runtime image needs no network.

Mirrors uni-voices' prefetch step: a failure here fails the docker build, which
guarantees the published image is fully offline-capable. Intentionally does NOT
import config.py — it reads the model id straight from the env (same default) so
that editing config/app code does not invalidate this layer's cache (no
re-download on every code change).
"""
import os

import torch
from transformers import pipeline

MODEL_ID = os.environ.get("WHISPER_MODEL_ID", "openai/whisper-large-v3-turbo")

if __name__ == "__main__":
    pipeline("automatic-speech-recognition", model=MODEL_ID, torch_dtype=torch.float16)
    print(f"prefetched {MODEL_ID}")
