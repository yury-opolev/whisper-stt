"""Download Whisper weights at BUILD time so the runtime image needs no network.

Mirrors uni-voices' prefetch step: a failure here fails the docker build, which
guarantees the published image is fully offline-capable.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch  # noqa: E402
from transformers import pipeline  # noqa: E402

import config  # noqa: E402

if __name__ == "__main__":
    pipeline(
        "automatic-speech-recognition",
        model=config.MODEL_ID,
        torch_dtype=torch.float16,
    )
    print(f"prefetched {config.MODEL_ID}")
