"""Transcription engine abstraction.

`TranscriptionEngine` is the contract the FastAPI surface depends on. `EchoEngine`
is a GPU-free stand-in so the API can be unit-tested and CI can run without a GPU
or model weights (mirrors uni-voices' EchoEngine). The real `WhisperEngine` lives
in whisper_engine.py and is imported lazily so importing this module never pulls
in torch.
"""
from typing import Protocol


class TranscriptionEngine(Protocol):
    device: str

    def transcribe(self, pcm: bytes, language: str, prompt: str | None) -> str | None:
        ...

    def transcribe_detailed(
        self, pcm: bytes, language: str, prompt: str | None
    ) -> tuple[str | None, list[tuple[str, int, int]]]:
        ...


class EchoEngine:
    """GPU-free stand-in for tests/CI. Returns a fixed marker for any non-empty input."""

    device = "cpu"

    def transcribe(self, pcm: bytes, language: str, prompt: str | None) -> str | None:
        return "[echo]" if pcm else None

    def transcribe_detailed(
        self, pcm: bytes, language: str, prompt: str | None
    ) -> tuple[str | None, list[tuple[str, int, int]]]:
        if not pcm:
            return None, []
        return "[echo]", [("[echo]", 0, 0)]
