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

    @property
    def model_loaded(self) -> bool:
        """Whether the model is currently resident and ready to transcribe."""
        ...

    @property
    def load_error(self) -> str | None:
        """Message from the last failed load, or None. Distinguishes a deliberate
        idle unload (healthy) from a device that refused to load us (not healthy)."""
        ...

    @property
    def healthy(self) -> bool:
        """False only when the last load attempt failed."""
        ...

    def transcribe(self, pcm: bytes, language: str, prompt: str | None) -> str | None:
        ...

    def transcribe_detailed(
        self, pcm: bytes, language: str, prompt: str | None
    ) -> tuple[str | None, list[tuple[str, int, int]]]:
        ...


class EchoEngine:
    """GPU-free stand-in for tests/CI. Returns a fixed marker for any non-empty input."""

    device = "cpu"

    # No weights to load, so it is always resident and never fails.
    model_loaded = True
    load_error = None
    healthy = True

    def transcribe(self, pcm: bytes, language: str, prompt: str | None) -> str | None:
        return "[echo]" if pcm else None
    def transcribe_detailed(
        self, pcm: bytes, language: str, prompt: str | None
    ) -> tuple[str | None, list[tuple[str, int, int]]]:
        if not pcm:
            return None, []
        return "[echo]", [("[echo]", 0, 0)]


def should_exit_on_load_failure(ever_loaded: bool, exit_enabled: bool) -> bool:
    """Whether a failed model load should take the process down.

    Kept here — deliberately free of torch — so CI covers it without a GPU or weights.

    True only when the model had already loaded successfully at least once. That is
    the signature of a device context that died underneath a healthy process (a host
    GPU driver update invalidating CUDA under a long-running container), which no
    amount of in-process retrying fixes: every later load raises
    "CUDA error: unknown error" forever. Exiting hands recovery to the container
    restart policy.

    A cold-boot failure returns False. That case means broken weights or a broken
    image, which the eager load in WhisperEngine.__init__ already surfaces by failing
    startup — exiting again there would only add a crash loop.
    """
    return ever_loaded and exit_enabled
