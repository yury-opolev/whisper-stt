"""Real Whisper engine backed by HF Transformers (`whisper-large-v3-turbo`).

Device is selected at construction: CUDA (Docker on Linux/Windows-WSL) -> MPS
(native macOS / Apple Silicon) -> CPU. A CPU fallback is logged LOUDLY because a
silently-CPU Whisper means multi-second transcripts.

Idle VRAM unload: when `idle_unload_seconds` > 0, a background reaper releases the
model from the GPU after that many seconds with no transcription, freeing VRAM.
The next real request lazy-reloads it (a few seconds, like Ollama's keep_alive
cold start). All load/unload/inference is serialized by a lock.

torch / transformers are imported here, never in engine.py, so the GPU-free test
path (EchoEngine) never loads them.
"""
import gc
import logging
import threading
import time

import torch
from transformers import pipeline

import config
from audio import is_silent, pcm16_to_float32

log = logging.getLogger("whisper_engine")


def _pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


class WhisperEngine:
    def __init__(self, model_id: str, idle_unload_seconds: int = 0):
        self.device = _pick_device()
        self._model_id = model_id
        self._idle_unload_seconds = idle_unload_seconds
        self._dtype = torch.float16 if self.device in ("cuda", "mps") else torch.float32
        self._pipe = None
        self._lock = threading.Lock()
        self._last_used = time.monotonic()

        if self.device == "cpu":
            log.warning(
                "whisper-stt is running on CPU — transcription will be SLOW. "
                "Check the CUDA (Docker) or MPS (native Mac) runtime."
            )

        # Eager initial load so the first request after a fresh start isn't slow
        # and a broken model surfaces at boot, not on the first utterance.
        with self._lock:
            self._load_locked()

        # Idle reaper: only meaningful on a GPU (nothing to reclaim on CPU).
        if self._idle_unload_seconds > 0 and self.device != "cpu":
            threading.Thread(target=self._reaper, name="idle-unload", daemon=True).start()
            log.info("idle unload enabled: %ds", self._idle_unload_seconds)

    @property
    def model_loaded(self) -> bool:
        return self._pipe is not None

    def _load_locked(self):
        if self._pipe is None:
            self._pipe = pipeline(
                "automatic-speech-recognition",
                model=self._model_id,
                torch_dtype=self._dtype,
                device=self.device,
            )
            log.info("whisper-stt model loaded to %s (dtype=%s)", self.device, self._dtype)

    def _unload_locked(self):
        if self._pipe is not None:
            self._pipe = None
            gc.collect()
            if self.device == "cuda":
                torch.cuda.empty_cache()
            log.info("whisper-stt model unloaded from %s (idle %ds)", self.device, self._idle_unload_seconds)

    def _reaper(self):
        # Check once a minute; unload when idle past the threshold.
        while True:
            time.sleep(60)
            with self._lock:
                idle = time.monotonic() - self._last_used
                if self._pipe is not None and idle >= self._idle_unload_seconds:
                    self._unload_locked()

    def _run(self, pcm: bytes, language: str, prompt: str | None, word_ts: bool):
        # Gate silence/non-speech BEFORE touching the model: the model hallucinates
        # stock phrases ("Thank you.") on empty audio, and a silent request must NOT
        # wake an idle-unloaded model.
        if len(pcm) < 2 or is_silent(pcm, config.SILENCE_RMS_THRESHOLD):
            return None

        audio = pcm16_to_float32(pcm)
        generate_kwargs = {"task": "transcribe"}
        if language and language != "auto":
            generate_kwargs["language"] = language

        with self._lock:
            self._load_locked()  # lazy reload if the reaper unloaded us
            self._last_used = time.monotonic()
            return self._pipe(
                {"array": audio, "sampling_rate": config.SAMPLE_RATE},
                return_timestamps="word" if word_ts else False,
                generate_kwargs=generate_kwargs,
            )

    def transcribe(self, pcm: bytes, language: str, prompt: str | None) -> str | None:
        out = self._run(pcm, language, prompt, word_ts=False)
        if out is None:
            return None
        text = (out.get("text") or "").strip()
        return text or None

    def transcribe_detailed(
        self, pcm: bytes, language: str, prompt: str | None
    ) -> tuple[str | None, list[tuple[str, int, int]]]:
        out = self._run(pcm, language, prompt, word_ts=True)
        if out is None:
            return None, []
        text = (out.get("text") or "").strip()
        if not text:
            return None, []
        tokens: list[tuple[str, int, int]] = []
        for chunk in out.get("chunks", []):
            ts = chunk.get("timestamp") or (None, None)
            start_ms = int((ts[0] or 0.0) * 1000)
            end_ms = int((ts[1] if ts[1] is not None else ts[0] or 0.0) * 1000)
            tokens.append((chunk.get("text", ""), start_ms, end_ms))
        return text, tokens
