"""Real Whisper engine backed by HF Transformers (`whisper-large-v3-turbo`).

Device is selected at construction: CUDA (Docker on Linux/Windows-WSL) -> MPS
(native macOS / Apple Silicon) -> CPU. A CPU fallback is logged LOUDLY because a
silently-CPU Whisper means multi-second transcripts (the failure mode this whole
sidecar is meant to make impossible to hide).

torch / transformers are imported here, never in engine.py, so the GPU-free test
path (EchoEngine) never loads them.
"""
import logging

import torch
from transformers import pipeline

import config
from audio import pcm16_to_float32

log = logging.getLogger("whisper_engine")


def _pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


class WhisperEngine:
    def __init__(self, model_id: str):
        self.device = _pick_device()
        if self.device == "cpu":
            log.warning(
                "whisper-stt is running on CPU — transcription will be SLOW. "
                "Check the CUDA (Docker) or MPS (native Mac) runtime."
            )
        dtype = torch.float16 if self.device in ("cuda", "mps") else torch.float32
        self._pipe = pipeline(
            "automatic-speech-recognition",
            model=model_id,
            torch_dtype=dtype,
            device=self.device,
        )
        log.info("whisper-stt loaded model=%s device=%s dtype=%s", model_id, self.device, dtype)

    def _run(self, pcm: bytes, language: str, prompt: str | None, word_ts: bool):
        if len(pcm) < 2:
            return None
        audio = pcm16_to_float32(pcm)
        generate_kwargs = {"task": "transcribe"}
        if language and language != "auto":
            generate_kwargs["language"] = language
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
