"""Audio decoding helpers for the STT sidecar."""
import numpy as np


def pcm16_to_float32(pcm: bytes) -> np.ndarray:
    """Decode little-endian int16 mono PCM bytes to float32 samples in [-1, 1].

    Matches the canonical input Cortex sends: 16 kHz mono signed 16-bit PCM.
    Returns an empty array for empty input.
    """
    if not pcm:
        return np.zeros(0, dtype=np.float32)
    samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32)
    return samples / 32768.0


def rms(samples: np.ndarray) -> float:
    """Root-mean-square amplitude of float32 samples (0.0 for empty)."""
    if samples.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))


def is_silent(pcm: bytes, threshold: float) -> bool:
    """True if the PCM is below the speech-energy threshold.

    Whisper hallucinates stock phrases ("Thank you." etc.) on silence/non-speech,
    so callers gate on this and skip transcription when it returns True. Real
    speech RMS is typically >0.02; digital silence/low noise is well under 0.005.
    """
    return rms(pcm16_to_float32(pcm)) < threshold
