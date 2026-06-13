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
