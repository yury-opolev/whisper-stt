import struct

import numpy as np

from audio import is_silent, pcm16_to_float32, rms


def test_pcm16_to_float32_scales_to_unit_range():
    # full-scale negative, zero, full-scale positive int16
    pcm = np.array([-32768, 0, 32767], dtype="<i2").tobytes()
    out = pcm16_to_float32(pcm)
    assert out.dtype == np.float32
    assert out[1] == 0.0
    assert -1.0 <= out[0] <= -0.999
    assert 0.999 <= out[2] <= 1.0


def test_pcm16_to_float32_empty_returns_empty():
    assert pcm16_to_float32(b"").shape == (0,)


def test_rms_empty_is_zero():
    assert rms(np.zeros(0, dtype=np.float32)) == 0.0


def test_rms_of_silence_is_zero():
    assert rms(np.zeros(1000, dtype=np.float32)) == 0.0


def test_rms_of_full_scale_is_one():
    assert abs(rms(np.ones(1000, dtype=np.float32)) - 1.0) < 1e-6


def test_is_silent_true_for_pure_silence():
    silence = b"\x00\x00" * 16000  # 1s of zeros
    assert is_silent(silence, 0.005) is True


def test_is_silent_true_for_tiny_noise():
    noise = b"".join(struct.pack("<h", (i % 7) - 3) for i in range(16000))  # |amp|<=3
    assert is_silent(noise, 0.005) is True


def test_is_silent_false_for_speech_level_signal():
    # A loud sine wave (~0.3 amplitude) is well above the silence gate.
    t = np.arange(16000)
    sine = (0.3 * np.sin(2 * np.pi * 220 * t / 16000) * 32767).astype("<i2").tobytes()
    assert is_silent(sine, 0.005) is False
