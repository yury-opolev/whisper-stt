import numpy as np

from audio import pcm16_to_float32


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
