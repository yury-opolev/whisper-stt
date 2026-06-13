from engine import EchoEngine


def test_echo_engine_transcribe_returns_marker():
    eng = EchoEngine()
    assert eng.transcribe(b"\x00\x00", "auto", None) == "[echo]"


def test_echo_engine_empty_audio_returns_none():
    eng = EchoEngine()
    assert eng.transcribe(b"", "auto", None) is None


def test_echo_engine_detailed_has_one_token():
    eng = EchoEngine()
    text, tokens = eng.transcribe_detailed(b"\x00\x00", "auto", None)
    assert text == "[echo]"
    assert tokens == [("[echo]", 0, 0)]


def test_echo_engine_detailed_empty_returns_none():
    eng = EchoEngine()
    text, tokens = eng.transcribe_detailed(b"", "auto", None)
    assert text is None
    assert tokens == []
