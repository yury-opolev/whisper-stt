from engine import EchoEngine, should_exit_on_load_failure


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


def test_echo_engine_reports_healthy():
    eng = EchoEngine()
    assert eng.model_loaded is True
    assert eng.healthy is True
    assert eng.load_error is None


# ── unrecoverable-load-failure exit guard ────────────────────────────────────
#
# A CUDA context invalidated by a host driver update never recovers in-process, so
# the engine takes the process down and lets the container restart policy rebuild
# it. This guard is what keeps that from becoming a crash loop. It lives in the
# torch-free module precisely so CI covers it without a GPU.


def test_reload_failure_after_successful_load_exits():
    # The real 2026-08-17→19 shape: the model had been serving fine, the idle reaper
    # unloaded it, and the reload then hit a dead device context.
    assert should_exit_on_load_failure(ever_loaded=True, exit_enabled=True) is True


def test_cold_boot_failure_does_not_exit():
    # Never loaded => broken weights/image, not a dead context. Exiting here would
    # pile a crash loop on top of an already-failing startup.
    assert should_exit_on_load_failure(ever_loaded=False, exit_enabled=True) is False


def test_exit_can_be_disabled():
    assert should_exit_on_load_failure(ever_loaded=True, exit_enabled=False) is False
    assert should_exit_on_load_failure(ever_loaded=False, exit_enabled=False) is False
