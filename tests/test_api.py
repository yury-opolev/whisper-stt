import os

os.environ["WHISPER_ENGINE"] = "echo"  # GPU-free engine; must be set before importing app

from fastapi.testclient import TestClient  # noqa: E402

from app import app  # noqa: E402

OCTET = {"content-type": "application/octet-stream"}


def _client() -> TestClient:
    # TestClient as a context manager runs the lifespan startup (builds the engine).
    return TestClient(app)


def test_health_ok():
    with _client() as client:
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["loaded"] is True
        assert r.json()["healthy"] is True


def test_health_reports_503_when_the_last_load_failed():
    """THE 2026-08-17→19 REGRESSION.

    A host GPU driver update invalidated the CUDA context under the running
    container. Every transcription failed with "CUDA error: unknown error", but
    /health returned {"loaded": true} because it only checked that the engine
    *object* existed — so Docker reported the container healthy for two days.
    """
    import app as app_module

    with _client() as client:
        broken = _BrokenEngine()
        original, app_module.engine = app_module.engine, broken
        try:
            r = client.get("/health")
            assert r.status_code == 503
            assert r.json()["healthy"] is False
            assert r.json()["loaded"] is False
            assert "CUDA error" in r.json()["error"]
        finally:
            app_module.engine = original


def test_health_stays_ok_while_idle_unloaded():
    """An idle-unloaded model is healthy — it reloads on the next request.

    Guards against the naive fix of wiring /health straight to `model_loaded`,
    which would flap the container health every idle period (default 8h).
    """
    import app as app_module

    with _client() as client:
        idle = _IdleUnloadedEngine()
        original, app_module.engine = app_module.engine, idle
        try:
            r = client.get("/health")
            assert r.status_code == 200
            assert r.json()["healthy"] is True
            assert r.json()["loaded"] is False
            assert "error" not in r.json()
        finally:
            app_module.engine = original


class _BrokenEngine:
    device = "cuda"
    model_loaded = False
    load_error = "RuntimeError: CUDA error: unknown error"
    healthy = False


class _IdleUnloadedEngine:
    device = "cuda"
    model_loaded = False
    load_error = None
    healthy = True


def test_info_shape():
    with _client() as client:
        j = client.get("/info").json()
        assert set(j) >= {"version", "modelId", "device", "modelLoaded", "loadError"}
        assert j["device"] == "cpu"  # EchoEngine
        assert j["modelLoaded"] is True
        assert j["loadError"] is None


def test_transcribe_returns_text():
    with _client() as client:
        r = client.post("/v1/transcribe", content=b"\x01\x00", headers=OCTET)
        assert r.status_code == 200
        assert r.json()["text"] == "[echo]"


def test_transcribe_empty_is_204():
    with _client() as client:
        r = client.post("/v1/transcribe", content=b"", headers=OCTET)
        assert r.status_code == 204


def test_detailed_returns_tokens():
    with _client() as client:
        r = client.post("/v1/transcribe/detailed", content=b"\x01\x00", headers=OCTET)
        assert r.status_code == 200
        body = r.json()
        assert body["text"] == "[echo]"
        assert body["tokens"][0] == {"text": "[echo]", "startMs": 0, "endMs": 0}
