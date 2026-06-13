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


def test_info_shape():
    with _client() as client:
        j = client.get("/info").json()
        assert set(j) >= {"version", "modelId", "device"}
        assert j["device"] == "cpu"  # EchoEngine


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
