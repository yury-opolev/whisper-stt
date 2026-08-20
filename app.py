"""FastAPI surface for the Whisper STT sidecar.

Routes (the contract Cortex's RemoteSpeechToText calls):
  GET  /health                     -> { loaded: bool, healthy: bool, error?: str } | 503
  GET  /info                       -> { version, modelId, device, modelLoaded, loadError }
  POST /v1/transcribe              -> { text } | 204
  POST /v1/transcribe/detailed     -> { text, tokens[] } | 204

Audio is the raw request body: 16 kHz mono signed-16-bit little-endian PCM.
`language` and `prompt` are optional query params.
"""
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request, Response

import config
from models import DetailedResponse, InfoResponse, Token, TranscribeResponse

_VERSION = json.loads((Path(__file__).parent / "version.json").read_text()).get("version", "0.0.0")

# Module-level engine handle, populated on startup.
engine = None


def _build_engine():
    """Pick the engine. WHISPER_ENGINE=echo selects the GPU-free stand-in (tests/CI)."""
    if os.environ.get("WHISPER_ENGINE") == "echo":
        from engine import EchoEngine

        return EchoEngine()
    from whisper_engine import WhisperEngine

    return WhisperEngine(config.MODEL_ID, idle_unload_seconds=config.IDLE_UNLOAD_SECONDS)


@asynccontextmanager
async def lifespan(_: FastAPI):
    global engine
    engine = _build_engine()
    yield


app = FastAPI(title="whisper-stt", lifespan=lifespan)


@app.get("/health")
def health(response: Response):
    """Liveness AND model-readiness for the container healthcheck.

    Previously this returned `{"loaded": engine is not None}` — the engine *object*,
    which exists from startup no matter what state the model is in. So it answered
    `{"loaded": true}` for two days (2026-08-17→19) while every transcription failed
    with a dead CUDA context, and Docker went on reporting the container healthy.

    `loaded` now means the model is actually resident. `healthy` is the signal the
    healthcheck keys on, and is deliberately NOT the same thing: an idle-unloaded
    model is healthy and reloads on demand, so only a *failed load attempt* is
    unhealthy. Unhealthy answers 503 so the Dockerfile HEALTHCHECK (which asserts
    status == 200) actually fails.
    """
    loaded = bool(getattr(engine, "model_loaded", engine is not None))
    error = getattr(engine, "load_error", None)
    healthy = engine is not None and getattr(engine, "healthy", True)

    if not healthy:
        response.status_code = 503

    body = {"loaded": loaded, "healthy": healthy}
    if error:
        body["error"] = error
    return body


@app.get("/info", response_model=InfoResponse)
def info():
    return InfoResponse(
        version=_VERSION,
        modelId=config.MODEL_ID,
        device=getattr(engine, "device", "unknown"),
        modelLoaded=getattr(engine, "model_loaded", True),
        loadError=getattr(engine, "load_error", None),
    )


@app.post("/v1/transcribe")
async def transcribe(
    request: Request,
    language: str = config.DEFAULT_LANGUAGE,
    prompt: str | None = None,
):
    pcm = await request.body()
    text = engine.transcribe(pcm, language, prompt or config.DEFAULT_PROMPT)
    if text is None:
        return Response(status_code=204)
    return TranscribeResponse(text=text)


@app.post("/v1/transcribe/detailed", response_model=DetailedResponse)
async def transcribe_detailed(
    request: Request,
    language: str = config.DEFAULT_LANGUAGE,
    prompt: str | None = None,
):
    pcm = await request.body()
    text, tokens = engine.transcribe_detailed(pcm, language, prompt or config.DEFAULT_PROMPT)
    if text is None:
        return Response(status_code=204)
    return DetailedResponse(
        text=text,
        tokens=[Token(text=t, startMs=s, endMs=e) for (t, s, e) in tokens],
    )
