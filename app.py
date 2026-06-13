"""FastAPI surface for the Whisper STT sidecar.

Routes (the contract Cortex's RemoteSpeechToText calls):
  GET  /health                     -> { loaded: bool }
  GET  /info                       -> { version, modelId, device }
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

    return WhisperEngine(config.MODEL_ID)


@asynccontextmanager
async def lifespan(_: FastAPI):
    global engine
    engine = _build_engine()
    yield


app = FastAPI(title="whisper-stt", lifespan=lifespan)


@app.get("/health")
def health():
    return {"loaded": engine is not None}


@app.get("/info", response_model=InfoResponse)
def info():
    return InfoResponse(
        version=_VERSION,
        modelId=config.MODEL_ID,
        device=getattr(engine, "device", "unknown"),
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
