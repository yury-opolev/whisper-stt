# whisper-stt

Whisper speech-to-text sidecar for [Cortex](https://github.com/yury-opolev/cortex).
A small FastAPI service wrapping HF Transformers `whisper-large-v3-turbo`.

- **Docker (CUDA)** on the Linux/Windows-WSL host (today).
- **Native venv (MPS)** on Apple Silicon (later) — same code, `scripts/run-native.sh`.

Cortex talks to it purely over HTTP, so where it runs is just one config value
(`RemoteSpeechToText` `BaseAddress`).

## HTTP API

Audio is the raw request **body**: 16 kHz mono signed-16-bit little-endian PCM.
`language` (ISO-639-1 or `auto`) and `prompt` are optional query params.

| Method | Path | Body | Response |
|---|---|---|---|
| GET | `/health` | — | `{ "loaded": true }` |
| GET | `/info` | — | `{ "version", "modelId", "device" }` where `device ∈ {cuda, mps, cpu}` |
| POST | `/v1/transcribe` | PCM | `{ "text": "..." }` or `204` (no speech) |
| POST | `/v1/transcribe/detailed` | PCM | `{ "text", "tokens": [{text,startMs,endMs}] }` or `204` |

`device == "cpu"` is logged loudly and surfaced in `/info` — a CPU fallback means
slow transcripts and should never pass silently.

## Run

### Docker (CUDA)
```bash
docker build -t whisper-stt:latest .
docker run --rm --gpus all -p 127.0.0.1:5300:8000 whisper-stt:latest
```

### Native (Apple Silicon / MPS)
```bash
PORT=5300 ./scripts/run-native.sh
```

## Develop / test (GPU-free)

Tests run against an `EchoEngine` stand-in, so no GPU or weights are needed:
```bash
pip install -r requirements-dev.txt
python -m pytest -q
```

## Layout

| File | Responsibility |
|---|---|
| `app.py` | FastAPI routes (HTTP surface) |
| `engine.py` | `TranscriptionEngine` protocol + GPU-free `EchoEngine` |
| `whisper_engine.py` | Real `WhisperEngine` (HF Transformers; cuda→mps→cpu) |
| `audio.py` | PCM s16le → float32 @16 kHz |
| `models.py` | pydantic request/response DTOs (the wire contract) |
| `config.py` | model id / sample rate / decoding defaults (env-overridable) |
| `scripts/prefetch_models.py` | bake weights at Docker build |
| `scripts/run-native.sh` | native-Mac launcher |

Published to GHCR as `ghcr.io/yury-opolev/whisper-stt:0.1` by `.github/workflows/release.yml`.
