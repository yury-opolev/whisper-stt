"""Runtime configuration, all overridable via environment variables.

The sidecar holds NO secrets; only model selection and decoding defaults.
"""
import os

MODEL_ID = os.environ.get("WHISPER_MODEL_ID", "openai/whisper-large-v3-turbo")
SAMPLE_RATE = 16000
DEFAULT_LANGUAGE = os.environ.get("WHISPER_LANGUAGE", "auto")
DEFAULT_PROMPT = os.environ.get("WHISPER_PROMPT", "") or None

# RMS amplitude below which audio is treated as silence/non-speech and NOT
# transcribed — prevents Whisper's stock-phrase hallucination ("Thank you.")
# on empty/silent clips. Real speech is well above this; tune via env if needed.
SILENCE_RMS_THRESHOLD = float(os.environ.get("WHISPER_SILENCE_RMS", "0.005"))

# Seconds of inactivity after which the model is released from VRAM (Ollama-style).
# The next real request lazy-reloads it (~a few seconds). 0 disables. Default 8h.
IDLE_UNLOAD_SECONDS = int(os.environ.get("WHISPER_IDLE_UNLOAD_SECONDS", str(8 * 3600)))

# Exit the process when a lazy reload fails after the model had previously loaded.
# A CUDA context invalidated underneath a long-running container (typically a host
# driver update) is unrecoverable in-process: every subsequent load raises
# "CUDA error: unknown error" forever. Exiting lets the container restart policy
# rebuild the context, turning a silent multi-day outage into a few seconds of
# downtime. Only ever triggers after a successful load, so a genuinely broken boot
# cannot crash-loop on this path. Set to 0 to disable.
EXIT_ON_LOAD_FAILURE = os.environ.get("WHISPER_EXIT_ON_LOAD_FAILURE", "1") not in ("0", "false", "False")
