"""Runtime configuration, all overridable via environment variables.

The sidecar holds NO secrets; only model selection and decoding defaults.
"""
import os

MODEL_ID = os.environ.get("WHISPER_MODEL_ID", "openai/whisper-large-v3-turbo")
SAMPLE_RATE = 16000
DEFAULT_LANGUAGE = os.environ.get("WHISPER_LANGUAGE", "auto")
DEFAULT_PROMPT = os.environ.get("WHISPER_PROMPT", "") or None
