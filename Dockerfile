# Whisper STT sidecar. One CUDA runtime serving HF Transformers
# whisper-large-v3-turbo. Requires NVIDIA Container Toolkit on the host (Docker
# Desktop on Windows handles this when WSL2 GPU passthrough is enabled).
FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/root/.cache/huggingface \
    PATH=/opt/venv/bin:$PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-venv python3-pip \
        ffmpeg libsndfile1 ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv && pip install --upgrade pip

WORKDIR /app

COPY requirements.txt .
# torch/torchaudio from the cu124 wheel index; everything else from PyPI in one call.
RUN pip install --extra-index-url https://download.pytorch.org/whl/cu124 -r requirements.txt

# Prefetch the weights BEFORE copying app code: it depends only on requirements +
# the prefetch script (which reads the model id from env, NOT config.py), so
# editing config/app code no longer re-triggers the multi-GB model download.
COPY scripts/ ./scripts/
RUN python scripts/prefetch_models.py

COPY *.py ./
COPY version.json ./

EXPOSE 8000

# Long start-period: first boot loads the model before /health reports loaded.
HEALTHCHECK --interval=30s --timeout=10s --start-period=900s --retries=3 \
    CMD python -c "import urllib.request,sys; \
        r=urllib.request.urlopen('http://localhost:8000/health'); \
        sys.exit(0 if r.status==200 else 1)" || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
