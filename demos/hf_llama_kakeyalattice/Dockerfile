# syntax=docker/dockerfile:1.6
#
# HuggingFace Space: kakeyalattice-demo (Docker SDK, Gradio inside)
#
# Build notes:
#   - CPU-first so this runs on a free HF Space (no GPU required).
#   - Pulls kakeyalattice + transformers + gradio from PyPI at build
#     time, so the Space is self-contained and reproducible.
#   - App runs Gradio on port 7860 (HF Space default).

FROM python:3.11-slim

# System deps (minimal — Gradio + torch-cpu don't need much)
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        && rm -rf /var/lib/apt/lists/*

# Create an unprivileged user (HF Spaces expect UID 1000)
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/home/user/.cache/huggingface

WORKDIR $HOME/app

# Copy requirements first to maximise Docker layer cache
COPY --chown=user:user requirements.txt .

# Install CPU-only torch + gradio + our package from PyPI
# NOTE: --extra-index-url pulls CPU-only torch (smaller image, faster cold start on free tier).
RUN pip install --no-cache-dir --user --upgrade pip && \
    pip install --no-cache-dir --user \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        -r requirements.txt

# Copy the app
COPY --chown=user:user app.py README.md ./

# HF Space default port
EXPOSE 7860

CMD ["python", "app.py"]
