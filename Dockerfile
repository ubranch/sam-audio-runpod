# RunPod Serverless Dockerfile for SAM Audio
# Based on PyTorch with CUDA support

FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip

# Copy pyproject.toml and install dependencies
COPY pyproject.toml /app/
RUN pip install --no-cache-dir .

# Copy handler code
COPY handler.py /app/handler.py

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Model loading options (in order of priority):
# 1. RunPod Cached Models: Set endpoint "Cached Models" to the HF model URL
#    (e.g., https://huggingface.co/facebook/sam-audio-large)
#    The handler auto-detects and uses /runpod-volume/huggingface-cache/hub
# 2. HF_TOKEN: Set at runtime for downloading gated models on-demand

# Start the handler
CMD ["python", "-u", "handler.py"]

