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

# Copy pyproject.toml and install base dependencies
COPY pyproject.toml /app/
RUN pip install --no-cache-dir .

# Install Facebook Research packages
# ImageBind needs its deps for import to work, others use --no-deps to avoid conflicts
RUN pip install --no-cache-dir --no-deps \
    git+https://github.com/facebookresearch/pytorchvideo.git@6cdc929315aab1b5674b6dcf73b16ec99147735f \
    git+https://github.com/facebookresearch/dacvae.git

# Install ImageBind WITH dependencies (required for import to succeed)
RUN pip install --no-cache-dir \
    git+https://github.com/facebookresearch/ImageBind.git

# Install remaining FB packages with --no-deps
RUN pip install --no-cache-dir --no-deps \
    git+https://github.com/facebookresearch/perception_models.git@unpin-deps \
    git+https://github.com/facebookresearch/sam-audio.git

# Verify ImageBind can be imported
RUN python -c "from imagebind import data; print('ImageBind import OK')"

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

