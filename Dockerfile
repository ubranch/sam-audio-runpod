# SAM-Audio RunPod Serverless Dockerfile
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

# Prevent interactive prompts during apt install
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3-pip \
    git \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Make python3.11 the default python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

WORKDIR /app

# Install PyTorch with CUDA 12.4
RUN pip install --no-cache-dir \
    torch==2.5.1 \
    torchaudio==2.5.1 \
    torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cu124

# Install sam-audio pinned to a specific commit
RUN pip install --no-cache-dir \
    git+https://github.com/facebookresearch/sam-audio.git@68b48d48fff1ad776d3afefbe634eb5f5d60ba7b

# Install RunPod handler dependencies (pinned)
RUN pip install --no-cache-dir runpod==1.8.2 requests==2.32.5

# Copy handler code
COPY handler.py ./

# Create non-root user for runtime
RUN useradd --create-home --shell /bin/bash appuser
USER appuser

# Environment
ENV PYTHONUNBUFFERED=1

# Health check for local Docker testing (RunPod manages health externally)
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import handler" || exit 1

# RunPod handler entrypoint
CMD ["python", "handler.py"]
