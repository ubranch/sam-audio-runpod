# SAM-Audio RunPod Serverless Dockerfile
FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04

# Prevent interactive prompts during apt install
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3-pip \
    git \
    curl \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# Make python3.11 the default python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

WORKDIR /app

# Install PyTorch with CUDA 12.8 first
RUN pip install --no-cache-dir \
    torch==2.5.1 \
    torchaudio==2.5.1 \
    torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cu124

# Install sam-audio with all its dependencies
RUN pip install --no-cache-dir \
    git+https://github.com/facebookresearch/sam-audio.git

# Install RunPod handler dependencies
RUN pip install --no-cache-dir runpod requests

# Copy handler code
COPY handler.py ./

# Environment
ENV PYTHONUNBUFFERED=1

# RunPod handler entrypoint
CMD ["python", "handler.py"]
