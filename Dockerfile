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

# Install base dependencies for sam-audio
RUN pip install --no-cache-dir \
    transformers scipy soundfile torchcodec torchdiffeq descript-audiotools eva-decord

# Install Facebook Research packages (--no-deps to avoid dependency conflicts)
RUN pip install --no-cache-dir --no-deps git+https://github.com/facebookresearch/sam-audio.git && \
    pip install --no-cache-dir --no-deps git+https://github.com/facebookresearch/perception_models.git@unpin-deps && \
    pip install --no-cache-dir --no-deps git+https://github.com/facebookresearch/ImageBind.git && \
    pip install --no-cache-dir --no-deps git+https://github.com/facebookresearch/dacvae.git && \
    pip install --no-cache-dir --no-deps git+https://github.com/facebookresearch/pytorchvideo.git@6cdc929315aab1b5674b6dcf73b16ec99147735f

# Install iopath (required by pytorchvideo)
RUN pip install --no-cache-dir iopath

# Install RunPod handler dependencies
RUN pip install --no-cache-dir runpod requests

# Copy handler code
COPY handler.py ./

# Environment
ENV PYTHONUNBUFFERED=1

# RunPod handler entrypoint
CMD ["python", "handler.py"]
