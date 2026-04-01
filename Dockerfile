# sam-audio runpod serverless
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3-pip \
    git \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# python 3.11 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

WORKDIR /app

# pytorch + xformers (cuda 12.4)
RUN pip install --no-cache-dir \
    torch==2.5.1 \
    torchaudio==2.5.1 \
    torchvision==0.20.1 \
    xformers \
    --index-url https://download.pytorch.org/whl/cu124

# sam-audio runtime deps (installed before facebook packages to satisfy transitive imports)
RUN pip install --no-cache-dir \
    transformers scipy soundfile torchcodec torchdiffeq descript-audiotools eva-decord \
    einops timm ftfy

# facebook research packages (--no-deps prevents pip from pulling conflicting versions)
RUN pip install --no-cache-dir --no-deps \
    git+https://github.com/facebookresearch/perception_models.git@unpin-deps
RUN pip install --no-cache-dir --no-deps \
    git+https://github.com/facebookresearch/ImageBind.git
RUN pip install --no-cache-dir --no-deps \
    git+https://github.com/facebookresearch/dacvae.git
RUN pip install --no-cache-dir --no-deps \
    git+https://github.com/facebookresearch/pytorchvideo.git@6cdc929315aab1b5674b6dcf73b16ec99147735f
RUN pip install --no-cache-dir iopath

# sam-audio: cloned from source because pip produces a broken empty wheel (UNKNOWN-0.0.0)
RUN git clone --filter=blob:none https://github.com/facebookresearch/sam-audio.git /opt/sam-audio && \
    cd /opt/sam-audio && \
    git checkout 68b48d48fff1ad776d3afefbe634eb5f5d60ba7b
ENV PYTHONPATH="/opt/sam-audio"

# verify sam_audio is discoverable (full import requires cuda libs only present at runtime)
RUN python -c "import importlib.util; spec = importlib.util.find_spec('sam_audio'); assert spec, 'sam_audio not found'; print('sam_audio OK:', spec.origin)"

# runpod handler deps
RUN pip install --no-cache-dir runpod==1.8.2 requests==2.32.5

COPY handler.py ./

RUN useradd --create-home --shell /bin/bash appuser
USER appuser

ENV PYTHONUNBUFFERED=1

CMD ["python", "handler.py"]
