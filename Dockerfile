# sam-audio runpod serverless
FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3-pip \
    git \
    curl \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# python 3.11 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

WORKDIR /app

# pytorch (cuda 12.4 wheels, forward-compatible with 12.8 runtime)
RUN pip install --no-cache-dir \
    torch==2.5.1 \
    torchaudio==2.5.1 \
    torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cu124

# all transitive deps for sam-audio + facebook research packages
RUN pip install --no-cache-dir \
    "transformers>=4.54.0" scipy soundfile torchcodec torchdiffeq \
    einops timm ftfy xformers pydub numpy audiobox_aesthetics \
    descript-audiotools eva-decord iopath

# facebook research packages (--no-deps to prevent torch/torchcodec version conflicts)
RUN pip install --no-cache-dir --no-deps \
    git+https://github.com/facebookresearch/perception_models.git@unpin-deps
RUN pip install --no-cache-dir --no-deps \
    git+https://github.com/facebookresearch/ImageBind.git
RUN pip install --no-cache-dir --no-deps \
    git+https://github.com/facebookresearch/dacvae.git
RUN pip install --no-cache-dir --no-deps \
    git+https://github.com/facebookresearch/pytorchvideo.git@6cdc929315aab1b5674b6dcf73b16ec99147735f
RUN pip install --no-cache-dir --no-deps \
    git+https://github.com/lematt1991/CLAP.git

# sam-audio source (pip produces broken UNKNOWN-0.0.0 wheel)
RUN git clone --filter=blob:none https://github.com/facebookresearch/sam-audio.git /opt/sam-audio
ENV PYTHONPATH="/opt/sam-audio"

# runpod handler deps
RUN pip install --no-cache-dir runpod==1.8.2 requests==2.32.5

COPY handler.py ./

RUN useradd --create-home --shell /bin/bash appuser
USER appuser

ENV PYTHONUNBUFFERED=1

CMD ["python", "handler.py"]
