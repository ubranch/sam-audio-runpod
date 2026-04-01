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
    curl \
    libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

# python 3.11 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

WORKDIR /app

ENV HF_HOME=/runpod-volume/huggingface-cache \
    HF_HUB_CACHE=/runpod-volume/huggingface-cache/hub \
    TRANSFORMERS_CACHE=/runpod-volume/huggingface-cache/hub \
    TORCH_HOME=/runpod-volume/torch-cache

# pytorch (cuda 12.4 wheels matching the container runtime)
RUN pip install --no-cache-dir \
    torch==2.5.1 \
    torchaudio==2.5.1 \
    torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cu124

# install the upstream runtime dependency set explicitly instead of relying on
# sam-audio's package metadata to pull everything in transitively.
RUN pip install --no-cache-dir \
    transformers>=4.54.0 \
    scipy \
    soundfile \
    torchcodec \
    torchdiffeq \
    descript-audiotools \
    eva-decord \
    einops \
    timm \
    ftfy \
    xformers \
    iopath

RUN pip install --no-cache-dir --no-deps \
    git+https://github.com/facebookresearch/perception_models.git@unpin-deps \
    git+https://github.com/facebookresearch/ImageBind.git \
    git+https://github.com/facebookresearch/dacvae.git \
    git+https://github.com/facebookresearch/pytorchvideo.git@6cdc929315aab1b5674b6dcf73b16ec99147735f \
    git+https://github.com/facebookresearch/sam-audio.git@68b48d48fff1ad776d3afefbe634eb5f5d60ba7b

# the upstream package install can leave a non-importable wheel in this image,
# so keep the pinned source tree on PYTHONPATH as the runtime import source.
RUN git clone --filter=blob:none https://github.com/facebookresearch/sam-audio.git /opt/sam-audio && \
    cd /opt/sam-audio && \
    git checkout 68b48d48fff1ad776d3afefbe634eb5f5d60ba7b
ENV PYTHONPATH="/opt/sam-audio"

# fail the image build if sam_audio's real runtime imports are still broken.
RUN python -c "import importlib.util; assert importlib.util.find_spec('core.audio_visual_encoder'), 'core.audio_visual_encoder not found'; from sam_audio import SAMAudio, SAMAudioProcessor; print('sam_audio runtime imports ok')"

# runpod handler deps
RUN pip install --no-cache-dir runpod==1.8.2 requests==2.32.5

COPY handler.py ./

RUN useradd --create-home --shell /bin/bash appuser
RUN mkdir -p /runpod-volume/huggingface-cache/hub /runpod-volume/torch-cache && \
    chown -R appuser:appuser /runpod-volume
USER appuser

ENV PYTHONUNBUFFERED=1

CMD ["python", "handler.py"]
