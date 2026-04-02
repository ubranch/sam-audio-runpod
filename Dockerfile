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

ENV HF_HOME=/runpod-volume/huggingface-cache \
    HF_HUB_CACHE=/runpod-volume/huggingface-cache/hub \
    TRANSFORMERS_CACHE=/runpod-volume/huggingface-cache/hub \
    TORCH_HOME=/runpod-volume/torch-cache

ARG SAM_AUDIO_REF=68b48d48fff1ad776d3afefbe634eb5f5d60ba7b

# pytorch + xformers from the cu124 index so all cuda binaries match.
# xformers is required by perception_models (core/transformer.py).
RUN python -m pip install --no-cache-dir \
    torch==2.5.1 \
    torchaudio==2.5.1 \
    torchvision==0.20.1 \
    xformers \
    --index-url https://download.pytorch.org/whl/cu124

# explicit runtime deps — sam-audio's pyproject.toml has broken metadata
# (installs as UNKNOWN-0.0.0 on stock pip), so we enumerate everything here.
RUN python -m pip install --no-cache-dir \
    huggingface_hub \
    "transformers>=4.54.0" \
    scipy \
    soundfile \
    torchdiffeq \
    descript-audiotools \
    eva-decord \
    einops \
    timm \
    ftfy \
    iopath \
    regex \
    fvcore \
    av \
    networkx \
    parameterized

# facebook packages (--no-deps to avoid pulling conflicting torch versions)
RUN python -m pip install --no-cache-dir --no-deps \
    git+https://github.com/facebookresearch/ImageBind.git \
    git+https://github.com/facebookresearch/dacvae.git \
    git+https://github.com/facebookresearch/pytorchvideo.git@6cdc929315aab1b5674b6dcf73b16ec99147735f

COPY patches/ /tmp/patches/

# perception_models: clone + checkout unpin-deps + apply audio-only patch
RUN git clone --filter=blob:none https://github.com/facebookresearch/perception_models.git /opt/perception-models && \
    cd /opt/perception-models && \
    git checkout unpin-deps && \
    git apply /tmp/patches/perception-models-audio-only.patch

# sam-audio: clone + pin to specific commit + apply audio-only patch
RUN git clone --filter=blob:none https://github.com/facebookresearch/sam-audio.git /opt/sam-audio && \
    cd /opt/sam-audio && \
    git checkout ${SAM_AUDIO_REF} && \
    git apply /tmp/patches/sam-audio-audio-only.patch
ENV PYTHONPATH="/opt/sam-audio:/opt/perception-models"

# build-time smoke test: verify torch wasn't overwritten and imports work
RUN python -c "\
import huggingface_hub, torch; \
assert torch.__version__.startswith('2.5.1'), f'torch version: {torch.__version__}'; \
assert torch.version.cuda == '12.4', f'cuda version: {torch.version.cuda}'; \
from sam_audio import SAMAudio, SAMAudioProcessor; \
from sam_audio.ranking.imagebind import __imagebind_exists__; \
from core.audio_visual_encoder.transforms import AudioProcessor, PEAudioFrameTransform; \
assert __imagebind_exists__, 'imagebind runtime deps missing'; \
print(f'torch {torch.__version__} cuda {torch.version.cuda}'); \
print('sam_audio runtime imports ok')"

# runpod handler deps
RUN python -m pip install --no-cache-dir runpod==1.8.2 requests==2.32.5

COPY handler.py ./

ENV PYTHONUNBUFFERED=1

CMD ["python", "handler.py"]
