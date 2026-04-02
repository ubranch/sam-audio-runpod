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

# use python -m pip so the image always installs into python 3.11, and upgrade
# the build frontend first so pyproject metadata and vcs dependencies resolve
# deterministically during the image build.
RUN python -m pip install --no-cache-dir --upgrade \
    pip \
    setuptools \
    wheel

# install pytorch first, matching the working reference repo's flow.
RUN python -m pip install --no-cache-dir \
    torch==2.5.1 \
    torchaudio==2.5.1 \
    torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cu124

# let upstream sam-audio resolve its own dependency tree instead of maintaining
# a custom binary stack here. use the named vcs requirement syntax recommended
# by pip so the resolver keeps the project metadata attached to the install.
RUN python -m pip install --no-cache-dir \
    "sam_audio @ git+https://github.com/facebookresearch/sam-audio.git@${SAM_AUDIO_REF}"

COPY patches/ /tmp/patches/

# keep pinned source trees on PYTHONPATH and patch them in-place so this image
# remains audio-only and never imports torchcodec during bootstrap.
RUN git clone --filter=blob:none https://github.com/facebookresearch/perception_models.git /opt/perception-models && \
    cd /opt/perception-models && \
    git checkout unpin-deps && \
    git apply /tmp/patches/perception-models-audio-only.patch

RUN git clone --filter=blob:none https://github.com/facebookresearch/sam-audio.git /opt/sam-audio && \
    cd /opt/sam-audio && \
    git checkout ${SAM_AUDIO_REF} && \
    git apply /tmp/patches/sam-audio-audio-only.patch
ENV PYTHONPATH="/opt/sam-audio:/opt/perception-models"

# fail the image build if the resolver upgraded torch away from the expected
# runtime, or if the patched import path is still broken.
RUN python -c "import huggingface_hub, torch, transformers; from sam_audio import SAMAudio, SAMAudioProcessor; from sam_audio.ranking.imagebind import __imagebind_exists__; from core.audio_visual_encoder.transforms import AudioProcessor, PEAudioFrameTransform; assert torch.__version__.startswith('2.5.1'), torch.__version__; assert torch.version.cuda == '12.4', torch.version.cuda; assert __imagebind_exists__, 'imagebind runtime deps missing'; print(f'torch {torch.__version__} cuda {torch.version.cuda}'); print(f'huggingface_hub {huggingface_hub.__version__} transformers {transformers.__version__}'); print('sam_audio runtime imports ok')"

# runpod handler deps
RUN python -m pip install --no-cache-dir runpod==1.8.2 requests==2.32.5

COPY handler.py ./

RUN useradd --create-home --shell /bin/bash appuser
RUN mkdir -p /runpod-volume/huggingface-cache/hub /runpod-volume/torch-cache && \
    chown -R appuser:appuser /runpod-volume
USER appuser

ENV PYTHONUNBUFFERED=1

CMD ["python", "handler.py"]
