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

# pytorch (cuda 12.4 wheels matching the container runtime)
RUN pip install --no-cache-dir \
    torch==2.5.1 \
    torchaudio==2.5.1 \
    torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cu124

# install sam-audio dependencies from the pinned upstream revision.
RUN pip install --no-cache-dir \
    git+https://github.com/facebookresearch/sam-audio.git@68b48d48fff1ad776d3afefbe634eb5f5d60ba7b

# the upstream package install can leave a non-importable wheel in this image,
# so keep the pinned source tree on PYTHONPATH as the runtime import source.
RUN git clone --filter=blob:none https://github.com/facebookresearch/sam-audio.git /opt/sam-audio && \
    cd /opt/sam-audio && \
    git checkout 68b48d48fff1ad776d3afefbe634eb5f5d60ba7b
ENV PYTHONPATH="/opt/sam-audio"

# fail the image build if sam_audio is still not importable.
RUN python -c "import importlib.util; spec = importlib.util.find_spec('sam_audio'); assert spec, 'sam_audio not found'; print('sam_audio ok:', spec.origin)"

# runpod handler deps
RUN pip install --no-cache-dir runpod==1.8.2 requests==2.32.5

COPY handler.py ./

RUN useradd --create-home --shell /bin/bash appuser
USER appuser

ENV PYTHONUNBUFFERED=1

CMD ["python", "handler.py"]
