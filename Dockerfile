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

# install sam-audio the same way as the working upstream fork and pin the
# currently published sam-audio revision to avoid future drift.
RUN pip install --no-cache-dir \
    git+https://github.com/facebookresearch/sam-audio.git@68b48d48fff1ad776d3afefbe634eb5f5d60ba7b

# runpod handler deps
RUN pip install --no-cache-dir runpod==1.8.2 requests==2.32.5

COPY handler.py ./

RUN useradd --create-home --shell /bin/bash appuser
USER appuser

ENV PYTHONUNBUFFERED=1

CMD ["python", "handler.py"]
