# RunPod Serverless Dockerfile for SAM Audio
# Based on PyTorch with CUDA support

FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install FFmpeg 7 from jellyfin repo (torchcodec works better with FFmpeg 5+)
RUN curl -fsSL https://repo.jellyfin.org/ubuntu/jellyfin_team.gpg.key | gpg --dearmor -o /usr/share/keyrings/jellyfin.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/jellyfin.gpg] https://repo.jellyfin.org/ubuntu jammy main" > /etc/apt/sources.list.d/jellyfin.list && \
    apt-get update && \
    apt-get install -y jellyfin-ffmpeg7 && \
    ln -s /usr/lib/jellyfin-ffmpeg/ffmpeg /usr/local/bin/ffmpeg && \
    ln -s /usr/lib/jellyfin-ffmpeg/ffprobe /usr/local/bin/ffprobe && \
    echo "/usr/lib/jellyfin-ffmpeg/lib" > /etc/ld.so.conf.d/jellyfin-ffmpeg.conf && \
    ldconfig && \
    rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip

# Upgrade PyTorch to 2.5+ (required for torch.nn.attention.flex_attention)
RUN pip install --no-cache-dir \
    torch==2.5.1 \
    torchvision==0.20.1 \
    torchaudio==2.5.1 \
    --index-url https://download.pytorch.org/whl/cu124

# Copy pyproject.toml and install base dependencies
COPY pyproject.toml /app/
RUN pip install --no-cache-dir .

# Install Facebook Research packages
# ImageBind needs its deps for import to work, others use --no-deps to avoid conflicts
RUN pip install --no-cache-dir --no-deps \
    git+https://github.com/facebookresearch/pytorchvideo.git@6cdc929315aab1b5674b6dcf73b16ec99147735f \
    git+https://github.com/facebookresearch/dacvae.git

# Install ImageBind WITH dependencies (required for import to succeed)
RUN pip install --no-cache-dir \
    git+https://github.com/facebookresearch/ImageBind.git

# Install remaining FB packages with --no-deps
RUN pip install --no-cache-dir --no-deps \
    git+https://github.com/facebookresearch/perception_models.git@unpin-deps \
    git+https://github.com/facebookresearch/sam-audio.git

# CRITICAL: Install PyTorch + torchcodec together from SAME cu124 index
# This ensures ABI compatibility. torchcodec MUST come from PyTorch's index, 
# NOT PyPI, otherwise you get "undefined symbol" errors at runtime.
RUN pip install --no-cache-dir --force-reinstall \
    torch==2.5.1 \
    torchvision==0.20.1 \
    torchaudio==2.5.1 \
    torchcodec \
    --index-url https://download.pytorch.org/whl/cu124

# Verify versions and imports
RUN echo "=== Version Check ===" && \
    python -c "import torch; print(f'PyTorch: {torch.__version__}')" && \
    python -c "import torchcodec; print(f'TorchCodec: {torchcodec.__version__}')" && \
    ffmpeg -version | head -1 && \
    echo "===================="

RUN python -c "import torch; assert torch.__version__.startswith('2.5'), f'Need PyTorch 2.5+, got {torch.__version__}'"
RUN python -c "import torchcodec; print(f'torchcodec {torchcodec.__version__} loaded successfully')"
RUN python -c "from imagebind import data; print('ImageBind import OK')"
RUN python -c "from sam_audio import SAMAudio, SAMAudioProcessor; print('SAM Audio import OK')" || \
    python -c "import traceback; exec(\"try:\\n    from sam_audio import SAMAudio\\nexcept:\\n    traceback.print_exc()\")"

# Copy handler code
COPY handler.py /app/handler.py

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Model loading options (in order of priority):
# 1. RunPod Cached Models: Set endpoint "Cached Models" to the HF model URL
#    (e.g., https://huggingface.co/facebook/sam-audio-large)
#    The handler auto-detects and uses /runpod-volume/huggingface-cache/hub
# 2. HF_TOKEN: Set at runtime for downloading gated models on-demand


# Start the handler
CMD ["python", "-u", "handler.py"]

