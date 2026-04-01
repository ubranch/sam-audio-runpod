# sam-audio runpod

audio source separation as a runpod serverless endpoint, powered by meta's [sam-audio](https://github.com/facebookresearch/sam-audio). isolate any sound from audio using text descriptions.

## what it does

- text-prompted audio separation — describe a sound, get it isolated
- batch processing — up to 16 audio items per request
- multiple output formats — wav, flac, ogg, mp3
- production-ready — input validation, ssrf protection, size limits

## model variants

| model | env value | vram |
|-------|-----------|------|
| small (default) | `facebook/sam-audio-small` | ~4 gb |
| base | `facebook/sam-audio-base` | ~8 gb |
| large | `facebook/sam-audio-large` | ~16 gb |

set via the `MODEL_NAME` environment variable in your runpod template.

## deployment

### build and push

```bash
docker build -t sam-audio-serverless .
docker tag sam-audio-serverless your-registry/sam-audio-serverless:latest
docker push your-registry/sam-audio-serverless:latest
```

### runpod setup

1. go to [runpod serverless](https://www.runpod.io/console/serverless)
2. create a new endpoint with your docker image
3. set environment variables:
   - `HF_TOKEN` — huggingface access token (required, model is gated)
   - `MODEL_NAME` — override model variant (optional, defaults to `facebook/sam-audio-small`)

the model requires huggingface access — request it at the [sam-audio-small](https://huggingface.co/facebook/sam-audio-small) repo page first.

## api

### request

```json
{
  "input": {
    "items": [
      {
        "audio_url": "https://example.com/audio.wav",
        "description": "drums"
      }
    ],
    "return_target": true,
    "return_residual": false,
    "output_format": "wav",
    "predict_spans": false,
    "reranking_candidates": 1
  }
}
```

| field | type | default | description |
|-------|------|---------|-------------|
| `items` | array | required | audio items to process (max 16) |
| `items[].audio_url` | string | — | url to audio file |
| `items[].audio_base64` | string | — | base64-encoded audio |
| `items[].description` | string | required | sound to isolate |
| `return_target` | bool | `true` | return isolated audio |
| `return_residual` | bool | `false` | return everything except target |
| `output_format` | string | `"wav"` | wav, flac, ogg, or mp3 |
| `predict_spans` | bool | `false` | auto-detect time spans |
| `reranking_candidates` | int | `1` | reranking candidates (1–16) |

each item needs either `audio_url` or `audio_base64`, plus a `description`.

### response

```json
{
  "results": [
    {
      "duration_seconds": 10.5,
      "target_base64": "<base64>",
      "residual_base64": "<base64>"
    }
  ],
  "sample_rate": 48000
}
```

### example client

```python
import runpod
import base64

runpod.api_key = "your_api_key"
endpoint = runpod.Endpoint("your_endpoint_id")

result = endpoint.run_sync({
    "input": {
        "items": [
            {
                "audio_url": "https://example.com/song.wav",
                "description": "drums"
            }
        ],
        "return_target": True,
        "output_format": "wav"
    }
})

if "results" in result:
    audio = base64.b64decode(result["results"][0]["target_base64"])
    with open("drums.wav", "wb") as f:
        f.write(audio)
```

### limits

| limit | value |
|-------|-------|
| batch size | 16 items |
| audio file size | 100 mb |
| sample rate | 48 khz (auto-resampled) |
| input formats | anything ffmpeg supports |

## local development

```bash
uv venv --python 3.11
source .venv/bin/activate

uv pip install torch torchaudio transformers scipy soundfile \
    torchcodec torchdiffeq descript-audiotools eva-decord \
    einops timm ftfy xformers

# facebook research packages (--no-deps to avoid conflicts)
uv pip install --no-deps git+https://github.com/facebookresearch/perception_models.git@unpin-deps
uv pip install --no-deps git+https://github.com/facebookresearch/ImageBind.git
uv pip install --no-deps git+https://github.com/facebookresearch/dacvae.git
uv pip install --no-deps git+https://github.com/facebookresearch/pytorchvideo.git@6cdc929315aab1b5674b6dcf73b16ec99147735f
uv pip install iopath

# sam-audio must be cloned (pip install produces a broken wheel)
git clone https://github.com/facebookresearch/sam-audio.git /opt/sam-audio
cd /opt/sam-audio && git checkout 68b48d48fff1ad776d3afefbe634eb5f5d60ba7b
export PYTHONPATH="/opt/sam-audio:$PYTHONPATH"
```

### tests

```bash
pip install -r requirements-dev.txt
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install runpod requests
pytest tests/ -v
```

## license

sam-audio is licensed under the [sam license](https://github.com/facebookresearch/sam-audio/blob/main/LICENSE).
