# SAM Audio Service

Audio source separation using Meta's [Segment Anything Audio (SAM-Audio)](https://github.com/facebookresearch/sam-audio) model. Isolate any sound from audio using text descriptions.

## Features

- **Text-prompted separation**: Describe what you want to isolate (e.g., "Drums", "Vocals", "A man speaking")
- **Batch processing**: Process multiple audio items in a single request (up to 16)
- **Multiple output formats**: WAV, FLAC, OGG, MP3
- **RunPod Serverless**: Deploy as a scalable serverless endpoint
- **Production hardened**: Input validation, SSRF protection, size limits, sanitized errors

## Local Development

### Setup

```bash
# Create virtual environment with Python 3.11+
uv venv --python 3.11
source .venv/bin/activate

# Install dependencies
uv pip install torch torchaudio transformers scipy soundfile torchcodec torchdiffeq descript-audiotools eva-decord

# Install Facebook Research packages
uv pip install --no-deps git+https://github.com/facebookresearch/sam-audio.git
uv pip install --no-deps git+https://github.com/facebookresearch/perception_models.git@unpin-deps
uv pip install --no-deps git+https://github.com/facebookresearch/ImageBind.git
uv pip install --no-deps git+https://github.com/facebookresearch/dacvae.git
uv pip install --no-deps git+https://github.com/facebookresearch/pytorchvideo.git@6cdc929315aab1b5674b6dcf73b16ec99147735f
uv pip install iopath
```

### Hugging Face Authentication

The SAM Audio model requires Hugging Face authentication:

1. Create an account at [huggingface.co](https://huggingface.co)
2. Request access to [facebook/sam-audio-large](https://huggingface.co/facebook/sam-audio-large)
3. Login via CLI:
   ```bash
   huggingface-cli login
   ```

### Running Tests

```bash
pip install -r requirements-dev.txt
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install runpod requests
pytest tests/ -v
```

### Linting

```bash
ruff check handler.py tests/
```

## RunPod Serverless Deployment

### Build Docker Image

```bash
docker build -t sam-audio-serverless .
docker tag sam-audio-serverless your-registry/sam-audio-serverless:latest
docker push your-registry/sam-audio-serverless:latest
```

### Deploy to RunPod

1. Go to [RunPod Serverless](https://www.runpod.io/console/serverless)
2. Create a new endpoint
3. Select your Docker image
4. Configure GPU (recommended: RTX 3090 or better for `large` model)
5. Set environment variables:
   - `HF_TOKEN`: Your Hugging Face access token (required for gated model access)
   - `MODEL_NAME` (optional): Override model ID (default: `facebook/sam-audio-large`)

## API Usage

### Request Format

The API accepts batch requests with one or more items:

```json
{
  "input": {
    "items": [
      {
        "audio_url": "https://example.com/audio.wav",
        "description": "Drums"
      },
      {
        "audio_base64": "<base64-encoded-audio>",
        "description": "Vocals"
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

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `items` | array | *required* | Array of items to process (max 16) |
| `items[].audio_url` | string | - | URL to audio file (must be http/https) |
| `items[].audio_base64` | string | - | Base64-encoded audio data |
| `items[].description` | string | *required* | Text description of the sound to isolate |
| `return_target` | bool | `true` | Return the isolated audio |
| `return_residual` | bool | `false` | Return the residual (everything except the target) |
| `output_format` | string | `"wav"` | Output format: `wav`, `flac`, `ogg`, or `mp3` |
| `predict_spans` | bool | `false` | Predict time spans of the target sound |
| `reranking_candidates` | int | `1` | Number of reranking candidates (1-16) |

Each item must provide either `audio_url` or `audio_base64`, plus a `description`.

### Response Format

```json
{
  "results": [
    {
      "duration_seconds": 10.5,
      "target_base64": "<base64-encoded-isolated-audio>",
      "residual_base64": "<base64-encoded-residual-audio>"
    }
  ],
  "sample_rate": 48000
}
```

Error responses:

```json
{
  "error": "Description of what went wrong"
}
```

### Limits

| Limit | Value |
|-------|-------|
| Max batch size | 16 items |
| Max audio file size | 100 MB |
| Sample rate | 48 kHz (auto-resampled) |
| Supported input formats | WAV, MP3, FLAC, OGG, and anything ffmpeg supports |

### Example Python Client

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
                "description": "Drums"
            }
        ],
        "return_target": True,
        "output_format": "wav"
    }
})

if "results" in result:
    target_audio = base64.b64decode(result["results"][0]["target_base64"])
    with open("drums_isolated.wav", "wb") as f:
        f.write(target_audio)
```

## Supported Descriptions

SAM-Audio can isolate various sounds:

- **Instruments**: "Drums", "Guitar", "Piano", "Bass", "Violin"
- **Vocals**: "Vocals", "A man speaking", "A woman singing"
- **Effects**: "Applause", "Crowd noise", "Wind"
- **General**: Any natural language description of the sound

## Model Sizes

| Model | VRAM | Speed | Quality |
|-------|------|-------|---------|
| small | ~4GB | Fast | Good |
| base | ~8GB | Medium | Better |
| large | ~16GB | Slower | Best |

Set via the `MODEL_NAME` environment variable (e.g., `facebook/sam-audio-small`).

## License

SAM-Audio is licensed under the [SAM License](https://github.com/facebookresearch/sam-audio/blob/main/LICENSE).
