# SAM Audio Service

Audio source separation using Meta's [Segment Anything Audio (SAM-Audio)](https://github.com/facebookresearch/sam-audio) model. Isolate any sound from audio using text descriptions.

## Features

- **Text-prompted separation**: Describe what you want to isolate (e.g., "Drums", "Vocals", "A man speaking")
- **Cloudflare R2 storage**: Input and output audio via R2 object keys (S3-compatible)
- **Batch processing**: Process multiple audio items in a single request
- **RunPod Serverless**: Deploy as a scalable serverless endpoint

## Local Development

### Setup

```bash
cd sam_service

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

### Environment Variables

Copy `.env.example` to `.env` and fill in your values:

```bash
cp .env.example .env
```

Required variables:

| Variable | Description |
|----------|-------------|
| `R2_ENDPOINT_URL` | Cloudflare R2 S3-compatible endpoint (e.g., `https://<account_id>.r2.cloudflarestorage.com`) |
| `R2_ACCESS_KEY_ID` | R2 API token access key ID |
| `R2_SECRET_ACCESS_KEY` | R2 API token secret access key |
| `R2_BUCKET_NAME` | R2 bucket name |

Optional variables:

| Variable | Description |
|----------|-------------|
| `HF_TOKEN` | Hugging Face token (required for gated model access) |
| `MODEL_NAME` | Override default model (defaults to `facebook/sam-audio-large`) |

### Hugging Face Authentication

The SAM Audio model requires Hugging Face authentication:

1. Create an account at [huggingface.co](https://huggingface.co)
2. Request access to [facebook/sam-audio-large](https://huggingface.co/facebook/sam-audio-large)
3. Login via CLI:
   ```bash
   huggingface-cli login
   ```

## RunPod Serverless Deployment

### Build Docker Image

```bash
# Build the image
docker build -t sam-audio-serverless .

# Tag for your registry
docker tag sam-audio-serverless your-registry/sam-audio-serverless:latest

# Push to registry
docker push your-registry/sam-audio-serverless:latest
```

### Deploy to RunPod

1. Go to [RunPod Serverless](https://www.runpod.io/console/serverless)
2. Create a new endpoint
3. Select your Docker image
4. Configure GPU (recommended: RTX 3090 or better for `large` model)
5. Set environment variables:
   - `R2_ENDPOINT_URL`: Your Cloudflare R2 S3-compatible endpoint
   - `R2_ACCESS_KEY_ID`: R2 API token access key ID
   - `R2_SECRET_ACCESS_KEY`: R2 API token secret access key
   - `R2_BUCKET_NAME`: R2 bucket name
   - `HF_TOKEN`: Your Hugging Face access token (required for gated model access)

### API Usage

#### Request Format

```json
{
  "input": {
    "items": [
      {
        "r2_key": "inputs/song1.wav",
        "description": "Drums"
      }
    ],
    "return_target": true,
    "return_residual": false,
    "output_format": "wav",
    "output_prefix": "outputs/"
  }
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `items[].r2_key` | string | *required* | R2 object key for the input audio file |
| `items[].description` | string | *required* | Text description of the sound to isolate |
| `return_target` | bool | `true` | Return the isolated audio |
| `return_residual` | bool | `false` | Return the background/residual audio |
| `output_format` | string | `"wav"` | Output audio format (`wav`, `flac`) |
| `output_prefix` | string | `"outputs/"` | R2 key prefix for output files |
| `predict_spans` | bool | `false` | Predict time spans of the target sound |
| `reranking_candidates` | int | `1` | Number of reranking candidates (1-16) |

#### Response Format

```json
{
  "results": [
    {
      "target_r2_key": "outputs/<job_id>/item_0_target.wav",
      "residual_r2_key": "outputs/<job_id>/item_0_residual.wav",
      "duration_seconds": 5.2
    }
  ],
  "sample_rate": 48000
}
```

Output files are uploaded to R2 at `{output_prefix}{job_id}/item_{i}_target.{format}`.

### Example Python Client

```python
import runpod

runpod.api_key = "your_api_key"

endpoint = runpod.Endpoint("your_endpoint_id")

# Run separation — input audio must already exist in R2
result = endpoint.run_sync({
    "input": {
        "items": [
            {
                "r2_key": "inputs/song.wav",
                "description": "Drums"
            }
        ],
        "return_target": True,
        "output_prefix": "separated/"
    }
})

# Result contains R2 keys where output was uploaded
for item in result["results"]:
    print(f"Target uploaded to: {item['target_r2_key']}")
    # Download from R2 using boto3 if needed
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

## License

SAM-Audio is licensed under the [SAM License](https://github.com/facebookresearch/sam-audio/blob/main/LICENSE).
