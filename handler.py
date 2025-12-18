"""
RunPod Serverless Handler for SAM Audio (Segment Anything Audio)
Separates audio sources based on text descriptions.

Input format:
{
    "input": {
        "audio_url": "https://example.com/audio.wav",  # URL to audio file
        # OR
        "audio_base64": "base64_encoded_audio_data",   # Base64 encoded audio
        "description": "Drums",                         # What sound to isolate
        "model_size": "large"                           # Optional: small, base, large (default: large)
    }
}

Output format:
{
    "target_base64": "base64_encoded_isolated_audio",
    "residual_base64": "base64_encoded_residual_audio",
    "sample_rate": 16000,
    "description": "Drums"
}
"""

import runpod
import torch
import torchaudio
import base64
import os
import tempfile
import requests
from pathlib import Path

# RunPod Cached Models location (HF cache layout)
CACHE_ROOT = Path("/runpod-volume/huggingface-cache/hub")

# Configure HF to use RunPod's cache if it exists
if CACHE_ROOT.exists():
    print(f"Using RunPod cached models from: {CACHE_ROOT}")
    # Set HF cache to RunPod's cache location
    os.environ["HF_HUB_CACHE"] = str(CACHE_ROOT)
    os.environ["HF_HOME"] = str(CACHE_ROOT.parent)
    # Force offline mode - only load from cache
    # os.environ["HF_HUB_OFFLINE"] = "1"
    # os.environ["TRANSFORMERS_OFFLINE"] = "1"
else:
    print(f"RunPod cache not found at {CACHE_ROOT}, will download models")

# Global model cache - loaded once on cold start
MODEL = None
PROCESSOR = None
DEVICE = None


def resolve_snapshot_path(model_id: str) -> Path | None:
    """
    Resolve the local snapshot path for a cached model.
    Returns None if not found (will fall back to downloading).
    """
    if not CACHE_ROOT.exists():
        return None
    
    # HF cache layout: models--{org}--{name}/snapshots/{revision}
    org, name = model_id.split("/", 1)
    model_dir = CACHE_ROOT / f"models--{org}--{name}"
    
    if not model_dir.exists():
        print(f"Model cache dir not found: {model_dir}")
        # Debug: show what exists
        try:
            existing = sorted([p.name for p in CACHE_ROOT.iterdir()])[:20]
            print(f"Available in cache: {existing}")
        except Exception:
            pass
        return None
    
    # Prefer refs/main when present
    ref_main = model_dir / "refs" / "main"
    if ref_main.exists():
        rev = ref_main.read_text().strip()
        snap = model_dir / "snapshots" / rev
        if snap.exists():
            print(f"Using cached snapshot: {snap}")
            return snap
    
    # Fallback: first snapshot directory
    snap_root = model_dir / "snapshots"
    snaps = sorted(snap_root.glob("*")) if snap_root.exists() else []
    if snaps:
        print(f"Using cached snapshot: {snaps[0]}")
        return snaps[0]
    
    return None


def load_model(model_size: str = "large"):
    """Load the SAM Audio model and processor."""
    global MODEL, PROCESSOR, DEVICE
    
    if MODEL is not None and PROCESSOR is not None:
        return MODEL, PROCESSOR, DEVICE
    
    from sam_audio import SAMAudio, SAMAudioProcessor
    from huggingface_hub import login
    
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_id = f"facebook/sam-audio-{model_size}"
    
    # Check for cached model first
    snapshot_path = resolve_snapshot_path(model_id)
    
    print(f"Snapshot path: {snapshot_path}")
    if snapshot_path:
        # Load from cached snapshot (offline)
        print(f"Loading SAM Audio model from cache (size: {model_size}) on device: {DEVICE}")
        MODEL = SAMAudio.from_pretrained(str(snapshot_path)).to(DEVICE).eval()
        PROCESSOR = SAMAudioProcessor.from_pretrained(str(snapshot_path))
    else:
        # Fall back to downloading (requires HF_TOKEN for gated models)
        hf_token = os.environ.get("HF_TOKEN")
        if hf_token:
            print("Authenticating with Hugging Face...")
            login(token=hf_token)
        else:
            print("Warning: HF_TOKEN not set. Model download may fail for gated models.")
        
        print(f"Downloading SAM Audio model (size: {model_size}) on device: {DEVICE}")
        MODEL = SAMAudio.from_pretrained(model_id).to(DEVICE).eval()
        PROCESSOR = SAMAudioProcessor.from_pretrained(model_id)
    
    print("Model loaded successfully!")
    return MODEL, PROCESSOR, DEVICE


def download_audio(url: str, temp_dir: str) -> str:
    """Download audio from URL to a temporary file."""
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    
    # Determine file extension from URL or content type
    content_type = response.headers.get("content-type", "")
    if "wav" in url.lower() or "wav" in content_type:
        ext = ".wav"
    elif "mp3" in url.lower() or "mp3" in content_type or "mpeg" in content_type:
        ext = ".mp3"
    elif "flac" in url.lower() or "flac" in content_type:
        ext = ".flac"
    else:
        ext = ".wav"  # Default to wav
    
    temp_path = os.path.join(temp_dir, f"input_audio{ext}")
    with open(temp_path, "wb") as f:
        f.write(response.content)
    
    return temp_path


def decode_base64_audio(audio_base64: str, temp_dir: str) -> str:
    """Decode base64 audio to a temporary file."""
    audio_data = base64.b64decode(audio_base64)
    temp_path = os.path.join(temp_dir, "input_audio.wav")
    with open(temp_path, "wb") as f:
        f.write(audio_data)
    return temp_path


def encode_audio_to_base64(audio_path: str) -> str:
    """Encode audio file to base64."""
    with open(audio_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def handler(job):
    """
    Process audio separation job using SAM Audio.
    
    Expected input:
    - audio_url OR audio_base64: The audio to process
    - description: Text description of the sound to isolate (e.g., "Drums", "Vocals")
    - model_size: Optional model size (small, base, large). Default: large
    """
    job_input = job["input"]
    
    # Validate input
    if "audio_url" not in job_input and "audio_base64" not in job_input:
        return {"error": "Missing 'audio_url' or 'audio_base64' in input"}
    
    if "description" not in job_input:
        return {"error": "Missing 'description' in input"}
    
    description = job_input["description"]
    model_size = job_input.get("model_size", "large")
    
    if model_size not in ["small", "base", "large"]:
        return {"error": f"Invalid model_size '{model_size}'. Must be: small, base, or large"}
    
    # Load model
    runpod.serverless.progress_update(job, "Loading model...")
    model, processor, device = load_model(model_size)
    
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            # Get audio file
            runpod.serverless.progress_update(job, "Downloading/decoding audio...")
            
            if "audio_url" in job_input:
                audio_path = download_audio(job_input["audio_url"], temp_dir)
            else:
                audio_path = decode_base64_audio(job_input["audio_base64"], temp_dir)
            
            # Convert to WAV if needed (using torchaudio)
            runpod.serverless.progress_update(job, "Processing audio...")
            
            # Process with SAM Audio
            runpod.serverless.progress_update(job, f"Separating audio: '{description}'...")
            
            batch = processor(audios=[audio_path], descriptions=[description]).to(device)
            
            with torch.inference_mode():
                result = model.separate(batch)
            
            # Save separated audio
            runpod.serverless.progress_update(job, "Encoding results...")
            
            sample_rate = processor.audio_sampling_rate
            
            target_path = os.path.join(temp_dir, "target.wav")
            residual_path = os.path.join(temp_dir, "residual.wav")
            
            torchaudio.save(target_path, result.target.cpu(), sample_rate)
            torchaudio.save(residual_path, result.residual.cpu(), sample_rate)
            
            # Encode to base64
            target_base64 = encode_audio_to_base64(target_path)
            residual_base64 = encode_audio_to_base64(residual_path)
            
            return {
                "target_base64": target_base64,
                "residual_base64": residual_base64,
                "sample_rate": sample_rate,
                "description": description,
                "status": "success"
            }
            
        except requests.RequestException as e:
            return {"error": f"Failed to download audio: {str(e)}"}
        except Exception as e:
            return {"error": f"Processing error: {str(e)}"}


# Pre-load model on cold start for faster inference
# This runs when the container starts
if os.environ.get("RUNPOD_POD_ID"):
    print("Running on RunPod - pre-loading model...")
    try:
        load_model("large")
    except Exception as e:
        print(f"Warning: Could not pre-load model: {e}")


# Start the serverless worker
runpod.serverless.start({"handler": handler})

