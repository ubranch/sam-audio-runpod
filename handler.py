#!/usr/bin/env python3
"""
SAM-Audio RunPod Serverless Handler

Performs GPU inference on batches of audio segments.
Chunking and stitching are client responsibilities.
"""

import io
import logging
import os
import tempfile
import warnings
from typing import Optional

# Suppress noisy warnings before importing libraries
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ---------------------------
# Model Store Configuration (MUST be before HuggingFace imports)
# ---------------------------

# RunPod Model Store cache location
HF_CACHE_ROOT = "/runpod-volume/huggingface-cache/hub"

# Default model ID (can be overridden by MODEL_NAME env var)
MODEL_ID = os.environ.get("MODEL_NAME", "facebook/sam-audio-large")


def resolve_snapshot_path(model_id: str) -> Optional[str]:
    """
    Convert a HF model id like 'facebook/sam-audio-large'
    into its local snapshot path inside Model Store cache.
    
    Returns None if not found (will fall back to downloading).
    """
    if "/" not in model_id:
        print(f"[ModelStore] MODEL_ID '{model_id}' is not in 'org/name' format")
        return None
    
    org, name = model_id.split("/", 1)
    model_root = os.path.join(HF_CACHE_ROOT, f"models--{org}--{name}")
    
    print(f"[ModelStore] MODEL_ID: {model_id}")
    print(f"[ModelStore] Model root: {model_root}")
    
    if not os.path.isdir(model_root):
        print(f"[ModelStore] Cache directory not found: {model_root}")
        # Debug: show what exists in cache
        if os.path.isdir(HF_CACHE_ROOT):
            try:
                existing = sorted(os.listdir(HF_CACHE_ROOT))[:10]
                print(f"[ModelStore] Available in cache: {existing}")
            except Exception:
                pass
        return None
    
    refs_main = os.path.join(model_root, "refs", "main")
    snapshots_dir = os.path.join(model_root, "snapshots")
    
    # 1) Preferred: use refs/main to get active snapshot hash
    if os.path.isfile(refs_main):
        with open(refs_main, "r") as f:
            snapshot_hash = f.read().strip()
        candidate = os.path.join(snapshots_dir, snapshot_hash)
        if os.path.isdir(candidate):
            print(f"[ModelStore] Using snapshot from refs/main: {candidate}")
            return candidate
        else:
            print(f"[ModelStore] Snapshot from refs/main not found: {candidate}")
    
    # 2) Fallback: list snapshots directory and pick first one
    if not os.path.isdir(snapshots_dir):
        print(f"[ModelStore] Snapshots directory not found: {snapshots_dir}")
        return None
    
    versions = [d for d in os.listdir(snapshots_dir)
                if os.path.isdir(os.path.join(snapshots_dir, d))]
    
    if not versions:
        print(f"[ModelStore] No snapshot subdirectories found under {snapshots_dir}")
        return None
    
    versions.sort()
    chosen = os.path.join(snapshots_dir, versions[0])
    print(f"[ModelStore] Using first available snapshot: {chosen}")
    return chosen


# Resolve model path BEFORE importing HuggingFace libraries
LOCAL_MODEL_PATH = resolve_snapshot_path(MODEL_ID)

if LOCAL_MODEL_PATH:
    # Force offline mode - MUST be set before importing transformers/huggingface_hub
    print("[ModelStore] Cache found, enabling offline mode")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
else:
    print("[ModelStore] No cache found, will download from HuggingFace")
    # Authenticate if HF_TOKEN provided
    hf_token = os.environ.get("HF_TOKEN")
    if hf_token:
        print("[ModelStore] HF_TOKEN found, will authenticate")

# ---------------------------
# Now safe to import HuggingFace libraries
# ---------------------------

import boto3
import runpod
import torch
import torchaudio

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger(__name__)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

# Constants
SAMPLE_RATE = 48000
R2_BUCKET_NAME = os.environ.get("R2_BUCKET_NAME", "")

# Validate R2 configuration at startup
_R2_REQUIRED = ["R2_ENDPOINT_URL", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_BUCKET_NAME"]
_r2_missing = [v for v in _R2_REQUIRED if not os.environ.get(v)]
if _r2_missing:
    raise RuntimeError(f"Missing required R2 environment variables: {', '.join(_r2_missing)}")

# Global model instances (loaded once during cold start)
model = None
processor = None

# R2 client singleton
_r2_client = None


def get_r2_client():
    """Get or create the R2 S3-compatible client."""
    global _r2_client
    if _r2_client is None:
        _r2_client = boto3.client(
            "s3",
            endpoint_url=os.environ["R2_ENDPOINT_URL"],
            aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
            aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
            region_name="auto",
        )
    return _r2_client


def load_model():
    """Load the SAMAudio model and processor."""
    global model, processor
    
    if model is not None:
        return
    
    from sam_audio import SAMAudio, SAMAudioProcessor
    
    if LOCAL_MODEL_PATH:
        # Load from cached snapshot (offline)
        log.info(f"Loading model from Model Store cache: {LOCAL_MODEL_PATH}")
        model = SAMAudio.from_pretrained(LOCAL_MODEL_PATH)
        processor = SAMAudioProcessor.from_pretrained(LOCAL_MODEL_PATH)
    else:
        # Download from HuggingFace
        log.info(f"Loading model from HuggingFace: {MODEL_ID}")
        hf_token = os.environ.get("HF_TOKEN")
        if hf_token:
            from huggingface_hub import login
            login(token=hf_token)
        model = SAMAudio.from_pretrained(MODEL_ID)
        processor = SAMAudioProcessor.from_pretrained(MODEL_ID)
    
    model = model.eval().half().cuda()
    log.info("Model loaded successfully")


def decode_audio(r2_key: str) -> torch.Tensor:
    """
    Download audio from R2 by object key.
    Returns tensor at 48kHz sample rate.
    """
    r2 = get_r2_client()
    response = r2.get_object(Bucket=R2_BUCKET_NAME, Key=r2_key)
    audio_bytes = response["Body"].read()

    # Write to temp file and load with torchaudio
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
        tmp.write(audio_bytes)
        tmp.flush()
        wav, sr = torchaudio.load(tmp.name)

    # Resample to 48kHz if needed
    if sr != SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)

    return wav


def upload_audio(wav: torch.Tensor, r2_key: str, output_format: str = "wav") -> str:
    """Encode audio tensor and upload to R2. Returns the R2 key."""
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)

    buffer = io.BytesIO()
    torchaudio.save(buffer, wav, SAMPLE_RATE, format=output_format)
    buffer.seek(0)

    content_type = "audio/flac" if output_format == "flac" else "audio/wav"
    r2 = get_r2_client()
    r2.put_object(
        Bucket=R2_BUCKET_NAME,
        Key=r2_key,
        Body=buffer.read(),
        ContentType=content_type,
    )
    return r2_key


def validate_input(job_input: dict) -> tuple[list, dict]:
    """
    Validate and parse input, returning items list and global options.
    """
    items = job_input.get("items", [])
    
    if not items:
        raise ValueError("At least one item is required in 'items' array")
    
    # Validate each item
    for i, item in enumerate(items):
        if not item.get("r2_key"):
            raise ValueError(f"Item {i}: 'r2_key' is required")
        if not item.get("description"):
            raise ValueError(f"Item {i}: 'description' is required")
    
    # Global options
    return_target = job_input.get("return_target", True)
    return_residual = job_input.get("return_residual", False)
    
    # At least one output must be requested
    if not return_target and not return_residual:
        return_target = True  # Default to target if neither specified
    
    options = {
        "predict_spans": job_input.get("predict_spans", False),
        "reranking_candidates": min(max(job_input.get("reranking_candidates", 1), 1), 16),
        "return_target": return_target,
        "return_residual": return_residual,
        "output_format": job_input.get("output_format", "wav"),
        "output_prefix": job_input.get("output_prefix", "outputs/"),
    }
    
    return items, options


def handler(job):
    """
    RunPod serverless handler for SAM-Audio batch inference.
    """
    try:
        job_input = job.get("input", {})
        
        # Validate input
        items, options = validate_input(job_input)
        
        # Load model (no-op if already loaded)
        load_model()
        
        # Decode all audio items
        log.info(f"Processing batch of {len(items)} items")
        audios = []
        descriptions = []
        
        for item in items:
            wav = decode_audio(r2_key=item["r2_key"])
            audios.append(wav)
            descriptions.append(item["description"])
        
        # Create batch and run inference
        batch = processor(
            audios=audios,
            descriptions=descriptions,
        ).to("cuda")
        batch.audios = batch.audios.half()
        
        torch.cuda.empty_cache()
        with torch.inference_mode():
            result = model.separate(
                batch,
                predict_spans=options["predict_spans"],
                reranking_candidates=options["reranking_candidates"],
            )
        
        # Process results and upload to R2
        job_id = job.get("id", "unknown")
        fmt = options["output_format"]
        results = []
        for i in range(len(items)):
            target = result.target[i] if isinstance(result.target, list) else result.target
            residual = result.residual[i] if isinstance(result.residual, list) else result.residual

            if target.dim() == 1:
                target = target.unsqueeze(0)
            if residual.dim() == 1:
                residual = residual.unsqueeze(0)

            target = target.float().cpu()
            residual = residual.float().cpu()

            base_key = f"{options['output_prefix']}{job_id}/item_{i}"
            item_result = {
                "duration_seconds": target.shape[-1] / SAMPLE_RATE,
            }

            if options["return_target"]:
                target_key = f"{base_key}_target.{fmt}"
                upload_audio(target, target_key, fmt)
                item_result["target_r2_key"] = target_key

            if options["return_residual"]:
                residual_key = f"{base_key}_residual.{fmt}"
                upload_audio(residual, residual_key, fmt)
                item_result["residual_r2_key"] = residual_key

            results.append(item_result)
        
        return {
            "results": results,
            "sample_rate": SAMPLE_RATE,
        }
    
    except Exception as e:
        log.exception("Handler error")
        return {"error": str(e)}


# ---------------------------
# Startup
# ---------------------------

print("[Handler] Initializing SAM-Audio handler...")
load_model()
print("[Handler] Model loaded, starting RunPod serverless...")

runpod.serverless.start({"handler": handler})
