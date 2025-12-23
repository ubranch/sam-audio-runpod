#!/usr/bin/env python3
"""
SAM-Audio RunPod Serverless Handler

Performs GPU inference on batches of audio segments.
Chunking and stitching are client responsibilities.
"""

import base64
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

import runpod
import requests
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

# Global model instances (loaded once during cold start)
model = None
processor = None


def load_model():
    """Load the SAMAudio model and processor from MODEL_PATH or HuggingFace."""
    global model, processor
    
    if model is not None:
        return
    
    from sam_audio import SAMAudio, SAMAudioProcessor
    
    # Use MODEL_PATH from RunPod cache, or fall back to HuggingFace
    model_path = os.environ.get("MODEL_PATH")
    if model_path and os.path.exists(model_path):
        log.info(f"Loading model from RunPod cache: {model_path}")
    else:
        # Default to small model if no cache path
        model_path = os.environ.get("MODEL_ID", "facebook/sam-audio-small")
        log.info(f"Loading model from HuggingFace: {model_path}")
    
    model = SAMAudio.from_pretrained(model_path)
    processor = SAMAudioProcessor.from_pretrained(model_path)
    model = model.eval().half().cuda()
    
    log.info("Model loaded successfully")


def decode_audio(audio_url: Optional[str] = None, audio_base64: Optional[str] = None) -> torch.Tensor:
    """
    Decode audio from URL or base64 string.
    Returns tensor at 48kHz sample rate.
    """
    if audio_url:
        # Download from URL
        response = requests.get(audio_url, timeout=60)
        response.raise_for_status()
        audio_bytes = response.content
    elif audio_base64:
        # Decode base64
        audio_bytes = base64.b64decode(audio_base64)
    else:
        raise ValueError("Either audio_url or audio_base64 must be provided")
    
    # Write to temp file and load with torchaudio
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
        tmp.write(audio_bytes)
        tmp.flush()
        wav, sr = torchaudio.load(tmp.name)
    
    # Resample to 48kHz if needed
    if sr != SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
    
    return wav


def encode_audio(wav: torch.Tensor, output_format: str = "wav") -> str:
    """Encode audio tensor to base64 string."""
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    
    buffer = io.BytesIO()
    torchaudio.save(buffer, wav, SAMPLE_RATE, format=output_format)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")


def validate_input(job_input: dict) -> tuple[list, dict]:
    """
    Validate and parse input, returning items list and global options.
    """
    items = job_input.get("items", [])
    
    if not items:
        raise ValueError("At least one item is required in 'items' array")
    
    # Validate each item
    for i, item in enumerate(items):
        if not item.get("audio_url") and not item.get("audio_base64"):
            raise ValueError(f"Item {i}: Either 'audio_url' or 'audio_base64' is required")
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
            wav = decode_audio(
                audio_url=item.get("audio_url"),
                audio_base64=item.get("audio_base64")
            )
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
        
        # Process results
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
            
            item_result = {
                "duration_seconds": target.shape[-1] / SAMPLE_RATE,
            }
            
            if options["return_target"]:
                item_result["target_base64"] = encode_audio(target, options["output_format"])
            
            if options["return_residual"]:
                item_result["residual_base64"] = encode_audio(residual, options["output_format"])
            
            results.append(item_result)
        
        return {
            "results": results,
            "sample_rate": SAMPLE_RATE,
        }
    
    except Exception as e:
        log.exception("Handler error")
        return {"error": str(e)}


# Pre-load model during container startup
log.info("Initializing SAM-Audio handler...")
load_model()

# Start RunPod serverless
runpod.serverless.start({"handler": handler})

