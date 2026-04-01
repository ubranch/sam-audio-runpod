#!/usr/bin/env python3
"""
sam-audio runpod serverless handler.
batch gpu inference for audio source separation.
"""

import base64
import io
import ipaddress
import logging
import os
import tempfile
import warnings
from typing import Optional
from urllib.parse import urlparse

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["TOKENIZERS_PARALLELISM"] = "false"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

HF_HOME_ROOT = "/runpod-volume/huggingface-cache"
HF_CACHE_ROOT = os.path.join(HF_HOME_ROOT, "hub")
TORCH_CACHE_ROOT = "/runpod-volume/torch-cache"
MODEL_ID = os.environ.get("MODEL_NAME", "mrfakename/sam-audio-large")
_MODEL_SOURCE_UNSET = object()


def configure_cache_environment() -> None:
    """force model and torch caches onto the runpod volume."""
    os.environ["HF_HOME"] = HF_HOME_ROOT
    os.environ["HF_HUB_CACHE"] = HF_CACHE_ROOT
    os.environ["TRANSFORMERS_CACHE"] = HF_CACHE_ROOT
    os.environ["TORCH_HOME"] = TORCH_CACHE_ROOT


configure_cache_environment()


def resolve_snapshot_path(model_id: str) -> Optional[str]:
    """resolve a huggingface model id to its local cache snapshot path, or None."""
    if "/" not in model_id:
        log.warning("MODEL_ID '%s' is not in 'org/name' format", model_id)
        return None
    
    org, name = model_id.split("/", 1)
    model_root = os.path.join(HF_CACHE_ROOT, f"models--{org}--{name}")
    
    log.info("cache probe: model id %s", model_id)
    log.info("cache probe: model root %s", model_root)
    
    if not os.path.isdir(model_root):
        log.info("cache probe: miss")
        if os.path.isdir(HF_CACHE_ROOT):
            try:
                existing = sorted(os.listdir(HF_CACHE_ROOT))[:10]
                log.info("cache probe: root contents %s", existing)
            except OSError:
                pass
        return None

    refs_main = os.path.join(model_root, "refs", "main")
    snapshots_dir = os.path.join(model_root, "snapshots")

    if os.path.isfile(refs_main):
        with open(refs_main, "r") as f:
            snapshot_hash = f.read().strip()
        candidate = os.path.join(snapshots_dir, snapshot_hash)
        if os.path.isdir(candidate):
            log.info("cache probe: hit")
            log.info("cache probe: using cached snapshot %s", candidate)
            return candidate
        else:
            log.warning("snapshot hash not found: %s", candidate)

    if not os.path.isdir(snapshots_dir):
        log.warning("no snapshots dir: %s", snapshots_dir)
        return None

    versions = [d for d in os.listdir(snapshots_dir)
                if os.path.isdir(os.path.join(snapshots_dir, d))]

    if not versions:
        log.warning("no snapshots found in %s", snapshots_dir)
        return None

    versions.sort()
    chosen = os.path.join(snapshots_dir, versions[0])
    log.info("cache probe: hit")
    log.info("cache probe: using snapshot %s", chosen)
    return chosen


LOCAL_MODEL_PATH = None

import runpod  # noqa: E402
import requests  # noqa: E402
import torch  # noqa: E402
import torchaudio  # noqa: E402

logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

SAMPLE_RATE = 48_000
MAX_AUDIO_BYTES = 100 * 1024 * 1024
MAX_BATCH_SIZE = 16
ALLOWED_OUTPUT_FORMATS = {"wav", "flac", "ogg", "mp3"}

model = None
processor = None


def prepare_model_access() -> Optional[str]:
    """resolve startup model access and prepare cache/offline flags."""
    global LOCAL_MODEL_PATH

    configure_cache_environment()
    LOCAL_MODEL_PATH = resolve_snapshot_path(MODEL_ID)
    hf_token = os.environ.get("HF_TOKEN")

    log.info("cache config: hf_home %s", HF_HOME_ROOT)
    log.info("cache config: hf_hub_cache %s", HF_CACHE_ROOT)
    log.info("cache config: torch_home %s", TORCH_CACHE_ROOT)
    log.info("hf_token: %s", "present" if hf_token else "missing")

    if LOCAL_MODEL_PATH:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        log.info("model source: /runpod-volume cache")
        log.info("cache reuse: enabled")
        return LOCAL_MODEL_PATH

    os.environ.pop("HF_HUB_OFFLINE", None)
    os.environ.pop("TRANSFORMERS_OFFLINE", None)
    log.info("model source: huggingface download")
    log.info("cache reuse: first-time download required")
    if hf_token:
        log.info("huggingface auth: enabled")
    else:
        log.info("huggingface auth: disabled, attempting anonymous download")

    return None


def load_model(model_source=_MODEL_SOURCE_UNSET):
    """load sam-audio model and processor onto gpu."""
    global model, processor, LOCAL_MODEL_PATH

    if model is not None:
        log.info("model already initialized")
        return

    if model_source is _MODEL_SOURCE_UNSET:
        model_source = prepare_model_access()

    from sam_audio import SAMAudio, SAMAudioProcessor

    if model_source:
        log.info("model load start: cached snapshot")
        model = SAMAudio.from_pretrained(model_source)
        processor = SAMAudioProcessor.from_pretrained(model_source)
    else:
        log.info("model load start: huggingface")
        hf_token = os.environ.get("HF_TOKEN")
        if hf_token:
            from huggingface_hub import login
            login(token=hf_token)
        model = SAMAudio.from_pretrained(MODEL_ID, cache_dir=HF_CACHE_ROOT)
        processor = SAMAudioProcessor.from_pretrained(MODEL_ID, cache_dir=HF_CACHE_ROOT)
        downloaded_snapshot = resolve_snapshot_path(MODEL_ID)
        if downloaded_snapshot:
            LOCAL_MODEL_PATH = downloaded_snapshot
            log.info("cache warmed: %s", downloaded_snapshot)

    model = model.eval()
    model = model.half().cuda()
    log.info("gpu move complete")
    log.info("model ready")


def bootstrap_worker():
    """warm the worker before registering it with runpod."""
    log.info("worker bootstrap start")
    model_source = prepare_model_access()
    load_model(model_source=model_source)
    log.info("worker ready")


def _validate_audio_url(url: str) -> None:
    """reject non-http schemes and private/loopback ips."""
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(f"Unsupported URL scheme: {parsed.scheme!r}")
    hostname = parsed.hostname
    if not hostname:
        raise ValueError("URL has no hostname")
    try:
        ip = ipaddress.ip_address(hostname)
        if ip.is_private or ip.is_loopback or ip.is_link_local:
            raise ValueError("URLs pointing to private/loopback addresses are not allowed")
    except ValueError as exc:
        if "not allowed" in str(exc):
            raise
        # hostname is a DNS name, not a raw IP — allow it through


def decode_audio(audio_url: Optional[str] = None, audio_base64: Optional[str] = None) -> torch.Tensor:
    """decode audio from url or base64 to a 48khz tensor."""
    if audio_url:
        _validate_audio_url(audio_url)
        response = requests.get(audio_url, timeout=60, stream=True)
        response.raise_for_status()
        chunks = []
        total = 0
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            total += len(chunk)
            if total > MAX_AUDIO_BYTES:
                response.close()
                raise ValueError(
                    f"Audio download exceeds {MAX_AUDIO_BYTES // (1024 * 1024)} MB limit"
                )
            chunks.append(chunk)
        audio_bytes = b"".join(chunks)
    elif audio_base64:
        if len(audio_base64) > MAX_AUDIO_BYTES * 4 // 3:
            raise ValueError("Base64 audio payload exceeds size limit")
        audio_bytes = base64.b64decode(audio_base64)
    else:
        raise ValueError("Either audio_url or audio_base64 must be provided")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
        tmp.write(audio_bytes)
        tmp.flush()
        wav, sr = torchaudio.load(tmp.name)

    if sr != SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)

    return wav


def encode_audio(wav: torch.Tensor, output_format: str = "wav") -> str:
    """encode audio tensor to base64."""
    if wav.dim() == 1:
        wav = wav.unsqueeze(0)
    
    buffer = io.BytesIO()
    torchaudio.save(buffer, wav, SAMPLE_RATE, format=output_format)
    buffer.seek(0)
    return base64.b64encode(buffer.read()).decode("utf-8")


def validate_input(job_input: dict) -> tuple[list, dict]:
    """validate request payload, return (items, options)."""
    items = job_input.get("items", [])

    if not items:
        raise ValueError("At least one item is required in 'items' array")

    if len(items) > MAX_BATCH_SIZE:
        raise ValueError(
            f"Batch size {len(items)} exceeds maximum of {MAX_BATCH_SIZE}"
        )

    for i, item in enumerate(items):
        if not item.get("audio_url") and not item.get("audio_base64"):
            raise ValueError(f"Item {i}: Either 'audio_url' or 'audio_base64' is required")
        if not item.get("description"):
            raise ValueError(f"Item {i}: 'description' is required")

    return_target = job_input.get("return_target", True)
    return_residual = job_input.get("return_residual", False)

    if not return_target and not return_residual:
        return_target = True

    output_format = job_input.get("output_format", "wav")
    if output_format not in ALLOWED_OUTPUT_FORMATS:
        raise ValueError(
            f"Unsupported output_format: {output_format!r}. "
            f"Allowed: {sorted(ALLOWED_OUTPUT_FORMATS)}"
        )

    options = {
        "predict_spans": job_input.get("predict_spans", False),
        "reranking_candidates": min(max(job_input.get("reranking_candidates", 1), 1), 16),
        "return_target": return_target,
        "return_residual": return_residual,
        "output_format": output_format,
    }

    return items, options


def handler(job):
    """runpod handler — batch audio separation."""
    job_id = job.get("id", "unknown")
    try:
        job_input = job.get("input", {})
        items, options = validate_input(job_input)

        load_model()

        log.info("[%s] separating %d items", job_id, len(items))
        audios = []
        descriptions = []
        
        for item in items:
            wav = decode_audio(
                audio_url=item.get("audio_url"),
                audio_base64=item.get("audio_base64")
            )
            audios.append(wav)
            descriptions.append(item["description"])
        
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
    
    except ValueError as e:
        log.warning("[%s] validation: %s", job_id, e)
        return {"error": str(e)}
    except Exception:
        log.exception("[%s] handler error", job_id)
        return {"error": "Internal processing error"}


def main():
    """bootstrap the worker, then hand control to runpod."""
    log.info("starting sam-audio handler...")
    try:
        bootstrap_worker()
    except Exception:
        log.exception("worker bootstrap failed")
        raise

    runpod.serverless.start({"handler": handler})


if __name__ == "__main__":
    main()
