"""
Unit tests for the SAM-Audio RunPod handler.

These tests cover validation, URL safety, encoding, and error handling.
No GPU or SAM model required.
"""

import base64
import io
import os
import sys
from unittest.mock import MagicMock, patch

import pytest
import torch
import torchaudio

# mock sam_audio so importing handler does not require the real package.
sys.modules["sam_audio"] = MagicMock()
import handler  # noqa: E402


# ---------------------------------------------------------------------------
# validate_input
# ---------------------------------------------------------------------------

class TestValidateInput:
    def test_rejects_empty_items(self):
        with pytest.raises(ValueError, match="At least one item"):
            handler.validate_input({"items": []})

    def test_rejects_missing_items(self):
        with pytest.raises(ValueError, match="At least one item"):
            handler.validate_input({})

    def test_rejects_oversized_batch(self):
        items = [
            {"audio_url": "https://example.com/a.wav", "description": "voice"}
            for _ in range(handler.MAX_BATCH_SIZE + 1)
        ]
        with pytest.raises(ValueError, match="exceeds maximum"):
            handler.validate_input({"items": items})

    def test_rejects_missing_audio(self):
        with pytest.raises(ValueError, match="audio_url"):
            handler.validate_input({"items": [{"description": "voice"}]})

    def test_rejects_missing_description(self):
        with pytest.raises(ValueError, match="description"):
            handler.validate_input({
                "items": [{"audio_url": "https://example.com/a.wav"}]
            })

    def test_rejects_bad_output_format(self):
        with pytest.raises(ValueError, match="Unsupported output_format"):
            handler.validate_input({
                "items": [{"audio_url": "https://example.com/a.wav", "description": "voice"}],
                "output_format": "exe",
            })

    def test_valid_input(self):
        items, options = handler.validate_input({
            "items": [
                {"audio_url": "https://example.com/a.wav", "description": "voice"},
            ],
            "output_format": "flac",
            "return_residual": True,
        })
        assert len(items) == 1
        assert options["output_format"] == "flac"
        assert options["return_residual"] is True
        assert options["return_target"] is True

    def test_defaults_to_target_when_neither_requested(self):
        _, options = handler.validate_input({
            "items": [{"audio_url": "https://example.com/a.wav", "description": "v"}],
            "return_target": False,
            "return_residual": False,
        })
        assert options["return_target"] is True

    def test_reranking_candidates_clamped(self):
        _, options = handler.validate_input({
            "items": [{"audio_url": "https://example.com/a.wav", "description": "v"}],
            "reranking_candidates": 100,
        })
        assert options["reranking_candidates"] == 16

        _, options = handler.validate_input({
            "items": [{"audio_url": "https://example.com/a.wav", "description": "v"}],
            "reranking_candidates": -5,
        })
        assert options["reranking_candidates"] == 1


# ---------------------------------------------------------------------------
# bootstrap / startup
# ---------------------------------------------------------------------------

class TestBootstrapWorker:
    def test_configure_cache_environment_uses_runpod_volume(self):
        with patch.dict("os.environ", {}, clear=True):
            handler.configure_cache_environment()

        assert os.environ["HF_HOME"] == handler.HF_HOME_ROOT
        assert os.environ["HF_HUB_CACHE"] == handler.HF_CACHE_ROOT
        assert os.environ["TRANSFORMERS_CACHE"] == handler.HF_CACHE_ROOT
        assert os.environ["TORCH_HOME"] == handler.TORCH_CACHE_ROOT

    def test_bootstrap_uses_cache_without_token(self, caplog):
        caplog.set_level("INFO")

        with patch.dict("os.environ", {"HF_TOKEN": ""}, clear=False):
            with patch.object(handler, "LOCAL_MODEL_PATH", None):
                with patch.object(
                    handler,
                    "resolve_snapshot_path",
                    return_value="/runpod-volume/cache/snapshot",
                ) as resolve_snapshot_path:
                    with patch.object(handler, "load_model") as load_model:
                        handler.bootstrap_worker()

        resolve_snapshot_path.assert_called_once_with(handler.MODEL_ID)
        load_model.assert_called_once_with(model_source="/runpod-volume/cache/snapshot")
        assert "hf_token: missing" in caplog.text
        assert "model source: /runpod-volume cache" in caplog.text
        assert "worker ready" in caplog.text

    def test_bootstrap_allows_public_download_without_token(self, caplog):
        caplog.set_level("INFO")

        with patch.dict("os.environ", {"HF_TOKEN": ""}, clear=False):
            with patch.object(handler, "LOCAL_MODEL_PATH", None):
                with patch.object(handler, "resolve_snapshot_path", return_value=None):
                    with patch.object(handler, "load_model") as load_model:
                        handler.bootstrap_worker()

        load_model.assert_called_once_with(model_source=None)
        assert "model source: huggingface download" in caplog.text
        assert "huggingface auth: disabled, attempting anonymous download" in caplog.text
        assert "worker ready" in caplog.text


class TestLoadModel:
    def test_load_model_initializes_once(self):
        fake_model = MagicMock()
        fake_model.eval.return_value = fake_model
        fake_model.half.return_value = fake_model
        fake_model.cuda.return_value = fake_model

        fake_processor = MagicMock()
        fake_module = MagicMock()
        fake_module.SAMAudio.from_pretrained.return_value = fake_model
        fake_module.SAMAudioProcessor.from_pretrained.return_value = fake_processor

        with patch.dict(sys.modules, {"sam_audio": fake_module}):
            with patch.object(handler, "model", None):
                with patch.object(handler, "processor", None):
                    handler.load_model(model_source="/runpod-volume/cache/snapshot")
                    handler.load_model(model_source="/runpod-volume/cache/snapshot")

        assert fake_module.SAMAudio.from_pretrained.call_count == 1
        assert fake_module.SAMAudioProcessor.from_pretrained.call_count == 1

    def test_load_model_downloads_into_runpod_volume_cache(self):
        fake_model = MagicMock()
        fake_model.eval.return_value = fake_model
        fake_model.half.return_value = fake_model
        fake_model.cuda.return_value = fake_model

        fake_processor = MagicMock()
        fake_module = MagicMock()
        fake_module.SAMAudio.from_pretrained.return_value = fake_model
        fake_module.SAMAudioProcessor.from_pretrained.return_value = fake_processor

        with patch.dict(sys.modules, {"sam_audio": fake_module}):
            with patch.dict("os.environ", {"HF_TOKEN": ""}, clear=False):
                with patch.object(handler, "model", None):
                    with patch.object(handler, "processor", None):
                        with patch.object(
                            handler,
                            "resolve_snapshot_path",
                            return_value="/runpod-volume/huggingface-cache/hub/models--mrfakename--sam-audio-large/snapshots/mock",
                        ):
                            handler.load_model(model_source=None)

        fake_module.SAMAudio.from_pretrained.assert_called_once_with(
            handler.MODEL_ID,
            cache_dir=handler.HF_CACHE_ROOT,
        )
        fake_module.SAMAudioProcessor.from_pretrained.assert_called_once_with(
            handler.MODEL_ID,
            cache_dir=handler.HF_CACHE_ROOT,
        )


class TestMain:
    def test_main_bootstraps_before_start(self):
        with patch.object(handler, "bootstrap_worker") as bootstrap_worker:
            with patch.object(handler.runpod.serverless, "start") as start:
                handler.main()

        bootstrap_worker.assert_called_once_with()
        start.assert_called_once_with({"handler": handler.handler})


# ---------------------------------------------------------------------------
# _validate_audio_url
# ---------------------------------------------------------------------------

class TestValidateAudioUrl:
    def test_rejects_file_scheme(self):
        with pytest.raises(ValueError, match="Unsupported URL scheme"):
            handler._validate_audio_url("file:///etc/passwd")

    def test_rejects_ftp_scheme(self):
        with pytest.raises(ValueError, match="Unsupported URL scheme"):
            handler._validate_audio_url("ftp://example.com/audio.wav")

    def test_rejects_loopback_ip(self):
        with pytest.raises(ValueError, match="private/loopback"):
            handler._validate_audio_url("http://127.0.0.1/audio.wav")

    def test_rejects_private_ip(self):
        with pytest.raises(ValueError, match="private/loopback"):
            handler._validate_audio_url("http://10.0.0.1/audio.wav")

    def test_rejects_link_local_ip(self):
        with pytest.raises(ValueError, match="private/loopback"):
            handler._validate_audio_url("http://169.254.1.1/audio.wav")

    def test_allows_https(self):
        handler._validate_audio_url("https://example.com/audio.wav")

    def test_allows_http(self):
        handler._validate_audio_url("http://example.com/audio.wav")

    def test_rejects_no_hostname(self):
        with pytest.raises(ValueError, match="no hostname"):
            handler._validate_audio_url("http:///audio.wav")


# ---------------------------------------------------------------------------
# encode_audio / decode_audio roundtrip
# ---------------------------------------------------------------------------

_can_save_audio = True
try:
    _buf = io.BytesIO()
    torchaudio.save(_buf, torch.zeros(1, 100), 48000, format="wav")
except (ImportError, RuntimeError):
    _can_save_audio = False

_skip_no_codec = pytest.mark.skipif(
    not _can_save_audio,
    reason="torchaudio.save requires torchcodec (not installed in test env)",
)


class TestEncodeAudio:
    @_skip_no_codec
    def test_roundtrip_wav(self):
        wav = torch.randn(1, 4800)
        encoded = handler.encode_audio(wav, "wav")

        raw = base64.b64decode(encoded)
        buffer = io.BytesIO(raw)
        decoded, sr = torchaudio.load(buffer)

        assert sr == handler.SAMPLE_RATE
        assert decoded.shape == wav.shape

    @_skip_no_codec
    def test_1d_tensor_gets_unsqueezed(self):
        wav = torch.randn(4800)
        encoded = handler.encode_audio(wav, "wav")
        raw = base64.b64decode(encoded)
        buffer = io.BytesIO(raw)
        decoded, sr = torchaudio.load(buffer)
        assert decoded.shape == (1, 4800)


# ---------------------------------------------------------------------------
# decode_audio size limits
# ---------------------------------------------------------------------------

class TestDecodeAudioLimits:
    def test_base64_size_limit(self):
        # Create a base64 string that exceeds the limit
        oversized = "A" * (handler.MAX_AUDIO_BYTES * 4 // 3 + 1)
        with pytest.raises(ValueError, match="size limit"):
            handler.decode_audio(audio_base64=oversized)

    def test_url_download_size_limit(self):
        def fake_iter_content(chunk_size):
            # Yield chunks that exceed the limit
            chunk = b"\x00" * (1024 * 1024)
            for _ in range(110):  # 110 MB > 100 MB limit
                yield chunk

        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        mock_response.iter_content = fake_iter_content

        with patch("handler.requests.get", return_value=mock_response):
            with pytest.raises(ValueError, match="exceeds.*limit"):
                handler.decode_audio(audio_url="https://example.com/big.wav")

    def test_rejects_no_audio_source(self):
        with pytest.raises(ValueError, match="audio_url or audio_base64"):
            handler.decode_audio()


# ---------------------------------------------------------------------------
# handler error sanitization
# ---------------------------------------------------------------------------

class TestHandlerErrors:
    def test_validation_error_returned(self):
        result = handler.handler({"id": "test-1", "input": {"items": []}})
        assert "error" in result
        assert "At least one item" in result["error"]

    def test_internal_error_sanitized(self):
        """Errors from model inference should not leak tracebacks."""
        with patch.object(handler, "validate_input", side_effect=RuntimeError("secret GPU info")):
            result = handler.handler({"id": "test-2", "input": {}})
        assert result["error"] == "Internal processing error"
        assert "secret" not in result["error"]

    def test_bad_format_error_returned(self):
        result = handler.handler({
            "id": "test-3",
            "input": {
                "items": [{"audio_url": "https://example.com/a.wav", "description": "v"}],
                "output_format": "exe",
            },
        })
        assert "error" in result
        assert "Unsupported output_format" in result["error"]
