"""
Test local VLM figure captioning with OpenAI-compatible endpoints (Ollama, vLLM, etc.)

This test verifies that the gpt4v module correctly supports local vision language models
in addition to Azure OpenAI. It tests configuration detection, endpoint connectivity,
and actual caption generation.

Configuration (environment variables):
    KH_VLM_ENDPOINT  - VLM API endpoint (default: http://localhost:11434/v1/chat/completions)
    KH_VLM_MODEL     - Vision model name (default: qwen2.5vl:7b)
    TEST_TIMEOUT     - Request timeout in seconds (default: 60)

Usage:
    # With local Ollama:
    KH_VLM_MODEL=qwen2.5vl:7b pytest test_local_vlm.py -v

    # Skip if no local VLM available:
    pytest test_local_vlm.py -v  # tests will be skipped if endpoint unreachable
"""
import base64
import os
import struct
import zlib

import pytest
import requests

# Test configuration with defaults
VLM_ENDPOINT = os.environ.get(
    "KH_VLM_ENDPOINT", "http://localhost:11434/v1/chat/completions"
)
VLM_MODEL = os.environ.get("KH_VLM_MODEL", "qwen2.5vl:7b")
TEST_TIMEOUT = int(os.environ.get("TEST_TIMEOUT", "60"))


def _vlm_endpoint_available() -> bool:
    """Check if the VLM endpoint is reachable."""
    try:
        base_url = VLM_ENDPOINT.rsplit("/v1/", 1)[0]
        resp = requests.get(f"{base_url}/api/tags", timeout=5)
        return resp.status_code == 200
    except Exception:
        return False


def _create_test_png(width: int, height: int, rgb: tuple) -> bytes:
    """Create a minimal valid PNG image for testing."""

    def chunk(chunk_type, data):
        return (
            struct.pack(">I", len(data))
            + chunk_type
            + data
            + struct.pack(">I", zlib.crc32(chunk_type + data) & 0xFFFFFFFF)
        )

    raw_data = b""
    for _ in range(height):
        raw_data += b"\x00"  # filter byte
        for _ in range(width):
            raw_data += bytes(rgb)

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw_data))
        + chunk(b"IEND", b"")
    )


# Skip all tests in this module if VLM endpoint is not available
pytestmark = pytest.mark.skipif(
    not _vlm_endpoint_available(),
    reason=f"VLM endpoint not available at {VLM_ENDPOINT}",
)


class TestLocalVLMConfig:
    """Test VLM configuration detection."""

    def test_local_mode_when_kh_vlm_model_set(self, monkeypatch):
        """Config should detect local mode when KH_VLM_MODEL is set."""
        monkeypatch.setenv("KH_VLM_MODEL", VLM_MODEL)
        monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)

        from kotaemon.loaders.utils.gpt4v import _get_vlm_config

        config = _get_vlm_config()

        assert config["model"] == VLM_MODEL
        assert "Authorization" in config["headers"]
        assert "api-key" not in config["headers"]

    def test_azure_mode_when_azure_key_set(self, monkeypatch):
        """Config should detect Azure mode when AZURE_OPENAI_API_KEY is set."""
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-azure-key")
        monkeypatch.delenv("KH_VLM_MODEL", raising=False)

        from kotaemon.loaders.utils.gpt4v import _get_vlm_config

        config = _get_vlm_config()

        assert config["model"] is None  # Azure embeds model in URL
        assert "api-key" in config["headers"]
        assert config["headers"]["api-key"] == "test-azure-key"

    def test_local_mode_takes_precedence(self, monkeypatch):
        """Local mode should take precedence when both configs are set."""
        monkeypatch.setenv("AZURE_OPENAI_API_KEY", "test-azure-key")
        monkeypatch.setenv("KH_VLM_MODEL", VLM_MODEL)

        from kotaemon.loaders.utils.gpt4v import _get_vlm_config

        config = _get_vlm_config()

        # Local mode should win
        assert config["model"] == VLM_MODEL
        assert "Authorization" in config["headers"]


class TestLocalVLMGeneration:
    """Test actual VLM caption generation with local endpoint."""

    def test_generate_caption_for_solid_color(self, monkeypatch):
        """VLM should correctly identify a solid color image."""
        monkeypatch.setenv("KH_VLM_MODEL", VLM_MODEL)
        monkeypatch.delenv("AZURE_OPENAI_API_KEY", raising=False)

        from kotaemon.loaders.utils.gpt4v import generate_gpt4v

        # Create a solid blue test image
        png_data = _create_test_png(50, 50, (30, 100, 200))
        image_b64 = base64.b64encode(png_data).decode()
        image_url = f"data:image/png;base64,{image_b64}"

        result = generate_gpt4v(
            endpoint=VLM_ENDPOINT,
            images=image_url,
            prompt="What color is this solid colored image? Reply with just the color name.",
            max_tokens=50,
        )

        assert result, "Expected non-empty response from VLM"
        assert isinstance(result, str)
        # The response should mention blue (or similar)
        assert any(
            color in result.lower() for color in ["blue", "azure", "cyan", "teal"]
        ), f"Expected blue-ish color, got: {result}"


class TestDoclingIntegration:
    """Test integration with DoclingReader."""

    def test_docling_reader_accepts_vlm_endpoint(self):
        """DoclingReader should accept VLM endpoint configuration."""
        from kotaemon.loaders import DoclingReader

        reader = DoclingReader()
        reader.vlm_endpoint = VLM_ENDPOINT

        assert reader.vlm_endpoint == VLM_ENDPOINT
