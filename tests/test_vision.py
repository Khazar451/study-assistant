import io
from types import SimpleNamespace
from unittest.mock import MagicMock
from PIL import Image
import pytest

from src.ingestion.vision import VisionDescriber


def _create_test_image_bytes(width: int, height: int, mode: str = "RGB", color: str = "red") -> bytes:
    """Helper to generate in-memory image bytes with Pillow."""
    img = Image.new(mode, (width, height), color=color)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_init_vision_describer_missing_key(monkeypatch):
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.delenv("NVIDIA_EMBEDDER_KEY", raising=False)
    describer = VisionDescriber(api_key=None)
    assert describer.client is None
    assert describer.model == "meta/llama-3.2-11b-vision-instruct"


def test_init_vision_describer_with_key():
    describer = VisionDescriber(api_key="nvapi-test-key")
    assert describer.api_key == "nvapi-test-key"
    assert describer.client is not None


def test_is_meaningful_diagram_rejects_small_bytes():
    describer = VisionDescriber(api_key="nvapi-test-key", min_bytes=5000)
    assert describer.is_meaningful_diagram(b"small") is False


def test_is_meaningful_diagram_rejects_small_dimensions():
    describer = VisionDescriber(api_key="nvapi-test-key", min_width=150, min_height=150, min_bytes=10)
    small_bytes = _create_test_image_bytes(50, 50)
    assert describer.is_meaningful_diagram(small_bytes) is False


def test_is_meaningful_diagram_rejects_extreme_aspect_ratio():
    describer = VisionDescriber(api_key="nvapi-test-key", min_width=150, min_height=10, min_bytes=10)
    # 500x20 divider line (aspect ratio 25.0)
    line_bytes = _create_test_image_bytes(500, 20)
    assert describer.is_meaningful_diagram(line_bytes) is False


def test_is_meaningful_diagram_accepts_valid_diagram():
    describer = VisionDescriber(api_key="nvapi-test-key", min_width=150, min_height=150, min_bytes=10)
    diagram_bytes = _create_test_image_bytes(400, 300)
    assert describer.is_meaningful_diagram(diagram_bytes) is True


def test_prepare_image_b64_rgba_transparency_handling():
    describer = VisionDescriber(api_key="nvapi-test-key")
    rgba_bytes = _create_test_image_bytes(200, 200, mode="RGBA", color=(255, 0, 0, 128))
    b64_str = describer.prepare_image_b64(rgba_bytes)
    assert isinstance(b64_str, str)
    assert len(b64_str) > 50


def test_describe_diagram_cache():
    describer = VisionDescriber(api_key="nvapi-test-key", min_bytes=10, min_width=50, min_height=50)
    mock_client = MagicMock()
    mock_choice = SimpleNamespace(message=SimpleNamespace(content="Pyramid architecture chart."))
    mock_client.chat.completions.create.return_value = SimpleNamespace(choices=[mock_choice])
    describer.client = mock_client

    img_bytes = _create_test_image_bytes(200, 200)

    # First call invokes client
    res1 = describer.describe_diagram(img_bytes, context_hint="Page 5")
    assert res1 == "Pyramid architecture chart."
    assert mock_client.chat.completions.create.call_count == 1

    # Second call uses cache without client invocation
    res2 = describer.describe_diagram(img_bytes, context_hint="Page 5")
    assert res2 == "Pyramid architecture chart."
    assert mock_client.chat.completions.create.call_count == 1


def test_describe_diagram_offline_fallback(monkeypatch):
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.delenv("NVIDIA_EMBEDDER_KEY", raising=False)
    describer = VisionDescriber(api_key=None, min_bytes=10, min_width=50, min_height=50)

    img_bytes = _create_test_image_bytes(200, 200)
    res = describer.describe_diagram(img_bytes, context_hint="Neural Network")
    assert "Neural Network" in res
