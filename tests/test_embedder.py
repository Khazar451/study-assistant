import os
import pytest
from unittest.mock import MagicMock
from types import SimpleNamespace

from src.ingestion.embedder import NvidiaEmbedder


def test_init_missing_api_key_raises_error(monkeypatch):
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.delenv("NVIDIA_EMBEDDER_KEY", raising=False)
    with pytest.raises(ValueError, match="API key not set"):
        NvidiaEmbedder(api_key=None)


def test_init_with_explicit_api_key():
    embedder = NvidiaEmbedder(api_key="nvapi-test-key")
    assert embedder.api_key == "nvapi-test-key"
    assert embedder.model == "nvidia/llama-nemotron-embed-vl-1b-v2"
    assert embedder.base_url == "https://integrate.api.nvidia.com/v1"


def test_embed_text_calls_api_with_passage():
    embedder = NvidiaEmbedder(api_key="nvapi-test-key")

    mock_client = MagicMock()
    mock_item = SimpleNamespace(embedding=[0.1, 0.2, 0.3])
    mock_response = SimpleNamespace(data=[mock_item])
    mock_client.embeddings.create.return_value = mock_response

    embedder.client = mock_client

    result = embedder.embed_text("Sample lecture text", input_type="passage")

    assert result == [0.1, 0.2, 0.3]
    mock_client.embeddings.create.assert_called_once_with(
        model="nvidia/llama-nemotron-embed-vl-1b-v2",
        input=["Sample lecture text"],
        encoding_format="float",
        extra_body={"input_type": "passage", "truncate": "NONE"},
    )


def test_embed_query_sets_query_input_type():
    embedder = NvidiaEmbedder(api_key="nvapi-test-key")

    mock_client = MagicMock()
    mock_item = SimpleNamespace(embedding=[0.4, 0.5, 0.6])
    mock_response = SimpleNamespace(data=[mock_item])
    mock_client.embeddings.create.return_value = mock_response

    embedder.client = mock_client

    result = embedder.embed_query("What is Newton's second law?")

    assert result == [0.4, 0.5, 0.6]
    mock_client.embeddings.create.assert_called_once_with(
        model="nvidia/llama-nemotron-embed-vl-1b-v2",
        input=["What is Newton's second law?"],
        encoding_format="float",
        extra_body={"input_type": "query", "truncate": "NONE"},
    )


def test_embed_batch_handles_multiple_texts():
    embedder = NvidiaEmbedder(api_key="nvapi-test-key")

    mock_client = MagicMock()
    mock_items = [
        SimpleNamespace(embedding=[0.1, 0.2]),
        SimpleNamespace(embedding=[0.3, 0.4]),
    ]
    mock_response = SimpleNamespace(data=mock_items)
    mock_client.embeddings.create.return_value = mock_response

    embedder.client = mock_client

    texts = ["Text 1", "Text 2"]
    results = embedder.embed_batch(texts, batch_size=2)

    assert results == [[0.1, 0.2], [0.3, 0.4]]
    mock_client.embeddings.create.assert_called_once_with(
        model="nvidia/llama-nemotron-embed-vl-1b-v2",
        input=["Text 1", "Text 2"],
        encoding_format="float",
        extra_body={"input_type": "passage", "truncate": "NONE"},
    )


def test_embed_chunks_supports_dicts_and_objects():
    embedder = NvidiaEmbedder(api_key="nvapi-test-key")

    mock_client = MagicMock()
    mock_items = [
        SimpleNamespace(embedding=[0.11, 0.22]),
        SimpleNamespace(embedding=[0.33, 0.44]),
    ]
    mock_response = SimpleNamespace(data=mock_items)
    mock_client.embeddings.create.return_value = mock_response

    embedder.client = mock_client

    # Test with dict
    chunk_dict = {"text": "Cell biology", "source": "bio.pdf"}
    # Test with object
    chunk_obj = SimpleNamespace(text="Mitochondria", source="bio.pdf", embedding=None)

    processed = embedder.embed_chunks([chunk_dict, chunk_obj])

    assert processed[0]["embedding"] == [0.11, 0.22]
    assert processed[1].embedding == [0.33, 0.44]
