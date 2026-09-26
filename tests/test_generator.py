import os
from unittest.mock import MagicMock, patch
import pytest

from src.generation.generator import StudyGenerator


@pytest.fixture
def mock_openai_client():
    client = MagicMock()
    # Default completion response
    mock_choice = MagicMock()
    mock_choice.message.content = (
        "According to Newton's First Law, an object remains at rest unless acted upon by a net force. "
        "[Source: physics_lecture1.pdf, Page: 4]"
    )
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    client.chat.completions.create.return_value = mock_response
    return client


def test_missing_api_keys_raises_value_error(monkeypatch):
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.delenv("NEBIUS_API_KEY", raising=False)

    with pytest.raises(ValueError, match="No API key found"):
        StudyGenerator()


def test_nvidia_key_is_primary_when_both_set(monkeypatch):
    monkeypatch.setenv("NVIDIA_API_KEY", "nv-test-key-123")
    monkeypatch.setenv("NEBIUS_API_KEY", "neb-test-key-456")

    generator = StudyGenerator()
    assert generator.api_key == "nv-test-key-123"
    assert generator.provider == "nvidia"
    assert "nvidia.com" in generator.base_url
    assert "llama" in generator.model.lower()


def test_nebius_key_fallback_when_nvidia_absent(monkeypatch):
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.setenv("NEBIUS_API_KEY", "neb-test-key-789")

    generator = StudyGenerator()
    assert generator.api_key == "neb-test-key-789"
    assert generator.provider == "nebius"
    assert "nebius.ai" in generator.base_url


def test_explicit_api_key_override(monkeypatch):
    monkeypatch.setenv("NVIDIA_API_KEY", "nv-env-key")

    generator = StudyGenerator(
        api_key="custom-key",
        base_url="https://api.studio.nebius.ai/v1",
        model="custom-model",
    )
    assert generator.api_key == "custom-key"
    assert generator.provider == "nebius"
    assert generator.model == "custom-model"


def test_prepare_context_with_chunk_list():
    generator = StudyGenerator(api_key="dummy-key", client=MagicMock())

    chunks = [
        {
            "text": "Photosynthesis produces glucose and oxygen.",
            "metadata": {"source": "bio_ch2.pdf", "page": 15},
        },
        {
            "text": "Cellular respiration consumes oxygen.",
            "metadata": {"source": "bio_ch2.pdf", "page": 18},
        },
    ]

    context = generator.prepare_context(chunks)
    assert "[Document 1] (Source: bio_ch2.pdf | Page: 15)" in context
    assert "Photosynthesis produces glucose and oxygen." in context
    assert "[Document 2] (Source: bio_ch2.pdf | Page: 18)" in context
    assert "---" in context


def test_prepare_context_passthrough_string():
    generator = StudyGenerator(api_key="dummy-key", client=MagicMock())
    raw_str = "[Document 1] Some existing context"
    assert generator.prepare_context(raw_str) == raw_str
    assert generator.prepare_context("") == ""
    assert generator.prepare_context([]) == ""


def test_generate_answer(mock_openai_client):
    generator = StudyGenerator(api_key="dummy-key", client=mock_openai_client)

    context = (
        "[Document 1] (Source: physics_lecture1.pdf | Page: 4)\n"
        "An object at rest stays at rest unless acted upon by a net force."
    )

    result = generator.generate(
        query="What is inertia?",
        context=context,
    )

    assert "Newton's First Law" in result["answer"]
    assert result["provider"] == "nvidia"
    assert "physics_lecture1.pdf" in result["sources"]

    # Verify messages sent to OpenAI client
    mock_openai_client.chat.completions.create.assert_called_once()
    call_kwargs = mock_openai_client.chat.completions.create.call_args.kwargs
    messages = call_kwargs["messages"]
    assert messages[0]["role"] == "system"
    assert "Grounding: Rely ONLY on the facts" in messages[0]["content"]
    assert "Student Question: What is inertia?" in messages[1]["content"]
    assert "Context Excerpts:" in messages[1]["content"]


def test_generate_empty_query():
    generator = StudyGenerator(api_key="dummy-key", client=MagicMock())
    res = generator.generate(query="", context="some context")
    assert "Please provide a valid question" in res["answer"]


def test_generate_stream(mock_openai_client):
    # Mock streaming response
    def create_chunk(content):
        chunk = MagicMock()
        chunk.choices = [MagicMock()]
        chunk.choices[0].delta.content = content
        return chunk

    mock_openai_client.chat.completions.create.return_value = iter([
        create_chunk("Newton's "),
        create_chunk("First "),
        create_chunk("Law."),
    ])

    generator = StudyGenerator(api_key="dummy-key", client=mock_openai_client)
    tokens = list(generator.generate_stream(query="Explain inertia", context="some context"))

    assert "".join(tokens) == "Newton's First Law."
    call_kwargs = mock_openai_client.chat.completions.create.call_args.kwargs
    assert call_kwargs["stream"] is True


def test_generate_stream_empty_query():
    generator = StudyGenerator(api_key="dummy-key", client=MagicMock())
    tokens = list(generator.generate_stream(query="   ", context="some context"))
    assert "".join(tokens) == "Please provide a valid question."
