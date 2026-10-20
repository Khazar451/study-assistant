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


def test_nebius_key_is_primary_when_both_set(monkeypatch):
    monkeypatch.setenv("NVIDIA_API_KEY", "nv-test-key-123")
    monkeypatch.setenv("NEBIUS_API_KEY", "neb-test-key-456")
    monkeypatch.delenv("NEBIUS_BASE_URL", raising=False)
    monkeypatch.delenv("NEBIUS_MODEL", raising=False)
    monkeypatch.setenv("LLM_MODEL", "meta/llama-3.2-11b-vision-instruct")

    generator = StudyGenerator()
    assert generator.api_key == "neb-test-key-456"
    assert generator.provider == "nebius"
    assert generator.provider_label == "Nebius Token Factory"
    assert "tokenfactory.nebius.com" in generator.base_url
    assert generator.model.startswith("nvidia/") and "nemotron" in generator.model.lower()


def test_nvidia_key_fallback_when_nebius_absent(monkeypatch):
    monkeypatch.delenv("NEBIUS_API_KEY", raising=False)
    monkeypatch.setenv("NVIDIA_API_KEY", "nv-test-key-789")
    monkeypatch.delenv("NVIDIA_BASE_URL", raising=False)

    generator = StudyGenerator()
    assert generator.api_key == "nv-test-key-789"
    assert generator.provider == "nvidia"
    assert generator.provider_label == "NVIDIA NIM"
    assert "nvidia.com" in generator.base_url


def test_generate_strips_inline_reasoning(mock_openai_client):
    mock_openai_client.chat.completions.create.return_value.choices[0].message.content = (
        "<think>The context mentions inertia on page 4.</think>\nInertia is resistance to motion changes."
    )
    generator = StudyGenerator(api_key="dummy-key", client=mock_openai_client)
    result = generator.generate(query="What is inertia?", context="some context")
    assert result["answer"] == "Inertia is resistance to motion changes."


def test_generate_stream_filters_reasoning_split_across_tokens(mock_openai_client):
    def create_chunk(content):
        chunk = MagicMock()
        chunk.choices = [MagicMock()]
        chunk.choices[0].delta.content = content
        return chunk

    mock_openai_client.chat.completions.create.return_value = iter([
        create_chunk("<thi"),
        create_chunk("nk>planning the answer</th"),
        create_chunk("ink>\nNewton's "),
        create_chunk("First Law."),
    ])
    generator = StudyGenerator(api_key="dummy-key", client=mock_openai_client)
    tokens = list(generator.generate_stream(query="Explain inertia", context="some context"))
    assert "".join(tokens) == "Newton's First Law."


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
    assert "Use only facts in the context excerpts" in messages[0]["content"]
    assert "Check-in" in messages[0]["content"]
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


def _nebius_generator(monkeypatch, client):
    monkeypatch.setenv("NEBIUS_API_KEY", "neb-test-key")
    monkeypatch.delenv("NEBIUS_MODEL", raising=False)
    monkeypatch.delenv("NEBIUS_REASONING_MODEL", raising=False)
    monkeypatch.delenv("ENABLE_REASONING_ROUTER", raising=False)
    return StudyGenerator(client=client)


def test_hard_math_question_routes_to_ultra(monkeypatch, mock_openai_client):
    generator = _nebius_generator(monkeypatch, mock_openai_client)
    result = generator.generate("Derive the integral of x*e^x", context="[Document 1] x")

    used = mock_openai_client.chat.completions.create.call_args.kwargs
    assert "ultra" in used["model"].lower()
    assert used["max_tokens"] >= 4000
    assert "ultra" in result["model"].lower()


def test_simple_question_stays_on_super(monkeypatch, mock_openai_client):
    generator = _nebius_generator(monkeypatch, mock_openai_client)
    result = generator.generate("What is Kepler's first law?", context="[Document 1] x")

    assert "super" in mock_openai_client.chat.completions.create.call_args.kwargs["model"].lower()
    assert "super" in result["model"].lower()


def test_ultra_failure_falls_back_to_super(monkeypatch, mock_openai_client):
    good = mock_openai_client.chat.completions.create.return_value
    calls = []

    def fake_create(**kwargs):
        calls.append(kwargs["model"])
        if "ultra" in kwargs["model"].lower():
            raise RuntimeError("model not found")
        return good

    mock_openai_client.chat.completions.create.side_effect = fake_create
    generator = _nebius_generator(monkeypatch, mock_openai_client)
    result = generator.generate("Prove that sqrt(2) is irrational", context="[Document 1] x")

    assert "ultra" in calls[0].lower() and "super" in calls[1].lower()
    assert "super" in result["model"].lower()


def test_router_can_be_disabled(monkeypatch, mock_openai_client):
    monkeypatch.setenv("ENABLE_REASONING_ROUTER", "false")
    monkeypatch.setenv("NEBIUS_API_KEY", "neb-test-key")
    generator = StudyGenerator(client=mock_openai_client)
    generator.generate("Derive the integral of x*e^x", context="[Document 1] x")

    assert "super" in mock_openai_client.chat.completions.create.call_args.kwargs["model"].lower()
