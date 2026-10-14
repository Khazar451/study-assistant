from types import SimpleNamespace
from unittest.mock import MagicMock
import pytest

from src.retrieval.query_augmenter import QueryAugmenter


def test_init_missing_api_key_raises_error(monkeypatch):
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.delenv("NEBIUS_API_KEY", raising=False)
    with pytest.raises(ValueError, match="No API key found"):
        QueryAugmenter(api_key=None)


def test_init_prefers_nebius_fast_model(monkeypatch):
    monkeypatch.setenv("NVIDIA_API_KEY", "nv-key")
    monkeypatch.setenv("NEBIUS_API_KEY", "neb-key")
    monkeypatch.delenv("NEBIUS_BASE_URL", raising=False)
    monkeypatch.delenv("NEBIUS_FAST_MODEL", raising=False)

    augmenter = QueryAugmenter()
    assert augmenter.api_key == "neb-key"
    assert "tokenfactory.nebius.com" in augmenter.base_url
    assert augmenter.model == "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B"
    assert augmenter.extra_body == {"chat_template_kwargs": {"enable_thinking": False}}


def test_explicit_key_sends_no_extra_body():
    augmenter = QueryAugmenter(api_key="test-api-key")
    assert augmenter.extra_body is None


def test_init_with_explicit_params():
    augmenter = QueryAugmenter(
        api_key="test-api-key",
        base_url="https://test.endpoint.com/v1",
        model="custom/llama-3-model",
        temperature=0.3,
    )
    assert augmenter.api_key == "test-api-key"
    assert augmenter.base_url == "https://test.endpoint.com/v1"
    assert augmenter.model == "custom/llama-3-model"
    assert augmenter.temperature == 0.3


def test_rewrite_query_success():
    augmenter = QueryAugmenter(api_key="test-api-key")

    mock_client = MagicMock()
    mock_choice = SimpleNamespace(message=SimpleNamespace(content="Newton's universal gravitation and orbital mechanics"))
    mock_response = SimpleNamespace(choices=[mock_choice])
    mock_client.chat.completions.create.return_value = mock_response

    augmenter.client = mock_client

    result = augmenter.rewrite_query("why planets stay in orbit")

    assert result == "Newton's universal gravitation and orbital mechanics"
    mock_client.chat.completions.create.assert_called_once()
    call_kwargs = mock_client.chat.completions.create.call_args.kwargs
    assert call_kwargs["model"] == augmenter.model
    assert "why planets stay in orbit" in call_kwargs["messages"][1]["content"]


def test_rewrite_query_empty_returns_empty():
    augmenter = QueryAugmenter(api_key="test-api-key")
    assert augmenter.rewrite_query("") == ""
    assert augmenter.rewrite_query("   ") == ""


def test_rewrite_query_fallback_on_exception():
    augmenter = QueryAugmenter(api_key="test-api-key")
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = RuntimeError("API connection timeout")
    augmenter.client = mock_client

    # Should gracefully return original query
    assert augmenter.rewrite_query("sample query") == "sample query"


def test_expand_query_success():
    augmenter = QueryAugmenter(api_key="test-api-key")

    mock_client = MagicMock()
    llm_output = (
        "1. First law of thermodynamics definition\n"
        "2. Conservation of energy in closed systems\n"
        "3. Internal energy heat and thermodynamic work\n"
    )
    mock_choice = SimpleNamespace(message=SimpleNamespace(content=llm_output))
    mock_response = SimpleNamespace(choices=[mock_choice])
    mock_client.chat.completions.create.return_value = mock_response

    augmenter.client = mock_client

    results = augmenter.expand_query("thermodynamics first law", num_queries=3)

    # First item is the original query
    assert results[0] == "thermodynamics first law"
    # Following items are parsed and stripped of numbering
    assert "First law of thermodynamics definition" in results
    assert "Conservation of energy in closed systems" in results
    assert "Internal energy heat and thermodynamic work" in results
    assert len(results) == 4


def test_expand_query_empty_returns_empty():
    augmenter = QueryAugmenter(api_key="test-api-key")
    assert augmenter.expand_query("") == []
    assert augmenter.expand_query("  \t ") == []


def test_expand_query_fallback_on_exception():
    augmenter = QueryAugmenter(api_key="test-api-key")
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = Exception("Rate limit exceeded")
    augmenter.client = mock_client

    assert augmenter.expand_query("quantum mechanics") == ["quantum mechanics"]


def test_generate_hyde_query_success():
    augmenter = QueryAugmenter(api_key="test-api-key")

    mock_client = MagicMock()
    hypothetical_passage = (
        "Mitochondria are double-membraned cell organelles responsible for generating most of the chemical energy "
        "needed to power the cell's biochemical reactions, stored in adenosine triphosphate (ATP)."
    )
    mock_choice = SimpleNamespace(message=SimpleNamespace(content=hypothetical_passage))
    mock_response = SimpleNamespace(choices=[mock_choice])
    mock_client.chat.completions.create.return_value = mock_response

    augmenter.client = mock_client

    result = augmenter.generate_hyde_query("What is the function of mitochondria?")

    assert "adenosine triphosphate" in result
    assert result == hypothetical_passage


def test_generate_hyde_query_empty():
    augmenter = QueryAugmenter(api_key="test-api-key")
    assert augmenter.generate_hyde_query("") == ""


def test_augment_modes():
    augmenter = QueryAugmenter(api_key="test-api-key")

    # Mock expand
    augmenter.expand_query = MagicMock(return_value=["q1", "q2"])
    res_expand = augmenter.augment("test", mode="expand", num_queries=2)
    assert res_expand == ["q1", "q2"]
    augmenter.expand_query.assert_called_once_with("test", num_queries=2)

    # Mock rewrite
    augmenter.rewrite_query = MagicMock(return_value="rewritten query")
    res_rewrite = augmenter.augment("test", mode="rewrite")
    assert res_rewrite == ["rewritten query"]

    # Mock hyde
    augmenter.generate_hyde_query = MagicMock(return_value="hypothetical answer")
    res_hyde = augmenter.augment("test", mode="hyde")
    assert res_hyde == ["hypothetical answer"]

    # Passthrough
    res_pass = augmenter.augment("test", mode="passthrough")
    assert res_pass == ["test"]

    # Unsupported mode
    with pytest.raises(ValueError, match="Unsupported augmentation mode"):
        augmenter.augment("test", mode="unknown_mode")


def test_condense_conversational_query_no_history_returns_original():
    augmenter = QueryAugmenter(api_key="test-api-key")
    res = augmenter.condense_conversational_query("How does it work?", history=[])
    assert res == "How does it work?"


def test_condense_conversational_query_with_history_resolves_context():
    augmenter = QueryAugmenter(api_key="test-api-key")
    mock_client = MagicMock()
    mock_choice = SimpleNamespace(message=SimpleNamespace(content="How does Kepler's First Law orbit work?"))
    mock_client.chat.completions.create.return_value = SimpleNamespace(choices=[mock_choice])
    augmenter.client = mock_client

    history = [
        {"role": "user", "content": "What is Kepler's First Law?"},
        {"role": "assistant", "content": "Kepler's First Law states that planets orbit the sun in ellipses with the sun at one focus."},
    ]
    res = augmenter.condense_conversational_query("How does it work?", history=history)
    assert res == "How does Kepler's First Law orbit work?"
    mock_client.chat.completions.create.assert_called_once()


def test_condense_conversational_query_fallback_on_error():
    augmenter = QueryAugmenter(api_key="test-api-key")
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = RuntimeError("API unavailable")
    augmenter.client = mock_client

    history = [{"role": "user", "content": "Calculus question"}]
    res = augmenter.condense_conversational_query("Explain step 2", history=history)
    assert res == "Explain step 2"

