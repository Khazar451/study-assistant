from unittest.mock import MagicMock, patch
import pytest

from src.retrieval.tavily_search import TavilySearchClient
from src.assistant import StudyAssistant


def test_tavily_client_init_no_key():
    """Ensure client initializes gracefully without API key."""
    with patch.dict("os.environ", {}, clear=True):
        client = TavilySearchClient(api_key=None, mock_mode=False)
        assert client.api_key is None
        assert client.is_available is False


def test_tavily_client_init_with_key():
    """Ensure client initializes with explicit API key."""
    with patch("src.retrieval.tavily_search.TavilyClient") as mock_tavily_cls:
        client = TavilySearchClient(api_key="tvly-test-123", mock_mode=False)
        assert client.api_key == "tvly-test-123"
        assert client.is_available is True
        mock_tavily_cls.assert_called_once_with(api_key="tvly-test-123")


def test_tavily_search_empty_query():
    """Ensure empty or whitespace queries return an empty list immediately."""
    client = TavilySearchClient(mock_mode=True)
    assert client.search("") == []
    assert client.search("   ") == []


def test_tavily_search_mock_mode():
    """Ensure mock mode returns structured chunks with correct metadata."""
    client = TavilySearchClient(mock_mode=True)
    chunks = client.search("Kepler laws of planetary motion", max_results=2)
    assert len(chunks) == 2

    first = chunks[0]
    assert "id" in first
    assert "text" in first
    assert "metadata" in first
    assert first["metadata"]["source_type"] == "web"
    assert "wikipedia.org" in first["metadata"]["source"]
    assert first["similarity_score"] > 0.0


def test_tavily_search_live_mocked_api_response():
    """Verify live client correctly maps Tavily API response dictionary to standard chunks."""
    with patch("src.retrieval.tavily_search.TavilyClient") as mock_cls:
        mock_instance = MagicMock()
        mock_cls.return_value = mock_instance
        mock_instance.search.return_value = {
            "results": [
                {
                    "url": "https://arxiv.org/abs/2305.18290",
                    "title": "Direct Preference Optimization",
                    "content": "We propose Direct Preference Optimization (DPO)...",
                    "score": 0.91,
                }
            ]
        }

        client = TavilySearchClient(api_key="tvly-key", mock_mode=False)
        chunks = client.search("DPO algorithm", max_results=1)

        assert len(chunks) == 1
        assert chunks[0]["metadata"]["source"] == "https://arxiv.org/abs/2305.18290"
        assert chunks[0]["metadata"]["domain"] == "arxiv.org"
        assert chunks[0]["metadata"]["source_type"] == "web"
        assert "Direct Preference Optimization" in chunks[0]["text"]
        mock_instance.search.assert_called_once_with(
            query="DPO algorithm",
            search_depth="basic",
            max_results=1,
            include_answer=False,
        )


def test_tavily_search_api_exception_fallback():
    """Verify client handles API errors gracefully by returning empty results."""
    with patch("src.retrieval.tavily_search.TavilyClient") as mock_cls:
        mock_instance = MagicMock()
        mock_cls.return_value = mock_instance
        mock_instance.search.side_effect = RuntimeError("Network timeout")

        client = TavilySearchClient(api_key="tvly-key", mock_mode=False)
        chunks = client.search("test query")
        assert chunks == []


def test_assistant_search_with_web_augment():
    """Verify StudyAssistant combines local vector chunks and Tavily web chunks."""
    mock_pipeline = MagicMock()
    mock_retriever = MagicMock()
    mock_reranker = MagicMock()
    mock_generator = MagicMock()
    mock_tavily = MagicMock()

    mock_retriever.retrieve.return_value = [
        {"id": "local_1", "text": "Local lecture note", "metadata": {"source": "lecture.pdf", "page": 1}}
    ]
    mock_tavily.is_available = True
    mock_tavily.search.return_value = [
        {"id": "web_1", "text": "Web article", "metadata": {"source": "https://nature.com", "source_type": "web"}}
    ]
    mock_reranker.rerank.side_effect = lambda query, chunks, top_n, min_score=None: chunks[:top_n]

    assistant = StudyAssistant(
        pipeline=mock_pipeline,
        retriever=mock_retriever,
        reranker=mock_reranker,
        generator=mock_generator,
        tavily=mock_tavily,
        persist_directory="./test_db",
    )

    results = assistant.search("Quantum gravity", top_n=5, use_web=True)
    assert len(results) == 2
    mock_tavily.search.assert_called_once_with(query="Quantum gravity", max_results=5)
    mock_reranker.rerank.assert_called_once()


def test_assistant_ask_fallback_on_empty_database():
    """Ensure StudyAssistant falls back to Tavily web search when vector DB is empty."""
    mock_pipeline = MagicMock()
    mock_retriever = MagicMock()
    mock_reranker = MagicMock()
    mock_generator = MagicMock()
    mock_tavily = MagicMock()
    mock_indexer = MagicMock()

    mock_indexer.count.return_value = 0
    mock_pipeline.indexer = mock_indexer

    mock_tavily.is_available = True
    mock_tavily.search.return_value = [
        {"id": "web_1", "text": "Web content on black holes", "metadata": {"source": "https://nasa.gov", "source_type": "web"}}
    ]
    mock_retriever.format_context.return_value = "[Document 1] (Web Source: https://nasa.gov)\nWeb content on black holes"
    mock_generator.generate.return_value = {
        "answer": "Black holes are regions of spacetime [Web Source: https://nasa.gov].",
        "sources": ["https://nasa.gov"],
        "model": "nvidia/model",
        "provider": "nvidia",
    }

    assistant = StudyAssistant(
        pipeline=mock_pipeline,
        retriever=mock_retriever,
        reranker=mock_reranker,
        generator=mock_generator,
        tavily=mock_tavily,
        persist_directory="./test_db",
    )
    assistant.indexer = mock_indexer

    res = assistant.ask("What is a black hole?", fallback_to_web=True)
    assert "Black holes are regions" in res["answer"]
    assert "https://nasa.gov" in res["sources"]
    mock_tavily.search.assert_called_once_with(query="What is a black hole?", max_results=5)
    mock_generator.generate.assert_called_once()
