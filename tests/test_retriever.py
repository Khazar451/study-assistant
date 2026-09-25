import uuid
from unittest.mock import MagicMock
import chromadb
import pytest
from langchain_core.documents import Document

from src.ingestion.indexer import ChromaIndexer
from src.retrieval.retriever import StudyRetriever


@pytest.fixture
def mock_embedder():
    """Mock embedder returning fixed query vector."""
    embedder = MagicMock()
    embedder.embed_query.return_value = [0.1, 0.2, 0.3, 0.4]
    return embedder


@pytest.fixture
def in_memory_indexer():
    """ChromaIndexer backed by an isolated in-memory collection."""
    client = chromadb.EphemeralClient()
    unique_name = f"test_retriever_{uuid.uuid4().hex[:8]}"
    indexer = ChromaIndexer(collection_name=unique_name, client=client)

    # Populate with sample study notes
    chunks = [
        Document(
            page_content="Newton's first law states that an object remains at rest unless acted upon by a net external force.",
            metadata={"source": "physics_lecture1.pdf", "page": 4, "subject": "physics", "id": "chunk_phys_1"},
        ),
        Document(
            page_content="The geometric interpretation of the derivative is the slope of the tangent line.",
            metadata={"source": "calculus_notes.pdf", "page": 12, "subject": "math", "id": "chunk_math_1"},
        ),
    ]
    # Add dummy embeddings matching test vectors
    chunks[0].metadata["embedding"] = [0.1, 0.2, 0.3, 0.4]
    chunks[1].metadata["embedding"] = [0.9, 0.8, 0.7, 0.6]

    indexer.add_chunks(chunks)
    return indexer


def test_retriever_empty_query_returns_empty_list(mock_embedder, in_memory_indexer):
    retriever = StudyRetriever(indexer=in_memory_indexer, embedder=mock_embedder)

    assert retriever.retrieve("") == []
    assert retriever.retrieve("   ") == []
    mock_embedder.embed_query.assert_not_called()


def test_retriever_basic_retrieval(mock_embedder, in_memory_indexer):
    retriever = StudyRetriever(indexer=in_memory_indexer, embedder=mock_embedder, top_k=2)

    results = retriever.retrieve("What is the law of inertia?")

    mock_embedder.embed_query.assert_called_once_with("What is the law of inertia?")
    assert len(results) == 2
    # The first result matches query embedding [0.1, 0.2, 0.3, 0.4]
    top_result = results[0]
    assert "Newton" in top_result["text"]
    assert top_result["metadata"]["source"] == "physics_lecture1.pdf"
    assert top_result["metadata"]["page"] == 4
    assert "similarity_score" in top_result
    assert top_result["similarity_score"] >= 0.99


def test_retriever_metadata_filtering(mock_embedder, in_memory_indexer):
    retriever = StudyRetriever(indexer=in_memory_indexer, embedder=mock_embedder)

    # Filter only math documents
    results = retriever.retrieve("sample search query", where={"subject": "math"})

    assert len(results) == 1
    assert "derivative" in results[0]["text"]
    assert results[0]["metadata"]["source"] == "calculus_notes.pdf"


def test_retriever_score_threshold(mock_embedder, in_memory_indexer):
    # Only return chunks with similarity >= 0.95
    retriever = StudyRetriever(indexer=in_memory_indexer, embedder=mock_embedder, score_threshold=0.95)

    results = retriever.retrieve("Inertia principle")

    # Only the first chunk matches closely enough
    assert len(results) == 1
    assert "Newton" in results[0]["text"]


def test_retriever_format_context():
    retriever = StudyRetriever(indexer=MagicMock(), embedder=MagicMock())

    chunks = [
        {
            "text": "Mitochondria is the powerhouse of the cell.",
            "metadata": {"source": "biology_ch3.pdf", "page": 7},
        },
        {
            "text": "Chloroplasts conduct photosynthesis in plant cells.",
            "metadata": {"source": "biology_ch3.pdf", "page": 9},
        },
    ]

    context = retriever.format_context(chunks)

    assert "[Document 1] (Source: biology_ch3.pdf | Page: 7)" in context
    assert "Mitochondria is the powerhouse of the cell." in context
    assert "[Document 2] (Source: biology_ch3.pdf | Page: 9)" in context
    assert "---" in context


def test_retriever_format_context_empty():
    retriever = StudyRetriever(indexer=MagicMock(), embedder=MagicMock())
    assert retriever.format_context([]) == ""
