import os
import pytest
from langchain_core.documents import Document
from langchain_community.embeddings import FakeEmbeddings

from src.ingestion.indexer import build_vector_store

# Generates 32 dimensioned mock vectors
@pytest.fixture
def mock_embeddings():
    return FakeEmbeddings(size = 32)

# Generates sample docs of study
@pytest.fixture
def sample_documents():
    return [
        Document(
            page_content = "Mathematical methods of calculation derivatives and integration",
            metadata = {"lecture": "mathematics", "topic": "analysis"}
        ),
        Document(
            page_content = "Law's of motion in Physics",
            metadata = {"lecture": "physics", "topic": "dynamic"}
        )
    ]

# Varifies that there is a in-memory ChromaDB and you can make search in it
def test_build_vector_store_in_memory(mock_embeddings, sample_documents):
    vector_store = build_vector_store(sample_documents, mock_embeddings)

    results = vector_store.similarity_search("derivative", k = 1)

    assert len(results) == 1
    assert "derivative" in results[0].page_content
    assert results[0].metadata["lecture"] == "mathematics"

# Varifies when it has given persist_directory it varifies that the database files are written on the disk
def test_build_vector_store_persistence(mock_embeddings, sample_documents, tmp_path):
    persist_dir = str(tmp_path / "test_chroma_store")

    vector_store = build_vector_store(
        docs = sample_documents,
        embeddings = mock_embeddings,
        persist_directory = persist_dir
    )

    assert os.path.exists(persist_dir)
    assert len(os.listdir(persist_dir)) > 0

# Varifies when it has given a filter it only returns the document that has matched
def test_build_vector_store_metadata_filtering(mock_embeddings, sample_documents):
    vector_store = build_vector_store(sample_documents, mock_embeddings)

    results = vector_store.similarity_search(
        "dynamic",
        k = 2,
        filter = {"lecture": "physics"}
    )

    assert len(results) == 1
    assert results[0].metadata["lecture"] == "physics"

# Varifies when it has given an empty doc list ValueError thrown
def test_build_vector_store_empty_docs(mock_embeddings):
    with pytest.raises(ValueError, match = "Document not found to indexed."):
        build_vector_store([], mock_embeddings)