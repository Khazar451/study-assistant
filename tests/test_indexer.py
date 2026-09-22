import os
import pytest
from langchain_core.documents import Document
from src.ingestion.indexer import ChromaIndexer


@pytest.fixture
def sample_embedding():
    # 4-dimensional mock vector for testing
    return [0.1, 0.2, 0.3, 0.4]


@pytest.fixture
def sample_chunks(sample_embedding):
    return [
        Document(
            page_content="Calculus derivatives and integral calculation methods.",
            metadata={"subject": "mathematics", "embedding": sample_embedding, "id": "doc_math"}
        ),
        Document(
            page_content="Physics dynamics and Newton's laws of motion.",
            metadata={"subject": "physics", "embedding": [0.9, 0.8, 0.7, 0.6], "id": "doc_phys"}
        )
    ]


def test_chroma_indexer_add_and_query(sample_chunks, sample_embedding):
    """Test in-memory indexing and similarity search retrieval."""
    indexer = ChromaIndexer(collection_name="test_mem", persist_directory=None)
    
    # Verify chunk addition
    added_count = indexer.add_chunks(sample_chunks)
    assert added_count == 2
    assert indexer.count() == 2

    # Verify querying
    results = indexer.query(query_embedding=sample_embedding, top_k=1)
    assert len(results) == 1
    assert results[0]["id"] == "doc_math"
    assert results[0]["metadata"]["subject"] == "mathematics"
    assert "Calculus" in results[0]["text"]


def test_chroma_indexer_metadata_filtering(sample_chunks):
    """Test retrieval filtering using the where parameter."""
    indexer = ChromaIndexer(collection_name="test_filter", persist_directory=None)
    indexer.add_chunks(sample_chunks)

    # Filter strictly for physics subject
    results = indexer.query(
        query_embedding=[0.1, 0.2, 0.3, 0.4],
        top_k=2,
        where={"subject": "physics"}
    )
    assert len(results) == 1
    assert results[0]["metadata"]["subject"] == "physics"
    assert results[0]["id"] == "doc_phys"


def test_chroma_indexer_persistence(sample_chunks, tmp_path):
    """Test disk persistence and directory creation."""
    persist_dir = str(tmp_path / "chroma_db_test")
    
    indexer = ChromaIndexer(
        collection_name="test_persist",
        persist_directory=persist_dir
    )
    indexer.add_chunks(sample_chunks)

    assert os.path.exists(persist_dir)
    assert len(os.listdir(persist_dir)) > 0


def test_chroma_indexer_delete_and_reset(sample_chunks):
    """Test deleting specific chunks and resetting the collection."""
    indexer = ChromaIndexer(collection_name="test_mgmt", persist_directory=None)
    indexer.add_chunks(sample_chunks)
    assert indexer.count() == 2

    # Delete single document by ID
    indexer.delete(ids=["doc_math"])
    assert indexer.count() == 1

    # Reset collection completely
    indexer.reset()
    assert indexer.count() == 0