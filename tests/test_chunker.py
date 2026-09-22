import pytest
from langchain_core.documents import Document

from src.ingestion.chunker import chunk_documents, chunk_text

# Testing Chunk Size
def test_chunk_documents_splits_long_text():
    long_text = "Study assistant data ingestion test." * 30
    docs = [Document(page_content = long_text, metadata = {"source": "notes.txt"})]

    chunks = chunk_documents(docs, chunk_size = 100, chunk_overlap = 20)

    assert len(chunks) > 1

    for chunk in chunks:
        assert len(chunk.page_content) <= 120

# Testing whether or not the metadata is preserved
def test_chunk_documents_preserves_metadata():
    long_text = "Notes of Math course and formulas." * 20
    docs = [Document(page_content = long_text, metadata = {"lecture": "math", "unite": 1})]

    chunks = chunk_documents(docs, chunk_size = 80, chunk_overlap = 10)

    for chunk in chunks:
        assert chunk.metadata["lecture"] == "math"
        assert chunk.metadata["unite"] == 1
        assert "start_index" in chunk.metadata

# Testing if it returns empty list when the text is empty
def test_chunk_text_empty_input():
    assert chunk_text("") == []
    assert chunk_text("   ") == []

# Testing if it returns empty list when the doc is empty
def test_chunk_documents_empty_listd():
    assert chunk_documents([]) == []

# Testing the overlap between two chunks
def test_chunk_overlap_preserves_context():
    sample_text = "word1 " * 20 + "word2 " * 25
    chunks = chunk_text(sample_text, chunk_size = 100, chunk_overlap = 30)

    assert len(chunks) >= 2
    assert len(chunks[1].page_content) > 0