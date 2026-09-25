import uuid
import chromadb
import pytest
from pypdf import PdfWriter
from unittest.mock import MagicMock

from langchain_core.documents import Document
from src.ingestion.indexer import ChromaIndexer
from src.ingestion.loader import DocumentLoader
from src.ingestion.pipeline import IngestionPipeline, DocumentChunker


@pytest.fixture
def mock_embedder():
    """Mock embedder returning deterministic 4-dimensional vectors."""
    embedder = MagicMock()
    # Support both embed_documents and embed_batch
    embedder.embed_documents.side_effect = lambda texts, **kwargs: [[0.1, 0.2, 0.3, 0.4] for _ in texts]
    embedder.embed_batch.side_effect = lambda texts, **kwargs: [[0.1, 0.2, 0.3, 0.4] for _ in texts]
    return embedder


@pytest.fixture
def in_memory_indexer():
    """In-memory ChromaDB client for isolated testing."""
    client = chromadb.EphemeralClient()
    unique_name = f"test_{uuid.uuid4().hex[:8]}"
    return ChromaIndexer(collection_name=unique_name, client=client)


def test_metadata_page_number_preservation_after_chunking(tmp_path, mock_embedder, in_memory_indexer):
    """Verify that a 3-page document split into 15+ chunks retains origin page numbers for each chunk."""
    pdf_path = tmp_path / "three_page_lecture.pdf"

    # Create a 3-page PDF where each page has multiple distinct sentences
    writer = PdfWriter()
    for _ in range(3):
        writer.add_blank_page(width=72, height=72)
    with open(pdf_path, "wb") as f:
        writer.write(f)

    # Mock the loader's output to simulate 3 distinct pages with content
    loader = DocumentLoader()
    page1_text = "Page 1 Content: Quantum mechanics intro. " * 10
    page2_text = "Page 2 Content: Wave function probabilities. " * 10
    page3_text = "Page 3 Content: Schrodinger equation solutions. " * 10

    mock_docs = [
        Document(page_content=page1_text, metadata={"source": pdf_path.name, "file_path": str(pdf_path), "file_type": "pdf", "page": 1}),
        Document(page_content=page2_text, metadata={"source": pdf_path.name, "file_path": str(pdf_path), "file_type": "pdf", "page": 2}),
        Document(page_content=page3_text, metadata={"source": pdf_path.name, "file_path": str(pdf_path), "file_type": "pdf", "page": 3}),
    ]
    loader.load = MagicMock(return_value=mock_docs)

    # Use small chunk_size to generate at least 15 chunks total
    chunker = DocumentChunker(chunk_size=80, chunk_overlap=10)
    pipeline = IngestionPipeline(
        loader=loader,
        chunker=chunker,
        embedder=mock_embedder,
        indexer=in_memory_indexer,
        batch_size=8,
    )

    result = pipeline.ingest_file(pdf_path)
    assert result["status"] == "success"
    assert result["chunks_count"] >= 15

    # Retrieve all chunks from indexer to verify page metadata preservation
    collection_items = in_memory_indexer.collection.get()
    metadatas = collection_items["metadatas"]
    documents = collection_items["documents"]

    assert len(metadatas) == result["chunks_count"]

    # Verify every chunk originating from page 1, 2, and 3 kept its respective page number
    for doc_text, meta in zip(documents, metadatas):
        assert "page" in meta
        if "Page 1" in doc_text:
            assert meta["page"] == 1
        elif "Page 2" in doc_text:
            assert meta["page"] == 2
        elif "Page 3" in doc_text:
            assert meta["page"] == 3


def test_idempotent_ingestion(tmp_path, mock_embedder, in_memory_indexer):
    """Assert repeated ingestion does not increase chunk count, and shortening a file removes ghost chunks."""
    test_file = tmp_path / "study_notes.md"
    test_file.write_text("Topic A sentence 1. Topic A sentence 2. Topic A sentence 3.", encoding="utf-8")

    chunker = DocumentChunker(chunk_size=30, chunk_overlap=5)
    pipeline = IngestionPipeline(
        chunker=chunker,
        embedder=mock_embedder,
        indexer=in_memory_indexer,
    )

    # 1. First ingestion
    res1 = pipeline.ingest_file(test_file)
    initial_count = in_memory_indexer.count()
    assert initial_count > 0
    assert res1["chunks_count"] == initial_count

    # 2. Second ingestion with identical file (Idempotency test)
    res2 = pipeline.ingest_file(test_file)
    assert in_memory_indexer.count() == initial_count
    assert res2["chunks_count"] == initial_count

    # 3. Third ingestion with shortened file (Ghost chunk elimination test)
    test_file.write_text("Short text.", encoding="utf-8")
    res3 = pipeline.ingest_file(test_file)
    new_count = in_memory_indexer.count()
    assert new_count < initial_count
    assert new_count == res3["chunks_count"]


def test_ingest_directory_continues_on_corrupt_file(tmp_path, mock_embedder, in_memory_indexer):
    """Verify ingest_directory indexes valid files and captures corrupt files in return report."""
    test_dir = tmp_path / "materials"
    test_dir.mkdir()

    # Valid file
    valid_file = test_dir / "valid_notes.md"
    valid_file.write_text("Valid notes on cellular respiration.", encoding="utf-8")

    # Corrupt zero-byte PDF file
    corrupt_file = test_dir / "corrupted_lecture.pdf"
    corrupt_file.write_bytes(b"")

    pipeline = IngestionPipeline(
        embedder=mock_embedder,
        indexer=in_memory_indexer,
    )

    results = pipeline.ingest_directory(test_dir)

    # Verify valid file succeeded
    assert len(results["successful"]) == 1
    assert results["successful"][0]["source"] == "valid_notes.md"
    assert in_memory_indexer.count() > 0

    # Verify corrupt file failed gracefully without aborting
    assert len(results["failed"]) == 1
    assert results["failed"][0]["source"] == "corrupted_lecture.pdf"
    assert "error" in results["failed"][0]
