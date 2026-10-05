from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
import pytest

from src.api.server import app, set_assistant
from src.assistant import StudyAssistant


@pytest.fixture
def mock_assistant():
    """Mock StudyAssistant for API tests."""
    assistant = MagicMock(spec=StudyAssistant)
    assistant.count.return_value = 42
    assistant.persist_directory = "./mock_chroma"
    assistant.generator = MagicMock()
    assistant.generator.model = "nvidia/Llama-3.1-Nemotron-70B-Instruct-HF"
    assistant.generator.provider = "NVIDIA NIM"
    assistant.reranker = MagicMock()
    assistant.reranker.mode = "llm_listwise"
    assistant.retriever = MagicMock()
    assistant.retriever.format_context.return_value = "Formatted context from Kepler's Laws"

    # Ingestion mock
    assistant.ingest.return_value = {
        "status": "success",
        "file": "test.txt",
        "chunks_created": 3,
    }

    # Ask mock
    assistant.ask.return_value = {
        "answer": "Planetary orbits are ellipses with the Sun at one focus [Source: kepler.txt, Page: 1].",
        "sources": ["kepler.txt (p. 1)"],
        "chunks": [{"text": "Kepler chunk", "metadata": {"source": "kepler.txt", "page": 1}}],
        "model": "nvidia/Llama-3.1-Nemotron-70B-Instruct-HF",
        "provider": "NVIDIA NIM",
    }

    # Search mock
    assistant.search.return_value = [
        {"text": "Kepler chunk 1", "metadata": {"source": "kepler.txt", "page": 1}, "rerank_score": 0.95}
    ]

    # Indexer mock
    mock_indexer = MagicMock()
    mock_indexer.collection.get.return_value = {
        "metadatas": [
            {"source": "astronomy_kepler.txt", "page": 1, "file_path": "sample_materials/astronomy_kepler.txt"},
            {"source": "astronomy_kepler.txt", "page": 2, "file_path": "sample_materials/astronomy_kepler.txt"},
        ]
    }
    assistant.indexer = mock_indexer

    set_assistant(assistant)
    return assistant


@pytest.fixture
def client(mock_assistant):
    return TestClient(app)


def test_api_status(client, mock_assistant):
    res = client.get("/api/status")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ready"
    assert data["indexed_chunks"] == 42
    assert "provider" in data


def test_api_documents(client, mock_assistant):
    res = client.get("/api/documents")
    assert res.status_code == 200
    data = res.json()
    assert data["total_chunks"] == 2
    assert data["total_documents"] == 1
    assert data["documents"][0]["source"] == "astronomy_kepler.txt"


def test_api_ask(client, mock_assistant):
    payload = {"query": "What is Kepler's first law?"}
    res = client.post("/api/ask", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert "Kepler's first law" in data["answer"] or "Planetary orbits" in data["answer"]
    assert "sources" in data
    assert "duration_ms" in data


def test_api_search(client, mock_assistant):
    payload = {"query": "elliptical orbits"}
    res = client.post("/api/search", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["total_chunks"] == 1
    assert len(data["chunks"]) == 1


def test_api_clear(client, mock_assistant):
    mock_assistant.count.return_value = 0
    res = client.post("/api/clear")
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is True
    assert data["count"] == 0


def test_api_ingest_file(client, mock_assistant):
    file_content = b"# Test Document\n\nSample test content for ingestion."
    files = {"file": ("test_lecture.txt", file_content, "text/plain")}
    res = client.post("/api/ingest/file", files=files)
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is True
    assert data["filename"] == "test_lecture.txt"


def test_api_ingest_file_invalid_ext(client, mock_assistant):
    files = {"file": ("bad_file.exe", b"binary", "application/octet-stream")}
    res = client.post("/api/ingest/file", files=files)
    assert res.status_code == 400
    assert "Unsupported file format" in res.json()["detail"]


def test_api_ingest_image_file(client, mock_assistant):
    image_bytes = b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x06\x00\x00\x00\x1f\x15c4\x00\x00\x00\nIDATx\x9cc\x00\x01\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
    files = {"file": ("slide_architecture.png", image_bytes, "image/png")}
    res = client.post("/api/ingest/file", files=files)
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is True
    assert data["filename"] == "slide_architecture.png"


def test_api_ingest_sample(client, mock_assistant):
    res = client.post("/api/ingest/sample")
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is True
    assert "Kepler" in data["material"]


def test_api_ask_stream(client, mock_assistant):
    mock_assistant.generator.generate_stream.return_value = iter(["Kepler's ", "first ", "law."])
    res = client.get("/api/ask/stream?query=kepler")
    assert res.status_code == 200
    text = res.text
    assert "data:" in text
    assert "Kepler's " in text
    assert "done" in text


def test_api_flashcards(client, mock_assistant):
    res = client.post("/api/flashcards", json={"count": 2})
    assert res.status_code == 200
    data = res.json()
    assert "flashcards" in data
    assert len(data["flashcards"]) > 0


def test_api_metrics(client, mock_assistant):
    res = client.get("/metrics")
    assert res.status_code == 200
    assert "text/plain" in res.headers["content-type"]
    text = res.text
    assert "rag_indexed_chunks_total" in text
    assert "rag_queries_total" in text


def test_api_ask_with_web_flag(client, mock_assistant):
    payload = {"query": "What is dark matter?", "use_web": True}
    res = client.post("/api/ask", json=payload)
    assert res.status_code == 200
    mock_assistant.ask.assert_called_with(
        query="What is dark matter?",
        stream=False,
        top_k=15,
        top_n=5,
        augment=True,
        augment_mode="expand",
        min_similarity=None,
        min_rerank_score=None,
        use_web=True,
        fallback_to_web=True,
    )


def test_api_ingest_file_background(client, mock_assistant):
    file_content = b"# Background Ingestion\nTesting background job execution."
    files = {"file": ("async_doc.txt", file_content, "text/plain")}
    res = client.post("/api/ingest/file?background=true", files=files)
    assert res.status_code == 200
    data = res.json()
    assert data["success"] is True
    assert "job_id" in data
    assert data["status"] == "processing"
    assert data["filename"] == "async_doc.txt"

    job_id = data["job_id"]
    status_res = client.get(f"/api/ingest/status/{job_id}")
    assert status_res.status_code == 200
    status_data = status_res.json()
    assert status_data["job_id"] == job_id
    assert status_data["filename"] == "async_doc.txt"

    jobs_res = client.get("/api/ingest/jobs")
    assert jobs_res.status_code == 200
    jobs_data = jobs_res.json()
    assert "jobs" in jobs_data
    assert any(j["job_id"] == job_id for j in jobs_data["jobs"])


def test_api_ingest_status_not_found(client, mock_assistant):
    res = client.get("/api/ingest/status/non-existent-job-id")
    assert res.status_code == 404

