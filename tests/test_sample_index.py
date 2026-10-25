"""A fresh demo must index the bundled sample without calling the embedding API."""

import json
from pathlib import Path
import shutil
import threading
from unittest.mock import MagicMock
import uuid

import chromadb
from langchain_core.documents import Document
import pytest
from fastapi.testclient import TestClient

from src.api import server
from src.api.server import SHARED_WORKSPACE, app, set_assistant
from src.assistant import StudyAssistant
from src.ingestion.indexer import ChromaIndexer
from src.ingestion.loader import DocumentLoader
from src.ingestion.pipeline import IngestionPipeline, text_key
from src.ingestion.sample_index import index_path, load_embedding_cache, sample_chunk_texts

REPO_SAMPLE = Path(__file__).resolve().parent.parent / "sample_materials" / "astronomy_kepler.txt"
MODEL = json.loads(index_path(REPO_SAMPLE).read_text(encoding="utf-8"))["model"]


def test_committed_index_matches_the_committed_sample():
    """Fails when the sample text or chunking changes without rebuilding the index file."""
    cache = load_embedding_cache(REPO_SAMPLE, MODEL)
    assert cache is not None
    assert {text_key(t) for t in sample_chunk_texts(REPO_SAMPLE)} <= set(cache)


@pytest.fixture
def demo(tmp_path, monkeypatch):
    """A real loader, chunker and in-memory Chroma; only the embedding API is a mock."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "sample_materials").mkdir()
    for path in (REPO_SAMPLE, index_path(REPO_SAMPLE)):
        shutil.copy(path, tmp_path / "sample_materials" / path.name)
    for name in ("WORKSPACE_ISOLATION", "DEMO_MODE", "RATE_LIMIT_PER_MIN", "DEMO_MAX_PER_MIN", "DEMO_MAX_PER_DAY"):
        monkeypatch.delenv(name, raising=False)

    embedder = MagicMock()
    embedder.model = MODEL
    embedder.embed_batch.side_effect = lambda texts, **_: [[0.5] * 2048 for _ in texts]
    indexer = ChromaIndexer(collection_name=f"sample-{uuid.uuid4().hex[:8]}", client=chromadb.EphemeralClient())
    pipeline = IngestionPipeline(loader=DocumentLoader(extract_visuals=False), embedder=embedder, indexer=indexer)
    assistant = StudyAssistant(
        pipeline=pipeline,
        retriever=MagicMock(),
        reranker=MagicMock(),
        generator=MagicMock(),
        tavily=MagicMock(is_available=False),
    )
    set_assistant(assistant)
    server._sample_state["error"] = None
    return assistant


def test_sample_is_indexed_from_the_precomputed_file(demo):
    client = TestClient(app)
    assert client.get("/api/status").json()["sample_ready"] is False

    assert client.post("/api/ingest/sample").status_code == 200

    demo.embedder.embed_batch.assert_not_called()
    stored = demo.indexer.collection.get(include=["embeddings", "metadatas", "documents"])
    cache = load_embedding_cache(REPO_SAMPLE, MODEL)
    assert len(stored["ids"]) == len(cache)
    for text, vector, meta in zip(stored["documents"], stored["embeddings"], stored["metadatas"]):
        assert list(vector) == pytest.approx(cache[text_key(text)])
        assert meta["workspace"] == SHARED_WORKSPACE
    assert client.get("/api/status").json()["sample_ready"] is True


def test_a_different_embedding_model_falls_back_to_live_ingest(demo):
    demo.embedder.model = "some/other-embedder"
    assert TestClient(app).post("/api/ingest/sample").status_code == 200
    demo.embedder.embed_batch.assert_called()


def test_startup_reports_why_the_sample_is_missing(demo, monkeypatch, caplog):
    monkeypatch.setenv("AUTO_INGEST_SAMPLE", "true")
    monkeypatch.setattr(server.time, "sleep", lambda _seconds: None)
    demo.embedder.model = "some/other-embedder"
    demo.embedder.embed_batch.side_effect = RuntimeError("Embedding request failed")

    server._auto_ingest_sample()
    for thread in threading.enumerate():
        if thread.name == "auto-ingest-sample":
            thread.join(timeout=10)

    status = TestClient(app).get("/api/status").json()
    assert status["sample_ready"] is False
    # Visitors see a reference, not the exception; the log holds the cause under the same reference.
    ref = status["sample_error"].split("(ref ")[1].split(")")[0]
    assert "Embedding request failed" not in status["sample_error"]
    logged = [r for r in caplog.records if f"(ref {ref})" in r.getMessage()]
    assert logged and "Embedding request failed" in str(logged[0].exc_info[1])


def test_pipeline_uses_cached_vectors_only_when_every_text_is_cached():
    loader = MagicMock()
    loader.load.return_value = [Document(page_content="Kepler orbit. " * 120, metadata={"source": "k.txt", "page": 1})]
    embedder = MagicMock()
    embedder.embed_batch.side_effect = lambda texts, **_: [[0.1, 0.2] for _ in texts]
    pipeline = IngestionPipeline(loader=loader, embedder=embedder, indexer=MagicMock())
    texts = [c.page_content for c in pipeline.chunker.split_documents(loader.load.return_value)]

    pipeline.ingest_file(Path("k.txt"), embedding_cache={text_key(t): [0.9, 0.9] for t in texts})
    embedder.embed_batch.assert_not_called()

    pipeline.ingest_file(Path("k.txt"), embedding_cache={text_key(texts[0]): [0.9, 0.9]})
    embedder.embed_batch.assert_called()
