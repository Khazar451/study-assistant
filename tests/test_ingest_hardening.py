"""Uploads are indexed one at a time, never lose the previous version, and respect per-visitor limits."""

from pathlib import Path
import threading
from unittest.mock import MagicMock
import uuid

import chromadb
from langchain_core.documents import Document
import pytest
from fastapi.testclient import TestClient

from src.api import server
from src.api.server import app, set_assistant
from src.assistant import StudyAssistant
from src.ingestion.indexer import ChromaIndexer
from src.ingestion.pipeline import IngestionPipeline
from src.memory import session_store


# --- B11: replacing a file's chunks ---


def _pipeline(text_holder, embedder):
    loader = MagicMock()
    loader.load.side_effect = lambda path: [
        Document(page_content=text_holder["text"], metadata={"source": "book.txt", "file_path": str(path), "page": 1})
    ]
    indexer = ChromaIndexer(collection_name=f"ingest-{uuid.uuid4().hex[:8]}", client=chromadb.EphemeralClient())
    return IngestionPipeline(loader=loader, embedder=embedder, indexer=indexer), indexer


def _vectors(texts, **_):
    return [[1.0, float(len(t) % 7)] for t in texts]


def test_failed_reingest_keeps_the_previous_version_searchable():
    text = {"text": "Kepler orbit ellipse. " * 900}  # well over one 16-chunk batch
    embedder = MagicMock()
    embedder.embed_batch.side_effect = _vectors
    pipeline, indexer = _pipeline(text, embedder)
    pipeline.ingest_file(Path("book.txt"))
    before = indexer.count()
    assert before > 16

    calls = {"n": 0}

    def fail_on_second_batch(texts, **kwargs):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("Embedding request failed")
        return _vectors(texts)

    embedder.embed_batch.side_effect = fail_on_second_batch
    text["text"] = "Newton gravity force. " * 900
    with pytest.raises(RuntimeError):
        pipeline.ingest_file(Path("book.txt"))

    assert indexer.count() == before


def test_shorter_reingest_removes_leftover_chunks():
    text = {"text": "Kepler orbit ellipse. " * 900}
    embedder = MagicMock()
    embedder.embed_batch.side_effect = _vectors
    pipeline, indexer = _pipeline(text, embedder)
    pipeline.ingest_file(Path("book.txt"))

    text["text"] = "Kepler orbit ellipse. " * 40
    result = pipeline.ingest_file(Path("book.txt"))
    assert indexer.count() == result["chunks_count"]


# --- B10, B12 and per-visitor limits: the upload endpoint ---


@pytest.fixture
def uploads(tmp_path, monkeypatch):
    """Run from a temporary folder with a mock assistant whose ingest can be held open."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(session_store, "_global_session_store", session_store.SessionStore(str(tmp_path / "c.db")))
    for name in ("WORKSPACE_ISOLATION", "DEMO_MODE", "RATE_LIMIT_PER_MIN", "DEMO_MAX_PER_MIN", "DEMO_MAX_PER_DAY",
                 "MAX_PENDING_UPLOADS", "MAX_FILES_PER_WORKSPACE"):
        monkeypatch.delenv(name, raising=False)

    gate = threading.Event()
    state = {"running": 0, "max_running": 0, "lock_held": []}
    guard = threading.Lock()

    def ingest(path, **_):
        state["lock_held"].append(server._ingest_lock.locked())
        with guard:
            state["running"] += 1
            state["max_running"] = max(state["max_running"], state["running"])
        gate.wait(timeout=10)
        with guard:
            state["running"] -= 1
        return {"status": "success"}

    assistant = MagicMock(spec=StudyAssistant)
    assistant.ingest.side_effect = ingest
    assistant.count.return_value = 1
    set_assistant(assistant)
    yield {"gate": gate, "state": state, "assistant": assistant}
    gate.set()


def _post(client, name, content=b"Kepler notes.", background=False):
    return client.post(
        "/api/ingest/file", params={"background": str(background).lower()}, files={"file": (name, content, "text/plain")}
    )


def _ingest_threads():
    return [t for t in threading.enumerate() if t.name.startswith("ingest")]


def test_synchronous_upload_holds_the_ingest_lock(uploads):
    uploads["gate"].set()
    assert _post(TestClient(app), "notes.txt").status_code == 200
    assert uploads["state"]["lock_held"] == [True]


def test_background_uploads_share_one_worker(uploads):
    client = TestClient(app)
    for i in range(3):
        assert _post(client, f"book{i}.txt", background=True).status_code == 200
    try:
        assert len(_ingest_threads()) == 1
    finally:
        uploads["gate"].set()


def test_visitor_with_three_uploads_in_progress_must_wait(uploads, monkeypatch):
    monkeypatch.setenv("WORKSPACE_ISOLATION", "true")
    client = TestClient(app)
    try:
        for i in range(3):
            assert _post(client, f"book{i}.txt", background=True).status_code == 200
        refused = _post(client, "book3.txt", background=True)
        assert refused.status_code == 429
        assert "wait" in refused.json()["detail"]
        assert TestClient(app).post(  # another visitor is not affected
            "/api/ingest/file", files={"file": ("other.txt", b"notes", "text/plain")}, params={"background": "true"}
        ).status_code == 200
    finally:
        uploads["gate"].set()


def test_visitor_file_cap_allows_replacing_an_existing_file(uploads, monkeypatch):
    monkeypatch.setenv("WORKSPACE_ISOLATION", "true")
    monkeypatch.setenv("MAX_FILES_PER_WORKSPACE", "2")
    uploads["gate"].set()
    client = TestClient(app)
    assert _post(client, "a.txt").status_code == 200
    assert _post(client, "b.txt").status_code == 200
    over = _post(client, "c.txt")
    assert over.status_code == 409
    assert "at most 2 files" in over.json()["detail"]
    assert _post(client, "a.txt", b"Updated notes.").status_code == 200


def test_local_development_has_no_upload_quota(uploads):
    uploads["gate"].set()
    client = TestClient(app)
    for i in range(7):
        assert _post(client, f"note{i}.txt").status_code == 200


@pytest.mark.parametrize(
    "name, content, ok",
    [
        ("lecture.pdf", b"%PDF-1.7\n1 0 obj", True),
        ("lecture.pdf", b"MZ\x90\x00 this is an executable", False),
        ("notes.txt", b"Kepler's laws", True),
        ("notes.txt", b"\x7fELF\x02\x01\x01\x00\x00\x00", False),
        ("slide.png", b"\x89PNG\r\n\x1a\n\x00\x00", True),
        ("slide.png", b"\xff\xd8\xff\xe0 a JPEG", False),
        ("deck.pptx", b"PK\x03\x04\x14\x00", True),
        ("deck.ppt", b"PK\x03\x04\x14\x00", False),
    ],
)
def test_file_content_must_match_its_extension(uploads, name, content, ok):
    uploads["gate"].set()
    res = _post(TestClient(app), name, content)
    assert res.status_code == (200 if ok else 400)
    assert Path("data/uploads", name).exists() is ok
