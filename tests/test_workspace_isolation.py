"""Visitors of a shared deployment must not see or change each other's chats, files or documents."""

from pathlib import Path
import sqlite3
from unittest.mock import MagicMock
import uuid

import chromadb
from langchain_core.documents import Document
import pytest
from fastapi.testclient import TestClient

from src.api.server import SHARED_WORKSPACE, WORKSPACE_COOKIE, _workspace_filter, app, set_assistant
from src.assistant import StudyAssistant
from src.ingestion.indexer import ChromaIndexer
from src.ingestion.pipeline import IngestionPipeline
from src.memory import session_store
from src.retrieval.retriever import StudyRetriever


@pytest.fixture
def store(tmp_path, monkeypatch):
    """A throwaway chat database instead of ./data/chat_sessions.db."""
    temp_store = session_store.SessionStore(db_path=str(tmp_path / "chat_sessions.db"))
    monkeypatch.setattr(session_store, "_global_session_store", temp_store)
    return temp_store


@pytest.fixture
def assistant():
    mock = MagicMock(spec=StudyAssistant)
    mock.count.return_value = 1
    mock.persist_directory = "./mock_chroma"
    mock.generator = MagicMock()
    mock.generator.model = "nvidia/nemotron-3-super-120b-a12b"
    mock.generator.generate_stream.return_value = iter(["Kepler."])
    mock.retriever = MagicMock()
    mock.retriever.augmenter = None
    mock.retriever.format_context.return_value = "[Document 1] (Source: kepler.txt | Page: 1)\nKepler"
    mock.augmenter = None
    mock.reranker = MagicMock()
    mock.search.return_value = [{"text": "Kepler", "metadata": {"source": "kepler.txt", "page": 1}, "rerank_score": 0.9}]
    mock.ask.return_value = {"answer": "Kepler.", "sources": []}
    mock.ingest.return_value = {"status": "success"}
    mock.indexer = MagicMock()
    mock.indexer.collection.get.return_value = {"ids": [], "metadatas": []}
    set_assistant(mock)
    return mock


@pytest.fixture
def isolation(monkeypatch, store, assistant):
    monkeypatch.setenv("WORKSPACE_ISOLATION", "true")
    for name in ("RATE_LIMIT_PER_MIN", "DEMO_MAX_PER_MIN", "DEMO_MAX_PER_DAY", "DEMO_PAUSED"):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def workdir(tmp_path, monkeypatch):
    """Run from a temporary folder so uploads never land in the real ./data/uploads."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "sample_materials").mkdir()
    (tmp_path / "sample_materials" / "astronomy_kepler.txt").write_text("Kepler's laws.")
    return tmp_path


def visitor() -> TestClient:
    """A browser of its own: every TestClient keeps a separate cookie jar."""
    return TestClient(app)


def upload(client: TestClient, name: str, content: bytes, background: bool = False):
    return client.post(
        "/api/ingest/file",
        params={"background": str(background).lower()},
        files={"file": (name, content, "text/plain")},
    )


def test_each_browser_gets_a_private_workspace_cookie(isolation):
    alice = visitor()
    first = alice.get("/api/chat/sessions")
    cookie = first.headers["set-cookie"]
    assert cookie.startswith(f"{WORKSPACE_COOKIE}=")
    assert "HttpOnly" in cookie
    assert "samesite=lax" in cookie.lower()

    again = alice.get("/api/chat/sessions")
    assert "set-cookie" not in again.headers
    assert visitor().get("/api/chat/sessions").cookies[WORKSPACE_COOKIE] != alice.cookies[WORKSPACE_COOKIE]


def test_visitors_cannot_list_read_rename_or_delete_each_others_chats(isolation, store):
    alice, bob = visitor(), visitor()
    sid = alice.post("/api/chat/sessions", json={"title": "Alice exam prep"}).json()["id"]
    store.add_message(sid, "user", "my private question")

    assert bob.get("/api/chat/sessions").json() == []
    assert bob.get(f"/api/chat/sessions/{sid}").status_code == 404
    assert bob.patch(f"/api/chat/sessions/{sid}", json={"title": "taken"}).status_code == 404
    assert bob.delete(f"/api/chat/sessions/{sid}").status_code == 404

    mine = alice.get(f"/api/chat/sessions/{sid}").json()
    assert mine["title"] == "Alice exam prep"
    assert [m["text"] for m in mine["messages"]] == ["my private question"]
    assert [s["id"] for s in alice.get("/api/chat/sessions").json()] == [sid]


def test_foreign_session_id_is_refused_when_asking(isolation, store, assistant):
    alice, bob = visitor(), visitor()
    sid = alice.post("/api/chat/sessions", json={"title": "Alice"}).json()["id"]
    store.add_message(sid, "user", "my private question")

    assert bob.post("/api/ask", json={"query": "orbits", "session_id": sid}).status_code == 404
    assistant.ask.assert_not_called()
    assert bob.get("/api/ask/stream", params={"query": "orbits", "session_id": sid}).status_code == 404
    assistant.generator.generate_stream.assert_not_called()
    assert len(store.get_session_messages(sid)) == 1


def test_new_session_id_belongs_to_the_caller(isolation, store):
    alice, bob = visitor(), visitor()
    assert bob.post("/api/ask", json={"query": "orbits", "session_id": "session_bob1"}).status_code == 200

    exists, owner = store.session_owner("session_bob1")
    assert exists and owner == bob.cookies[WORKSPACE_COOKIE]
    assert alice.get("/api/chat/sessions/session_bob1").status_code == 404


def test_taken_session_id_cannot_be_created_again(isolation):
    alice, bob = visitor(), visitor()
    sid = alice.post("/api/chat/sessions", json={"title": "Alice"}).json()["id"]
    assert bob.post("/api/chat/sessions", json={"title": "Bob", "session_id": sid}).status_code == 409


def test_without_isolation_one_shared_history_and_no_cookie(monkeypatch, store, assistant):
    monkeypatch.delenv("WORKSPACE_ISOLATION", raising=False)
    monkeypatch.setenv("DEMO_MODE", "false")
    first, second = visitor(), visitor()
    created = first.post("/api/chat/sessions", json={"title": "Local notes"})
    assert "set-cookie" not in created.headers
    assert [s["title"] for s in second.get("/api/chat/sessions").json()] == ["Local notes"]


def test_existing_database_gains_workspace_column(tmp_path):
    db = tmp_path / "old.db"
    conn = sqlite3.connect(db)
    conn.executescript(
        """
        CREATE TABLE sessions (id TEXT PRIMARY KEY, title TEXT NOT NULL, created_at REAL NOT NULL, updated_at REAL NOT NULL);
        INSERT INTO sessions VALUES ('session_old', 'Before workspaces', 1.0, 1.0);
        """
    )
    conn.close()

    upgraded = session_store.SessionStore(db_path=str(db))
    assert [s["id"] for s in upgraded.list_sessions()] == ["session_old"]
    assert upgraded.list_sessions(workspace="a" * 32) == []
    assert upgraded.session_owner("session_old") == (True, None)


# --- Uploads, documents, files and jobs (B2, B5, B6) ---


def test_same_name_uploads_stay_separate(isolation, workdir, assistant):
    alice, bob = visitor(), visitor()
    assert upload(alice, "notes.txt", b"ALICE NOTES").status_code == 200
    assert upload(bob, "notes.txt", b"BOB NOTES").status_code == 200

    alice_ws, bob_ws = alice.cookies[WORKSPACE_COOKIE], bob.cookies[WORKSPACE_COOKIE]
    uploads = workdir / "data" / "uploads"
    assert (uploads / alice_ws / "notes.txt").read_bytes() == b"ALICE NOTES"
    assert (uploads / bob_ws / "notes.txt").read_bytes() == b"BOB NOTES"
    stamped = [call.kwargs.get("metadata") for call in assistant.ingest.call_args_list]
    assert stamped == [{"workspace": alice_ws}, {"workspace": bob_ws}]


def test_visitors_cannot_download_or_render_each_others_files(isolation, workdir):
    alice, bob = visitor(), visitor()
    upload(alice, "notes.txt", b"ALICE NOTES")
    path = str((workdir / "data" / "uploads" / alice.cookies[WORKSPACE_COOKIE] / "notes.txt").resolve())

    assert alice.get("/api/materials/file", params={"path": path}).content == b"ALICE NOTES"
    assert bob.get("/api/materials/file", params={"path": path}).status_code == 404
    refused = bob.get("/api/materials/page", params={"path": path, "page": 1})
    assert refused.status_code == 404
    assert refused.json()["detail"] == "File is not in the course library."
    sample = str((workdir / "sample_materials" / "astronomy_kepler.txt").resolve())
    assert bob.get("/api/materials/file", params={"path": sample}).status_code == 200


def test_ingestion_jobs_are_private(isolation, workdir):
    alice, bob = visitor(), visitor()
    job_id = upload(alice, "notes.txt", b"ALICE NOTES", background=True).json()["job_id"]

    assert alice.get(f"/api/ingest/status/{job_id}").status_code == 200
    assert bob.get(f"/api/ingest/status/{job_id}").status_code == 404
    assert job_id in [job["job_id"] for job in alice.get("/api/ingest/jobs").json()["jobs"]]
    assert job_id not in [job["job_id"] for job in bob.get("/api/ingest/jobs").json()["jobs"]]


def test_every_retrieval_path_is_limited_to_own_workspace_and_sample(isolation, assistant):
    bob = visitor()
    bob.get("/api/status")
    expected = {"workspace": {"$in": [bob.cookies[WORKSPACE_COOKIE], SHARED_WORKSPACE]}}

    bob.post("/api/search", json={"query": "orbits"})
    assert assistant.search.call_args.kwargs["where"] == expected
    bob.post("/api/flashcards", json={"topic": "orbits"})
    assert assistant.search.call_args.kwargs["where"] == expected
    bob.get("/api/ask/stream", params={"query": "orbits", "augment": "false"})
    assert assistant.search.call_args.kwargs["where"] == expected
    bob.post("/api/ask", json={"query": "orbits"})
    assert assistant.ask.call_args.kwargs["where"] == expected
    bob.get("/api/documents")
    assert assistant.indexer.collection.get.call_args.kwargs["where"] == expected


def test_sample_is_indexed_into_the_shared_workspace(isolation, workdir, assistant):
    assert visitor().post("/api/ingest/sample").status_code == 200
    assert assistant.ingest.call_args.kwargs["metadata"] == {"workspace": SHARED_WORKSPACE}


def test_without_isolation_retrieval_is_unfiltered(monkeypatch, store, assistant):
    monkeypatch.delenv("WORKSPACE_ISOLATION", raising=False)
    monkeypatch.setenv("DEMO_MODE", "false")
    visitor().post("/api/search", json={"query": "orbits"})
    assert "where" not in assistant.search.call_args.kwargs


def test_chroma_filter_returns_own_chunks_and_sample_only():
    """The filter shape is evaluated by a real (in-memory) Chroma, not just passed along."""
    indexer = ChromaIndexer(collection_name=f"ws-{uuid.uuid4().hex[:8]}", client=chromadb.EphemeralClient())
    alice, bob = "a" * 32, "b" * 32
    indexer.upsert(
        ids=["alice_0", "bob_0", "sample_0"],
        documents=["Alice's Kepler notes", "Bob's Kepler notes", "Kepler sample"],
        embeddings=[[1.0, 0.0], [1.0, 0.01], [1.0, 0.02]],
        metadatas=[{"workspace": alice}, {"workspace": bob}, {"workspace": SHARED_WORKSPACE}],
    )
    embedder = MagicMock()
    embedder.embed_query.return_value = [1.0, 0.0]
    retriever = StudyRetriever(indexer=indexer, embedder=embedder)

    found = retriever.retrieve("Kepler", top_k=10, where=_workspace_filter(alice))
    assert sorted(item["id"] for item in found) == ["alice_0", "sample_0"]


def test_pipeline_stamps_extra_metadata_on_every_chunk():
    loader = MagicMock()
    loader.load.return_value = [Document(page_content="Kepler " * 300, metadata={"source": "notes.txt", "page": 1})]
    embedder = MagicMock()
    embedder.embed_batch.side_effect = lambda texts, **_: [[0.1, 0.2] for _ in texts]
    indexer = MagicMock()
    pipeline = IngestionPipeline(loader=loader, embedder=embedder, indexer=indexer)

    pipeline.ingest_file(Path("notes.txt"), extra_metadata={"workspace": "a" * 32})

    metadatas = [m for call in indexer.upsert.call_args_list for m in call.kwargs["metadatas"]]
    assert len(metadatas) > 1
    assert all(m["workspace"] == "a" * 32 for m in metadatas)


def test_assistant_passes_the_filter_down_to_the_retriever():
    retriever = MagicMock()
    retriever.retrieve.return_value = []
    tavily = MagicMock(is_available=False)
    assistant = StudyAssistant(
        pipeline=MagicMock(), retriever=retriever, reranker=MagicMock(), generator=MagicMock(), tavily=tavily
    )
    where = _workspace_filter("a" * 32)
    assistant.search("Kepler", where=where)
    assert retriever.retrieve.call_args.kwargs["where"] == where
