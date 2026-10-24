"""Visitors of a shared deployment must not see or change each other's chats."""

import sqlite3
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from src.api.server import WORKSPACE_COOKIE, app, set_assistant
from src.assistant import StudyAssistant
from src.memory import session_store


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
    set_assistant(mock)
    return mock


@pytest.fixture
def isolation(monkeypatch, store, assistant):
    monkeypatch.setenv("WORKSPACE_ISOLATION", "true")
    for name in ("RATE_LIMIT_PER_MIN", "DEMO_MAX_PER_MIN", "DEMO_MAX_PER_DAY", "DEMO_PAUSED"):
        monkeypatch.delenv(name, raising=False)


def visitor() -> TestClient:
    """A browser of its own: every TestClient keeps a separate cookie jar."""
    return TestClient(app)


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
