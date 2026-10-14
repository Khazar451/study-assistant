import os
from pathlib import Path
import tempfile
import pytest

from src.memory.session_store import SessionStore


@pytest.fixture
def temp_store():
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "test_chat.db"
        store = SessionStore(db_path=str(db_path))
        yield store


def test_session_lifecycle(temp_store: SessionStore):
    # 1. Create session
    sess = temp_store.create_session(title="Math Homework")
    assert sess["id"].startswith("session_")
    assert sess["title"] == "Math Homework"
    assert sess["message_count"] == 0

    # 2. Add user message
    msg1 = temp_store.add_message(
        session_id=sess["id"],
        sender="user",
        text="What is integration by parts?",
    )
    assert msg1["sender"] == "user"
    assert msg1["text"] == "What is integration by parts?"

    # 3. Add tutor message with citations
    msg2 = temp_store.add_message(
        session_id=sess["id"],
        sender="tutor",
        text=r"Integration by parts rewrites $\int u\,dv = uv - \int v\,du$ [1].",
        sources=["Calculus, page 475"],
        references=[{"n": 1, "label": "Calculus, page 475"}],
        grounding="course",
    )
    assert msg2["sender"] == "tutor"
    assert len(msg2["sources"]) == 1

    # 4. List sessions
    sessions = temp_store.list_sessions()
    assert len(sessions) == 1
    assert sessions[0]["id"] == sess["id"]
    assert sessions[0]["message_count"] == 2

    # 5. Fetch messages
    messages = temp_store.get_session_messages(sess["id"])
    assert len(messages) == 2
    assert messages[0]["sender"] == "user"
    assert messages[1]["sender"] == "tutor"
    assert messages[1]["sources"] == ["Calculus, page 475"]

    # 6. Recent history for LLM
    history = temp_store.get_recent_history(sess["id"], turns=1)
    # turns=1 -> 2 messages (user + assistant)
    assert len(history) == 2
    assert history[0]["role"] == "user"
    assert history[1]["role"] == "assistant"

    # 7. Rename session
    renamed = temp_store.rename_session(sess["id"], "Calculus Ch 8")
    assert renamed is True
    updated_sess = temp_store.get_session(sess["id"])
    assert updated_sess is not None
    assert updated_sess["title"] == "Calculus Ch 8"

    # 8. Delete session cascades to messages
    deleted = temp_store.delete_session(sess["id"])
    assert deleted is True
    assert temp_store.get_session(sess["id"]) is None
    assert len(temp_store.get_session_messages(sess["id"])) == 0


def test_auto_titling_on_first_message(temp_store: SessionStore):
    # Session starts as "New Chat"
    sess = temp_store.create_session()
    assert sess["title"] == "New Chat"

    # First user message auto-titles
    temp_store.add_message(
        session_id=sess["id"],
        sender="user",
        text="Explain Kepler's Third Law in orbital mechanics",
    )

    updated = temp_store.get_session(sess["id"])
    assert updated is not None
    assert "Kepler's Third Law" in updated["title"]


def test_limit_recent_messages(temp_store: SessionStore):
    sess = temp_store.create_session(title="Long Chat")
    for i in range(10):
        temp_store.add_message(sess["id"], "user", f"Question {i}")
        temp_store.add_message(sess["id"], "tutor", f"Answer {i}")

    # Total 20 messages
    all_msgs = temp_store.get_session_messages(sess["id"])
    assert len(all_msgs) == 20

    # Fetch last 4 messages
    limited = temp_store.get_session_messages(sess["id"], limit=4)
    assert len(limited) == 4
    # Chronological: should be Question 8, Answer 8, Question 9, Answer 9
    assert limited[0]["text"] == "Question 8"
    assert limited[-1]["text"] == "Answer 9"
