"""Math checks have a time limit, visitors never see raw exceptions, and flashcards survive model quirks."""

import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from src.api import server
from src.api.server import app, set_assistant
from src.assistant import StudyAssistant
from src.generation.generator import StudyGenerator
from src.memory import session_store
from src.verification.time_limited import TimeLimitedVerifier

NORMAL = r"$$\frac{d}{dx}\sin(x^2) = 2x\cos(x^2)$$"
RUNAWAY = r"$$(a+b+c+d+e+f)^{25} = a^{25}$$"  # SymPy ran for over 60 s on this when measured


def _ready(verifier: TimeLimitedVerifier, deadline: float = 60.0) -> None:
    """Wait until the helper has started and answers a normal check."""
    verifier.start()
    end = time.time() + deadline
    while time.time() < end:
        if verifier.verify(NORMAL):
            return
        time.sleep(0.2)
    pytest.fail("verifier helper did not become ready")


@pytest.fixture
def verifier():
    helper = TimeLimitedVerifier(timeout=2.0)
    _ready(helper)
    yield helper
    helper.close()


# --- B16: time limit ---


def test_runaway_check_is_stopped_and_the_helper_recovers(verifier):
    start = time.perf_counter()
    assert verifier.verify(RUNAWAY) is None
    assert time.perf_counter() - start < 4.0
    _ready(verifier)  # restarted helper answers normal checks again
    assert verifier.verify(NORMAL)["status"] == "verified"


def test_check_is_skipped_while_the_helper_is_busy(verifier):
    worker = threading.Thread(target=verifier.verify, args=(RUNAWAY,), daemon=True)
    worker.start()
    time.sleep(0.3)
    start = time.perf_counter()
    assert verifier.verify(NORMAL) is None
    assert time.perf_counter() - start < 1.5
    worker.join(timeout=5)


@pytest.fixture
def api(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(session_store, "_global_session_store", session_store.SessionStore(str(tmp_path / "c.db")))
    for name in ("WORKSPACE_ISOLATION", "DEMO_MODE", "RATE_LIMIT_PER_MIN", "DEMO_MAX_PER_MIN", "DEMO_MAX_PER_DAY"):
        monkeypatch.delenv(name, raising=False)
    assistant = MagicMock(spec=StudyAssistant)
    assistant.count.return_value = 1
    assistant.persist_directory = "./mock_chroma"
    assistant.generator = MagicMock()
    assistant.generator.model = "nvidia/nemotron-3-super-120b-a12b"
    assistant.retriever = MagicMock()
    assistant.retriever.augmenter = None
    assistant.retriever.format_context.return_value = "[Document 1] (Source: k.txt | Page: 1)\nKepler"
    assistant.augmenter = None
    assistant.reranker = MagicMock()
    assistant.indexer = MagicMock()
    assistant.search.return_value = [{"text": "Kepler", "metadata": {"source": "k.txt", "page": 1}, "rerank_score": 0.9}]
    set_assistant(assistant)
    return assistant


def test_runaway_math_in_an_answer_does_not_hold_the_response(api, monkeypatch, verifier):
    monkeypatch.setattr(server, "_math_verifier", verifier)
    api.ask.return_value = {"answer": f"Expanding gives {RUNAWAY}", "sources": []}
    done = {}
    request = threading.Thread(
        target=lambda: done.setdefault("res", TestClient(app).post("/api/ask", json={"query": "expand it"})),
        daemon=True,
    )
    request.start()
    request.join(timeout=8)
    assert "res" in done, "the answer was held by the math check"
    assert done["res"].status_code == 200


# --- B15: no raw exception text for visitors ---

SECRET = "upstream said: invalid key nvapi-abc123 at /app/data/uploads/x"


def _ref_and_logged(message: str, caplog) -> None:
    assert SECRET not in message and "(ref " in message
    ref = message.split("(ref ")[1].split(")")[0]
    assert any(f"(ref {ref})" in r.getMessage() and SECRET in str(r.exc_info[1]) for r in caplog.records)


def test_stream_error_is_a_safe_message(api, caplog):
    api.search.side_effect = RuntimeError(SECRET)
    body = TestClient(app).get("/api/ask/stream", params={"query": "orbits", "augment": "false"}).text
    assert SECRET not in body
    error = next(line for line in body.splitlines() if '"type": "error"' in line)
    _ref_and_logged(error, caplog)


def test_failed_upload_is_a_safe_message(api, caplog):
    api.ingest.side_effect = RuntimeError(SECRET)
    res = TestClient(app).post("/api/ingest/file", files={"file": ("notes.txt", b"Kepler", "text/plain")})
    assert res.status_code == 500
    _ref_and_logged(res.json()["detail"], caplog)
    job = TestClient(app).get("/api/ingest/jobs").json()["jobs"][-1]
    _ref_and_logged(job["error"], caplog)


def test_document_listing_error_is_a_safe_message(api, caplog):
    api.indexer.collection.get.side_effect = RuntimeError(SECRET)
    _ref_and_logged(TestClient(app).get("/api/documents").json()["error"], caplog)


def test_deliberate_upload_refusals_keep_their_message(api):
    res = TestClient(app).post("/api/ingest/file", files={"file": ("fake.pdf", b"MZ not a pdf", "text/plain")})
    assert res.status_code == 400
    assert "does not match its .pdf extension" in res.json()["detail"]


# --- B14: flashcards ---

CARDS = '[{"question": "What shape is an orbit?", "answer": "An ellipse.", "category": "Kepler", "difficulty": "Easy", "source": "k.txt p.1"}]'


@pytest.mark.parametrize(
    "reply",
    [
        CARDS,
        f"```json\n{CARDS}\n```",
        f"Here are your flashcards:\n{CARDS}\nGood luck!",
    ],
)
def test_flashcards_parse_wrapped_model_output(api, reply):
    api.generator.complete_text.return_value = reply
    data = TestClient(app).post("/api/flashcards", json={"count": 1}).json()
    assert "fallback" not in data
    assert data["flashcards"][0]["question"] == "What shape is an orbit?"


def test_flashcards_fall_back_to_excerpts_when_the_reply_is_not_json(api):
    api.generator.complete_text.return_value = "Sorry, I cannot help with that."
    data = TestClient(app).post("/api/flashcards", json={"count": 1}).json()
    assert data["fallback"] is True and data["flashcards"]


def test_complete_text_uses_nano_with_thinking_off_and_strips_reasoning(monkeypatch):
    monkeypatch.setenv("NEBIUS_API_KEY", "neb-test-key")
    for name in ("NEBIUS_MODEL", "NEBIUS_FAST_MODEL", "NEBIUS_REASONING_MODEL"):
        monkeypatch.delenv(name, raising=False)
    client = MagicMock()
    client.chat.completions.create.return_value = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=f"<think>planning the cards</think>{CARDS}"))]
    )
    generator = StudyGenerator(client=client)

    assert generator.complete_text("system", "user", max_tokens=900) == CARDS
    call = client.chat.completions.create.call_args.kwargs
    assert "nano" in call["model"].lower()
    assert call["max_tokens"] == 900
    assert call["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}
