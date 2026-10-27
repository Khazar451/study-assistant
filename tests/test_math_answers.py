"""Math answers are not cut off when Super stands in for Ultra, and a cut-off answer ends cleanly."""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from src import providers
from src.api.server import app, set_assistant
from src.assistant import StudyAssistant
from src.generation.generator import CUT_SHORT_NOTE, StudyGenerator

MATH = "Prove the power rule for derivatives"
PLAIN = "What is Kepler's first law?"


def _reply(content: str, finish: str = "stop"):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content), finish_reason=finish)])


def _chunk(text: str = "", finish=None):
    return SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=text), finish_reason=finish)])


@pytest.fixture
def nebius(monkeypatch):
    monkeypatch.setenv("NEBIUS_API_KEY", "neb-test-key")
    for name in ("NEBIUS_MODEL", "NEBIUS_FAST_MODEL", "NEBIUS_REASONING_MODEL", "ENABLE_REASONING_ROUTER"):
        monkeypatch.delenv(name, raising=False)
    providers._reasoning_calls.clear()
    client = MagicMock()
    client.chat.completions.create.return_value = _reply("Answer.")
    yield StudyGenerator(client=client), client
    providers._reasoning_calls.clear()


def _calls(client):
    return [c.kwargs for c in client.chat.completions.create.call_args_list]


def test_capped_math_question_gets_the_large_budget_on_super(nebius, monkeypatch):
    generator, client = nebius
    monkeypatch.setenv("ULTRA_MAX_PER_HOUR", "1")
    generator.generate(MATH, context="[Document 1] x")  # uses the hour's only Ultra call
    generator.generate(MATH, context="[Document 1] x")
    second = _calls(client)[-1]
    assert "super" in second["model"].lower()
    assert second["max_tokens"] >= StudyGenerator.REASONING_MAX_TOKENS


def test_failed_ultra_falls_back_to_super_with_the_large_budget(nebius):
    generator, client = nebius

    def create(**kwargs):
        if "ultra" in kwargs["model"].lower():
            raise TimeoutError("no first token")
        return _reply("Answer.")

    client.chat.completions.create.side_effect = create
    generator.generate(MATH, context="[Document 1] x")
    fallback = _calls(client)[-1]
    assert "super" in fallback["model"].lower()
    assert fallback["max_tokens"] >= StudyGenerator.REASONING_MAX_TOKENS


def test_ordinary_questions_keep_the_normal_budget(nebius):
    generator, client = nebius
    generator.generate(PLAIN, context="[Document 1] x")
    assert _calls(client)[-1]["max_tokens"] == generator.max_tokens


def test_ultra_call_has_its_own_longer_timeout(nebius):
    generator, client = nebius
    generator.generate(MATH, context="[Document 1] x")
    ultra = _calls(client)[0]
    assert "ultra" in ultra["model"].lower()
    assert ultra["timeout"] == StudyGenerator.REASONING_TIMEOUT_S == 300.0


def test_cap_fallback_is_logged_as_a_warning(nebius, monkeypatch, caplog):
    generator, _ = nebius
    monkeypatch.setenv("ULTRA_MAX_PER_HOUR", "1")
    generator.generate(MATH, context="[Document 1] x")
    with caplog.at_level("WARNING", logger="src.generation.generator"):
        generator.generate(MATH, context="[Document 1] x")
    assert any("Hourly cap" in r.getMessage() and r.levelname == "WARNING" for r in caplog.records)


def test_cut_off_stream_closes_the_formula_and_says_so(nebius):
    generator, client = nebius
    client.chat.completions.create.return_value = iter(
        [_chunk("Starting from the limit: "), _chunk("$$\\lim_{z\\to x} \\frac{z^n"), _chunk("", finish="length")]
    )
    text = "".join(generator.generate_stream(PLAIN, context="[Document 1] x"))
    assert text.count("$$") % 2 == 0
    assert text.rstrip().endswith(CUT_SHORT_NOTE)
    assert text.index(CUT_SHORT_NOTE) > text.rindex("$$")  # the note sits after the closed formula


def test_complete_stream_gets_no_note(nebius):
    generator, client = nebius
    client.chat.completions.create.return_value = iter([_chunk("Ellipses. "), _chunk("Done.", finish="stop")])
    assert CUT_SHORT_NOTE not in "".join(generator.generate_stream(PLAIN, context="[Document 1] x"))


def test_cut_off_answer_without_streaming_ends_cleanly(nebius):
    generator, client = nebius
    client.chat.completions.create.return_value = _reply("Using the definition $$\\frac{d}{dx} x^n = ", finish="length")
    answer = generator.generate(PLAIN, context="[Document 1] x")["answer"]
    assert answer.count("$$") % 2 == 0
    assert answer.rstrip().endswith(CUT_SHORT_NOTE)


def test_status_reports_ultra_usage(monkeypatch):
    monkeypatch.setenv("ULTRA_MAX_PER_HOUR", "20")
    providers._reasoning_calls.clear()
    assistant = MagicMock(spec=StudyAssistant)
    assistant.count.return_value = 1
    assistant.persist_directory = "./mock_chroma"
    assistant.generator = MagicMock()
    assistant.retriever = MagicMock()
    assistant.reranker = MagicMock()
    assistant.indexer = MagicMock()
    set_assistant(assistant)
    client = TestClient(app)
    before = client.get("/api/status").json()
    providers.reserve_reasoning_call()
    after = client.get("/api/status").json()
    providers._reasoning_calls.clear()
    assert (before["ultra_calls_last_hour"], before["ultra_cap"]) == (0, 20)
    assert after["ultra_calls_last_hour"] == 1
