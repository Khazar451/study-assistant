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
    mock_assistant.tavily = MagicMock()
    mock_assistant.tavily.is_available = True
    res = client.get("/api/status")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "ready"
    assert data["indexed_chunks"] == 42
    assert "provider" in data
    assert data["tavily_available"] is True


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


@pytest.mark.anyio
async def test_api_ask_stream_disconnect(mock_assistant):
    from unittest.mock import AsyncMock
    from src.api.server import _stream_answer

    mock_assistant.generator.generate_stream.return_value = ["token1 ", "token2 ", "token3 "]
    chunks = [{"text": "Kepler chunk", "metadata": {"source": "kepler.txt", "page": 1}}]

    mock_req = AsyncMock()
    mock_req.is_disconnected.side_effect = [False, True, True]

    events = []
    async for event in _stream_answer(mock_assistant, "query", chunks, "course", request=mock_req):
        events.append(event)

    assert len(events) == 1
    assert "token1" in events[0]
    assert not any("done" in e for e in events)


@pytest.mark.anyio
async def test_ask_stream_does_not_block_other_requests(mock_assistant):
    """A slow retrieval in one stream must not stall unrelated requests on the shared event loop."""
    import asyncio
    import time

    import httpx

    def slow_search(**_):
        time.sleep(2.0)
        return [{"text": "Kepler chunk", "metadata": {"source": "kepler.txt", "page": 1}, "rerank_score": 0.9}]

    mock_assistant.search.side_effect = slow_search
    mock_assistant.generator.generate_stream.return_value = iter(["Kepler."])
    status_done_at: list = []

    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as ac:
        async def ask():
            res = await ac.get("/api/ask/stream", params={"query": "kepler", "augment": "false"})
            assert '"type": "done"' in res.text

        async def status():
            await asyncio.sleep(0.3)  # let the stream reach its slow retrieval first
            res = await ac.get("/api/status")
            status_done_at.append(time.perf_counter() - start)
            assert res.status_code == 200

        # Timed from the start: a blocked loop delays even the sleep above, so status would finish after 2 s.
        start = time.perf_counter()
        await asyncio.gather(ask(), status())

    assert status_done_at[0] < 1.5


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
        session_id=None,
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


def test_api_chat_sessions_crud(client, mock_assistant):
    # 1. Create session
    create_res = client.post("/api/chat/sessions", json={"title": "Calculus Notes"})
    assert create_res.status_code == 200
    sess = create_res.json()
    assert sess["title"] == "Calculus Notes"
    assert "id" in sess
    sid = sess["id"]

    # 2. List sessions
    list_res = client.get("/api/chat/sessions")
    assert list_res.status_code == 200
    sessions = list_res.json()
    assert any(s["id"] == sid for s in sessions)

    # 3. Get session details
    get_res = client.get(f"/api/chat/sessions/{sid}")
    assert get_res.status_code == 200
    sess_details = get_res.json()
    assert sess_details["id"] == sid
    assert sess_details["title"] == "Calculus Notes"
    assert "messages" in sess_details

    # 4. Rename session
    patch_res = client.patch(f"/api/chat/sessions/{sid}", json={"title": "Calculus Exam Prep"})
    assert patch_res.status_code == 200
    assert patch_res.json()["title"] == "Calculus Exam Prep"

    # 5. Delete session
    del_res = client.delete(f"/api/chat/sessions/{sid}")
    assert del_res.status_code == 200
    assert del_res.json()["success"] is True

    # 6. Verify 404 after deletion
    get_404 = client.get(f"/api/chat/sessions/{sid}")
    assert get_404.status_code == 404




def test_clear_blocked_in_demo_mode(client, mock_assistant, monkeypatch):
    monkeypatch.setenv("DEMO_MODE", "true")
    monkeypatch.delenv("ADMIN_TOKEN", raising=False)
    assert client.post("/api/clear").status_code == 403
    mock_assistant.clear.assert_not_called()


def test_clear_requires_admin_token_when_configured(client, mock_assistant, monkeypatch):
    monkeypatch.setenv("ADMIN_TOKEN", "s3cret")
    assert client.post("/api/clear").status_code == 403
    assert client.post("/api/clear", headers={"X-Admin-Token": "wrong"}).status_code == 403
    mock_assistant.count.return_value = 0
    assert client.post("/api/clear", headers={"X-Admin-Token": "s3cret"}).status_code == 200


def test_rate_limit_blocks_llm_endpoint(client, monkeypatch):
    from src.api import server

    server._rate_hits.clear()
    monkeypatch.setenv("RATE_LIMIT_PER_MIN", "2")
    statuses = [client.post("/api/search", json={"query": "orbits"}).status_code for _ in range(3)]
    assert statuses[:2] == [200, 200]
    assert statuses[2] == 429
    server._rate_hits.clear()


@pytest.fixture
def fresh_limits(monkeypatch):
    """Empty limiter state and no limit settings, restored after the test."""
    from src.api import server

    for name in ("RATE_LIMIT_PER_MIN", "TRUSTED_PROXY_HOPS", "DEMO_MAX_PER_MIN", "DEMO_MAX_PER_DAY", "DEMO_PAUSED"):
        monkeypatch.delenv(name, raising=False)
    server._rate_hits.clear()
    server._global_hits.clear()
    yield server
    server._rate_hits.clear()
    server._global_hits.clear()


def test_rate_limit_ignores_spoofed_forwarded_for(client, fresh_limits, monkeypatch):
    # The proxy appends the real address last; whatever the client sent comes before it.
    monkeypatch.setenv("RATE_LIMIT_PER_MIN", "2")
    monkeypatch.setenv("TRUSTED_PROXY_HOPS", "1")
    statuses = [
        client.post(
            "/api/search", json={"query": "orbits"}, headers={"X-Forwarded-For": f"10.0.0.{i}, 203.0.113.7"}
        ).status_code
        for i in range(3)
    ]
    assert statuses == [200, 200, 429]


def test_rate_limit_table_stays_bounded(client, fresh_limits, monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_PER_MIN", "5")
    monkeypatch.setenv("TRUSTED_PROXY_HOPS", "1")
    monkeypatch.setattr(fresh_limits, "_MAX_RATE_KEYS", 10)
    for i in range(40):
        client.post("/api/search", json={"query": "orbits"}, headers={"X-Forwarded-For": f"198.51.100.{i}"})
    assert len(fresh_limits._rate_hits) <= 10


def test_global_daily_cap_returns_503(client, fresh_limits, monkeypatch):
    monkeypatch.setenv("DEMO_MAX_PER_DAY", "2")
    responses = [
        client.post("/api/search", json={"query": "orbits"}, headers={"X-Forwarded-For": f"198.51.100.{i}"})
        for i in range(3)
    ]
    assert [r.status_code for r in responses] == [200, 200, 503]
    assert "daily question limit" in responses[2].json()["detail"]
    assert int(responses[2].headers["Retry-After"]) > 0


def test_global_minute_cap_returns_503(client, fresh_limits, monkeypatch):
    monkeypatch.setenv("DEMO_MAX_PER_MIN", "1")
    assert client.post("/api/search", json={"query": "orbits"}).status_code == 200
    busy = client.post("/api/search", json={"query": "orbits"})
    assert busy.status_code == 503
    assert "busy" in busy.json()["detail"]


def test_refused_requests_do_not_spend_the_global_budget(client, fresh_limits, monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_PER_MIN", "1")
    monkeypatch.setenv("DEMO_MAX_PER_DAY", "5")
    for _ in range(4):
        client.post("/api/search", json={"query": "orbits"})
    assert len(fresh_limits._global_hits) == 1


def test_demo_paused_blocks_spend_endpoints_only(client, fresh_limits, monkeypatch):
    monkeypatch.setenv("DEMO_PAUSED", "true")
    paused = client.post("/api/ask", json={"query": "orbits"})
    assert paused.status_code == 503
    assert "paused" in paused.json()["detail"]
    assert client.get("/api/status").status_code == 200


def test_sample_ingest_is_rate_limited(client, fresh_limits, monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_PER_MIN", "1")
    assert client.post("/api/ingest/sample").status_code == 200
    assert client.post("/api/ingest/sample").status_code == 429


def test_sample_ingest_skips_when_already_indexed(client, mock_assistant):
    mock_assistant.indexer.collection.get.return_value = {"ids": ["sample_0"]}
    res = client.post("/api/ingest/sample")
    assert res.status_code == 200
    assert res.json()["result"]["status"] == "already_indexed"
    mock_assistant.ingest.assert_not_called()


def test_oversized_question_is_rejected(client, mock_assistant, fresh_limits):
    from src.api.server import MAX_QUERY_CHARS

    too_long = "x" * (MAX_QUERY_CHARS + 1)
    assert client.post("/api/ask", json={"query": too_long}).status_code == 422
    assert client.post("/api/search", json={"query": too_long}).status_code == 422
    assert client.get("/api/ask/stream", params={"query": too_long}).status_code == 422
    mock_assistant.ask.assert_not_called()
    assert client.post("/api/ask", json={"query": "x" * MAX_QUERY_CHARS}).status_code == 200
