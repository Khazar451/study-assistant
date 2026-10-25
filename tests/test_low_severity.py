"""Low-severity fixes: client timeouts, retry budget, score keys, None IDs, CORS default, /metrics guard."""

import json
from pathlib import Path
import subprocess
import sys
import uuid
from unittest.mock import MagicMock

import chromadb
import pytest
from fastapi.testclient import TestClient

from src.api import server
from src.api.server import OfflineFallbackReranker, app, set_assistant
from src.assistant import StudyAssistant
from src.generation.generator import StudyGenerator
from src.ingestion import embedder as embedder_module
from src.ingestion.embedder import NvidiaEmbedder
from src.ingestion.indexer import ChromaIndexer
from src.ingestion.vision import VisionDescriber
from src.retrieval.query_augmenter import QueryAugmenter

REPO = Path(__file__).resolve().parent.parent


def _read_timeout(client) -> float:
    timeout = client.timeout
    return float(getattr(timeout, "read", timeout))


# --- B17 ---


@pytest.mark.parametrize(
    "build, seconds",
    [
        (lambda: StudyGenerator(api_key="k", base_url="https://api.tokenfactory.nebius.com/v1/"), 120.0),
        (lambda: QueryAugmenter(api_key="k", base_url="https://api.tokenfactory.nebius.com/v1/"), 20.0),
        (lambda: VisionDescriber(api_key="k"), 60.0),
    ],
)
def test_llm_clients_have_explicit_timeouts(build, seconds):
    client = build().client
    assert _read_timeout(client) == seconds
    assert client.max_retries == 1


# --- B18 ---


def test_query_embedding_gives_up_after_three_attempts(monkeypatch):
    monkeypatch.setattr(embedder_module.time, "sleep", lambda _s: None)
    embedder = NvidiaEmbedder(api_key="k")
    embedder.client = MagicMock()
    embedder.client.embeddings.create.side_effect = TimeoutError("stalled")
    with pytest.raises(TimeoutError):
        embedder.embed_query("Kepler")
    assert embedder.client.embeddings.create.call_count == 3


def test_query_embedding_stops_retrying_when_the_budget_is_spent(monkeypatch):
    monkeypatch.setattr(embedder_module.time, "sleep", lambda _s: None)
    embedder = NvidiaEmbedder(api_key="k")
    embedder.QUERY_BUDGET = 0.0
    embedder.client = MagicMock()
    embedder.client.embeddings.create.side_effect = TimeoutError("stalled")
    with pytest.raises(TimeoutError):
        embedder.embed_query("Kepler")
    assert embedder.client.embeddings.create.call_count == 1


# --- B19 ---


def test_offline_reranker_orders_by_retriever_similarity():
    chunks = [{"text": t, "similarity_score": s} for t, s in (("low", 0.2), ("high", 0.9), ("mid", 0.5))]
    ranked = OfflineFallbackReranker().rerank("q", chunks, top_n=3)
    assert [c["text"] for c in ranked] == ["high", "mid", "low"]
    assert [c["rerank_score"] for c in ranked] == [0.9, 0.5, 0.2]


def test_stream_previews_show_the_similarity_score(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    for name in ("WORKSPACE_ISOLATION", "DEMO_MODE", "RATE_LIMIT_PER_MIN", "DEMO_MAX_PER_MIN", "DEMO_MAX_PER_DAY"):
        monkeypatch.delenv(name, raising=False)
    assistant = MagicMock(spec=StudyAssistant)
    assistant.count.return_value = 1
    assistant.generator = MagicMock()
    assistant.generator.generate_stream.return_value = iter(["Kepler."])
    assistant.retriever = MagicMock()
    assistant.retriever.augmenter = None
    assistant.retriever.format_context.return_value = "[Document 1] (Source: k.txt | Page: 1)\nKepler"
    assistant.augmenter = None
    assistant.search.return_value = [{"text": "Kepler", "metadata": {"source": "k.txt", "page": 1}, "similarity_score": 0.77}]
    set_assistant(assistant)

    body = TestClient(app).get("/api/ask/stream", params={"query": "orbits", "augment": "false"}).text
    chunks_event = next(json.loads(line[6:]) for line in body.splitlines() if '"type": "chunks"' in line)
    assert chunks_event["chunks"][0]["score"] == 0.77
    assert '"type": "done"' in body


# --- B20 ---


def test_chunks_with_a_none_id_get_their_own_ids():
    indexer = ChromaIndexer(collection_name=f"ids-{uuid.uuid4().hex[:8]}", client=chromadb.EphemeralClient())
    indexer.add_chunks([{"text": f"chunk {i}", "id": None, "embedding": [0.1 * (i + 1), 0.2]} for i in range(3)])
    assert indexer.count() == 3
    assert "None" not in indexer.collection.get()["ids"]


# --- B21 ---


def test_no_cross_origin_access_by_default():
    """Checked in a fresh interpreter: CORS is configured when the server module is imported."""
    code = (
        "from fastapi.testclient import TestClient\n"
        "from src.api.server import app\n"
        "c = TestClient(app)\n"
        "simple = c.get('/api/status', headers={'Origin': 'https://evil.example'})\n"
        "pre = c.options('/api/ask', headers={'Origin': 'https://evil.example', 'Access-Control-Request-Method': 'POST'})\n"
        "print(simple.headers.get('access-control-allow-origin'), pre.headers.get('access-control-allow-origin'))\n"
    )
    env = {**__import__("os").environ, "CORS_ORIGINS": "", "NEBIUS_API_KEY": "", "NVIDIA_API_KEY": "",
           "NVIDIA_EMBEDDER_KEY": "", "TAVILY_API_KEY": "", "AUTO_INGEST_SAMPLE": "false"}
    out = subprocess.run([sys.executable, "-c", code], cwd=REPO, env=env, capture_output=True, text=True, timeout=180)
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.split()[-2:] == ["None", "None"]


# --- B22 ---


def test_metrics_closed_in_the_demo_without_a_token(monkeypatch):
    monkeypatch.setenv("DEMO_MODE", "true")
    monkeypatch.delenv("ADMIN_TOKEN", raising=False)
    assert TestClient(app).get("/metrics").status_code == 403


def test_metrics_need_the_admin_token_when_one_is_set(monkeypatch):
    monkeypatch.setenv("ADMIN_TOKEN", "s3cret")
    client = TestClient(app)
    assert client.get("/metrics").status_code == 403
    assert client.get("/metrics", headers={"X-Admin-Token": "s3cret"}).status_code == 200
