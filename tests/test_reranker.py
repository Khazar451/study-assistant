import copy
import math
from unittest.mock import MagicMock, patch
import pytest

from src.retrieval.reranker import StudyReranker


@pytest.fixture
def sample_chunks():
    return [
        {
            "id": "chunk_1",
            "text": "Cellular respiration occurs in mitochondria.",
            "similarity_score": 0.85,
            "metadata": {"source": "bio.pdf", "page": 10},
        },
        {
            "id": "chunk_2",
            "text": "Photosynthesis occurs in chloroplasts and generates glucose.",
            "similarity_score": 0.65,
            "metadata": {"source": "bio.pdf", "page": 12},
        },
        {
            "id": "chunk_3",
            "text": "Newton's laws govern classical mechanics.",
            "similarity_score": 0.40,
            "metadata": {"source": "physics.pdf", "page": 4},
        },
    ]


def test_missing_api_keys_raises_value_error(monkeypatch):
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.delenv("NEBIUS_API_KEY", raising=False)

    with pytest.raises(ValueError, match="No API key found"):
        StudyReranker()


def test_explicit_api_key_override(monkeypatch):
    monkeypatch.setenv("NVIDIA_API_KEY", "env_key")
    reranker = StudyReranker(api_key="explicit_key", client=MagicMock())
    assert reranker.api_key == "explicit_key"
    assert reranker.llm_api_key == "explicit_key"


def test_key_precedence():
    reranker = StudyReranker(
        ranking_api_key="ranking_special",
        llm_api_key="llm_special",
        client=MagicMock(),
    )
    assert reranker.ranking_api_key == "ranking_special"
    assert reranker.llm_api_key == "llm_special"


def test_rerank_llm_listwise_success(sample_chunks):
    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.choices = [
        MagicMock(
            message=MagicMock(
                content='[{"id": "chunk_2", "relevance_score": 0.95}, {"id": "chunk_1", "relevance_score": 0.40}, {"id": "chunk_3", "relevance_score": 0.05}]'
            )
        )
    ]
    mock_client.chat.completions.create.return_value = mock_resp

    reranker = StudyReranker(api_key="test_key", client=mock_client, mode="llm_listwise")
    results = reranker.rerank(query="How does photosynthesis work?", chunks=sample_chunks)

    assert len(results) == 3
    # chunk_2 should be ranked first because it scored 0.95
    assert results[0]["id"] == "chunk_2"
    assert results[0]["rerank_score"] == 0.95
    assert results[1]["id"] == "chunk_1"
    assert results[2]["id"] == "chunk_3"


def test_rerank_llm_listwise_markdown_code_fences(sample_chunks):
    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.choices = [
        MagicMock(
            message=MagicMock(
                content='Here is the evaluation:\n```json\n[{"id": "chunk_1", "relevance_score": 0.90}, {"id": "chunk_2", "relevance_score": 0.10}]\n```'
            )
        )
    ]
    mock_client.chat.completions.create.return_value = mock_resp

    reranker = StudyReranker(api_key="test_key", client=mock_client)
    results = reranker.rerank(query="Mitochondria function", chunks=sample_chunks)

    assert results[0]["id"] == "chunk_1"
    assert results[0]["rerank_score"] == 0.90


def test_omitted_id_monotonic_imputation(sample_chunks):
    mock_client = MagicMock()
    # LLM only scores chunk_1 and chunk_3, omitting chunk_2 entirely
    mock_resp = MagicMock()
    mock_resp.choices = [
        MagicMock(
            message=MagicMock(
                content='[{"id": "chunk_1", "relevance_score": 0.80}, {"id": "chunk_3", "relevance_score": 0.30}]'
            )
        )
    ]
    mock_client.chat.completions.create.return_value = mock_resp

    reranker = StudyReranker(api_key="test_key", client=mock_client)
    results = reranker.rerank(query="Cell biology", chunks=sample_chunks)

    scored_ids = [r["id"] for r in results]
    assert "chunk_2" in scored_ids

    # Verify chunk_2 was imputed strictly below the minimum scored chunk (chunk_3 = 0.30)
    chunk_2_res = next(r for r in results if r["id"] == "chunk_2")
    chunk_3_res = next(r for r in results if r["id"] == "chunk_3")
    assert chunk_2_res["rerank_score"] < chunk_3_res["rerank_score"]
    assert chunk_2_res["rerank_score"] == pytest.approx(0.30 - 1e-4, rel=1e-5)


def test_hallucinated_id_ignored(sample_chunks):
    mock_client = MagicMock()
    mock_resp = MagicMock()
    # Returns chunk_phantom which does not exist in input
    mock_resp.choices = [
        MagicMock(
            message=MagicMock(
                content='[{"id": "chunk_phantom", "relevance_score": 0.99}, {"id": "chunk_1", "relevance_score": 0.70}]'
            )
        )
    ]
    mock_client.chat.completions.create.return_value = mock_resp

    reranker = StudyReranker(api_key="test_key", client=mock_client)
    results = reranker.rerank(query="Cell biology", chunks=sample_chunks)

    assert not any(r["id"] == "chunk_phantom" for r in results)
    assert results[0]["id"] == "chunk_1"


def test_unparseable_json_fallback(sample_chunks):
    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock(message=MagicMock(content="I cannot rank these documents."))]
    mock_client.chat.completions.create.return_value = mock_resp

    reranker = StudyReranker(api_key="test_key", client=mock_client)
    results = reranker.rerank(query="Cell biology", chunks=sample_chunks)

    # Falls back to original similarity_score ranking (chunk_1: 0.85, chunk_2: 0.65, chunk_3: 0.40)
    assert results[0]["id"] == "chunk_1"
    assert results[1]["id"] == "chunk_2"
    assert results[2]["id"] == "chunk_3"
    assert results[0]["rerank_score"] == 0.85


def test_missing_similarity_score_fallback():
    chunks = [
        {"id": "c1", "text": "First chunk"},  # No similarity_score
        {"id": "c2", "text": "Second chunk", "similarity_score": 0.7},
    ]
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_effect = Exception("API down")

    reranker = StudyReranker(api_key="test_key", client=mock_client, max_retries=0)
    results = reranker.rerank(query="query", chunks=chunks)

    # c2 has similarity 0.7, c1 defaults to 0.0
    assert results[0]["id"] == "c2"
    assert results[1]["id"] == "c1"
    assert results[1]["rerank_score"] == 0.0


def test_stable_sort_tiebreaker():
    chunks = [
        {"id": "a", "text": "Alpha text"},
        {"id": "b", "text": "Beta text"},
        {"id": "c", "text": "Gamma text"},
    ]
    mock_client = MagicMock()
    # All chunks get identical score
    mock_resp = MagicMock()
    mock_resp.choices = [
        MagicMock(
            message=MagicMock(
                content='[{"id": "a", "relevance_score": 0.5}, {"id": "b", "relevance_score": 0.5}, {"id": "c", "relevance_score": 0.5}]'
            )
        )
    ]
    mock_client.chat.completions.create.return_value = mock_resp

    reranker = StudyReranker(api_key="test_key", client=mock_client)
    results = reranker.rerank(query="test", chunks=chunks)

    # Stable sort preserves original order [a, b, c]
    assert [r["id"] for r in results] == ["a", "b", "c"]


def test_min_score_filter(sample_chunks):
    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.choices = [
        MagicMock(
            message=MagicMock(
                content='[{"id": "chunk_1", "relevance_score": 0.90}, {"id": "chunk_2", "relevance_score": 0.50}, {"id": "chunk_3", "relevance_score": 0.10}]'
            )
        )
    ]
    mock_client.chat.completions.create.return_value = mock_resp

    reranker = StudyReranker(api_key="test_key", client=mock_client)
    results = reranker.rerank(query="biology", chunks=sample_chunks, min_score=0.60)

    # Only chunk_1 (0.90) passes the min_score threshold of 0.60
    assert len(results) == 1
    assert results[0]["id"] == "chunk_1"


def test_min_score_eliminates_all_returns_empty(sample_chunks):
    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.choices = [
        MagicMock(
            message=MagicMock(
                content='[{"id": "chunk_1", "relevance_score": 0.40}, {"id": "chunk_2", "relevance_score": 0.30}]'
            )
        )
    ]
    mock_client.chat.completions.create.return_value = mock_resp

    reranker = StudyReranker(api_key="test_key", client=mock_client)
    results = reranker.rerank(query="astronomy", chunks=sample_chunks, min_score=0.80)

    # All chunks scored below 0.80 -> returns [] without padding
    assert results == []


def test_top_n_truncation(sample_chunks):
    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.choices = [
        MagicMock(
            message=MagicMock(
                content='[{"id": "chunk_1", "relevance_score": 0.90}, {"id": "chunk_2", "relevance_score": 0.80}, {"id": "chunk_3", "relevance_score": 0.70}]'
            )
        )
    ]
    mock_client.chat.completions.create.return_value = mock_resp

    reranker = StudyReranker(api_key="test_key", client=mock_client, top_n=2)
    results = reranker.rerank(query="biology", chunks=sample_chunks, top_n=2)

    assert len(results) == 2
    assert results[0]["id"] == "chunk_1"
    assert results[1]["id"] == "chunk_2"


def test_empty_inputs(sample_chunks):
    reranker = StudyReranker(api_key="test_key", client=MagicMock())
    assert reranker.rerank("", sample_chunks) == []
    assert reranker.rerank("   ", sample_chunks) == []
    assert reranker.rerank("query", []) == []


def test_immutability(sample_chunks):
    original_copy = copy.deepcopy(sample_chunks)
    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.choices = [
        MagicMock(message=MagicMock(content='[{"id": "chunk_1", "relevance_score": 0.9}]'))
    ]
    mock_client.chat.completions.create.return_value = mock_resp

    reranker = StudyReranker(api_key="test_key", client=mock_client)
    results = reranker.rerank(query="biology", chunks=sample_chunks)

    # Input sample_chunks must not be mutated
    assert "rerank_score" not in sample_chunks[0]
    assert sample_chunks == original_copy
    # Output must have rerank_score
    assert "rerank_score" in results[0]


def test_transient_retry_429(sample_chunks):
    mock_client = MagicMock()
    mock_resp = MagicMock()
    mock_resp.choices = [
        MagicMock(message=MagicMock(content='[{"id": "chunk_1", "relevance_score": 0.95}]'))
    ]
    # First attempt raises 429 RateLimit, second attempt succeeds
    mock_client.chat.completions.create.side_effect = [
        Exception("Error code: 429 - Rate limit reached"),
        mock_resp,
    ]

    reranker = StudyReranker(api_key="test_key", client=mock_client, max_retries=1)
    with patch("time.sleep", return_value=None):
        results = reranker.rerank(query="biology", chunks=sample_chunks)

    assert len(results) > 0
    assert results[0]["id"] == "chunk_1"
    assert results[0]["rerank_score"] == 0.95
    assert mock_client.chat.completions.create.call_count == 2


@patch("requests.post")
def test_ranking_api_index_mapping_and_sigmoid(mock_post, sample_chunks):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    # NVIDIA ranking API returns indices and raw logits: index 1 gets +2.0, index 0 gets -2.0
    mock_resp.json.return_value = {
        "rankings": [
            {"index": 1, "logit": 2.0},
            {"index": 0, "logit": -2.0},
        ]
    }
    mock_post.return_value = mock_resp

    reranker = StudyReranker(api_key="test_key", mode="ranking_api")
    results = reranker.rerank(query="photosynthesis", chunks=sample_chunks)

    # index 1 maps to chunk_2 (Photosynthesis)
    assert results[0]["id"] == "chunk_2"
    expected_sigmoid_chunk_2 = round(1.0 / (1.0 + math.exp(-2.0)), 4)
    assert results[0]["rerank_score"] == expected_sigmoid_chunk_2

    # index 0 maps to chunk_1
    assert results[1]["id"] == "chunk_1"
    expected_sigmoid_chunk_1 = round(1.0 / (1.0 + math.exp(2.0)), 4)
    assert results[1]["rerank_score"] == expected_sigmoid_chunk_1


@patch("requests.post")
def test_negative_ranking_logits_do_not_outrank_scored(mock_post, sample_chunks):
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    # Both scored chunks have negative logits: chunk 0 has -1.0, chunk 1 has -5.0. chunk 2 is omitted.
    mock_resp.json.return_value = {
        "rankings": [
            {"index": 0, "logit": -1.0},
            {"index": 1, "logit": -5.0},
        ]
    }
    mock_post.return_value = mock_resp

    reranker = StudyReranker(api_key="test_key", mode="ranking_api")
    results = reranker.rerank(query="negative logit test", chunks=sample_chunks)

    chunk_3_res = next(r for r in results if r["id"] == "chunk_3")
    chunk_2_res = next(r for r in results if r["id"] == "chunk_2")

    # chunk_3 (omitted) must strictly rank below chunk_2 (lowest scored chunk)
    assert chunk_3_res["rerank_score"] < chunk_2_res["rerank_score"]


@patch("requests.post")
def test_auto_mode_falls_back_on_404(mock_post, sample_chunks):
    # ranking API returns 404
    mock_post_resp = MagicMock()
    mock_post_resp.status_code = 404
    mock_post_resp.text = "Function not found"
    mock_post.return_value = mock_post_resp

    # LLM listwise mock
    mock_client = MagicMock()
    mock_llm_resp = MagicMock()
    mock_llm_resp.choices = [
        MagicMock(message=MagicMock(content='[{"id": "chunk_3", "relevance_score": 0.99}]'))
    ]
    mock_client.chat.completions.create.return_value = mock_llm_resp

    reranker = StudyReranker(api_key="test_key", mode="auto", client=mock_client)
    results = reranker.rerank(query="physics", chunks=sample_chunks)

    # Succeeded via LLM listwise fallback
    assert results[0]["id"] == "chunk_3"
    assert results[0]["rerank_score"] == 0.99


@patch("requests.post")
def test_auto_mode_does_not_retry_400(mock_post, sample_chunks):
    # 400 Bad Request should not be retried
    mock_post_resp = MagicMock()
    mock_post_resp.status_code = 400
    mock_post_resp.text = "Bad Request schema"
    mock_post.return_value = mock_post_resp

    mock_client = MagicMock()
    mock_llm_resp = MagicMock()
    mock_llm_resp.choices = [
        MagicMock(message=MagicMock(content='[{"id": "chunk_1", "relevance_score": 0.88}]'))
    ]
    mock_client.chat.completions.create.return_value = mock_llm_resp

    reranker = StudyReranker(api_key="test_key", mode="auto", client=mock_client, max_retries=2)
    results = reranker.rerank(query="biology", chunks=sample_chunks)

    # Verify requests.post was called only once (no retries on 400)
    assert mock_post.call_count == 1
    assert results[0]["id"] == "chunk_1"
