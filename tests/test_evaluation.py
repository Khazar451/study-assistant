"""Unit tests for the Quantitative Academic Evaluation & Benchmark Suite."""

import json
from pathlib import Path
import pytest

from evaluation.benchmark import BenchmarkRunner
from evaluation.dataset import BenchmarkItem, load_default_dataset
from evaluation.metrics import (
    EvaluationScores,
    aggregate_scores,
    compute_context_precision,
    compute_context_recall,
    compute_faithfulness,
    compute_mrr,
)
from evaluation.report_generator import (
    generate_markdown_report,
    save_json_results,
    save_markdown_report,
)


# --- 1. Metric Tests ---

def test_context_precision_perfect():
    chunks = [
        {"text": "Kepler discovered elliptical orbits with the Sun at one focus."},
        {"text": "Elliptical orbits have focal points."},
    ]
    facts = ["elliptical", "focus"]
    assert compute_context_precision(chunks, facts, k=2) == 1.0


def test_context_precision_partial():
    chunks = [
        {"text": "Relevant chunk discussing elliptical orbits."},
        {"text": "Completely unrelated text about French history."},
        {"text": "Unrelated paragraph discussing economics."},
        {"text": "Second relevant chunk mentioning focal points."},
    ]
    facts = ["elliptical", "focal points"]
    # 2 out of 4 chunks are relevant
    assert compute_context_precision(chunks, facts, k=4) == 0.5


def test_context_precision_empty_and_zero_division():
    assert compute_context_precision([], ["fact"]) == 0.0
    assert compute_context_precision([{"text": "text"}], [], k=1) == 1.0


def test_context_recall_complete():
    chunks = [
        {"text": "Planets move in elliptical orbits with the Sun at one of the focal points."},
        {"text": "When eccentricity equals zero, the ellipse is a circle."},
    ]
    facts = ["elliptical", "focal points", "eccentricity", "circle"]
    assert compute_context_recall(chunks, facts, k=2) == 1.0


def test_context_recall_partial():
    chunks = [
        {"text": "Only talks about elliptical orbits and focal points."},
    ]
    facts = ["elliptical", "focal points", "semi-major axis", "harmonic period"]
    # 2 out of 4 facts present
    assert compute_context_recall(chunks, facts, k=1) == 0.5


def test_context_recall_empty():
    assert compute_context_recall([], ["fact"]) == 0.0
    assert compute_context_recall([{"text": "sample"}], []) == 1.0


def test_mrr_first_rank():
    chunks = [
        {"text": "Target text containing Kepler's second law.", "metadata": {"source": "kepler.pdf"}},
        {"text": "Other chunk."},
    ]
    assert compute_mrr(chunks, ["Kepler's second law"], target_source="kepler.pdf") == 1.0


def test_mrr_third_rank():
    chunks = [
        {"text": "Irrelevant chunk 1."},
        {"text": "Irrelevant chunk 2."},
        {"text": "Found it here: Kepler's harmonic law.", "metadata": {"source": "notes.txt"}},
    ]
    assert compute_mrr(chunks, ["harmonic law"], target_source="notes.txt") == pytest.approx(0.3333, abs=0.01)


def test_mrr_no_match_or_empty():
    assert compute_mrr([], ["fact"]) == 0.0
    chunks = [{"text": "Nothing relevant here."}]
    assert compute_mrr(chunks, ["missing_fact"]) == 0.0


def test_mrr_source_mismatch_skips():
    chunks = [
        {"text": "Has fact, but wrong source.", "metadata": {"source": "wrong_source.pdf"}},
        {"text": "Has fact with correct source.", "metadata": {"source": "target_source.pdf"}},
    ]
    # Should skip rank 1 because source mismatch, and match rank 2 -> MRR = 0.5
    assert compute_mrr(chunks, ["Has fact"], target_source="target_source.pdf") == 0.5


def test_faithfulness_grounded_answer():
    context = (
        "Kepler's first law states that planets orbit the Sun in elliptical paths, "
        "with the Sun positioned at one focus. Eccentricity measures the elongation."
    )
    answer = (
        "According to Kepler's first law, planets travel in elliptical orbits. "
        "The Sun is located at one focus of the ellipse."
    )
    score = compute_faithfulness(answer, context)
    assert score >= 0.80


def test_faithfulness_hallucinated_answer():
    context = "Kepler's first law states that planets orbit in ellipses."
    answer = (
        "Albert Einstein discovered relativity in Zurich in 1905 with quantum mechanics. "
        "Black holes warp spacetime geometry completely."
    )
    score = compute_faithfulness(answer, context)
    assert score <= 0.20


def test_faithfulness_empty_inputs():
    assert compute_faithfulness("", "Some context") == 0.0
    assert compute_faithfulness("Some answer", "") == 0.0


def test_aggregate_scores():
    scores = [
        EvaluationScores(precision=0.8, recall=1.0, mrr=1.0, faithfulness=0.9, latency_ms=100.0),
        EvaluationScores(precision=0.6, recall=0.8, mrr=0.5, faithfulness=0.9, latency_ms=200.0),
    ]
    agg = aggregate_scores(scores)
    assert agg["precision"] == 0.7
    assert agg["recall"] == 0.9
    assert agg["mrr"] == 0.75
    assert agg["faithfulness"] == 0.9
    assert agg["latency_ms"] == 150.0


def test_aggregate_scores_empty():
    agg = aggregate_scores([])
    assert agg["precision"] == 0.0
    assert agg["recall"] == 0.0


# --- 2. Dataset & Runner Tests ---

def test_benchmark_dataset_integrity():
    dataset = load_default_dataset()
    assert len(dataset) >= 5

    for item in dataset:
        assert isinstance(item, BenchmarkItem)
        assert item.id.startswith("astro_")
        assert len(item.formal_query) > 10
        assert len(item.colloquial_query) > 10
        assert len(item.required_facts) >= 2
        assert item.target_page >= 1


def test_benchmark_runner_mock_mode():
    runner = BenchmarkRunner(mock_mode=True)
    results = runner.run(query_mode="colloquial")

    assert results["total_queries"] == len(runner.dataset)
    assert "summary" in results
    assert "naive_rag" in results["summary"]
    assert "augmented_rag" in results["summary"]
    assert "advanced_rag" in results["summary"]

    # In mock mode, advanced RAG must strictly outperform naive RAG
    naive_rec = results["summary"]["naive_rag"]["recall"]
    adv_rec = results["summary"]["advanced_rag"]["recall"]
    assert adv_rec > naive_rec

    adv_mrr = results["summary"]["advanced_rag"]["mrr"]
    naive_mrr = results["summary"]["naive_rag"]["mrr"]
    assert adv_mrr > naive_mrr


def test_report_generator_markdown_and_json(tmp_path):
    runner = BenchmarkRunner(mock_mode=True)
    results = runner.run(query_mode="formal")

    # Generate Markdown
    md_content = generate_markdown_report(results)
    assert "# 📊 Academic RAG Benchmark Evaluation Report" in md_content
    assert "Executive Comparison" in md_content
    assert "Context Recall@5" in md_content

    # Save Markdown
    md_file = save_markdown_report(md_content, str(tmp_path / "report.md"))
    assert md_file.exists()
    assert len(md_file.read_text(encoding="utf-8")) > 200

    # Save JSON
    json_file = save_json_results(results, str(tmp_path / "results.json"))
    assert json_file.exists()
    loaded_json = json.loads(json_file.read_text(encoding="utf-8"))
    assert loaded_json["total_queries"] == len(runner.dataset)
