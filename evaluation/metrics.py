"""Evaluation Metrics for Academic RAG Systems.

Provides deterministic, reproducible calculations for:
- Context Precision@K
- Context Recall@K
- Mean Reciprocal Rank (MRR)
- Groundedness / Faithfulness
- End-to-End Latency
"""

from dataclasses import dataclass
import re
from typing import Any, Dict, List, Optional, Set


STOP_WORDS: Set[str] = {
    "the", "and", "is", "in", "it", "of", "to", "a", "an", "that",
    "this", "with", "for", "as", "on", "by", "at", "from", "or",
    "are", "was", "were", "be", "been", "have", "has", "had", "do",
    "does", "did", "but", "not", "which", "their", "there", "then"
}


@dataclass
class EvaluationScores:
    """Evaluation scores across all key academic RAG metrics."""

    precision: float
    recall: float
    mrr: float
    faithfulness: float
    latency_ms: float


def compute_context_precision(
    chunks: List[Dict[str, Any]],
    required_facts: List[str],
    k: Optional[int] = None,
) -> float:
    """Calculate Context Precision@K.

    Proportion of retrieved chunks in top-K that contain at least one required fact.
    """
    if not chunks:
        return 0.0

    eval_pool = chunks[:k] if k is not None else chunks
    if not eval_pool:
        return 0.0

    if not required_facts:
        return 1.0

    relevant_count = 0
    for chunk in eval_pool:
        text = (chunk.get("text") or chunk.get("page_content") or "").lower()
        if any(fact.lower() in text for fact in required_facts):
            relevant_count += 1

    return round(relevant_count / len(eval_pool), 4)


def compute_context_recall(
    chunks: List[Dict[str, Any]],
    required_facts: List[str],
    k: Optional[int] = None,
) -> float:
    """Calculate Context Recall@K.

    Proportion of required ground-truth facts present anywhere in the top-K chunks.
    """
    if not required_facts:
        return 1.0
    if not chunks:
        return 0.0

    eval_pool = chunks[:k] if k is not None else chunks
    if not eval_pool:
        return 0.0

    aggregated_text = " ".join(
        (c.get("text") or c.get("page_content") or "").lower()
        for c in eval_pool
    )

    captured_facts = sum(
        1 for fact in required_facts if fact.lower() in aggregated_text
    )

    return round(captured_facts / len(required_facts), 4)


def compute_mrr(
    chunks: List[Dict[str, Any]],
    required_facts: List[str],
    target_source: Optional[str] = None,
) -> float:
    """Calculate Mean Reciprocal Rank (MRR).

    Returns 1.0 / rank of the first chunk that contains required facts (and matches target source).
    """
    if not chunks:
        return 0.0

    for rank, chunk in enumerate(chunks, 1):
        text = (chunk.get("text") or chunk.get("page_content") or "").lower()
        metadata = chunk.get("metadata") or {}
        source = str(metadata.get("source", "")).lower()

        source_matches = True if not target_source else (target_source.lower() in source)
        fact_matches = any(fact.lower() in text for fact in required_facts) if required_facts else True

        if source_matches and fact_matches:
            return round(1.0 / rank, 4)

    return 0.0


def compute_faithfulness(answer: str, context: str) -> float:
    """Calculate Faithfulness / Groundedness score (0.0 to 1.0).

    Verifies the proportion of informative sentences in the answer that are
    substantiated by the retrieved context chunks.
    """
    if not answer or not answer.strip():
        return 0.0
    if not context or not context.strip():
        return 0.0

    clean_context = context.lower()

    # Split into candidate sentences
    raw_sentences = re.split(r"(?<=[.!?])\s+|\n+", answer.strip())
    substantive_sentences: List[str] = []

    for s in raw_sentences:
        s_clean = s.strip()
        # Filter boilerplate markers, headers, and citation lines
        if not s_clean or len(s_clean) < 15:
            continue
        if s_clean.startswith(("===", "#", "* [Source:", "Source:", "**Source:")):
            continue
        substantive_sentences.append(s_clean)

    if not substantive_sentences:
        return 1.0

    supported_count = 0
    for sentence in substantive_sentences:
        words = re.findall(r"[a-zA-Z0-9_\^]{3,}", sentence.lower())
        content_words = [w for w in words if w not in STOP_WORDS]
        if not content_words:
            supported_count += 1
            continue

        # Check keyword support in context
        matches = sum(1 for w in content_words if w in clean_context)
        overlap_ratio = matches / len(content_words)

        if overlap_ratio >= 0.50:
            supported_count += 1

    return round(supported_count / len(substantive_sentences), 4)


def aggregate_scores(scores_list: List[EvaluationScores]) -> Dict[str, float]:
    """Compute statistical mean across a list of EvaluationScores."""
    if not scores_list:
        return {
            "precision": 0.0,
            "recall": 0.0,
            "mrr": 0.0,
            "faithfulness": 0.0,
            "latency_ms": 0.0,
        }

    n = len(scores_list)
    return {
        "precision": round(sum(s.precision for s in scores_list) / n, 4),
        "recall": round(sum(s.recall for s in scores_list) / n, 4),
        "mrr": round(sum(s.mrr for s in scores_list) / n, 4),
        "faithfulness": round(sum(s.faithfulness for s in scores_list) / n, 4),
        "latency_ms": round(sum(s.latency_ms for s in scores_list) / n, 2),
    }
