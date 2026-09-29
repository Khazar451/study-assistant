"""Quantitative Academic Evaluation & Benchmark Suite.

Provides dataset curation, precision/recall/MRR metrics, and comparative benchmarking.
"""

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

__all__ = [
    "BenchmarkRunner",
    "BenchmarkItem",
    "load_default_dataset",
    "EvaluationScores",
    "compute_context_precision",
    "compute_context_recall",
    "compute_mrr",
    "compute_faithfulness",
    "aggregate_scores",
    "generate_markdown_report",
    "save_markdown_report",
    "save_json_results",
]
