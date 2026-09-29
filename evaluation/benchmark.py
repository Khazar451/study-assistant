"""Benchmark Runner for Academic RAG Systems.

Executes comparative benchmarks across:
1. Naive Baseline RAG (Vector-only, K=5, No Augmentation, No Rerank)
2. Augmented RAG (Multi-Query Expansion, K=15 -> Top-5 by Cosine, No Rerank)
3. Full Advanced RAG (Multi-Query Expansion + Semantic Cross-Encoder Reranker)
"""

import argparse
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

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
from src.assistant import StudyAssistant


class BenchmarkRunner:
    """Orchestrates multi-pipeline comparative evaluation across academic benchmarks."""

    def __init__(
        self,
        assistant: Optional[StudyAssistant] = None,
        dataset: Optional[List[BenchmarkItem]] = None,
        mock_mode: bool = False,
    ):
        self.mock_mode = mock_mode
        self.dataset = dataset or load_default_dataset()
        if not mock_mode:
            self.assistant = assistant or StudyAssistant()
        else:
            self.assistant = None

    def _mock_pipeline_eval(
        self, item: BenchmarkItem, query: str, config: str
    ) -> Tuple[EvaluationScores, Dict[str, Any]]:
        """Simulate realistic pipeline execution for fast offline tests & CI."""
        is_colloquial = query == item.colloquial_query

        if config == "naive_rag":
            # Naive fails significantly on colloquial terms due to vocabulary gap
            recall = 0.50 if is_colloquial else 0.75
            precision = 0.40 if is_colloquial else 0.60
            mrr = 0.33 if is_colloquial else 0.50
            faithfulness = 0.90
            latency = 120.0
            chunks = [{"text": f"Sample naive chunk for {item.required_facts[0]}"}]
            answer = f"Naive answer discussing {item.required_facts[0]}."

        elif config == "augmented_rag":
            # Augmentation fixes recall on colloquial, but precision suffers without reranking
            recall = 0.95
            precision = 0.55
            mrr = 0.65
            faithfulness = 0.92
            latency = 280.0
            chunks = [{"text": f"Augmented chunk with {f}"} for f in item.required_facts]
            answer = f"Augmented answer explaining {', '.join(item.required_facts)}."

        else:  # advanced_rag
            # Full system achieves high recall AND high precision + high MRR
            recall = 1.00
            precision = 0.85
            mrr = 0.95
            faithfulness = 0.98
            latency = 450.0
            chunks = [{"text": f"Reranked top chunk with {f}"} for f in item.required_facts]
            answer = f"Advanced grounded answer strictly detailing {', '.join(item.required_facts)}."

        scores = EvaluationScores(
            precision=precision,
            recall=recall,
            mrr=mrr,
            faithfulness=faithfulness,
            latency_ms=latency,
        )
        metadata = {
            "chunks_count": len(chunks),
            "answer_preview": answer[:80] + "...",
        }
        return scores, metadata

    def evaluate_query(
        self, item: BenchmarkItem, query: str, config: str
    ) -> Tuple[EvaluationScores, Dict[str, Any]]:
        """Evaluate a single query under a specific pipeline configuration."""
        if self.mock_mode:
            return self._mock_pipeline_eval(item, query, config)

        t_start = time.perf_counter()

        if config == "naive_rag":
            # 1. Naive Baseline: No augmentation, K=5, direct cosine search
            chunks = self.assistant.retriever.retrieve(
                query=query,
                top_k=5,
                score_threshold=None,
                augment=False,
            )
            context = self.assistant.retriever.format_context(chunks)
            gen_res = self.assistant.generator.generate(query=query, context=context)
            answer = gen_res.get("answer", "")

        elif config == "augmented_rag":
            # 2. Augmented: Multi-query expansion, K=15, top-5 by raw cosine (no reranker)
            raw_candidates = self.assistant.retriever.retrieve(
                query=query,
                top_k=15,
                score_threshold=None,
                augment=True,
                augment_mode="expand",
            )
            chunks = raw_candidates[:5]
            context = self.assistant.retriever.format_context(chunks)
            gen_res = self.assistant.generator.generate(query=query, context=context)
            answer = gen_res.get("answer", "")

        elif config == "advanced_rag":
            # 3. Full Advanced System: Query expansion + Semantic Reranker
            chunks = self.assistant.search(
                query=query,
                top_k=15,
                top_n=5,
                augment=True,
                augment_mode="expand",
            )
            context = self.assistant.retriever.format_context(chunks)
            gen_res = self.assistant.generator.generate(query=query, context=context)
            answer = gen_res.get("answer", "")

        else:
            raise ValueError(f"Unknown pipeline configuration: {config}")

        latency_ms = (time.perf_counter() - t_start) * 1000.0

        precision = compute_context_precision(chunks, item.required_facts, k=5)
        recall = compute_context_recall(chunks, item.required_facts, k=5)
        mrr = compute_mrr(chunks, item.required_facts, target_source=item.target_source)
        faithfulness = compute_faithfulness(answer, context)

        scores = EvaluationScores(
            precision=precision,
            recall=recall,
            mrr=mrr,
            faithfulness=faithfulness,
            latency_ms=round(latency_ms, 2),
        )

        metadata = {
            "chunks_count": len(chunks),
            "answer_preview": answer[:100] + "..." if answer else "",
        }
        return scores, metadata

    def run(self, query_mode: str = "both") -> Dict[str, Any]:
        """Execute the full benchmark suite across all configurations.

        Args:
            query_mode: 'formal', 'colloquial', or 'both'.
        """
        configurations = ["naive_rag", "augmented_rag", "advanced_rag"]
        results_by_config: Dict[str, List[EvaluationScores]] = {cfg: [] for cfg in configurations}
        results_by_category: Dict[str, Dict[str, List[EvaluationScores]]] = {}
        query_logs: List[Dict[str, Any]] = []

        queries_to_test: List[Tuple[BenchmarkItem, str, str]] = []
        for item in self.dataset:
            if query_mode in ("formal", "both"):
                queries_to_test.append((item, item.formal_query, "formal"))
            if query_mode in ("colloquial", "both"):
                queries_to_test.append((item, item.colloquial_query, "colloquial"))

        print(f"\n🚀 Running Academic RAG Benchmark ({len(queries_to_test)} test cases)...")
        print("=" * 65)

        for idx, (item, query, phrasing) in enumerate(queries_to_test, 1):
            category_key = f"{item.category} ({phrasing})"
            if category_key not in results_by_category:
                results_by_category[category_key] = {cfg: [] for cfg in configurations}

            print(f"[{idx}/{len(queries_to_test)}] Evaluating: \"{query[:45]}...\"")
            item_log: Dict[str, Any] = {
                "id": item.id,
                "category": item.category,
                "phrasing": phrasing,
                "subject": item.subject,
                "query": query,
                "target_source": item.target_source,
                "target_page": item.target_page,
                "required_facts": item.required_facts,
                "results": {},
            }

            for cfg in configurations:
                scores, meta = self.evaluate_query(item, query, cfg)
                results_by_config[cfg].append(scores)
                results_by_category[category_key][cfg].append(scores)

                item_log["results"][cfg] = {
                    "recall": scores.recall,
                    "precision": scores.precision,
                    "mrr": scores.mrr,
                    "faithfulness": scores.faithfulness,
                    "latency_ms": scores.latency_ms,
                    "meta": meta,
                }

            query_logs.append(item_log)

        # Aggregate overall stats
        summary_by_config: Dict[str, Dict[str, float]] = {}
        for cfg in configurations:
            summary_by_config[cfg] = aggregate_scores(results_by_config[cfg])

        # Aggregate category stats
        category_summary: Dict[str, Dict[str, Dict[str, float]]] = {}
        for cat, cfgs in results_by_category.items():
            category_summary[cat] = {
                cfg: aggregate_scores(scores_list) for cfg, scores_list in cfgs.items()
            }

        benchmark_payload = {
            "total_queries": len(queries_to_test),
            "summary": summary_by_config,
            "by_category": category_summary,
            "queries": query_logs,
        }

        return benchmark_payload


def main():
    """CLI entrypoint for running the academic evaluation benchmark."""
    parser = argparse.ArgumentParser(
        description="Academic RAG Evaluation & Benchmark Suite for Study Assistant"
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run in offline mock mode (instantaneous, zero API calls)",
    )
    parser.add_argument(
        "--query-mode",
        type=str,
        default="both",
        choices=["formal", "colloquial", "both"],
        help="Which query phrasing variations to evaluate (default: both)",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="evaluation",
        help="Directory to save report and json files (default: evaluation)",
    )

    args = parser.parse_args()

    runner = BenchmarkRunner(mock_mode=args.mock)
    results = runner.run(query_mode=args.query_mode)

    # Generate and save reports
    md_report = generate_markdown_report(results)
    md_path = save_markdown_report(md_report, f"{args.out_dir}/benchmark_report.md")
    json_path = save_json_results(results, f"{args.out_dir}/benchmark_results.json")

    print("\n" + "=" * 65)
    print("✅ BENCHMARK COMPLETE!")
    print(f"📄 Markdown Report: {md_path}")
    print(f"📊 JSON Dataset:    {json_path}")
    print("=" * 65 + "\n")
    print(md_report)


if __name__ == "__main__":
    main()
