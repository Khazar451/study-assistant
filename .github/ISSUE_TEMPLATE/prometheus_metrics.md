---
name: Prometheus Metrics & Observability
about: Add GET /metrics endpoint and RAG telemetry for Prometheus and Grafana
title: 'feat(telemetry): Add Prometheus /metrics endpoint and RAG observability instrumentation'
labels: 'enhancement, observability, backend'
assignees: ''
---

## Summary

Implement a dedicated `GET /metrics` Prometheus endpoint and end-to-end RAG pipeline instrumentation within the FastAPI backend (`src/api/server.py`). This will allow monitoring agents and Prometheus scrapers to collect real-time telemetry into Prometheus and Grafana to evaluate system health, latency bottlenecks, and answer quality.

## Problem Statement

As observed during server operations, monitoring infrastructure is actively scraping `GET /metrics`:
```text
INFO: 172.18.0.2:41646 - "GET /metrics HTTP/1.1" 200 OK
INFO: 172.18.0.2:59108 - "GET /metrics HTTP/1.1" 200 OK
```

Currently, the FastAPI application does not expose a standard Prometheus exposition endpoint or domain-specific RAG metrics. Without structured metrics, there is no real-time visibility to evaluate:
1. Latency bottlenecks across the individual RAG stages (Query Augmentation, ChromaDB Retrieval, Cross-Encoder Reranking, and LLM Generation).
2. Streaming responsiveness, specifically Time-to-First-Token (TTFT).
3. Primary vs. fallback provider distribution (NVIDIA NIM vs. Nebius AI Studio).
4. Vector database size and index health.
5. Error rates and empty retrieval events (questions where no relevant chunks passed threshold).

## Proposed Solution

Integrate `prometheus-client` into the FastAPI backend to expose standard OpenMetrics/Prometheus format at `GET /metrics` with custom RAG pipeline instrumentation.

### 1. HTTP and Infrastructure Metrics
- `http_requests_total`: Counter by method, endpoint, and HTTP status code.
- `http_request_duration_seconds`: Histogram measuring total request latency.
- `rag_indexed_chunks_total`: Gauge tracking total documents/chunks persisted in ChromaDB.

### 2. RAG Pipeline Stage Latency (Histograms)
Track latency per stage to identify bottlenecks in real-time:
- `rag_stage_duration_seconds{stage="query_augmentation"}`: Time spent rewriting, expanding, or generating HyDE passages.
- `rag_stage_duration_seconds{stage="vector_retrieval"}`: Time spent computing query embeddings and querying ChromaDB HNSW index.
- `rag_stage_duration_seconds{stage="reranking"}`: Time spent in listwise semantic cross-scoring.
- `rag_stage_duration_seconds{stage="generation"}`: Time spent generating grounded LLM answers.
- `rag_llm_time_to_first_token_seconds`: Time from query submission to the first streamed token.

### 3. Retrieval Quality and Pipeline Counters
- `rag_queries_total{augmented="true|false", mode="rewrite|expand|hyde"}`: Total student queries processed.
- `rag_retrieval_chunks_count{type="retrieved|retained"}`: Histogram of candidate vs. reranked chunks.
- `rag_similarity_score`: Histogram distribution of cosine similarity and rerank scores.
- `rag_empty_retrievals_total`: Counter for queries that yielded zero context above threshold.

### 4. Provider and Resilience Metrics
- `rag_provider_requests_total{provider="nvidia_nim|nebius|offline"}`: Tracks primary vs. fallback utilization.
- `rag_provider_fallbacks_total`: Counter triggered whenever NVIDIA NIM fails or times out and Nebius is engaged.

## Implementation Plan

1. **Dependency:** Add `prometheus-client` to `requirements.txt`.
2. **Instrumentation Module (`src/api/metrics.py`):**
   - Define metric registries and collectors.
   - Provide lightweight context managers to time pipeline steps.
3. **Endpoint Registration (`src/api/server.py`):**
   - Register route `GET /metrics` returning `Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)`.
   - Update `StudyAssistant.ask()` and `ask_stream()` to record stage timings and chunk counts.
4. **Unit and Integration Tests (`tests/test_api.py`):**
   - Verify `GET /metrics` returns HTTP 200 with valid Prometheus metric text.
   - Verify counter and histogram increments after `/api/ask` and `/api/ask/stream`.
5. **Code Style & Formatting:**
   - Adhere strictly to the repository zero-emoji standard in code, logs, and metric descriptions.

## Acceptance Criteria

- [ ] `GET /metrics` returns HTTP 200 with valid `text/plain; version=0.0.4` payload.
- [ ] Prometheus metrics scrape passes without errors from Docker or external scrapers.
- [ ] Custom stage latencies (`query_augmentation`, `vector_retrieval`, `reranking`, `generation`) are tracked accurately.
- [ ] Provider fallback counter correctly increments on failover.
- [ ] All 119 existing unit tests continue to pass with zero regressions.
- [ ] New unit tests added in `tests/test_api.py` for `/metrics` endpoint.
