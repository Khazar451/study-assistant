---
name: Tavily Search API Integration
about: Integrate Tavily Search API for hybrid web grounding and fallback retrieval
title: 'feat(retrieval): Integrate Tavily Search API for hybrid web grounding and fallback search'
labels: 'enhancement, retrieval, integration'
assignees: ''
---

## Summary

Integrate the Tavily Search API into the Study Assistant retrieval pipeline to enable hybrid retrieval and intelligent fallback search. This allows the tutor to combine local ChromaDB course materials with real-time, clean web search results when course documents lack sufficient detail.

## Motivation & Problem Statement

Currently, Study Assistant operates strictly as a closed-book document RAG platform. When a student asks a question about:
1. Prerequisite concepts not covered in the uploaded lecture slides.
2. Real-world applications or current research literature beyond the syllabus.
3. Inquiries made before any course materials have been ingested into the vector database.

The tutor currently responds with:
> *"Based on the provided study materials, there is not enough information to answer this question."*

While strict grounding prevents hallucinations, this creates a dead-end user experience when students need broader academic context.

## Proposed Architecture & Capabilities

Integrate Tavily Search as an optional or fallback retrieval provider alongside ChromaDB:

```text
                                    +--> ChromaDB (Local Course Materials) ---+
Student Question -> Query Augmenter |                                         |--> StudyReranker --> Grounded Answer
                                    +--> Tavily Search API (Clean Web Data) --+   (Distinct Course vs. Web Citations)
```

### 1. Retrieval Strategies
- **Smart Fallback (Default):** Searches local ChromaDB first. If retrieved context similarity scores fall below the minimum threshold (or if the database is empty), automatically queries Tavily to answer the question using authoritative academic web sources.
- **Hybrid Augmentation:** Concurrently queries ChromaDB and Tavily, passing both local chunks and web passages to `StudyReranker` for unified semantic ranking.
- **On-Demand Toggle:** User-facing flag via CLI (`--web`) and Next.js frontend toggle.

### 2. Citation Integrity
Maintain strict attribution standards by differentiating source types:
- Local Course Materials: `[Course Document: lecture3.pdf, Page 14]`
- Web Sources: `[Web Source: en.wikipedia.org/wiki/Kepler, Domain: wikipedia.org]`

### 3. Dedicated Client Module (`src/retrieval/tavily_search.py`)
- Lightweight REST client implemented using `httpx` (avoiding heavy external dependencies).
- Normalizes Tavily search response objects into the existing Study Assistant chunk dictionary schema:
  ```python
  {
      "id": f"web_{hash}",
      "text": result["content"],
      "metadata": {
          "source": result["url"],
          "title": result["title"],
          "source_type": "web",
          "score": result.get("score", 0.0),
      },
  }
  ```
- Deterministic mock mode for offline testing and CI workflows without consuming live API credits.

## Implementation Steps

1. **Configuration:**
   - Add `TAVILY_API_KEY` and `ENABLE_WEB_SEARCH=false` to `.env.example`.
2. **Client Implementation (`src/retrieval/tavily_search.py`):**
   - Implement `TavilySearchClient` with `search(query, max_results=5, search_depth="basic")`.
   - Add graceful fallback when API key is missing or network times out.
3. **Orchestrator Integration (`src/assistant.py`):**
   - Add `use_web: bool = False` and `fallback_to_web: bool = True` arguments to `StudyAssistant.ask()` and `search()`.
4. **Backend Server Integration (`src/api/server.py`):**
   - Update `AskRequest` Pydantic model with `use_web: Optional[bool] = False`.
   - Thread parameter into `/api/ask` and `/api/ask/stream`.
5. **Frontend Toggle (`frontend/app/page.tsx`):**
   - Add a "Web Search" pill toggle in the query bar.
6. **Automated Unit Testing (`tests/test_tavily.py`):**
   - Test client initialization, payload construction, error handling, citation formatting, and offline mocking.
7. **Standards:**
   - Maintain zero emojis across all code, docstrings, logs, and UI components.

## Acceptance Criteria

- [ ] `TavilySearchClient` successfully retrieves and parses web passages via Tavily API.
- [ ] Local ChromaDB sources and Tavily web results are clearly distinguished in final citations.
- [ ] Fallback mode seamlessly activates when ChromaDB returns zero chunks above threshold.
- [ ] Zero API credits consumed during automated test suite execution (mocked client in CI).
- [ ] All 119 existing unit tests continue to pass without regression.
- [ ] New unit tests in `tests/test_tavily.py` pass with 100% assertion coverage.
