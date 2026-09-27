import copy
import json
import logging
import math
import os
import re
import time
from typing import Any, Dict, List, Optional, Union
from dotenv import load_dotenv
from openai import OpenAI
import requests

# Load environment variables
load_dotenv(".env.local")
load_dotenv()

logger = logging.getLogger(__name__)


class StudyReranker:
    """Semantic reranker for candidate document chunks in RAG pipelines.

    Supports:
    1. 'llm_listwise' (Default): Sub-second single-roundtrip listwise scoring using NVIDIA NIM instruction models.
    2. 'ranking_api': Dedicated cross-encoder microservice via /v1/ranking with sigmoid logit normalization.
    3. 'auto': Attempts ranking_api, with cached fallback to llm_listwise on 404/401/403.
    """

    DEFAULT_LLM_MODEL = "meta/llama-3.2-11b-vision-instruct"
    DEFAULT_LLM_BASE_URL = "https://integrate.api.nvidia.com/v1"

    DEFAULT_RANKING_MODEL = "nvidia/nv-rerankqa-mistral-4b-v3"
    DEFAULT_RANKING_BASE_URL = "https://ai.api.nvidia.com/v1/retrieval/nvidia"

    # Cached flag across instances to prevent repeated 404 penalties on unentitled accounts
    _ranking_api_available: bool = True

    def __init__(
        self,
        api_key: Optional[str] = None,
        ranking_api_key: Optional[str] = None,
        llm_api_key: Optional[str] = None,
        ranking_base_url: Optional[str] = None,
        llm_base_url: Optional[str] = None,
        ranking_model: Optional[str] = None,
        llm_model: Optional[str] = None,
        mode: str = "llm_listwise",
        top_n: int = 5,
        min_score: Optional[float] = None,
        max_retries: int = 2,
        timeout: float = 15.0,
        client: Optional[OpenAI] = None,
    ):
        """Initialize StudyReranker.

        Args:
            api_key: General API key fallback (NVIDIA or Nebius).
            ranking_api_key: Dedicated key for /v1/ranking endpoint.
            llm_api_key: Dedicated key for LLM listwise ranking.
            ranking_base_url: Base endpoint URL for dedicated cross-encoder.
            llm_base_url: Base endpoint URL for LLM listwise provider.
            ranking_model: Model identifier for ranking microservice.
            llm_model: Model identifier for LLM listwise reranker.
            mode: 'llm_listwise' (default), 'ranking_api', or 'auto'.
            top_n: Maximum number of top reranked chunks to return.
            min_score: Minimum normalized score threshold [0.0, 1.0].
            max_retries: Retry count for transient errors (429, 503, timeout).
            timeout: Network request timeout in seconds.
            client: Optional pre-configured OpenAI client for LLM ranking.
        """
        self.mode = mode.lower()
        if self.mode not in {"llm_listwise", "ranking_api", "auto"}:
            raise ValueError(f"Unsupported mode: '{mode}'. Must be 'llm_listwise', 'ranking_api', or 'auto'.")

        self.top_n = top_n
        self.min_score = min_score
        self.max_retries = max_retries
        self.timeout = timeout
        self.ranking_api_available: bool = True

        # Resolve credentials: parameter > specific env > NVIDIA_API_KEY > NEBIUS_API_KEY
        self.api_key = (
            api_key
            or llm_api_key
            or ranking_api_key
            or os.getenv("NVIDIA_API_KEY")
            or os.getenv("NEBIUS_API_KEY")
        )
        if not self.api_key:
            raise ValueError(
                "No API key found. Please set NVIDIA_API_KEY (or NEBIUS_API_KEY) "
                "or pass api_key to StudyReranker."
            )

        self.llm_api_key = llm_api_key or self.api_key
        self.ranking_api_key = ranking_api_key or self.api_key

        self.llm_base_url = llm_base_url or os.getenv("NVIDIA_BASE_URL", self.DEFAULT_LLM_BASE_URL)
        self.ranking_base_url = ranking_base_url or os.getenv("RANKING_BASE_URL", self.DEFAULT_RANKING_BASE_URL)

        self.llm_model = llm_model or os.getenv("LLM_MODEL", self.DEFAULT_LLM_MODEL)
        self.ranking_model = ranking_model or os.getenv("RANKING_MODEL", self.DEFAULT_RANKING_MODEL)

        self.client = client or OpenAI(
            api_key=self.llm_api_key,
            base_url=self.llm_base_url,
            timeout=self.timeout,
        )

    def _normalize_chunk(self, chunk: Any, fallback_id: str) -> Dict[str, Any]:
        """Convert chunk into a standardized, immutable internal dictionary."""
        if hasattr(chunk, "page_content"):
            text = chunk.page_content
            meta = copy.deepcopy(getattr(chunk, "metadata", {}))
            cid = str(meta.get("id") or meta.get("chunk_id") or fallback_id)
            sim_score = meta.get("similarity_score")
        elif isinstance(chunk, dict):
            text = chunk.get("text", chunk.get("page_content", ""))
            meta = copy.deepcopy(chunk.get("metadata", {}))
            cid = str(chunk.get("id") or chunk.get("chunk_id") or meta.get("id") or fallback_id)
            sim_score = chunk.get("similarity_score", meta.get("similarity_score"))
        else:
            text = getattr(chunk, "text", "")
            meta = copy.deepcopy(getattr(chunk, "metadata", {}))
            cid = str(getattr(chunk, "id", None) or fallback_id)
            sim_score = getattr(chunk, "similarity_score", None)

        return {
            "id": cid,
            "text": text.strip(),
            "metadata": meta,
            "similarity_score": float(sim_score) if sim_score is not None else None,
            "rerank_score": None,
        }

    def _score_llm_listwise(self, query: str, chunks: List[Dict[str, Any]]) -> Dict[str, float]:
        """Perform listwise relevance scoring using the active instruction LLM."""
        passages_payload = [
            {"id": c["id"], "text": c["text"][:1200]}  # Bound passage length
            for c in chunks
        ]

        prompt = (
            "You are an expert search relevance evaluator for academic study materials.\n"
            f"Student Query: {query}\n\n"
            "Evaluate each passage's direct relevance to answering the query. "
            "Assign a relevance_score between 0.0 (completely irrelevant) and 1.0 (directly answers the query).\n\n"
            "Passages to evaluate:\n"
            f"{json.dumps(passages_payload, indent=2)}\n\n"
            "Instructions:\n"
            "1. Output ONLY a valid JSON list of objects containing 'id' and 'relevance_score'.\n"
            "2. Score every passage in the input list.\n"
            "3. Do not include markdown preamble, commentary, or text outside the JSON list.\n\n"
            "Example format:\n"
            "[\n"
            '  {"id": "doc1", "relevance_score": 0.95},\n'
            '  {"id": "doc2", "relevance_score": 0.10}\n'
            "]"
        )

        for attempt in range(self.max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=self.llm_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                    max_tokens=400,
                )
                raw_text = response.choices[0].message.content.strip()

                # Extract JSON even if wrapped in markdown code fences
                match = re.search(r"\[\s*\{.*?\}\s*\]", raw_text, re.DOTALL)
                json_str = match.group(0) if match else raw_text

                parsed = json.loads(json_str)
                if not isinstance(parsed, list):
                    raise ValueError("Model output did not parse into a list.")

                # Build scores map, strictly filtering hallucinated IDs
                valid_ids = {c["id"] for c in chunks}
                scores_map: Dict[str, float] = {}

                for item in parsed:
                    if isinstance(item, dict) and "id" in item and "relevance_score" in item:
                        item_id = str(item["id"])
                        if item_id in valid_ids:
                            try:
                                score = float(item["relevance_score"])
                                scores_map[item_id] = max(0.0, min(1.0, score))
                            except (ValueError, TypeError):
                                continue

                # Defensive Monotonic Imputation: missing chunks rank strictly below all scored chunks
                if scores_map:
                    min_val = min(scores_map.values())
                    imputed = max(0.0, min_val - 1e-4)
                else:
                    imputed = 0.0

                for c in chunks:
                    if c["id"] not in scores_map:
                        scores_map[c["id"]] = imputed

                return scores_map

            except Exception as exc:
                err_str = str(exc)
                is_transient = "429" in err_str or "503" in err_str or "timeout" in err_str.lower()
                if is_transient and attempt < self.max_retries:
                    time.sleep(0.5 * (2 ** attempt))
                    continue
                logger.warning(f"LLM listwise reranker call failed: {exc}")
                break

        return {}

    def _score_ranking_api(self, query: str, chunks: List[Dict[str, Any]]) -> Dict[str, float]:
        """Perform cross-encoder reranking via dedicated NVIDIA NIM /v1/ranking endpoint."""
        url = f"{self.ranking_base_url.rstrip('/')}/ranking"
        headers = {
            "Authorization": f"Bearer {self.ranking_api_key}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

        payload = {
            "model": self.ranking_model,
            "query": {"text": query},
            "passages": [{"text": c["text"]} for c in chunks],
            "truncate": "END",
        }

        for attempt in range(self.max_retries + 1):
            try:
                resp = requests.post(url, headers=headers, json=payload, timeout=self.timeout)

                # Non-transient errors (404 not provisioned, 401/403 bad auth, 400 bad schema)
                if resp.status_code in {404, 401, 403, 400, 410}:
                    self.ranking_api_available = False
                    logger.warning(
                        f"NVIDIA ranking API returned {resp.status_code}: {resp.text[:100]}. "
                        "Marking ranking API unavailable."
                    )
                    return {}

                # Transient errors
                if resp.status_code in {429, 503}:
                    if attempt < self.max_retries:
                        time.sleep(0.5 * (2 ** attempt))
                        continue
                    return {}

                resp.raise_for_status()
                data = resp.json()

                # NVIDIA NIM ranking format: {"rankings": [{"index": 0, "logit": -1.24}, ...]}
                rankings = data.get("rankings", [])
                if not rankings:
                    return {}

                scores_map: Dict[str, float] = {}
                for item in rankings:
                    idx = item.get("index")
                    logit = item.get("logit")
                    if idx is not None and 0 <= idx < len(chunks) and logit is not None:
                        # Sigmoid normalization: logit -> [0.0, 1.0]
                        sigmoid_score = 1.0 / (1.0 + math.exp(-float(logit)))
                        chunk_id = chunks[idx]["id"]
                        scores_map[chunk_id] = round(sigmoid_score, 4)

                # Impute any missing chunks
                if scores_map:
                    min_val = min(scores_map.values())
                    imputed = max(0.0, min_val - 1e-4)
                else:
                    imputed = 0.0

                for c in chunks:
                    if c["id"] not in scores_map:
                        scores_map[c["id"]] = imputed

                return scores_map

            except Exception as exc:
                if attempt < self.max_retries:
                    time.sleep(0.5 * (2 ** attempt))
                    continue
                logger.warning(f"NVIDIA ranking API request failed: {exc}")
                return {}

        return {}

    def rerank(
        self,
        query: str,
        chunks: List[Any],
        top_n: Optional[int] = None,
        min_score: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Rerank candidate document chunks according to relevance to the query.

        Args:
            query: The student's question.
            chunks: List of retrieved chunks (dicts, LangChain Documents, or custom objects).
            top_n: Maximum number of chunks to return (defaults to self.top_n).
            min_score: Minimum normalized score threshold [0.0, 1.0].

        Returns:
            New list of chunks sorted descending by 'rerank_score', filtered and truncated.
        """
        if not query or not query.strip() or not chunks:
            return []

        limit = top_n if top_n is not None else self.top_n
        threshold = min_score if min_score is not None else self.min_score

        # 1. Normalize and copy input chunks, disambiguating any duplicate IDs
        seen_ids = set()
        normalized_chunks: List[Dict[str, Any]] = []

        for idx, item in enumerate(chunks):
            chunk_copy = self._normalize_chunk(item, fallback_id=f"chunk_{idx}")
            cid = chunk_copy["id"]
            if cid in seen_ids:
                cid = f"{cid}_{idx}"
                chunk_copy["id"] = cid
            seen_ids.add(cid)
            normalized_chunks.append(chunk_copy)

        # Single chunk trivial pass-through
        if len(normalized_chunks) == 1:
            single = normalized_chunks[0]
            single["rerank_score"] = single.get("similarity_score", 1.0) or 1.0
            if threshold is not None and single["rerank_score"] < threshold:
                return []
            return [single]

        # Limit candidate pool to top 10 chunks to prevent latency bloat
        eval_pool = normalized_chunks[:10]

        # 2. Execute Reranking Strategy
        scores_map: Dict[str, float] = {}

        if self.mode == "ranking_api":
            scores_map = self._score_ranking_api(query, eval_pool)
        elif self.mode == "auto":
            if self.ranking_api_available:
                scores_map = self._score_ranking_api(query, eval_pool)
            if not scores_map:
                scores_map = self._score_llm_listwise(query, eval_pool)
        else:  # default: 'llm_listwise'
            scores_map = self._score_llm_listwise(query, eval_pool)

        # 3. Attach rerank_score (fallback to similarity_score or 0.0 if scoring failed completely)
        for c in normalized_chunks:
            if c["id"] in scores_map:
                c["rerank_score"] = scores_map[c["id"]]
            else:
                sim = c.get("similarity_score")
                c["rerank_score"] = float(sim) if sim is not None else 0.0

        # 4. Stable Sort: rerank_score DESC (Python sort preserves original relative order on equal keys)
        normalized_chunks.sort(key=lambda x: x.get("rerank_score", 0.0), reverse=True)

        # 5. Filter by min_score
        if threshold is not None:
            normalized_chunks = [
                c for c in normalized_chunks if c.get("rerank_score", 0.0) >= threshold
            ]

        # 6. Truncate to top_n
        return normalized_chunks[:limit]
