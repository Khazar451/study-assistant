from __future__ import annotations

from collections import deque
import hmac
import json
import logging
import os
from pathlib import Path
import re
import secrets
import threading
import time
from typing import Any, Deque, Dict, Iterator, List, Optional
import uuid
from dotenv import load_dotenv
from fastapi import BackgroundTasks, Depends, FastAPI, File, Header, HTTPException, Query, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from starlette.concurrency import iterate_in_threadpool, run_in_threadpool
from pydantic import BaseModel, Field
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

from src.assistant import StudyAssistant
from src.generation.plots import embed_plots, ensure_requested_plot
from src.ingestion.page_images import attach_page_images, render_pdf_page
from src.generation.tutor_format import allowed_source_keys, bind_numbered_citations, infer_grounding, polish_tutor_answer
from src import providers
from src.providers import PROVIDER_LABELS
from src.verification.symbolic_verifier import verify_math_solution

# Load environment configuration
load_dotenv(".env.local")
load_dotenv()

app = FastAPI(
    title="Study Assistant RAG API",
    description="Grounded Academic RAG Tutor API powered by NVIDIA NIM and Nebius AI Studio",
    version="1.0.0",
)

# --- Deployment hardening (all optional; local development stays open) ---
# CORS_ORIGINS: comma-separated allowed origins ("*" by default for local dev).
# DEMO_MODE=true: the public demo cannot wipe the shared index unless ADMIN_TOKEN matches.
# ADMIN_TOKEN: when set, destructive endpoints require the X-Admin-Token header.
# RATE_LIMIT_PER_MIN: per-client cap on the endpoints that spend LLM credits (0 disables).
# TRUSTED_PROXY_HOPS: proxies in front of the app that append to X-Forwarded-For (0 = use the socket peer).
# DEMO_MAX_PER_MIN / DEMO_MAX_PER_DAY: caps on those endpoints across all visitors together (0 disables).
# DEMO_PAUSED=true: those endpoints return 503, so credits stop being spent at once.
# WORKSPACE_ISOLATION: give each browser its own chats (defaults to DEMO_MODE; local development stays shared).
# MAX_UPLOAD_MB: upload size cap (default 200).
_cors_origins = [o.strip() for o in os.getenv("CORS_ORIGINS", "*").split(",") if o.strip()] or ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials="*" not in _cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

_RATE_LIMITED_PATHS = {
    "/api/ask",
    "/api/ask/stream",
    "/api/search",
    "/api/flashcards",
    "/api/ingest/file",
    "/api/ingest/sample",
}
_MAX_RATE_KEYS = 10_000
_rate_hits: Dict[str, List[float]] = {}
_global_hits: Deque[float] = deque()
_rate_lock = threading.Lock()


def _env_flag(name: str) -> bool:
    return os.getenv(name, "false").strip().lower() in {"1", "true", "yes"}


def _env_int(name: str, default: int = 0) -> int:
    try:
        return max(0, int(os.getenv(name, str(default))))
    except ValueError:
        return default


def _demo_mode() -> bool:
    return _env_flag("DEMO_MODE")


def _rate_limit_per_min() -> int:
    return _env_int("RATE_LIMIT_PER_MIN")


def _client_key(request: Request) -> str:
    """Client address for rate limiting.

    Only entries appended by our own proxies are trusted, counted from the right end of
    X-Forwarded-For; anything a client sends in that header sits further left and is ignored.
    """
    peer = request.client.host if request.client else "unknown"
    hops = _env_int("TRUSTED_PROXY_HOPS")
    if hops:
        forwarded = [part.strip() for part in request.headers.get("x-forwarded-for", "").split(",") if part.strip()]
        if len(forwarded) >= hops:
            return forwarded[-hops]
    return peer


def _refusal(status_code: int, detail: str, retry_after: float) -> JSONResponse:
    retry = max(1, int(retry_after) + 1)
    return JSONResponse(status_code=status_code, content={"detail": detail}, headers={"Retry-After": str(retry)})


def _prune_rate_keys(now: float) -> None:
    """Keep the per-client table bounded even when callers rotate addresses."""
    for key in [k for k, hits in _rate_hits.items() if not hits or now - hits[-1] >= 60]:
        del _rate_hits[key]
    if len(_rate_hits) > _MAX_RATE_KEYS:
        by_last_hit = sorted(_rate_hits, key=lambda k: _rate_hits[k][-1])
        for key in by_last_hit[: len(_rate_hits) - _MAX_RATE_KEYS]:
            del _rate_hits[key]


def _check_spend_limits(key: str, now: float) -> Optional[JSONResponse]:
    """Apply the per-client and the global caps; a refused request does not count against either."""
    per_client = _rate_limit_per_min()
    per_min = _env_int("DEMO_MAX_PER_MIN")
    per_day = _env_int("DEMO_MAX_PER_DAY")
    with _rate_lock:
        hits = [t for t in _rate_hits.get(key, []) if now - t < 60]
        if per_client and len(hits) >= per_client:
            _rate_hits[key] = hits
            wait = 60 - (now - hits[0])
            return _refusal(429, f"Rate limit reached ({per_client}/min). Try again in {int(wait) + 1}s.", wait)

        while _global_hits and now - _global_hits[0] >= 86_400:
            _global_hits.popleft()
        if per_day and len(_global_hits) >= per_day:
            wait = 86_400 - (now - _global_hits[0])
            return _refusal(503, f"The demo reached its daily question limit. It reopens in about {int(wait // 3600) + 1} h.", wait)
        last_minute = [t for t in _global_hits if now - t < 60]
        if per_min and len(last_minute) >= per_min:
            wait = 60 - (now - last_minute[0])
            return _refusal(503, f"The demo is busy right now. Try again in {int(wait) + 1}s.", wait)

        if per_client:
            hits.append(now)
            _rate_hits[key] = hits
            if len(_rate_hits) > _MAX_RATE_KEYS:
                _prune_rate_keys(now)
        if per_min or per_day:
            _global_hits.append(now)
    return None


@app.middleware("http")
async def rate_limit_llm_endpoints(request: Request, call_next):
    if request.url.path in _RATE_LIMITED_PATHS and request.method != "OPTIONS":
        if _env_flag("DEMO_PAUSED"):
            return JSONResponse(status_code=503, content={"detail": "The demo is paused right now. Please try again later."})
        refusal = _check_spend_limits(_client_key(request), time.time())
        if refusal is not None:
            return refusal
    return await call_next(request)


WORKSPACE_COOKIE = "sa_workspace"
_WORKSPACE_RE = re.compile(r"[0-9a-f]{32}")


def _workspace_isolation() -> bool:
    if os.getenv("WORKSPACE_ISOLATION") is not None:
        return _env_flag("WORKSPACE_ISOLATION")
    return _demo_mode()


@app.middleware("http")
async def assign_workspace(request: Request, call_next):
    """Give each browser an unguessable workspace ID in a cookie when isolation is on.

    A cookie (not a header) because citations open files and page images through plain links,
    which cannot carry custom headers.
    """
    if not _workspace_isolation():
        request.state.workspace = None
        return await call_next(request)
    current = request.cookies.get(WORKSPACE_COOKIE, "")
    is_new = not _WORKSPACE_RE.fullmatch(current)
    request.state.workspace = secrets.token_hex(16) if is_new else current
    response = await call_next(request)
    if is_new:
        response.set_cookie(
            WORKSPACE_COOKIE,
            request.state.workspace,
            max_age=30 * 24 * 3600,
            httponly=True,
            samesite="lax",
            secure=request.url.scheme == "https" or request.headers.get("x-forwarded-proto") == "https",
        )
    return response


def current_workspace(request: Request) -> Optional[str]:
    """The caller's workspace ID, or None when every visitor shares one library (local development)."""
    return getattr(request.state, "workspace", None)


def _claim_session(session_id: Optional[str], workspace: Optional[str]) -> None:
    """Make sure a client-supplied session ID is the caller's own before it is read or written.

    An unknown ID is created for the caller; one owned by another workspace answers 404, the same
    as a session that does not exist, so IDs cannot be probed.
    """
    if not session_id or workspace is None:
        return
    from src.memory.session_store import get_session_store

    store = get_session_store()
    exists, owner = store.session_owner(session_id)
    if not exists:
        store.create_session(session_id=session_id, workspace=workspace)
    elif owner != workspace:
        raise HTTPException(status_code=404, detail="Session not found")


def require_admin(x_admin_token: Optional[str] = Header(None)) -> None:
    """Guard destructive endpoints on a shared deployment; open for local development."""
    expected = os.getenv("ADMIN_TOKEN", "")
    if expected:
        if not x_admin_token or not hmac.compare_digest(x_admin_token, expected):
            raise HTTPException(status_code=403, detail="Admin token required.")
    elif _demo_mode():
        raise HTTPException(status_code=403, detail="Disabled in the public demo.")



# --- Prometheus Metrics Definitions ---
HTTP_REQUESTS_TOTAL = Counter(
    "http_requests_total",
    "Total HTTP requests handled by Study Assistant",
    ["method", "endpoint", "status_code"],
)
HTTP_REQUEST_DURATION_SECONDS = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency in seconds",
    ["endpoint"],
)
RAG_INDEXED_CHUNKS = Gauge(
    "rag_indexed_chunks_total",
    "Total document chunks currently persisted in vector store",
)
RAG_QUERIES_TOTAL = Counter(
    "rag_queries_total",
    "Total student academic queries processed",
    ["mode", "use_web"],
)

# Global singleton instance (guarded: the startup sample-ingest thread and the first request can race)
_assistant_instance: Optional[StudyAssistant] = None
_assistant_lock = threading.Lock()


class OfflineFallbackEmbedder:
    """Deterministic local embedder when NVIDIA_API_KEY is not yet configured."""

    def __init__(self, dim: int = 2048):
        self.dim = dim

    def _hash_vector(self, text: str) -> List[float]:
        import hashlib
        import math
        vec = [0.0] * self.dim
        for i, word in enumerate(text.lower().split()):
            h = int(hashlib.md5(f"{word}_{i % 64}".encode()).hexdigest(), 16)
            idx = h % self.dim
            vec[idx] += 1.0
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]

    def embed_text(self, text: str, input_type: str = "passage") -> List[float]:
        return self._hash_vector(text)

    def embed_query(self, query: str) -> List[float]:
        return self._hash_vector(query)

    def embed_batch(self, texts: List[str], input_type: str = "passage", batch_size: int = 32, **_: Any) -> List[List[float]]:
        return [self._hash_vector(t) for t in texts]


class OfflineFallbackGenerator:
    """Grounded tutor generator when cloud LLM credentials are not yet configured."""

    def __init__(self):
        self.model = "offline-fallback-tutor"
        self.provider = "Offline Demo Mode"

    def generate(self, query: str, context: str) -> Dict[str, Any]:
        citations = []
        if context:
            citations = re.findall(r"\[Source:\s*([^,\]]+),\s*Page:\s*(\d+)\]", context)
            unique_citations = [f"{src} (p. {page})" for src, page in set(citations)]
        else:
            unique_citations = []

        if not context or "not enough information" in context.lower():
            answer = "Based on the provided study materials, there is not enough information to answer this question."
        else:
            first_citation = f" [Source: {citations[0][0]}, Page: {citations[0][1]}]" if citations else ""
            answer = (
                f"**Grounded Explanation for:** *{query}*\n\n"
                f"According to the ingested course materials:\n\n"
                f"{context[:400].strip()}...\n\n"
                f"*(Note: Running in zero-key offline mode. To enable live NVIDIA NIM or Nebius 70B generation, set NVIDIA_API_KEY or NEBIUS_API_KEY in your .env file).*{first_citation}"
            )

        return {
            "answer": answer,
            "sources": unique_citations,
            "model": self.model,
            "provider": self.provider,
        }

    def generate_stream(self, query: str, context: str) -> Iterator[str]:
        res = self.generate(query, context)
        tokens = res["answer"].split(" ")
        for i, token in enumerate(tokens):
            yield token + (" " if i < len(tokens) - 1 else "")
            time.sleep(0.015)


class OfflineFallbackReranker:
    """Pass-through reranker when LLM API keys are not yet configured."""

    def __init__(self, mode: str = "offline"):
        self.mode = mode

    def rerank(self, query: str, chunks: List[Dict[str, Any]], top_n: int = 5, min_score: Optional[float] = None) -> List[Dict[str, Any]]:
        sorted_chunks = sorted(chunks, key=lambda c: c.get("similarity", 1.0), reverse=True)
        for i, c in enumerate(sorted_chunks):
            if "rerank_score" not in c:
                c["rerank_score"] = c.get("similarity", 1.0 - (i * 0.05))
        return sorted_chunks[:top_n]


def get_assistant(db_path: str = "./chroma_db") -> StudyAssistant:
    """Retrieve or initialize the singleton StudyAssistant orchestrator."""
    global _assistant_instance
    if _assistant_instance is not None:
        return _assistant_instance
    with _assistant_lock:
        if _assistant_instance is not None:
            return _assistant_instance
        try:
            _assistant_instance = StudyAssistant(persist_directory=db_path)
        except ValueError:
            # NVIDIA / Nebius API key not set in environment: initialize resilient offline assistant
            from src.ingestion.indexer import ChromaIndexer
            from src.ingestion.pipeline import IngestionPipeline
            from src.retrieval.retriever import StudyRetriever

            fallback_embedder = OfflineFallbackEmbedder()
            fallback_indexer = ChromaIndexer(persist_directory=db_path)
            fallback_pipeline = IngestionPipeline(embedder=fallback_embedder, indexer=fallback_indexer)
            fallback_retriever = StudyRetriever(indexer=fallback_indexer, embedder=fallback_embedder)
            fallback_generator = OfflineFallbackGenerator()
            fallback_reranker = OfflineFallbackReranker()

            _assistant_instance = StudyAssistant(
                pipeline=fallback_pipeline,
                retriever=fallback_retriever,
                reranker=fallback_reranker,
                generator=fallback_generator,
                persist_directory=db_path,
            )
            _assistant_instance.embedder = fallback_embedder
    return _assistant_instance



def set_assistant(assistant: StudyAssistant) -> None:
    """Set assistant instance (used for dependency injection during testing)."""
    global _assistant_instance
    _assistant_instance = assistant



# --- Pydantic Request / Response Models ---

# Questions reach several paid model calls, so their size is capped (the UI enforces the same limit).
MAX_QUERY_CHARS = 2000


class AskRequest(BaseModel):
    query: str = Field(..., max_length=MAX_QUERY_CHARS, description="Student's academic question")
    top_k: int = Field(15, ge=1, le=50, description="Initial vector candidates")
    top_n: int = Field(5, ge=1, le=20, description="Reranked context chunks")
    augment: bool = Field(True, description="Enable query intelligence (rewrite/expand/HyDE)")
    augment_mode: str = Field("expand", description="Strategy: expand, rewrite, or hyde")
    min_similarity: Optional[float] = Field(None, description="Optional minimum cosine similarity")
    min_rerank_score: Optional[float] = Field(None, description="Optional minimum reranker threshold")
    use_web: bool = Field(False, description="Enable Tavily web search integration")
    fallback_to_web: bool = Field(True, description="Fallback to web search if local context is insufficient")
    session_id: Optional[str] = Field(None, description="Optional chat session ID for multi-turn conversational memory")
    source_mode: Optional[str] = Field(None, description="Optional source mode: notes, web, or both")


class CreateSessionRequest(BaseModel):
    title: Optional[str] = Field("New Chat", max_length=120, description="Title of the conversation session")
    session_id: Optional[str] = Field(None, description="Optional custom session ID")


class RenameSessionRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=120, description="New title for the session")


class SearchRequest(BaseModel):
    query: str = Field(..., max_length=MAX_QUERY_CHARS, description="Search query")
    top_k: int = Field(15, ge=1, le=50)
    top_n: int = Field(5, ge=1, le=20)
    augment: bool = Field(True)
    augment_mode: str = Field("expand")
    use_web: bool = Field(False)


class IngestPathRequest(BaseModel):
    path: str = Field(..., description="Local path to file or directory to ingest")
    batch_size: int = Field(32, ge=1, le=128)


class FlashcardItem(BaseModel):
    id: str
    question: str
    answer: str
    category: str
    difficulty: str
    source: str


class FlashcardsRequest(BaseModel):
    topic: Optional[str] = Field(None, max_length=200, description="Optional focus topic")
    count: int = Field(4, ge=1, le=10, description="Number of flashcards to generate")


# --- API Routes ---


@app.get("/metrics")
def get_metrics() -> Response:
    """Prometheus exposition endpoint for system and RAG observability."""
    try:
        assistant = get_assistant()
        RAG_INDEXED_CHUNKS.set(assistant.count())
    except Exception:
        pass
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/api/status")
def get_status() -> Dict[str, Any]:
    """Return system readiness, cloud providers, and vector store metrics."""
    assistant = get_assistant()
    chunk_count = assistant.count()

    nvidia_key = bool(os.getenv("NVIDIA_API_KEY"))
    nebius_key = bool(os.getenv("NEBIUS_API_KEY"))
    tavily = getattr(assistant, "tavily", None)
    tavily_available = bool(getattr(tavily, "is_available", False)) if tavily is not None else False

    generator = assistant.generator
    augmenter = getattr(assistant.retriever, "augmenter", None)

    return {
        "status": "ready",
        "provider": _provider_label(generator),
        "nvidia_configured": nvidia_key,
        "nebius_configured": nebius_key,
        "tavily_configured": bool(os.getenv("TAVILY_API_KEY")),
        "tavily_available": tavily_available,
        "indexed_chunks": chunk_count,
        "persist_directory": assistant.persist_directory,
        "generation_model": _str_attr(generator, "model"),
        "augment_model": _str_attr(augmenter, "model"),
        "rerank_model": _str_attr(assistant.reranker, "llm_model"),
        "embedding_model": _str_attr(getattr(assistant, "embedder", None), "model"),
        "rerank_mode": getattr(assistant.reranker, "mode", "llm_listwise"),
    }


def _str_attr(obj: Any, name: str) -> str:
    value = getattr(obj, name, "")
    return value if isinstance(value, str) else ""


def _source_mode(source_mode: str, use_web: bool, fallback_to_web: bool) -> str:
    """Honor an explicit student choice. Legacy clients still use the old flags."""
    if source_mode in {"notes", "web", "both"}:
        return source_mode
    if use_web and fallback_to_web:
        return "both"
    if use_web:
        return "web"
    if fallback_to_web:
        return "both"
    return "notes"


def _retrieve_for_mode(
    assistant: Any,
    query: str,
    mode: str,
    top_k: int,
    top_n: int,
    augment: bool,
    augment_mode: str,
    workspace: Optional[str] = None,
) -> List[Dict[str, Any]]:
    tavily = getattr(assistant, "tavily", None)
    web_ok = bool(tavily and getattr(tavily, "is_available", False))

    if mode == "web":
        if not web_ok:
            return []
        return tavily.search(query=query.strip(), max_results=top_n)

    if mode == "notes" and assistant.count() == 0:
        return []

    chunks = assistant.search(
        query=query,
        top_k=top_k,
        top_n=top_n,
        augment=augment,
        augment_mode=augment_mode,
        use_web=mode == "both" and web_ok,
        **_where_kwargs(workspace),
    )
    return chunks


def _where_kwargs(workspace: Optional[str]) -> Dict[str, Any]:
    """`where=` for assistant calls when the caller is limited to a workspace; nothing otherwise."""
    where = _workspace_filter(workspace)
    return {"where": where} if where else {}


def _empty_retrieval_notice(mode: str) -> str:
    if mode == "web":
        return "No academic web page matched this question. Try different wording, or switch the source to your notes."
    if mode == "both":
        return "Neither your notes nor the academic web had a verified passage for this question."
    return (
        "Based on the available course materials, there is not enough verified information to answer this question. "
        "Please upload the relevant lecture slides or textbook chapters."
    )


async def _stream_answer(
    assistant: Any,
    query: str,
    chunks: List[Dict[str, Any]],
    grounding: str,
    request: Optional[Request] = None,
    session_id: Optional[str] = None,
    history: Optional[List[Dict[str, str]]] = None,
):
    """Stream tokens, then replace the answer with a citation-checked final draft and persist to session.

    The model stream and the post-processing are blocking (HTTP reads, PyMuPDF, matplotlib, SymPy,
    SQLite), so they run in the threadpool; on the event loop they would stall every other visitor.
    """
    context = assistant.retriever.format_context(chunks) if chunks else ""
    stream = await run_in_threadpool(
        assistant.generator.generate_stream,
        query=query.strip(), context=context, grounding=grounding, history=history,
    )
    full_text: List[str] = []
    async for token in iterate_in_threadpool(stream):
        if request is not None and await request.is_disconnected():
            return
        full_text.append(token)
        yield f"data: {json.dumps({'type': 'token', 'token': token})}\n\n"

    if request is not None and await request.is_disconnected():
        return

    done_event = await run_in_threadpool(
        _finalize_answer, assistant, query, "".join(full_text), chunks, context, grounding, session_id
    )
    yield f"data: {json.dumps(done_event)}\n\n"


def _finalize_answer(
    assistant: Any,
    query: str,
    raw_answer: str,
    chunks: List[Dict[str, Any]],
    context: str,
    grounding: str,
    session_id: Optional[str],
) -> Dict[str, Any]:
    """Polish, cite, illustrate and verify a streamed answer, persist the turn, and build the done event."""
    answer = polish_tutor_answer(
        raw_answer,
        allowed_sources=allowed_source_keys(chunks, context),
        grounding=grounding,
    )
    references: List[Dict[str, Any]] = []
    if grounding != "general":
        answer, references = bind_numbered_citations(answer, chunks)
        attach_page_images(references, answer)
    answer = ensure_requested_plot(query, embed_plots(answer))
    sources = [ref["label"] for ref in references]
    verification = verify_math_solution(answer)

    # Persist conversation turn to SQLite session store
    if session_id:
        from src.memory.session_store import get_session_store
        store = get_session_store()
        store.add_message(session_id, "user", query.strip())
        store.add_message(
            session_id,
            "tutor",
            answer,
            sources=sources,
            references=references,
            grounding=grounding,
            verification=verification,
        )

    return {
        "type": "done",
        "answer": answer,
        "sources": sources,
        "references": references,
        "grounding": grounding,
        "model": providers.answer_model(assistant.generator),
        "provider": _provider_label(assistant.generator),
        "verification": verification,
        "session_id": session_id,
    }


def _provider_label(generator: Any) -> str:
    label = getattr(generator, "provider_label", None)
    if isinstance(label, str) and label:
        return label
    provider = getattr(generator, "provider", "")
    return PROVIDER_LABELS.get(provider, provider) if isinstance(provider, str) else ""


@app.get("/api/documents")
def get_documents(workspace: Optional[str] = Depends(current_workspace)) -> Dict[str, Any]:
    """Inspect the documents the caller can use (their own uploads and the shared sample)."""
    assistant = get_assistant()
    try:
        # ChromaDB collection inspection
        where = _workspace_filter(workspace)
        raw = assistant.indexer.collection.get(include=["metadatas"], **({"where": where} if where else {}))
        metadatas = raw.get("metadatas", []) or []

        doc_map: Dict[str, Dict[str, Any]] = {}
        for m in metadatas:
            if not isinstance(m, dict):
                continue
            src = m.get("source") or m.get("file_name") or m.get("file_path") or "Unknown"
            page = m.get("page")

            if src not in doc_map:
                doc_map[src] = {
                    "source": src,
                    "file_path": m.get("file_path", src),
                    "chunk_count": 0,
                    "pages": set(),
                    "has_visuals": False,
                    "visual_count": 0,
                }
            doc_map[src]["chunk_count"] += 1
            if page is not None:
                doc_map[src]["pages"].add(page)
            if m.get("has_visuals"):
                doc_map[src]["has_visuals"] = True
                doc_map[src]["visual_count"] += int(m.get("visual_count", 1))

        doc_list = []
        for src, d in doc_map.items():
            doc_list.append({
                "source": d["source"],
                "file_path": d["file_path"],
                "chunk_count": d["chunk_count"],
                "page_count": len(d["pages"]) if d["pages"] else 1,
                "has_visuals": d["has_visuals"],
                "visual_count": d["visual_count"],
            })

        return {
            "total_chunks": len(metadatas),
            "total_documents": len(doc_list),
            "documents": doc_list,
        }
    except Exception as e:
        return {
            "total_chunks": assistant.count(),
            "total_documents": 0,
            "documents": [],
            "error": str(e),
        }


# --- Visitor workspaces: whose documents, files and jobs a caller may use ---
# The bundled sample is indexed into this workspace so every visitor can read it.
SHARED_WORKSPACE = "shared"


def _workspace_filter(workspace: Optional[str]) -> Optional[Dict[str, Any]]:
    """Chroma filter for the caller's chunks plus the shared sample; None = no limit (local development)."""
    if workspace is None:
        return None
    return {"workspace": {"$in": [workspace, SHARED_WORKSPACE]}}


def _upload_dir(workspace: Optional[str]) -> Path:
    # Workspace IDs are 32 hex characters (checked when the cookie is read), so they are safe as a folder name.
    root = Path("./data/uploads").resolve()
    return root / workspace if workspace else root


def _in_library(path: Path, workspace: Optional[str]) -> bool:
    """True for files in the caller's upload folder or the bundled samples."""
    target = path.resolve()
    roots = [_upload_dir(workspace), Path("./sample_materials").resolve()]
    return target.is_file() and any(target == root or root in target.parents for root in roots)


def _ingest_kwargs(assistant: Any, workspace: Optional[str], progress_callback: Optional[Any] = None) -> Dict[str, Any]:
    import inspect

    kwargs: Dict[str, Any] = {}
    if progress_callback is not None and "progress_callback" in inspect.signature(assistant.ingest).parameters:
        kwargs["progress_callback"] = progress_callback
    if workspace is not None:
        kwargs["metadata"] = {"workspace": workspace}
    return kwargs


# Ingestion Job Tracker
ingestion_jobs: Dict[str, Dict[str, Any]] = {}
_job_workspace: Dict[str, Optional[str]] = {}
_ingest_lock = threading.Lock()
_log = logging.getLogger("study.ingest")
MAX_UPLOAD_BYTES = int(float(os.getenv("MAX_UPLOAD_MB", "200")) * 1024 * 1024)


def _run_ingestion_worker(job_id: str, target_path: Path, filename: str, workspace: Optional[str] = None) -> None:
    """Index one book at a time so ChromaDB is not written by two uploads at once."""
    assistant = get_assistant()
    try:
        with _ingest_lock:
            def on_progress(processed: int, total: int) -> None:
                if job_id in ingestion_jobs:
                    ingestion_jobs[job_id]["processed_chunks"] = processed
                    ingestion_jobs[job_id]["total_chunks"] = total
                    ingestion_jobs[job_id]["progress"] = int((processed / total) * 100) if total > 0 else 0
                    ingestion_jobs[job_id]["updated_at"] = time.time()

            if hasattr(assistant, "ingest"):
                result = assistant.ingest(target_path, **_ingest_kwargs(assistant, workspace, on_progress))
            else:
                result = {"status": "success", "file": filename}

            if job_id in ingestion_jobs:
                ingestion_jobs[job_id]["status"] = "completed"
                ingestion_jobs[job_id]["progress"] = 100
                ingestion_jobs[job_id]["result"] = result
                ingestion_jobs[job_id]["completed_at"] = time.time()
                ingestion_jobs[job_id]["total_indexed_chunks"] = (
                    assistant.count() if hasattr(assistant, "count") else 0
                )
    except Exception as e:
        _log.exception("Ingestion failed for %s", filename)
        if job_id in ingestion_jobs:
            ingestion_jobs[job_id]["status"] = "failed"
            ingestion_jobs[job_id]["error"] = str(e)
            ingestion_jobs[job_id]["completed_at"] = time.time()


async def _save_upload(file: UploadFile, target_path: Path) -> int:
    """Write an upload to disk in chunks so a textbook is not held entirely in memory."""
    size = 0
    try:
        with open(target_path, "wb") as handle:
            while True:
                block = await file.read(1024 * 1024)
                if not block:
                    break
                size += len(block)
                if size > MAX_UPLOAD_BYTES:
                    raise HTTPException(
                        status_code=413,
                        detail=f"This file is larger than {MAX_UPLOAD_BYTES // (1024 * 1024)} MB. Split the book or upload a smaller PDF.",
                    )
                handle.write(block)
    except Exception:
        if target_path.exists():
            target_path.unlink(missing_ok=True)
        raise
    if size == 0:
        target_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="The uploaded file is empty.")
    return size


@app.post("/api/ingest/file")
async def ingest_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    background: bool = Query(False),
    workspace: Optional[str] = Depends(current_workspace),
) -> Dict[str, Any]:
    """Upload and ingest a course file (PDF, TXT, MD, PNG, JPG, JPEG) into the vector store.

    If background=True, starts a background ingestion job and returns job_id immediately.
    If background=False, processes ingestion synchronously in worker threadpool without blocking asyncio loop.
    """
    allowed_exts = {".pdf", ".txt", ".md", ".png", ".jpg", ".jpeg", ".pptx", ".ppt"}
    filename = file.filename or "upload.txt"
    ext = Path(filename).suffix.lower()

    if ext not in allowed_exts:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format '{ext}'. Supported formats: {', '.join(sorted(allowed_exts))}",
        )

    upload_dir = _upload_dir(workspace)
    upload_dir.mkdir(parents=True, exist_ok=True)
    safe_name = Path(filename).name
    target_path = upload_dir / safe_name
    await _save_upload(file, target_path)

    job_id = str(uuid.uuid4())
    _job_workspace[job_id] = workspace
    ingestion_jobs[job_id] = {
        "job_id": job_id,
        "filename": safe_name,
        "status": "processing",
        "progress": 0,
        "processed_chunks": 0,
        "total_chunks": 0,
        "result": None,
        "error": None,
        "started_at": time.time(),
        "completed_at": None,
    }

    assistant = get_assistant()

    if background:
        threading.Thread(
            target=_run_ingestion_worker,
            args=(job_id, target_path, safe_name, workspace),
            name=f"ingest-{job_id[:8]}",
            daemon=True,
        ).start()
        return {
            "success": True,
            "job_id": job_id,
            "status": "processing",
            "filename": safe_name,
            "total_indexed_chunks": assistant.count() if hasattr(assistant, "count") else 0,
        }

    try:
        def on_progress(processed: int, total: int) -> None:
            if job_id in ingestion_jobs:
                ingestion_jobs[job_id]["processed_chunks"] = processed
                ingestion_jobs[job_id]["total_chunks"] = total
                ingestion_jobs[job_id]["progress"] = int((processed / total) * 100) if total > 0 else 0
                ingestion_jobs[job_id]["updated_at"] = time.time()

        result = await run_in_threadpool(
            assistant.ingest, target_path, **_ingest_kwargs(assistant, workspace, on_progress)
        )

        ingestion_jobs[job_id]["status"] = "completed"
        ingestion_jobs[job_id]["progress"] = 100
        ingestion_jobs[job_id]["result"] = result
        ingestion_jobs[job_id]["completed_at"] = time.time()
        return {
            "success": True,
            "job_id": job_id,
            "filename": filename,
            "result": result,
            "total_indexed_chunks": assistant.count() if hasattr(assistant, "count") else 0,
        }
    except Exception as e:
        ingestion_jobs[job_id]["status"] = "failed"
        ingestion_jobs[job_id]["error"] = str(e)
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


@app.get("/api/ingest/status/{job_id}")
def get_ingestion_status(job_id: str, workspace: Optional[str] = Depends(current_workspace)) -> Dict[str, Any]:
    """Retrieve real-time progress and completion status for an ingestion job."""
    if job_id not in ingestion_jobs or _job_workspace.get(job_id) != workspace:
        raise HTTPException(status_code=404, detail=f"Ingestion job '{job_id}' not found.")
    return ingestion_jobs[job_id]


@app.get("/api/ingest/jobs")
def list_ingestion_jobs(workspace: Optional[str] = Depends(current_workspace)) -> Dict[str, Any]:
    """List the caller's recent ingestion jobs."""
    mine = [job for job_id, job in ingestion_jobs.items() if _job_workspace.get(job_id) == workspace]
    return {"jobs": mine[-25:]}


@app.post("/api/ingest/sample")
def ingest_sample() -> Dict[str, Any]:
    """Ingest the bundled Kepler Astronomy sample material for instant testing."""
    sample_path = Path("./sample_materials/astronomy_kepler.txt").resolve()
    if not sample_path.exists():
        raise HTTPException(status_code=404, detail="Sample material file not found.")

    assistant = get_assistant()
    if _is_indexed(assistant, sample_path):
        # Already in the library: re-embedding it would only spend credits.
        return {
            "success": True,
            "material": "Kepler's Laws of Planetary Motion",
            "result": {"source": sample_path.name, "status": "already_indexed"},
            "total_indexed_chunks": assistant.count(),
        }
    try:
        result = assistant.ingest(sample_path, metadata={"workspace": SHARED_WORKSPACE})
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}") from exc
    return {
        "success": True,
        "material": "Kepler's Laws of Planetary Motion",
        "result": result,
        "total_indexed_chunks": assistant.count(),
    }


def _is_indexed(assistant: Any, path: Path) -> bool:
    try:
        found = assistant.indexer.collection.get(where={"file_path": str(path)}, limit=1)
    except Exception:
        return False
    return bool(isinstance(found, dict) and found.get("ids"))


@app.post("/api/ask")
def ask_question(req: AskRequest, workspace: Optional[str] = Depends(current_workspace)) -> Dict[str, Any]:
    """Generate a grounded academic response with verified inline citations."""
    _claim_session(req.session_id, workspace)
    assistant = get_assistant()
    start_time = time.time()

    RAG_QUERIES_TOTAL.labels(mode=req.augment_mode, use_web=str(req.use_web).lower()).inc()

    res = assistant.ask(
        query=req.query,
        stream=False,
        top_k=req.top_k,
        top_n=req.top_n,
        augment=req.augment,
        augment_mode=req.augment_mode,
        min_similarity=req.min_similarity,
        min_rerank_score=req.min_rerank_score,
        use_web=req.use_web,
        fallback_to_web=req.fallback_to_web,
        session_id=req.session_id,
        **_where_kwargs(workspace),
    )

    duration_ms = round((time.time() - start_time) * 1000, 1)
    verification = verify_math_solution(res.get("answer", ""))

    return {
        **res,
        "provider_label": _provider_label(assistant.generator),
        "duration_ms": duration_ms,
        "verification": verification,
    }


@app.post("/api/search")
def search_documents(req: SearchRequest, workspace: Optional[str] = Depends(current_workspace)) -> Dict[str, Any]:
    """Retrieve and rerank candidate document chunks without LLM synthesis."""
    assistant = get_assistant()
    chunks = assistant.search(
        query=req.query,
        top_k=req.top_k,
        top_n=req.top_n,
        augment=req.augment,
        augment_mode=req.augment_mode,
        use_web=req.use_web,
        **_where_kwargs(workspace),
    )
    return {
        "query": req.query,
        "total_chunks": len(chunks),
        "chunks": chunks,
    }


def _history_and_chunks(
    assistant: Any,
    query: str,
    session_id: Optional[str],
    mode: str,
    top_k: int,
    top_n: int,
    augment: bool,
    augment_mode: str,
    workspace: Optional[str] = None,
) -> tuple[List[Dict[str, str]], List[Dict[str, Any]]]:
    """Load chat history, condense a follow-up question, and retrieve chunks (blocking I/O)."""
    from src.memory.session_store import get_session_store

    history = get_session_store().get_recent_history(session_id, turns=4) if session_id else []

    search_query = query.strip()
    augmenter = getattr(assistant, "augmenter", None) or getattr(assistant.retriever, "augmenter", None)
    if history and augmenter and hasattr(augmenter, "condense_conversational_query"):
        search_query = augmenter.condense_conversational_query(search_query, history=history)

    chunks = _retrieve_for_mode(
        assistant,
        query=search_query,
        mode=mode,
        top_k=top_k,
        top_n=top_n,
        augment=augment,
        augment_mode=augment_mode,
        workspace=workspace,
    )
    return history, chunks


def _save_notice_turn(session_id: str, query: str, notice: str, mode: str) -> None:
    from src.memory.session_store import get_session_store

    store = get_session_store()
    store.add_message(session_id, "user", query.strip())
    store.add_message(session_id, "tutor", notice, grounding=mode if mode != "both" else "course")


@app.get("/api/ask/stream")
async def ask_stream(
    request: Request,
    query: str = Query(..., max_length=MAX_QUERY_CHARS, description="Student question"),
    top_k: int = Query(15, ge=1, le=50),
    top_n: int = Query(5, ge=1, le=20),
    augment: bool = Query(True),
    augment_mode: str = Query("expand"),
    use_web: bool = Query(False),
    fallback_to_web: bool = Query(True),
    source_mode: str = Query("", description="notes, web, or both. Overrides use_web when set."),
    session_id: Optional[str] = Query(None, description="Optional chat session ID for multi-turn conversational memory"),
    workspace: Optional[str] = Depends(current_workspace),
):
    """Server-Sent Events (SSE) token-by-token streaming endpoint for real-time typewriter UI."""
    # Checked before streaming starts, so a foreign session ID gets a plain 404 instead of an event stream.
    await run_in_threadpool(_claim_session, session_id, workspace)
    assistant = await run_in_threadpool(get_assistant)

    async def event_generator():
        try:
            # Check empty guards
            if not query or not query.strip():
                yield f"data: {json.dumps({'type': 'token', 'token': 'Please provide a valid question.'})}\n\n"
                yield f"data: {json.dumps({'type': 'done', 'sources': []})}\n\n"
                return

            if await request.is_disconnected():
                return

            mode = _source_mode(source_mode, use_web, fallback_to_web)
            history, chunks = await run_in_threadpool(
                _history_and_chunks,
                assistant,
                query=query,
                session_id=session_id,
                mode=mode,
                top_k=top_k,
                top_n=top_n,
                augment=augment,
                augment_mode=augment_mode,
                workspace=workspace,
            )
            if not chunks:
                notice = _empty_retrieval_notice(mode)
                yield f"data: {json.dumps({'type': 'token', 'token': notice})}\n\n"
                if session_id:
                    await run_in_threadpool(_save_notice_turn, session_id, query, notice, mode)
                yield f"data: {json.dumps({'type': 'done', 'answer': notice, 'sources': [], 'references': [], 'grounding': mode if mode != 'both' else 'course', 'session_id': session_id})}\n\n"
                return

            if await request.is_disconnected():
                return

            # Send candidate chunks event
            chunk_previews = [
                {
                    "source": c.get("metadata", {}).get("source", "Unknown"),
                    "page": c.get("metadata", {}).get("page", 1),
                    "score": c.get("rerank_score", c.get("similarity", 0.0)),
                    "preview": c.get("text", "")[:180] + "...",
                    "source_type": c.get("metadata", {}).get("source_type", "document"),
                }
                for c in chunks
            ]
            yield f"data: {json.dumps({'type': 'chunks', 'chunks': chunk_previews})}\n\n"
            async for event in _stream_answer(
                assistant,
                query,
                chunks,
                infer_grounding(chunks),
                request=request,
                session_id=session_id,
                history=history,
            ):
                yield event
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'error': str(e)})}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/materials/file")
def get_material_file(
    path: str = Query(..., description="Absolute path of an indexed course file"),
    workspace: Optional[str] = Depends(current_workspace),
):
    """Serve a course file so a citation can open the PDF at a page."""
    target = Path(path).resolve()
    if not _in_library(target, workspace):
        raise HTTPException(status_code=404, detail="File is not in the course library.")
    return FileResponse(target)


@app.get("/api/materials/page")
def get_material_page(
    path: str = Query(..., description="Absolute path of an indexed PDF"),
    page: int = Query(1, ge=1),
    workspace: Optional[str] = Depends(current_workspace),
):
    """Return a JPEG of one cited book page."""
    target = Path(path)
    if not _in_library(target, workspace):
        raise HTTPException(status_code=404, detail="File is not in the course library.")
    image = render_pdf_page(str(target), page)
    if image is None:
        raise HTTPException(status_code=404, detail="That page could not be drawn.")
    return FileResponse(image, media_type="image/jpeg")


@app.get("/api/plots/{name}")
def get_plot(name: str):
    """Serve a graph rendered for an explanation."""
    if not re.fullmatch(r"[0-9a-f]{20}\.png", name):
        raise HTTPException(status_code=404, detail="Plot not found.")
    image = Path("./data/plot_cache") / name
    if not image.is_file():
        raise HTTPException(status_code=404, detail="Plot not found.")
    return FileResponse(image, media_type="image/png")


@app.post("/api/flashcards")
def generate_flashcards(req: FlashcardsRequest, workspace: Optional[str] = Depends(current_workspace)) -> Dict[str, Any]:
    """Generate grounded active-recall flashcards from indexed course materials."""
    assistant = get_assistant()
    if assistant.count() == 0:
        raise HTTPException(
            status_code=400,
            detail="Vector database is empty. Ingest course documents first before generating flashcards.",
        )

    topic = req.topic or "core principles and key formulas"
    chunks = assistant.search(query=topic, top_k=10, top_n=5, **_where_kwargs(workspace))
    context = assistant.retriever.format_context(chunks)

    # Prompt structured flashcards
    prompt = (
        f"You are an expert academic tutor. Based strictly on the provided context below, generate {req.count} "
        "high-yield active recall flashcards for exam preparation.\n\n"
        "Return ONLY a valid JSON array of objects with the following schema, and no other text:\n"
        '[\n  {\n    "question": "Clear concept question or definition prompt",\n'
        '    "answer": "Concise, accurate pedagogical explanation with key formulas if any",\n'
        '    "category": "Topic Name",\n'
        '    "difficulty": "Easy|Medium|Hard",\n'
        '    "source": "Filename and Page from context"\n  }\n]\n\n'
        f"Context:\n{context}\n\nFlashcards JSON:"
    )

    try:
        llm_client = assistant.generator.client
        model = assistant.generator.model
        resp = llm_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a precise academic study assistant that outputs strict JSON."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        content = resp.choices[0].message.content.strip()
        # Parse JSON
        if content.startswith("```json"):
            content = content.replace("```json", "", 1).rsplit("```", 1)[0].strip()
        elif content.startswith("```"):
            content = content.replace("```", "", 1).rsplit("```", 1)[0].strip()

        data = json.loads(content)
        cards = []
        for idx, item in enumerate(data):
            cards.append({
                "id": f"card-{idx + 1}",
                "question": item.get("question", ""),
                "answer": item.get("answer", ""),
                "category": item.get("category", "General"),
                "difficulty": item.get("difficulty", "Medium"),
                "source": item.get("source", "Course Materials"),
            })
        return {"flashcards": cards, "total": len(cards)}
    except Exception as e:
        # Fallback heuristic flashcards if LLM JSON format fails
        cards = []
        for idx, chunk in enumerate(chunks[:req.count]):
            source = chunk.get("metadata", {}).get("source", "Course Material")
            page = chunk.get("metadata", {}).get("page", 1)
            text_lines = [line.strip() for line in chunk.get("text", "").split("\n") if line.strip()]
            header = text_lines[0] if text_lines else f"Concept {idx + 1}"
            explanation = " ".join(text_lines[1:]) if len(text_lines) > 1 else text_lines[0]
            cards.append({
                "id": f"card-{idx + 1}",
                "question": f"Explain: {header}",
                "answer": explanation[:300] + ("..." if len(explanation) > 300 else ""),
                "category": "Course Concept",
                "difficulty": "Medium",
                "source": f"{source} (p. {page})",
            })
        return {"flashcards": cards, "total": len(cards), "fallback": True}


@app.post("/api/clear", dependencies=[Depends(require_admin)])
def clear_database() -> Dict[str, Any]:
    """Clear all indexed document chunks from ChromaDB."""
    assistant = get_assistant()
    assistant.clear()
    return {"success": True, "count": assistant.count()}


# --- Chat Memory & Session Routes ---


@app.get("/api/chat/sessions")
def list_chat_sessions(workspace: Optional[str] = Depends(current_workspace)) -> List[Dict[str, Any]]:
    """List the caller's saved chat sessions ordered by last updated."""
    from src.memory.session_store import get_session_store
    return get_session_store().list_sessions(workspace=workspace)


@app.post("/api/chat/sessions")
def create_chat_session(
    req: Optional[CreateSessionRequest] = None,
    workspace: Optional[str] = Depends(current_workspace),
) -> Dict[str, Any]:
    """Create a new chat session."""
    from src.memory.session_store import get_session_store
    store = get_session_store()
    title = req.title if req and req.title else "New Chat"
    session_id = req.session_id if req else None
    if session_id and store.session_owner(session_id)[0]:
        raise HTTPException(status_code=409, detail="That session ID is already in use.")
    return store.create_session(session_id=session_id, title=title, workspace=workspace)


@app.get("/api/chat/sessions/{session_id}")
def get_chat_session(session_id: str, workspace: Optional[str] = Depends(current_workspace)) -> Dict[str, Any]:
    """Fetch a chat session and its full message history."""
    from src.memory.session_store import get_session_store
    store = get_session_store()
    session = store.get_session(session_id, workspace=workspace)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    messages = store.get_session_messages(session_id)
    return {
        **session,
        "messages": messages,
    }


@app.patch("/api/chat/sessions/{session_id}")
def rename_chat_session(
    session_id: str, req: RenameSessionRequest, workspace: Optional[str] = Depends(current_workspace)
) -> Dict[str, Any]:
    """Rename a chat session title."""
    from src.memory.session_store import get_session_store
    store = get_session_store()
    success = store.rename_session(session_id, req.title, workspace=workspace)
    if not success:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"success": True, "id": session_id, "title": req.title}


@app.delete("/api/chat/sessions/{session_id}")
def delete_chat_session(session_id: str, workspace: Optional[str] = Depends(current_workspace)) -> Dict[str, Any]:
    """Delete a chat session and all its messages."""
    from src.memory.session_store import get_session_store
    store = get_session_store()
    success = store.delete_session(session_id, workspace=workspace)
    if not success:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"success": True, "id": session_id}


# --- Single-process hosting: serve the static frontend build and pre-index the sample ---
# `STATIC_EXPORT=1 npm run build` in frontend/ produces frontend/out. When present, FastAPI serves it
# at "/" (after every API route, so /api/* and /metrics win). AUTO_INGEST_SAMPLE=true indexes the
# bundled sample on startup so a fresh, ephemeral deployment is never empty.
_FRONTEND_DIR = Path(os.getenv("FRONTEND_DIR", "./frontend/out"))


@app.on_event("startup")
def _auto_ingest_sample() -> None:
    if os.getenv("AUTO_INGEST_SAMPLE", "false").strip().lower() not in {"1", "true", "yes"}:
        return

    def worker() -> None:
        log = logging.getLogger(__name__)
        for attempt in range(1, 4):  # the hosted embedding API stalls now and then; retry
            try:
                if get_assistant().count() == 0:
                    ingest_sample()
                return
            except Exception as exc:  # best effort: the demo still works without the sample
                log.warning("Sample auto-ingest attempt %d failed: %s", attempt, exc)
                time.sleep(5 * attempt)

    threading.Thread(target=worker, name="auto-ingest-sample", daemon=True).start()


if _FRONTEND_DIR.is_dir():
    from fastapi.staticfiles import StaticFiles

    app.mount("/", StaticFiles(directory=str(_FRONTEND_DIR), html=True), name="frontend")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
