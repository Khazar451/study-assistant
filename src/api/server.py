from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import re
import threading
import time
from typing import Any, Dict, Iterator, List, Optional
import uuid
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, File, HTTPException, Query, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from starlette.concurrency import run_in_threadpool
from pydantic import BaseModel, Field
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

from src.assistant import StudyAssistant
from src.generation.plots import embed_plots, ensure_requested_plot
from src.ingestion.page_images import attach_page_images, is_library_file, render_pdf_page
from src.generation.tutor_format import allowed_source_keys, bind_numbered_citations, infer_grounding, polish_tutor_answer
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

# Enable CORS for Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

# Global singleton instance
_assistant_instance: Optional[StudyAssistant] = None


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

    def embed_batch(self, texts: List[str], input_type: str = "passage", batch_size: int = 32) -> List[List[float]]:
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
    if _assistant_instance is None:
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


class AskRequest(BaseModel):
    query: str = Field(..., description="Student's academic question")
    top_k: int = Field(15, ge=1, le=50, description="Initial vector candidates")
    top_n: int = Field(5, ge=1, le=20, description="Reranked context chunks")
    augment: bool = Field(True, description="Enable query intelligence (rewrite/expand/HyDE)")
    augment_mode: str = Field("expand", description="Strategy: expand, rewrite, or hyde")
    min_similarity: Optional[float] = Field(None, description="Optional minimum cosine similarity")
    min_rerank_score: Optional[float] = Field(None, description="Optional minimum reranker threshold")
    use_web: bool = Field(False, description="Enable Tavily web search integration")
    fallback_to_web: bool = Field(True, description="Fallback to web search if local context is insufficient")


class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query")
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
    topic: Optional[str] = Field(None, description="Optional focus topic")
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
    )
    return chunks


def _empty_retrieval_notice(mode: str) -> str:
    if mode == "web":
        return "No academic web page matched this question. Try different wording, or switch the source to your notes."
    if mode == "both":
        return "Neither your notes nor the academic web had a verified passage for this question."
    return (
        "Based on the available course materials, there is not enough verified information to answer this question. "
        "Please upload the relevant lecture slides or textbook chapters."
    )


def _stream_answer(assistant: Any, query: str, chunks: List[Dict[str, Any]], grounding: str):
    """Stream tokens, then replace the answer with a citation-checked final draft."""
    context = assistant.retriever.format_context(chunks) if chunks else ""
    stream = assistant.generator.generate_stream(query=query.strip(), context=context, grounding=grounding)
    full_text: List[str] = []
    for token in stream:
        full_text.append(token)
        yield f"data: {json.dumps({'type': 'token', 'token': token})}\n\n"

    answer = polish_tutor_answer(
        "".join(full_text),
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
    done_event = {
        "type": "done",
        "answer": answer,
        "sources": sources,
        "references": references,
        "grounding": grounding,
        "model": getattr(assistant.generator, "model", ""),
        "provider": _provider_label(assistant.generator),
        "verification": verification,
    }
    yield f"data: {json.dumps(done_event)}\n\n"


def _provider_label(generator: Any) -> str:
    label = getattr(generator, "provider_label", None)
    if isinstance(label, str) and label:
        return label
    provider = getattr(generator, "provider", "")
    return PROVIDER_LABELS.get(provider, provider) if isinstance(provider, str) else ""


@app.get("/api/documents")
def get_documents() -> Dict[str, Any]:
    """Inspect all documents indexed in the persistent ChromaDB collection."""
    assistant = get_assistant()
    try:
        # ChromaDB collection inspection
        raw = assistant.indexer.collection.get(include=["metadatas"])
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


# Ingestion Job Tracker
ingestion_jobs: Dict[str, Dict[str, Any]] = {}
_ingest_lock = threading.Lock()
_log = logging.getLogger("study.ingest")
MAX_UPLOAD_BYTES = 200 * 1024 * 1024


def _run_ingestion_worker(job_id: str, target_path: Path, filename: str) -> None:
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
                import inspect
                sig = inspect.signature(assistant.ingest)
                if "progress_callback" in sig.parameters:
                    result = assistant.ingest(target_path, progress_callback=on_progress)
                else:
                    result = assistant.ingest(target_path)
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
                        detail="This file is larger than 200 MB. Split the book or upload a smaller PDF.",
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

    upload_dir = Path("./data/uploads").resolve()
    upload_dir.mkdir(parents=True, exist_ok=True)
    safe_name = Path(filename).name
    target_path = upload_dir / safe_name
    await _save_upload(file, target_path)

    job_id = str(uuid.uuid4())
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
            args=(job_id, target_path, safe_name),
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

        import inspect
        sig = inspect.signature(assistant.ingest)
        if "progress_callback" in sig.parameters:
            result = await run_in_threadpool(assistant.ingest, target_path, progress_callback=on_progress)
        else:
            result = await run_in_threadpool(assistant.ingest, target_path)

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
def get_ingestion_status(job_id: str) -> Dict[str, Any]:
    """Retrieve real-time progress and completion status for an ingestion job."""
    if job_id not in ingestion_jobs:
        raise HTTPException(status_code=404, detail=f"Ingestion job '{job_id}' not found.")
    return ingestion_jobs[job_id]


@app.get("/api/ingest/jobs")
def list_ingestion_jobs() -> Dict[str, Any]:
    """List recent ingestion jobs."""
    return {"jobs": list(ingestion_jobs.values())[-25:]}


@app.post("/api/ingest/sample")
def ingest_sample() -> Dict[str, Any]:
    """Ingest the bundled Kepler Astronomy sample material for instant testing."""
    sample_path = Path("./sample_materials/astronomy_kepler.txt").resolve()
    if not sample_path.exists():
        raise HTTPException(status_code=404, detail="Sample material file not found.")

    assistant = get_assistant()
    try:
        result = assistant.ingest(sample_path)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {exc}") from exc
    return {
        "success": True,
        "material": "Kepler's Laws of Planetary Motion",
        "result": result,
        "total_indexed_chunks": assistant.count(),
    }


@app.post("/api/ask")
def ask_question(req: AskRequest) -> Dict[str, Any]:
    """Generate a grounded academic response with verified inline citations."""
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
def search_documents(req: SearchRequest) -> Dict[str, Any]:
    """Retrieve and rerank candidate document chunks without LLM synthesis."""
    assistant = get_assistant()
    chunks = assistant.search(
        query=req.query,
        top_k=req.top_k,
        top_n=req.top_n,
        augment=req.augment,
        augment_mode=req.augment_mode,
        use_web=req.use_web,
    )
    return {
        "query": req.query,
        "total_chunks": len(chunks),
        "chunks": chunks,
    }


@app.get("/api/ask/stream")
def ask_stream(
    query: str = Query(..., description="Student question"),
    top_k: int = Query(15, ge=1, le=50),
    top_n: int = Query(5, ge=1, le=20),
    augment: bool = Query(True),
    augment_mode: str = Query("expand"),
    use_web: bool = Query(False),
    fallback_to_web: bool = Query(True),
    source_mode: str = Query("", description="notes, web, or both. Overrides use_web when set."),
):
    """Server-Sent Events (SSE) token-by-token streaming endpoint for real-time typewriter UI."""
    assistant = get_assistant()

    def event_generator():
        try:
            # Check empty guards
            if not query or not query.strip():
                yield f"data: {json.dumps({'type': 'token', 'token': 'Please provide a valid question.'})}\n\n"
                yield f"data: {json.dumps({'type': 'done', 'sources': []})}\n\n"
                return

            mode = _source_mode(source_mode, use_web, fallback_to_web)
            chunks = _retrieve_for_mode(
                assistant,
                query=query,
                mode=mode,
                top_k=top_k,
                top_n=top_n,
                augment=augment,
                augment_mode=augment_mode,
            )
            if not chunks:
                notice = _empty_retrieval_notice(mode)
                yield f"data: {json.dumps({'type': 'token', 'token': notice})}\n\n"
                yield f"data: {json.dumps({'type': 'done', 'answer': notice, 'sources': [], 'references': [], 'grounding': mode if mode != 'both' else 'course'})}\n\n"
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
            yield from _stream_answer(assistant, query, chunks, infer_grounding(chunks))
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
def get_material_file(path: str = Query(..., description="Absolute path of an indexed course file")):
    """Serve a course file so a citation can open the PDF at a page."""
    target = Path(path).resolve()
    roots = [Path("./data/uploads").resolve(), Path("./sample_materials").resolve()]
    allowed = target.is_file() and any(target == root or root in target.parents for root in roots)
    if not allowed:
        raise HTTPException(status_code=404, detail="File is not in the course library.")
    return FileResponse(target)


@app.get("/api/materials/page")
def get_material_page(
    path: str = Query(..., description="Absolute path of an indexed PDF"),
    page: int = Query(1, ge=1),
):
    """Return a JPEG of one cited book page."""
    target = Path(path)
    if not is_library_file(target):
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
def generate_flashcards(req: FlashcardsRequest) -> Dict[str, Any]:
    """Generate grounded active-recall flashcards from indexed course materials."""
    assistant = get_assistant()
    if assistant.count() == 0:
        raise HTTPException(
            status_code=400,
            detail="Vector database is empty. Ingest course documents first before generating flashcards.",
        )

    topic = req.topic or "core principles and key formulas"
    chunks = assistant.search(query=topic, top_k=10, top_n=5)
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


@app.post("/api/clear")
def clear_database() -> Dict[str, Any]:
    """Clear all indexed document chunks from ChromaDB."""
    assistant = get_assistant()
    assistant.clear()
    return {"success": True, "count": assistant.count()}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
