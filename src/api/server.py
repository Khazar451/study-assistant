import json
import os
from pathlib import Path
import re
import time
from typing import Any, Dict, List, Optional
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from src.assistant import StudyAssistant

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


class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query")
    top_k: int = Field(15, ge=1, le=50)
    top_n: int = Field(5, ge=1, le=20)
    augment: bool = Field(True)
    augment_mode: str = Field("expand")


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


@app.get("/api/status")
def get_status() -> Dict[str, Any]:
    """Return system readiness, cloud providers, and vector store metrics."""
    assistant = get_assistant()
    chunk_count = assistant.count()

    nvidia_key = bool(os.getenv("NVIDIA_API_KEY"))
    nebius_key = bool(os.getenv("NEBIUS_API_KEY"))

    active_provider = "NVIDIA NIM" if nvidia_key else ("Nebius AI Studio" if nebius_key else "Mock / Offline")

    return {
        "status": "ready",
        "provider": active_provider,
        "nvidia_configured": nvidia_key,
        "nebius_configured": nebius_key,
        "indexed_chunks": chunk_count,
        "persist_directory": assistant.persist_directory,
        "generation_model": getattr(assistant.generator, "model", "nvidia/Llama-3.1-Nemotron-70B-Instruct-HF"),
        "rerank_mode": getattr(assistant.reranker, "mode", "llm_listwise"),
    }


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
                }
            doc_map[src]["chunk_count"] += 1
            if page is not None:
                doc_map[src]["pages"].add(page)

        doc_list = []
        for src, d in doc_map.items():
            doc_list.append({
                "source": d["source"],
                "file_path": d["file_path"],
                "chunk_count": d["chunk_count"],
                "page_count": len(d["pages"]) if d["pages"] else 1,
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


@app.post("/api/ingest/file")
async def ingest_file(file: UploadFile = File(...)) -> Dict[str, Any]:
    """Upload and ingest a course file (PDF, TXT, MD) into the vector store."""
    allowed_exts = {".pdf", ".txt", ".md"}
    filename = file.filename or "upload.txt"
    ext = Path(filename).suffix.lower()

    if ext not in allowed_exts:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format '{ext}'. Supported formats: {', '.join(allowed_exts)}",
        )

    upload_dir = Path("./data/uploads").resolve()
    upload_dir.mkdir(parents=True, exist_ok=True)
    target_path = upload_dir / filename

    content = await file.read()
    with open(target_path, "wb") as f:
        f.write(content)

    assistant = get_assistant()
    try:
        result = assistant.ingest(target_path)
        return {
            "success": True,
            "filename": filename,
            "result": result,
            "total_indexed_chunks": assistant.count(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


@app.post("/api/ingest/sample")
def ingest_sample() -> Dict[str, Any]:
    """Ingest the bundled Kepler Astronomy sample material for instant testing."""
    sample_path = Path("./sample_materials/astronomy_kepler.txt").resolve()
    if not sample_path.exists():
        raise HTTPException(status_code=404, detail="Sample material file not found.")

    assistant = get_assistant()
    result = assistant.ingest(sample_path)
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

    res = assistant.ask(
        query=req.query,
        stream=False,
        top_k=req.top_k,
        top_n=req.top_n,
        augment=req.augment,
        augment_mode=req.augment_mode,
        min_similarity=req.min_similarity,
        min_rerank_score=req.min_rerank_score,
    )

    duration_ms = round((time.time() - start_time) * 1000, 1)

    return {
        **res,
        "duration_ms": duration_ms,
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

            if assistant.count() == 0:
                yield f"data: {json.dumps({'type': 'token', 'token': 'Vector database is empty. Please ingest course materials first.'})}\n\n"
                yield f"data: {json.dumps({'type': 'done', 'sources': []})}\n\n"
                return

            # Retrieve and rerank
            chunks = assistant.search(
                query=query,
                top_k=top_k,
                top_n=top_n,
                augment=augment,
                augment_mode=augment_mode,
            )

            # Send candidate chunks event
            chunk_previews = [
                {
                    "source": c.get("metadata", {}).get("source", "Unknown"),
                    "page": c.get("metadata", {}).get("page", 1),
                    "score": c.get("rerank_score", c.get("similarity", 0.0)),
                    "preview": c.get("text", "")[:180] + "...",
                }
                for c in chunks
            ]
            yield f"data: {json.dumps({'type': 'chunks', 'chunks': chunk_previews})}\n\n"

            # Stream generation
            context = assistant.retriever.format_context(chunks)
            stream = assistant.generator.generate_stream(query=query.strip(), context=context)

            full_text = []
            for token in stream:
                full_text.append(token)
                yield f"data: {json.dumps({'type': 'token', 'token': token})}\n\n"

            answer_content = "".join(full_text)
            # Extract citations from generated text
            citations = re.findall(r"\[Source:\s*([^,\]]+),\s*Page:\s*(\d+)\]", answer_content)
            unique_sources = sorted(list(set(f"{src} (p. {page})" for src, page in citations)))

            yield f"data: {json.dumps({'type': 'done', 'sources': unique_sources, 'model': getattr(assistant.generator, 'model', '')})}\n\n"
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
