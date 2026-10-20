import os
import time
from typing import Any, List, Optional, Union
from dotenv import load_dotenv
from openai import OpenAI

# Automatically load environment variables from .env.local or .env if present
load_dotenv(".env.local")
load_dotenv()


class NvidiaEmbedder:
    """Embedding client for NVIDIA NIM retrieval models (e.g. llama-nemotron-embed-vl-1b-v2)."""

    DEFAULT_MODEL = "nvidia/llama-nemotron-embed-vl-1b-v2"
    DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ):
        self.api_key = (
            api_key
            or os.getenv("NVIDIA_EMBEDDER_KEY")
            or os.getenv("NVIDIA_API_KEY")
        )
        if not self.api_key:
            raise ValueError(
                "NVIDIA API key not set. Please set NVIDIA_EMBEDDER_KEY or "
                "NVIDIA_API_KEY environment variable, or pass api_key."
            )

        self.base_url = base_url or os.getenv("NVIDIA_BASE_URL", self.DEFAULT_BASE_URL)
        self.model = model or os.getenv("EMBEDDING_MODEL", self.DEFAULT_MODEL)

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=15.0,  # stalled requests are retried; single queries use the shorter QUERY_TIMEOUT below
            max_retries=0,
        )

    # The hosted NIM endpoint intermittently stalls a request (a normal reply takes ~0.3 s),
    # so a single embedding fails fast and is retried instead of hanging the student for minutes.
    QUERY_TIMEOUT = 4.0

    def embed_text(self, text: str, input_type: str = "passage", max_retries: int = 6) -> List[float]:
        last_err: Optional[Exception] = None
        for attempt in range(max_retries):
            try:
                response = self.client.embeddings.create(
                    model=self.model,
                    input=[text],
                    encoding_format="float",
                    extra_body={"input_type": input_type, "truncate": "NONE"},
                    timeout=self.QUERY_TIMEOUT,
                )
                return response.data[0].embedding
            except Exception as exc:
                last_err = exc
                if attempt < max_retries - 1:
                    time.sleep(0.5)
        raise last_err or RuntimeError("Embedding request failed")

    def embed_query(self, query: str) -> List[float]:
        return self.embed_text(query, input_type="query")

    def embed_batch(
        self,
        texts: List[str],
        input_type: str = "passage",
        batch_size: int = 64,
        max_retries: int = 3,
        truncate: str = "NONE",
    ) -> List[List[float]]:
        if not texts:
            return []

        all_embeddings: List[List[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = [self._clip(text) for text in texts[i : i + batch_size]]
            all_embeddings.extend(
                self._embed_resilient(
                    batch,
                    input_type=input_type,
                    truncate=truncate,
                    max_retries=max_retries,
                )
            )

        return all_embeddings

    @staticmethod
    def _clip(text: str, limit: int = 6000) -> str:
        cleaned = (text or "").strip()
        if len(cleaned) <= limit:
            return cleaned or " "
        return cleaned[:limit]

    def _embed_resilient(
        self,
        batch: List[str],
        input_type: str,
        truncate: str,
        max_retries: int,
    ) -> List[List[float]]:
        """Retry a batch, then split it, so one rejected passage does not abort the book."""
        last_err: Optional[Exception] = None
        for attempt in range(max_retries):
            try:
                response = self.client.embeddings.create(
                    model=self.model,
                    input=batch,
                    encoding_format="float",
                    extra_body={"input_type": input_type, "truncate": truncate},
                )
                ordered = sorted(response.data, key=lambda item: getattr(item, "index", 0))
                return [item.embedding for item in ordered]
            except Exception as exc:
                last_err = exc
                if attempt < max_retries - 1:
                    time.sleep(min(20.0, 1.5 * (2 ** attempt)))
        if len(batch) == 1:
            raise last_err or RuntimeError("Embedding request failed")
        mid = max(1, len(batch) // 2)
        return self._embed_resilient(batch[:mid], input_type, truncate, max_retries) + self._embed_resilient(
            batch[mid:], input_type, truncate, max_retries
        )

    def embed_documents(self, texts: List[str], batch_size: int = 64) -> List[List[float]]:
        """Generate embeddings for documents with input_type='passage' (LangChain naming convention)."""
        return self.embed_batch(texts, input_type="passage", batch_size=batch_size)

    def embed_chunks(
        self,
        chunks: List[Union[dict, Any]],
        input_type: str = "passage",
        batch_size: int = 64,
    ) -> List[Union[dict, Any]]:
        if not chunks:
            return []

        # Extract text from dict, LangChain Document, or custom object
        texts = []
        for chunk in chunks:
            if hasattr(chunk, "page_content"):
                texts.append(chunk.page_content)
            elif isinstance(chunk, dict):
                texts.append(chunk.get("text", chunk.get("page_content", "")))
            else:
                texts.append(getattr(chunk, "text", ""))

        embeddings = self.embed_batch(
            texts, input_type=input_type, batch_size=batch_size
        )

        for chunk, emb in zip(chunks, embeddings):
            if isinstance(chunk, dict):
                chunk["embedding"] = emb
            else:
                try:
                    chunk.embedding = emb
                except (AttributeError, ValueError):
                    if hasattr(chunk, "metadata") and isinstance(chunk.metadata, dict):
                        chunk.metadata["embedding"] = emb
                    else:
                        raise

        return chunks
