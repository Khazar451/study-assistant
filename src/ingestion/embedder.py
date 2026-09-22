import os
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
        self.api_key = api_key or os.getenv("NVIDIA_API_KEY")
        if not self.api_key:
            raise ValueError(
                "NVIDIA_API_KEY is not set. Please set the environment variable "
                "or pass api_key to NvidiaEmbedder."
            )

        self.base_url = base_url or os.getenv("NVIDIA_BASE_URL", self.DEFAULT_BASE_URL)
        self.model = model or os.getenv("EMBEDDING_MODEL", self.DEFAULT_MODEL)

        self.client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
        )

    def embed_text(self, text: str, input_type: str = "passage") -> List[float]:
        response = self.client.embeddings.create(
            model=self.model,
            input=[text],
            encoding_format="float",
            extra_body={"input_type": input_type, "truncate": "NONE"},
        )
        return response.data[0].embedding

    def embed_query(self, query: str) -> List[float]:
        return self.embed_text(query, input_type="query")

    def embed_batch(
        self,
        texts: List[str],
        input_type: str = "passage",
        batch_size: int = 32,
    ) -> List[List[float]]:
        if not texts:
            return []

        all_embeddings: List[List[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            response = self.client.embeddings.create(
                model=self.model,
                input=batch,
                encoding_format="float",
                extra_body={"input_type": input_type, "truncate": "NONE"},
            )
            all_embeddings.extend([item.embedding for item in response.data])

        return all_embeddings

    def embed_chunks(
        self,
        chunks: List[Union[dict, Any]],
        input_type: str = "passage",
        batch_size: int = 32,
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
