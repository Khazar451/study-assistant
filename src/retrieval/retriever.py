from typing import Any, Dict, List, Optional
from src.ingestion.embedder import NvidiaEmbedder
from src.ingestion.indexer import ChromaIndexer


class StudyRetriever:
    """Retrieves relevant document chunks from ChromaDB for a given user query."""

    def __init__(
        self,
        indexer: Optional[ChromaIndexer] = None,
        embedder: Optional[NvidiaEmbedder] = None,
        top_k: int = 5,
        score_threshold: Optional[float] = None,
    ):
        """Initialize the retriever component.

        Args:
            indexer: ChromaIndexer instance managing ChromaDB operations.
            embedder: NvidiaEmbedder instance to vectorize user queries.
            top_k: Default number of relevant chunks to retrieve.
            score_threshold: Minimum cosine similarity score threshold (between 0.0 and 1.0).
        """
        self.indexer = indexer or ChromaIndexer()
        self.embedder = embedder or NvidiaEmbedder()
        self.top_k = top_k
        self.score_threshold = score_threshold

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        where: Optional[Dict[str, Any]] = None,
        score_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve the most relevant document chunks for a natural language query.

        Args:
            query: The user query string (e.g., 'What are Newton's laws of motion?').
            top_k: Number of chunks to retrieve for this specific query (falls back to self.top_k).
            where: Metadata filter (e.g., {'source': 'physics_lecture1.pdf'}).
            score_threshold: Optional minimum similarity threshold for filtering low-relevance chunks.

        Returns:
            List of ranked chunk dictionaries containing id, text, metadata, distance, and similarity_score.
        """
        # 1. Safely handle empty or whitespace-only queries
        if not query or not query.strip():
            return []

        # 2. Resolve query-level overrides vs default configurations
        limit = top_k if top_k is not None else self.top_k
        threshold = score_threshold if score_threshold is not None else self.score_threshold

        # 3. Vectorize query using NVIDIA NIM model (embed_query automatically uses input_type='query')
        query_vector = self.embedder.embed_query(query.strip())

        # 4. Query vector store (ChromaDB) for top nearest neighbors
        raw_results = self.indexer.query(
            query_embedding=query_vector,
            top_k=limit,
            where=where,
        )

        # 5. Enrich results: convert cosine distance to similarity score [0.0, 1.0]
        enriched_results: List[Dict[str, Any]] = []
        for item in raw_results:
            distance = item.get("distance")
            # For cosine space: distance = 1 - cosine_similarity => similarity = 1 - distance
            if distance is not None:
                similarity = round(max(0.0, min(1.0, 1.0 - distance)), 4)
            else:
                similarity = 1.0

            item["similarity_score"] = similarity

            # 6. Apply similarity threshold filter if configured
            if threshold is not None and similarity < threshold:
                continue

            enriched_results.append(item)

        return enriched_results

    def format_context(self, retrieved_chunks: List[Dict[str, Any]]) -> str:
        """Format retrieved chunks into a clean, structured context string for LLM prompting.

        Attaches origin metadata (source filename and page number) to each chunk,
        enabling the LLM to provide grounded answers with exact academic citations.

        Args:
            retrieved_chunks: List of chunk dictionaries returned by retrieve().

        Returns:
            Formatted context string separated by dividers.
        """
        if not retrieved_chunks:
            return ""

        context_blocks: List[str] = []
        for idx, chunk in enumerate(retrieved_chunks, start=1):
            metadata = chunk.get("metadata", {})
            source = metadata.get("source", "Unknown Source")
            page = metadata.get("page", 1)
            text = chunk.get("text", "").strip()

            block = (
                f"[Document {idx}] (Source: {source} | Page: {page})\n"
                f"{text}"
            )
            context_blocks.append(block)

        return "\n\n---\n\n".join(context_blocks)
