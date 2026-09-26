from typing import Any, Dict, List, Optional
from src.ingestion.embedder import NvidiaEmbedder
from src.ingestion.indexer import ChromaIndexer
from src.retrieval.query_augmenter import QueryAugmenter


class StudyRetriever:
    """Retrieves relevant document chunks from ChromaDB for a given user query."""

    def __init__(
        self,
        indexer: Optional[ChromaIndexer] = None,
        embedder: Optional[NvidiaEmbedder] = None,
        augmenter: Optional[QueryAugmenter] = None,
        top_k: int = 5,
        score_threshold: Optional[float] = None,
    ):
        """Initialize the retriever component.

        Args:
            indexer: ChromaIndexer instance managing ChromaDB operations.
            embedder: NvidiaEmbedder instance to vectorize user queries.
            augmenter: QueryAugmenter instance for query rewriting and expansion.
            top_k: Default number of relevant chunks to retrieve.
            score_threshold: Minimum cosine similarity score threshold (between 0.0 and 1.0).
        """
        self.indexer = indexer or ChromaIndexer()
        self.embedder = embedder or NvidiaEmbedder()
        self.augmenter = augmenter
        self.top_k = top_k
        self.score_threshold = score_threshold

    def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        where: Optional[Dict[str, Any]] = None,
        score_threshold: Optional[float] = None,
        augment: bool = False,
        augment_mode: str = "expand",
    ) -> List[Dict[str, Any]]:
        """Retrieve the most relevant document chunks for a natural language query.

        Args:
            query: The user query string (e.g., 'What are Newton's laws of motion?').
            top_k: Number of chunks to retrieve for this specific query (falls back to self.top_k).
            where: Metadata filter (e.g., {'source': 'physics_lecture1.pdf'}).
            score_threshold: Optional minimum similarity threshold for filtering low-relevance chunks.
            augment: If True, uses the QueryAugmenter to expand or rewrite query before retrieval.
            augment_mode: Augmentation strategy ('expand', 'rewrite', 'hyde').

        Returns:
            List of ranked chunk dictionaries containing id, text, metadata, distance, and similarity_score.
        """
        # If augmentation is requested and an augmenter is available, route to retrieve_augmented
        if augment and self.augmenter is not None:
            return self.retrieve_augmented(
                query=query,
                mode=augment_mode,
                top_k=top_k,
                where=where,
                score_threshold=score_threshold,
            )

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

    def retrieve_augmented(
        self,
        query: str,
        mode: str = "expand",
        top_k: Optional[int] = None,
        where: Optional[Dict[str, Any]] = None,
        score_threshold: Optional[float] = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """Retrieve chunks using query augmentation (multi-query expansion, rewrite, or HyDE).

        Expands or rewrites the query, searches vector store for each variation,
        deduplicates chunks, and ranks them by similarity score.

        Args:
            query: Original user query.
            mode: Augmentation strategy ('expand', 'rewrite', 'hyde').
            top_k: Maximum number of merged chunks to return.
            where: Metadata filter.
            score_threshold: Minimum similarity score threshold.
            **kwargs: Additional parameters passed to QueryAugmenter.augment() (e.g. num_queries).

        Returns:
            Deduplicated, ranked list of chunks.
        """
        if not query or not query.strip():
            return []

        limit = top_k if top_k is not None else self.top_k
        threshold = score_threshold if score_threshold is not None else self.score_threshold

        # If no augmenter is configured, fall back to standard single-query retrieval
        if self.augmenter is None:
            return self.retrieve(
                query=query,
                top_k=limit,
                where=where,
                score_threshold=threshold,
                augment=False,
            )

        # 1. Generate augmented query variations
        augmented_queries = self.augmenter.augment(query=query, mode=mode, **kwargs)

        if not augmented_queries:
            augmented_queries = [query.strip()]

        # 2. Retrieve chunks for each query and aggregate with max-score pooling
        merged_chunks: Dict[str, Dict[str, Any]] = {}

        for sub_query in augmented_queries:
            sub_results = self.retrieve(
                query=sub_query,
                top_k=limit,
                where=where,
                score_threshold=None,  # Filter threshold after pooling
                augment=False,
            )

            for chunk in sub_results:
                chunk_id = (
                    chunk.get("id")
                    or (chunk.get("metadata") or {}).get("id")
                    or chunk.get("text", "")
                )

                if chunk_id not in merged_chunks:
                    merged_chunks[chunk_id] = dict(chunk)
                else:
                    existing_score = merged_chunks[chunk_id].get("similarity_score", 0.0)
                    new_score = chunk.get("similarity_score", 0.0)
                    if new_score > existing_score:
                        merged_chunks[chunk_id]["similarity_score"] = new_score
                        if "distance" in chunk:
                            merged_chunks[chunk_id]["distance"] = chunk["distance"]

        # 3. Apply score threshold filtering
        final_results = list(merged_chunks.values())
        if threshold is not None:
            final_results = [
                item for item in final_results
                if item.get("similarity_score", 0.0) >= threshold
            ]

        # 4. Sort by similarity score descending and limit to top_k
        final_results.sort(
            key=lambda x: x.get("similarity_score", 0.0),
            reverse=True,
        )

        return final_results[:limit]

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
