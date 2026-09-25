import json
import uuid
from typing import Any, Dict, List, Optional
import chromadb
from chromadb.api import ClientAPI


class ChromaIndexer:
    #Vector database indexer and retrieval manager powered by ChromaDB.

    def __init__(
        self,
        collection_name: str = "study_assistant",
        persist_directory: Optional[str] = "./chroma_db",
        client: Optional[ClientAPI] = None,
        distance_metric: str = "cosine",
    ):
        # Initialize ChromaDB client and collection.
        
        self.collection_name = collection_name
        self.persist_directory = persist_directory
        self.distance_metric = distance_metric

        if client is not None:
            self.client = client
        elif persist_directory:
            self.client = chromadb.PersistentClient(path=persist_directory)
        else:
            self.client = chromadb.Client()

        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": self.distance_metric},
        )

    def _extract_chunk_data(self, chunk: Any) -> Dict[str, Any]:
        #Normalize chunk data from LangChain Document, dict, or custom object.
        
        # 1. Extract text / page_content
        if hasattr(chunk, "page_content"):
            text = chunk.page_content
        elif isinstance(chunk, dict):
            text = chunk.get("text", chunk.get("page_content", ""))
        else:
            text = getattr(chunk, "text", "")

        # 2. Extract embedding
        if hasattr(chunk, "embedding") and chunk.embedding is not None:
            embedding = chunk.embedding
        elif isinstance(chunk, dict) and "embedding" in chunk and chunk["embedding"] is not None:
            embedding = chunk["embedding"]
        elif hasattr(chunk, "metadata") and isinstance(chunk.metadata, dict) and "embedding" in chunk.metadata:
            embedding = chunk.metadata["embedding"]
        else:
            embedding = None

        # 3. Extract or generate ID
        chunk_id = None
        if hasattr(chunk, "id") and chunk.id:
            chunk_id = str(chunk.id)
        elif hasattr(chunk, "chunk_id") and chunk.chunk_id:
            chunk_id = str(chunk.chunk_id)
        elif hasattr(chunk, "metadata") and isinstance(chunk.metadata, dict):
            chunk_id = str(chunk.metadata.get("id", chunk.metadata.get("chunk_id", ""))) or None
        elif isinstance(chunk, dict):
            chunk_id = str(chunk.get("id", chunk.get("chunk_id", ""))) or None

        if not chunk_id:
            chunk_id = str(uuid.uuid4())

        # 4. Extract and sanitize metadata
        raw_meta = {}
        if hasattr(chunk, "metadata") and isinstance(chunk.metadata, dict):
            raw_meta = chunk.metadata
        elif isinstance(chunk, dict):
            # Include keys other than 'text', 'page_content', 'embedding', 'id', 'chunk_id'
            reserved = {"text", "page_content", "embedding", "id", "chunk_id"}
            raw_meta = {k: v for k, v in chunk.items() if k not in reserved}

        # ChromaDB requires metadata values to be str, int, float, or bool
        clean_meta = {}
        for k, v in raw_meta.items():
            if k == "embedding":
                continue
            if isinstance(v, (str, int, float, bool)):
                clean_meta[k] = v
            elif v is not None:
                clean_meta[k] = json.dumps(v)

        # ChromaDB requires non-empty metadata dicts if metadatas list is provided
        if not clean_meta:
            clean_meta = {"indexed": True}

        return {
            "id": chunk_id,
            "text": text,
            "embedding": embedding,
            "metadata": clean_meta,
        }

    def add_chunks(self, chunks: List[Any]) -> int:
        # Upsert a list of document chunks into the collection.
        if not chunks:
            return 0

        ids: List[str] = []
        documents: List[str] = []
        embeddings: List[List[float]] = []
        metadatas: List[Dict[str, Any]] = []

        has_embeddings = False
        for chunk in chunks:
            data = self._extract_chunk_data(chunk)
            ids.append(data["id"])
            documents.append(data["text"])
            metadatas.append(data["metadata"])
            if data["embedding"] is not None:
                has_embeddings = True
                embeddings.append(data["embedding"])

        if has_embeddings and len(embeddings) == len(chunks):
            self.collection.upsert(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas,
            )
        else:
            self.collection.upsert(
                ids=ids,
                documents=documents,
                metadatas=metadatas,
            )

        return len(chunks)

    def query(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        total_items = self.count()
        if total_items == 0:
            return []

        n_results = min(top_k, total_items)

        query_kwargs: Dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": n_results,
        }
        if where:
            query_kwargs["where"] = where

        results = self.collection.query(**query_kwargs)

        formatted_results: List[Dict[str, Any]] = []
        if results and results.get("ids") and results["ids"][0]:
            num_matches = len(results["ids"][0])
            for i in range(num_matches):
                formatted_results.append({
                    "id": results["ids"][0][i],
                    "text": results["documents"][0][i] if results.get("documents") else "",
                    "metadata": results["metadatas"][0][i] if results.get("metadatas") else {},
                    "distance": results["distances"][0][i] if results.get("distances") else None,
                })

        return formatted_results

    def count(self) -> int:
        # Return the total number of chunks currently stored in the collection.
        return self.collection.count()

    def delete(
        self,
        ids: Optional[List[str]] = None,
        where: Optional[Dict[str, Any]] = None,
    ) -> None:
        # Delete specific chunks by ID or metadata filter.
        if ids or where:
            self.collection.delete(ids=ids, where=where)

    def delete_by_path(self, file_path: str) -> None:
        """Delete all chunks originating from a specific file to prevent ghost chunks."""
        self.collection.delete(where={"file_path": str(file_path)})

    def upsert(
        self,
        ids: List[str],
        documents: List[str],
        embeddings: Optional[List[List[float]]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Direct upsert into collection."""
        kwargs: Dict[str, Any] = {"ids": ids, "documents": documents}
        if embeddings is not None:
            kwargs["embeddings"] = embeddings
        if metadatas is not None:
            kwargs["metadatas"] = metadatas
        self.collection.upsert(**kwargs)

    def reset(self) -> None:
        # Clear all chunks from the current collection.
        self.client.delete_collection(name=self.collection_name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": self.distance_metric},
        )
