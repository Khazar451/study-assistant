import argparse
import hashlib
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from langchain_core.documents import Document

from src.ingestion.chunker import chunk_documents
from src.ingestion.embedder import NvidiaEmbedder
from src.ingestion.indexer import ChromaIndexer
from src.ingestion.loader import DocumentLoader


class DocumentChunker:
    """Configurable wrapper around chunk_documents."""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_documents(self, documents: List[Document]) -> List[Document]:
        return chunk_documents(
            documents, chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )


class IngestionPipeline:
    """Orchestrates: DocumentLoader -> DocumentChunker -> NvidiaEmbedder -> ChromaIndexer."""

    def __init__(
        self,
        loader: Optional[DocumentLoader] = None,
        chunker: Optional[Union[DocumentChunker, Callable]] = None,
        embedder: Optional[NvidiaEmbedder] = None,
        indexer: Optional[ChromaIndexer] = None,
        batch_size: int = 32,
    ):
        self.loader = loader or DocumentLoader()
        self.chunker = chunker or DocumentChunker()
        self.embedder = embedder or NvidiaEmbedder()
        self.indexer = indexer or ChromaIndexer()
        self.batch_size = batch_size

    def _generate_chunk_id(self, file_path: str, index: int) -> str:
        """Deterministic chunk ID to guarantee idempotency across re-ingestion."""
        path_hash = hashlib.sha256(file_path.encode("utf-8")).hexdigest()[:12]
        return f"{path_hash}_{index}"

    def ingest_file(
        self,
        file_path: Union[str, Path],
        chunk_size: Optional[int] = None,
        chunk_overlap: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Ingest a single document into the vector database.

        1. Loads raw documents with uniform metadata schema.
        2. Splits documents into chunks, preserving page numbers.
        3. Deletes any pre-existing chunks for this file to eliminate ghost chunks.
        4. Assigns deterministic chunk IDs.
        5. Embeds and upserts in safe batch sizes.
        """
        path = Path(file_path).resolve()
        raw_docs = self.loader.load(path)

        # Apply custom chunk settings if provided
        if chunk_size is not None and hasattr(self.chunker, "chunk_size"):
            self.chunker.chunk_size = chunk_size
        if chunk_overlap is not None and hasattr(self.chunker, "chunk_overlap"):
            self.chunker.chunk_overlap = chunk_overlap

        # Split documents
        if hasattr(self.chunker, "split_documents"):
            chunks = self.chunker.split_documents(raw_docs)
        elif callable(self.chunker):
            chunks = self.chunker(raw_docs)
        else:
            chunks = chunk_documents(raw_docs)

        if not chunks:
            return {"source": path.name, "chunks_count": 0, "status": "empty"}

        # Eliminate ghost chunks by removing prior entries for this file
        if hasattr(self.indexer, "delete_by_path"):
            self.indexer.delete_by_path(str(path))
        elif hasattr(self.indexer, "delete"):
            self.indexer.delete(where={"file_path": str(path)})

        # Assign deterministic IDs, chunk indices, and ensure no None metadata values
        for idx, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = idx
            chunk.metadata["chunk_id"] = self._generate_chunk_id(str(path), idx)
            # Remove any keys with None value for ChromaDB compliance
            chunk.metadata = {k: v for k, v in chunk.metadata.items() if v is not None}

        # Embed and index in safe batch sizes
        for i in range(0, len(chunks), self.batch_size):
            batch = chunks[i : i + self.batch_size]
            texts = [c.page_content for c in batch]

            if hasattr(self.embedder, "embed_documents"):
                embeddings = self.embedder.embed_documents(texts)
            elif hasattr(self.embedder, "embed_batch"):
                embeddings = self.embedder.embed_batch(texts)
            else:
                embeddings = None

            chunk_ids = [c.metadata["chunk_id"] for c in batch]
            metadatas = [c.metadata for c in batch]

            if hasattr(self.indexer, "upsert"):
                self.indexer.upsert(
                    ids=chunk_ids,
                    documents=texts,
                    embeddings=embeddings,
                    metadatas=metadatas,
                )
            else:
                self.indexer.add_chunks(batch)

        return {
            "source": path.name,
            "chunks_count": len(chunks),
            "status": "success",
        }

    def ingest_directory(
        self, dir_path: Union[str, Path], recursive: bool = True
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Ingest all supported documents in a directory with error isolation."""
        target = Path(dir_path).resolve()
        if not target.is_dir():
            raise NotADirectoryError(f"Directory not found: {target}")

        pattern = "**/*" if recursive else "*"
        results: Dict[str, List[Dict[str, Any]]] = {"successful": [], "failed": []}

        files = sorted(target.glob(pattern))
        for file_path in files:
            if (
                file_path.is_file()
                and file_path.suffix.lower() in DocumentLoader.SUPPORTED_EXTENSIONS
            ):
                try:
                    summary = self.ingest_file(file_path)
                    results["successful"].append(summary)
                except Exception as exc:
                    results["failed"].append(
                        {"source": file_path.name, "error": str(exc)}
                    )

        return results


def main():
    parser = argparse.ArgumentParser(description="Study Assistant Data Ingestion Pipeline")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--file", "-f", type=str, help="Path to a single file to ingest")
    group.add_argument("--dir", "-d", type=str, help="Path to directory of files to ingest")
    parser.add_argument("--batch-size", "-b", type=int, default=32, help="Embedding batch size")
    args = parser.parse_args()

    pipeline = IngestionPipeline(batch_size=args.batch_size)

    if args.file:
        res = pipeline.ingest_file(args.file)
        print(f"✅ Ingested {res['source']}: {res['chunks_count']} chunks ({res['status']})")
    elif args.dir:
        res = pipeline.ingest_directory(args.dir)
        print(
            f"✅ Ingestion complete: {len(res['successful'])} files succeeded, {len(res['failed'])} files failed."
        )


if __name__ == "__main__":
    main()
