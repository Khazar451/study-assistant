import argparse
import os
from pathlib import Path
import sys
import time
from typing import Any, Dict, Iterator, List, Optional, Union
from dotenv import load_dotenv

# Load environment variables
load_dotenv(".env.local")
load_dotenv()

from src.ingestion.embedder import NvidiaEmbedder
from src.ingestion.indexer import ChromaIndexer
from src.ingestion.pipeline import IngestionPipeline
from src.retrieval.query_augmenter import QueryAugmenter
from src.retrieval.reranker import StudyReranker
from src.retrieval.retriever import StudyRetriever
from src.generation.generator import StudyGenerator


class StudyAssistant:
    """Unified Facade for the Academic Study Assistant RAG System.

    Coordinates:
    1. Document Ingestion (DocumentLoader -> Chunker -> NvidiaEmbedder -> ChromaDB)
    2. Query Intelligence (Rewrite / Expand / HyDE via QueryAugmenter)
    3. Dense Vector Retrieval (Cosine Similarity & Max-Score Pooling via StudyRetriever)
    4. Cross-Attention Semantic Reranking (via StudyReranker)
    5. Grounded Answer Generation (Grounded Citations via StudyGenerator)
    """

    def __init__(
        self,
        pipeline: Optional[IngestionPipeline] = None,
        retriever: Optional[StudyRetriever] = None,
        reranker: Optional[StudyReranker] = None,
        generator: Optional[StudyGenerator] = None,
        persist_directory: str = "./chroma_db",
    ):
        """Initialize the StudyAssistant with shared or injected subcomponents."""
        self.persist_directory = persist_directory

        # 1. Coordinate Ingestion & Retrieval storage
        if pipeline is not None:
            self.pipeline = pipeline
            self.indexer = getattr(pipeline, "indexer", None) or ChromaIndexer(persist_directory=persist_directory)
            self.embedder = getattr(pipeline, "embedder", None) or NvidiaEmbedder()
        else:
            self.indexer = ChromaIndexer(persist_directory=persist_directory)
            self.embedder = NvidiaEmbedder()
            self.pipeline = IngestionPipeline(
                embedder=self.embedder,
                indexer=self.indexer,
            )

        # 2. Coordinate Retriever & Augmenter
        if retriever is not None:
            self.retriever = retriever
        else:
            self.augmenter = QueryAugmenter()
            self.retriever = StudyRetriever(
                indexer=self.indexer,
                embedder=self.embedder,
                augmenter=self.augmenter,
            )

        # 3. Coordinate Reranker
        self.reranker = reranker or StudyReranker(mode="llm_listwise")

        # 4. Coordinate Generator
        self.generator = generator or StudyGenerator()

    def ingest(self, path: Union[str, Path], batch_size: int = 32) -> Dict[str, Any]:
        """Ingest a single document or an entire directory of course materials.

        Args:
            path: File or directory path to ingest.
            batch_size: Embedding batch size.

        Returns:
            Dictionary detailing ingestion summary.
        """
        target = Path(path).resolve()
        if not target.exists():
            raise FileNotFoundError(f"Path does not exist: {target}")

        if target.is_dir():
            res = self.pipeline.ingest_directory(target)
            return {
                "source": target.name,
                "type": "directory",
                "successful_count": len(res.get("successful", [])),
                "failed_count": len(res.get("failed", [])),
                "details": res,
            }
        else:
            return self.pipeline.ingest_file(target)

    def search(
        self,
        query: str,
        top_k: int = 15,
        top_n: int = 5,
        augment: bool = True,
        augment_mode: str = "expand",
        min_similarity: Optional[float] = None,
        min_rerank_score: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve and rerank candidate document chunks without generating an answer.

        Useful for source inspection and semantic document discovery.
        """
        if not query or not query.strip():
            return []

        # 1. Retrieve candidates from ChromaDB
        candidates = self.retriever.retrieve(
            query=query.strip(),
            top_k=top_k,
            score_threshold=min_similarity,
            augment=augment,
            augment_mode=augment_mode,
        )

        if not candidates:
            return []

        # 2. Rerank candidates with cross-attention scoring
        reranked = self.reranker.rerank(
            query=query.strip(),
            chunks=candidates,
            top_n=top_n,
            min_score=min_rerank_score,
        )

        return reranked

    def ask(
        self,
        query: str,
        stream: bool = False,
        top_k: int = 15,
        top_n: int = 5,
        augment: bool = True,
        augment_mode: str = "expand",
        min_similarity: Optional[float] = None,
        min_rerank_score: Optional[float] = None,
    ) -> Union[Dict[str, Any], Iterator[str]]:
        """Ask a question and receive a grounded answer with page-level citations.

        Args:
            query: The student's question.
            stream: If True, yields token chunks iteratively for real-time typewriter effect.
            top_k: Number of candidate chunks to fetch in the vector retrieval stage.
            top_n: Number of top chunks to preserve after semantic reranking.
            augment: Whether to apply query augmentation (rewrite, expand, hyde).
            augment_mode: Augmentation strategy ('expand', 'rewrite', 'hyde').
            min_similarity: Optional minimum cosine similarity threshold for vector retrieval.
            min_rerank_score: Optional minimum score threshold for semantic reranker.

        Returns:
            Dictionary with 'answer', 'sources', 'chunks', 'model', and 'provider',
            or an Iterator[str] yielding tokens if stream=True.
        """
        # Guard: Empty or whitespace query
        if not query or not query.strip():
            msg = "Please provide a valid question."
            if stream:
                return iter([msg])
            return {
                "answer": msg,
                "sources": [],
                "chunks": [],
                "model": getattr(self.generator, "model", ""),
                "provider": getattr(self.generator, "provider", ""),
            }

        # Guard: Empty database
        if self.count() == 0:
            msg = "Vector database is empty. Please ingest your study notes or course materials first using assistant.ingest()."
            if stream:
                return iter([msg])
            return {
                "answer": msg,
                "sources": [],
                "chunks": [],
                "model": getattr(self.generator, "model", ""),
                "provider": getattr(self.generator, "provider", ""),
            }

        # 1. Search and Rerank
        reranked_chunks = self.search(
            query=query,
            top_k=top_k,
            top_n=top_n,
            augment=augment,
            augment_mode=augment_mode,
            min_similarity=min_similarity,
            min_rerank_score=min_rerank_score,
        )

        # 2. Format Context for Pedagogical Generator
        context = self.retriever.format_context(reranked_chunks)

        # 3. Generate Answer (Streaming vs Non-Streaming)
        if stream:
            return self.generator.generate_stream(query=query.strip(), context=context)

        gen_result = self.generator.generate(query=query.strip(), context=context)

        return {
            "answer": gen_result.get("answer", ""),
            "sources": gen_result.get("sources", []),
            "chunks": reranked_chunks,
            "model": gen_result.get("model", ""),
            "provider": gen_result.get("provider", ""),
        }

    def count(self) -> int:
        """Return total number of chunks currently indexed in the vector store."""
        return self.indexer.count()

    def clear(self) -> None:
        """Reset and clear all indexed document chunks."""
        self.indexer.reset()


def run_interactive_chat(assistant: StudyAssistant):
    """Run an interactive study chat session in the terminal."""
    total_chunks = assistant.count()
    print("\n" + "=" * 60)
    print("🎓 STUDY ASSISTANT: GROUNDED ACADEMIC TUTOR")
    print(f"📚 Vector Database: {total_chunks} chunks indexed ({assistant.persist_directory})")
    print("💡 Commands: Type '/exit' to quit, '/count' to inspect database.")
    print("=" * 60 + "\n")

    if total_chunks == 0:
        print("⚠️ Warning: Your vector database is currently empty.")
        print("   Run `python -m src.assistant --ingest path/to/document.pdf` to add materials.\n")

    while True:
        try:
            query = input("\n📖 Student: ").strip()
            if not query:
                continue
            if query.lower() in {"/exit", "exit", "quit"}:
                print("\nGoodbye and happy studying! 🚀\n")
                break
            if query.lower() == "/count":
                print(f"📊 Total indexed chunks: {assistant.count()}")
                continue

            print("\n🤖 Tutor: ", end="", flush=True)
            stream = assistant.ask(query, stream=True)
            for token in stream:
                sys.stdout.write(token)
                sys.stdout.flush()
            print()

        except (KeyboardInterrupt, EOFError):
            print("\n\nSession ended. Good luck with your studies!")
            break


def main():
    """Command-line entrypoint for StudyAssistant."""
    parser = argparse.ArgumentParser(
        description="Study Assistant: Grounded Academic RAG Tutor powered by NVIDIA NIM & Nebius AI Studio"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--chat", "-c", action="store_true", help="Launch interactive terminal study session")
    group.add_argument("--ask", "-a", type=str, help="Ask a single question and display answer")
    group.add_argument("--ingest", "-i", type=str, help="Ingest a file or directory of course materials")

    parser.add_argument("--top-k", type=int, default=15, help="Candidate chunks to retrieve (default: 15)")
    parser.add_argument("--top-n", type=int, default=5, help="Final chunks to retain after reranking (default: 5)")
    parser.add_argument("--no-augment", action="store_true", help="Disable query augmentation")
    parser.add_argument("--mode", type=str, default="expand", choices=["expand", "rewrite", "hyde"], help="Augmentation mode")
    parser.add_argument("--db-path", type=str, default="./chroma_db", help="Path to persistent ChromaDB directory")

    args = parser.parse_args()

    assistant = StudyAssistant(persist_directory=args.db_path)

    if args.ingest:
        print(f"📄 Ingesting course materials from: {args.ingest} ...")
        res = assistant.ingest(args.ingest)
        print(f"✅ Ingestion complete: {res}")
    elif args.ask:
        print(f"\n📖 Question: {args.ask}")
        print("🤖 Generating answer...\n")
        response = assistant.ask(
            query=args.ask,
            stream=False,
            top_k=args.top_k,
            top_n=args.top_n,
            augment=not args.no_augment,
            augment_mode=args.mode,
        )
        print("=== TUTOR ANSWER ===")
        print(response["answer"])
        print("\n=== CITED SOURCES ===")
        print(response["sources"] if response["sources"] else "None")
    elif args.chat:
        run_interactive_chat(assistant)


if __name__ == "__main__":
    main()
