import argparse
import os
from pathlib import Path
import sys
import time
from typing import Any, Callable, Dict, Iterator, List, Optional, Union
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
from src.retrieval.tavily_search import TavilySearchClient
from src import providers
from src.generation.generator import StudyGenerator
from src.generation.tutor_format import allowed_source_keys, infer_grounding


class StudyAssistant:
    """Unified Facade for the Academic Study Assistant RAG System.

    Coordinates:
    1. Document Ingestion (DocumentLoader -> Chunker -> NvidiaEmbedder -> ChromaDB)
    2. Query Intelligence (Rewrite / Expand / HyDE via QueryAugmenter)
    3. Dense Vector Retrieval (Cosine Similarity & Max-Score Pooling via StudyRetriever)
    4. Cross-Attention Semantic Reranking (via StudyReranker)
    5. Grounded Answer Generation (Grounded Citations via StudyGenerator)
    6. Real-Time Web Search & Grounding (via TavilySearchClient)
    """

    def __init__(
        self,
        pipeline: Optional[IngestionPipeline] = None,
        retriever: Optional[StudyRetriever] = None,
        reranker: Optional[StudyReranker] = None,
        generator: Optional[StudyGenerator] = None,
        tavily: Optional[TavilySearchClient] = None,
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

        # 5. Coordinate Tavily Search Client
        self.tavily = tavily or TavilySearchClient()

    def ingest(
        self,
        path: Union[str, Path],
        batch_size: int = 64,
        progress_callback: Optional[Callable[[int, int], None]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Ingest a single document or an entire directory of course materials.

        Args:
            path: File or directory path to ingest.
            batch_size: Embedding batch size.
            progress_callback: Optional callback reporting (processed_chunks, total_chunks).
            metadata: Optional fields stamped on every chunk of a single file (e.g. its workspace).

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
            kwargs: Dict[str, Any] = {}
            if progress_callback is not None:
                kwargs["progress_callback"] = progress_callback
            if metadata:
                kwargs["extra_metadata"] = metadata
            return self.pipeline.ingest_file(target, **kwargs)

    def search(
        self,
        query: str,
        top_k: int = 15,
        top_n: int = 5,
        augment: bool = True,
        augment_mode: str = "expand",
        min_similarity: Optional[float] = None,
        min_rerank_score: Optional[float] = None,
        use_web: bool = False,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Retrieve and rerank candidate document chunks without generating an answer.

        Useful for source inspection and semantic document discovery. `where` is a Chroma
        metadata filter (the public demo limits each visitor to their own workspace).
        """
        if not query or not query.strip():
            return []

        # 1. Retrieve candidates from ChromaDB
        retrieve_kwargs: Dict[str, Any] = {"where": where} if where else {}
        candidates = self.retriever.retrieve(
            query=query.strip(),
            top_k=top_k,
            score_threshold=min_similarity,
            augment=augment,
            augment_mode=augment_mode,
            **retrieve_kwargs,
        )

        # 2. Optionally augment with Tavily web search
        if use_web and self.tavily and self.tavily.is_available:
            web_candidates = self.tavily.search(query=query.strip(), max_results=top_n)
            candidates = list(candidates) + web_candidates

        if not candidates:
            return []

        # 3. Rerank candidates with cross-attention scoring
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
        use_web: bool = False,
        fallback_to_web: bool = False,
        session_id: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None,
        where: Optional[Dict[str, Any]] = None,
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
            use_web: Whether to include Tavily web search results.
            fallback_to_web: Whether to query Tavily if local knowledge base is empty or lacks context.
            session_id: Optional session identifier for conversation persistence.
            history: Optional list of past conversation turns.
            where: Optional Chroma metadata filter limiting which indexed chunks may be used.

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
                "model": providers.answer_model(self.generator),
                "provider": getattr(self.generator, "provider", ""),
                "session_id": session_id,
            }

        # Resolve conversation history
        from src.memory.session_store import get_session_store
        store = get_session_store()
        if history is None and session_id:
            history = store.get_recent_history(session_id, turns=4)

        # Condense conversational query if conversation history is present
        search_query = query.strip()
        augmenter = getattr(self, "augmenter", None) or getattr(self.retriever, "augmenter", None)
        if history and augmenter and hasattr(augmenter, "condense_conversational_query"):
            search_query = augmenter.condense_conversational_query(search_query, history=history)

        # Guard: Empty database
        if self.count() == 0:
            if (use_web or fallback_to_web) and self.tavily and self.tavily.is_available:
                web_chunks = self.tavily.search(query=search_query, max_results=top_n)
                if web_chunks:
                    context = self.retriever.format_context(web_chunks)
                    web_gen_kwargs = {"history": history} if history is not None else {}
                    if stream:
                        return self.generator.generate_stream(
                            query=query.strip(), context=context, **web_gen_kwargs
                        )
                    gen_result = self.generator.generate(
                        query=query.strip(), context=context, **web_gen_kwargs
                    )
                    answer = gen_result.get("answer", "")
                    if session_id:
                        store.add_message(session_id, "user", query.strip())
                        store.add_message(
                            session_id, "tutor", answer, sources=gen_result.get("sources", []), grounding="web"
                        )
                    return {
                        "answer": answer,
                        "sources": gen_result.get("sources", []),
                        "chunks": web_chunks,
                        "model": gen_result.get("model", ""),
                        "provider": gen_result.get("provider", ""),
                        "session_id": session_id,
                    }
            msg = "Vector database is empty. Please ingest your study notes or course materials first using assistant.ingest()."
            if stream:
                return iter([msg])
            return {
                "answer": msg,
                "sources": [],
                "chunks": [],
                "model": providers.answer_model(self.generator),
                "provider": getattr(self.generator, "provider", ""),
                "session_id": session_id,
            }

        # 1. Search and Rerank with condensed search query
        reranked_chunks = self.search(
            query=search_query,
            top_k=top_k,
            top_n=top_n,
            augment=augment,
            augment_mode=augment_mode,
            min_similarity=min_similarity,
            min_rerank_score=min_rerank_score,
            use_web=use_web,
            where=where,
        )

        # Fallback to Tavily if local search returned no relevant chunks and fallback_to_web is enabled
        if not reranked_chunks and fallback_to_web and not use_web and self.tavily and self.tavily.is_available:
            web_chunks = self.tavily.search(query=search_query, max_results=top_n)
            if web_chunks:
                reranked_chunks = self.reranker.rerank(query=search_query, chunks=web_chunks, top_n=top_n)

        # 2. Format Context for Pedagogical Generator
        context = self.retriever.format_context(reranked_chunks)
        grounding = infer_grounding(reranked_chunks)

        # 3. Generate Answer (Streaming vs Non-Streaming)
        gen_kwargs = {"history": history} if history is not None else {}
        if stream:
            return self.generator.generate_stream(
                query=query.strip(), context=context, grounding=grounding, **gen_kwargs
            )

        gen_result = self.generator.generate(
            query=query.strip(),
            context=context,
            grounding=grounding,
            allowed_sources=allowed_source_keys(reranked_chunks, context),
            **gen_kwargs,
        )

        answer = gen_result.get("answer", "")
        sources = gen_result.get("sources", [])
        if session_id:
            store.add_message(session_id, "user", query.strip())
            store.add_message(session_id, "tutor", answer, sources=sources, grounding=grounding)

        return {
            "answer": answer,
            "sources": sources,
            "chunks": reranked_chunks,
            "grounding": grounding,
            "model": gen_result.get("model", ""),
            "provider": gen_result.get("provider", ""),
            "session_id": session_id,
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
    print("STUDY ASSISTANT: GROUNDED ACADEMIC TUTOR")
    print(f"Vector Database: {total_chunks} chunks indexed ({assistant.persist_directory})")
    print("Commands: Type '/exit' to quit, '/count' to inspect database.")
    print("=" * 60 + "\n")

    if total_chunks == 0:
        print("Warning: Your vector database is currently empty.")
        print("   Run `python -m src.assistant --ingest path/to/document.pdf` to add materials.\n")

    while True:
        try:
            query = input("\nStudent: ").strip()
            if not query:
                continue
            if query.lower() in {"/exit", "exit", "quit"}:
                print("\nGoodbye and happy studying!\n")
                break
            if query.lower() == "/count":
                print(f"Total indexed chunks: {assistant.count()}")
                continue

            print("\nTutor: ", end="", flush=True)
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
    parser.add_argument("--web", action="store_true", help="Enable Tavily web search integration")
    parser.add_argument("--db-path", type=str, default="./chroma_db", help="Path to persistent ChromaDB directory")

    args = parser.parse_args()

    assistant = StudyAssistant(persist_directory=args.db_path)

    if args.ingest:
        print(f"Ingesting course materials from: {args.ingest} ...")
        res = assistant.ingest(args.ingest)
        print(f"Ingestion complete: {res}")
    elif args.ask:
        print(f"\nQuestion: {args.ask}")
        print("Generating answer...\n")
        ask_kwargs = {
            "query": args.ask,
            "stream": False,
            "top_k": args.top_k,
            "top_n": args.top_n,
            "augment": not args.no_augment,
            "augment_mode": args.mode,
        }
        if args.web:
            ask_kwargs["use_web"] = True

        response = assistant.ask(**ask_kwargs)
        print("=== TUTOR ANSWER ===")
        print(response["answer"])
        print("\n=== CITED SOURCES ===")
        print(response["sources"] if response["sources"] else "None")
    elif args.chat:
        run_interactive_chat(assistant)


if __name__ == "__main__":
    main()
