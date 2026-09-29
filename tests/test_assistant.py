from pathlib import Path
from unittest.mock import MagicMock, patch
import pytest

from src.assistant import StudyAssistant, main, run_interactive_chat


@pytest.fixture
def mock_subcomponents():
    """Provide mock instances for all StudyAssistant dependencies."""
    pipeline = MagicMock()
    indexer = MagicMock()
    embedder = MagicMock()
    pipeline.indexer = indexer
    pipeline.embedder = embedder

    retriever = MagicMock()
    reranker = MagicMock()
    generator = MagicMock()

    generator.model = "meta/llama-3.2-11b-vision-instruct"
    generator.provider = "nvidia"

    return {
        "pipeline": pipeline,
        "indexer": indexer,
        "embedder": embedder,
        "retriever": retriever,
        "reranker": reranker,
        "generator": generator,
    }


@pytest.fixture
def assistant(mock_subcomponents):
    """StudyAssistant initialized with injected mock subcomponents."""
    return StudyAssistant(
        pipeline=mock_subcomponents["pipeline"],
        retriever=mock_subcomponents["retriever"],
        reranker=mock_subcomponents["reranker"],
        generator=mock_subcomponents["generator"],
        persist_directory="/mock/db",
    )


def test_init_with_defaults():
    """Ensure default initialization coordinates all subcomponents."""
    with patch("src.assistant.ChromaIndexer") as mock_indexer_cls,          patch("src.assistant.NvidiaEmbedder") as mock_embedder_cls,          patch("src.assistant.IngestionPipeline") as mock_pipe_cls,          patch("src.assistant.QueryAugmenter") as mock_aug_cls,          patch("src.assistant.StudyRetriever") as mock_ret_cls,          patch("src.assistant.StudyReranker") as mock_rerank_cls,          patch("src.assistant.StudyGenerator") as mock_gen_cls:

        assistant = StudyAssistant(persist_directory="./custom_chroma")

        mock_indexer_cls.assert_called_once_with(persist_directory="./custom_chroma")
        mock_embedder_cls.assert_called_once()
        mock_pipe_cls.assert_called_once()
        mock_aug_cls.assert_called_once()
        mock_ret_cls.assert_called_once()
        mock_rerank_cls.assert_called_once_with(mode="llm_listwise")
        mock_gen_cls.assert_called_once()
        assert assistant.persist_directory == "./custom_chroma"


def test_init_with_injected_components(mock_subcomponents):
    """Validate clean dependency injection without instantiating defaults."""
    assistant = StudyAssistant(
        pipeline=mock_subcomponents["pipeline"],
        retriever=mock_subcomponents["retriever"],
        reranker=mock_subcomponents["reranker"],
        generator=mock_subcomponents["generator"],
    )
    assert assistant.pipeline is mock_subcomponents["pipeline"]
    assert assistant.indexer is mock_subcomponents["indexer"]
    assert assistant.embedder is mock_subcomponents["embedder"]
    assert assistant.retriever is mock_subcomponents["retriever"]
    assert assistant.reranker is mock_subcomponents["reranker"]
    assert assistant.generator is mock_subcomponents["generator"]


def test_ingest_file_delegation(assistant, mock_subcomponents, tmp_path):
    """Verify file ingestion delegates to IngestionPipeline.ingest_file."""
    file_path = tmp_path / "lecture.pdf"
    file_path.write_text("sample physics text")

    mock_subcomponents["pipeline"].ingest_file.return_value = {
        "status": "success",
        "file": str(file_path),
        "chunks_added": 3,
    }

    res = assistant.ingest(file_path)

    mock_subcomponents["pipeline"].ingest_file.assert_called_once_with(file_path.resolve())
    assert res["status"] == "success"
    assert res["chunks_added"] == 3


def test_ingest_directory_delegation(assistant, mock_subcomponents, tmp_path):
    """Verify directory ingestion delegates to IngestionPipeline.ingest_directory."""
    doc_dir = tmp_path / "notes"
    doc_dir.mkdir()
    (doc_dir / "note1.txt").write_text("Text 1")
    (doc_dir / "note2.txt").write_text("Text 2")

    mock_subcomponents["pipeline"].ingest_directory.return_value = {
        "successful": [str(doc_dir / "note1.txt"), str(doc_dir / "note2.txt")],
        "failed": [],
    }

    res = assistant.ingest(doc_dir)

    mock_subcomponents["pipeline"].ingest_directory.assert_called_once_with(doc_dir.resolve())
    assert res["type"] == "directory"
    assert res["successful_count"] == 2
    assert res["failed_count"] == 0


def test_ask_non_streaming(assistant, mock_subcomponents):
    """Verify end-to-end question answering pipeline in non-streaming mode."""
    mock_subcomponents["indexer"].count.return_value = 10
    mock_chunks = [
        {"id": "c1", "text": "Kepler discovered elliptical orbits.", "similarity_score": 0.82}
    ]
    reranked_chunks = [
        {"id": "c1", "text": "Kepler discovered elliptical orbits.", "rerank_score": 0.95}
    ]

    mock_subcomponents["retriever"].retrieve.return_value = mock_chunks
    mock_subcomponents["reranker"].rerank.return_value = reranked_chunks
    mock_subcomponents["retriever"].format_context.return_value = "[1] (kepler.pdf, p. 5): Kepler discovered..."
    mock_subcomponents["generator"].generate.return_value = {
        "answer": "Planets orbit in ellipses [1].",
        "sources": ["kepler.pdf (p. 5)"],
        "model": "meta/llama-3.2-11b-vision-instruct",
        "provider": "nvidia",
    }

    response = assistant.ask(
        query="How do planets orbit?",
        stream=False,
        top_k=10,
        top_n=3,
        augment=True,
        augment_mode="expand",
    )

    mock_subcomponents["retriever"].retrieve.assert_called_once_with(
        query="How do planets orbit?",
        top_k=10,
        score_threshold=None,
        augment=True,
        augment_mode="expand",
    )
    mock_subcomponents["reranker"].rerank.assert_called_once_with(
        query="How do planets orbit?",
        chunks=mock_chunks,
        top_n=3,
        min_score=None,
    )
    mock_subcomponents["retriever"].format_context.assert_called_once_with(reranked_chunks)
    mock_subcomponents["generator"].generate.assert_called_once_with(
        query="How do planets orbit?",
        context="[1] (kepler.pdf, p. 5): Kepler discovered...",
    )

    assert response["answer"] == "Planets orbit in ellipses [1]."
    assert response["sources"] == ["kepler.pdf (p. 5)"]
    assert response["chunks"] == reranked_chunks
    assert response["model"] == "meta/llama-3.2-11b-vision-instruct"
    assert response["provider"] == "nvidia"


def test_ask_streaming(assistant, mock_subcomponents):
    """Verify streaming tokens yield from generator.generate_stream."""
    mock_subcomponents["indexer"].count.return_value = 5
    mock_subcomponents["retriever"].retrieve.return_value = [{"id": "c1"}]
    mock_subcomponents["reranker"].rerank.return_value = [{"id": "c1"}]
    mock_subcomponents["retriever"].format_context.return_value = "Formatted context"
    mock_subcomponents["generator"].generate_stream.return_value = iter(["Token1 ", "Token2"])

    stream = assistant.ask("What is calculus?", stream=True)
    tokens = list(stream)

    assert tokens == ["Token1 ", "Token2"]
    mock_subcomponents["generator"].generate_stream.assert_called_once_with(
        query="What is calculus?",
        context="Formatted context",
    )


def test_ask_empty_query_guard(assistant, mock_subcomponents):
    """Ensure empty or whitespace queries return guidance without invoking retriever/LLM."""
    mock_subcomponents["indexer"].count.return_value = 10

    # Non-streaming
    res_empty = assistant.ask("   ", stream=False)
    assert "valid question" in res_empty["answer"].lower()
    mock_subcomponents["retriever"].retrieve.assert_not_called()
    mock_subcomponents["generator"].generate.assert_not_called()

    # Streaming
    res_stream = list(assistant.ask("", stream=True))
    assert len(res_stream) == 1
    assert "valid question" in res_stream[0].lower()
    mock_subcomponents["generator"].generate_stream.assert_not_called()


def test_ask_empty_database_warning(assistant, mock_subcomponents):
    """Ensure query on empty ChromaDB alerts user to ingest documents first."""
    mock_subcomponents["indexer"].count.return_value = 0

    # Non-streaming
    res = assistant.ask("What is Newton's law?", stream=False)
    assert "database is empty" in res["answer"].lower()
    mock_subcomponents["retriever"].retrieve.assert_not_called()

    # Streaming
    res_stream = list(assistant.ask("What is Newton's law?", stream=True))
    assert "database is empty" in "".join(res_stream).lower()


def test_search_returns_reranked_chunks(assistant, mock_subcomponents):
    """Verify search method retrieves and reranks chunks without calling LLM generator."""
    mock_subcomponents["retriever"].retrieve.return_value = [
        {"id": "c1", "similarity_score": 0.5},
        {"id": "c2", "similarity_score": 0.7},
    ]
    mock_subcomponents["reranker"].rerank.return_value = [
        {"id": "c2", "rerank_score": 0.92},
        {"id": "c1", "rerank_score": 0.65},
    ]

    results = assistant.search(query="Quantum entanglement", top_k=8, top_n=2)

    mock_subcomponents["retriever"].retrieve.assert_called_once_with(
        query="Quantum entanglement",
        top_k=8,
        score_threshold=None,
        augment=True,
        augment_mode="expand",
    )
    mock_subcomponents["reranker"].rerank.assert_called_once()
    mock_subcomponents["generator"].generate.assert_not_called()
    mock_subcomponents["generator"].generate_stream.assert_not_called()

    assert len(results) == 2
    assert results[0]["id"] == "c2"


def test_count_and_clear_delegation(assistant, mock_subcomponents):
    """Verify index count and clear delegating to indexer."""
    mock_subcomponents["indexer"].count.return_value = 42

    assert assistant.count() == 42
    mock_subcomponents["indexer"].count.assert_called()

    assistant.clear()
    mock_subcomponents["indexer"].reset.assert_called_once()


def test_cli_main_ingest_argument(capsys):
    """Verify --ingest flag invokes assistant.ingest."""
    with patch("sys.argv", ["src.assistant", "--ingest", "test_doc.pdf"]),          patch("src.assistant.StudyAssistant") as mock_cls:
        mock_instance = MagicMock()
        mock_cls.return_value = mock_instance
        mock_instance.ingest.return_value = {"status": "success", "chunks_count": 5}
        main()
        mock_cls.assert_called_once_with(persist_directory="./chroma_db")
        mock_instance.ingest.assert_called_once_with("test_doc.pdf")
        out = capsys.readouterr().out
        assert "Ingestion complete" in out


def test_cli_main_ask_argument(capsys):
    """Verify --ask flag forwards arguments and prints answer and sources."""
    with patch("sys.argv", ["src.assistant", "--ask", "What is gravity?", "--top-k", "8", "--top-n", "3", "--no-augment", "--db-path", "/custom/db"]),          patch("src.assistant.StudyAssistant") as mock_cls:
        mock_instance = MagicMock()
        mock_cls.return_value = mock_instance
        mock_instance.ask.return_value = {
            "answer": "Gravity is curvature of spacetime.",
            "sources": ["einstein.pdf (p. 2)"],
        }
        main()
        mock_cls.assert_called_once_with(persist_directory="/custom/db")
        mock_instance.ask.assert_called_once_with(
            query="What is gravity?",
            stream=False,
            top_k=8,
            top_n=3,
            augment=False,
            augment_mode="expand",
        )
        out = capsys.readouterr().out
        assert "=== TUTOR ANSWER ===" in out
        assert "Gravity is curvature of spacetime." in out
        assert "einstein.pdf (p. 2)" in out


def test_cli_main_chat_argument():
    """Verify --chat flag launches run_interactive_chat."""
    with patch("sys.argv", ["src.assistant", "--chat"]),          patch("src.assistant.StudyAssistant") as mock_cls,          patch("src.assistant.run_interactive_chat") as mock_chat:
        mock_instance = MagicMock()
        mock_cls.return_value = mock_instance
        main()
        mock_cls.assert_called_once_with(persist_directory="./chroma_db")
        mock_chat.assert_called_once_with(mock_instance)


def test_cli_main_mutually_exclusive_required():
    """Verify missing action flag raises SystemExit with code 2."""
    with patch("sys.argv", ["src.assistant"]):
        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 2


def test_run_interactive_chat_exit(assistant, capsys):
    """Verify interactive chat loop terminates cleanly on /exit."""
    assistant.count = MagicMock(return_value=3)
    with patch("builtins.input", side_effect=["/exit"]):
        run_interactive_chat(assistant)
    out = capsys.readouterr().out
    assert "Goodbye and happy studying!" in out


def test_run_interactive_chat_count_command(assistant, capsys):
    """Verify /count command inspects chunk count without exiting."""
    assistant.count = MagicMock(return_value=12)
    with patch("builtins.input", side_effect=["/count", "quit"]):
        run_interactive_chat(assistant)
    out = capsys.readouterr().out
    assert "Total indexed chunks: 12" in out


def test_run_interactive_chat_ask_stream(assistant, capsys):
    """Verify student query triggers streaming ask and prints tokens."""
    assistant.count = MagicMock(return_value=5)
    assistant.ask = MagicMock(return_value=iter(["Newton's ", "first ", "law."]))
    with patch("builtins.input", side_effect=["", "What is inertia?", "exit"]):
        run_interactive_chat(assistant)
    out = capsys.readouterr().out
    assert "Newton's first law." in out
    assistant.ask.assert_called_once_with("What is inertia?", stream=True)


def test_run_interactive_chat_keyboard_interrupt(assistant, capsys):
    """Verify KeyboardInterrupt exits gracefully with friendly message."""
    assistant.count = MagicMock(return_value=0)
    with patch("builtins.input", side_effect=KeyboardInterrupt):
        run_interactive_chat(assistant)
    out = capsys.readouterr().out
    assert "Session ended. Good luck with your studies!" in out
