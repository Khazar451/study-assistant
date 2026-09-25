from pathlib import Path
import pytest
from pypdf import PdfWriter
from src.ingestion.loader import DocumentLoader


def test_load_txt_file_uniform_metadata(tmp_path):
    loader = DocumentLoader()
    txt_file = tmp_path / "lecture_notes.txt"
    txt_file.write_text("Introduction to Calculus and limits.", encoding="utf-8")

    docs = loader.load(txt_file)

    assert len(docs) == 1
    assert docs[0].page_content == "Introduction to Calculus and limits."
    assert docs[0].metadata["source"] == "lecture_notes.txt"
    assert docs[0].metadata["file_path"] == str(txt_file.resolve())
    assert docs[0].metadata["file_type"] == "txt"
    assert docs[0].metadata["page"] == 1


def test_load_md_file_uniform_metadata(tmp_path):
    loader = DocumentLoader()
    md_file = tmp_path / "physics_summary.md"
    md_file.write_text("# Physics\nNewton's first law of motion.", encoding="utf-8")

    docs = loader.load(md_file)

    assert len(docs) == 1
    assert "Newton's first law" in docs[0].page_content
    assert docs[0].metadata["source"] == "physics_summary.md"
    assert docs[0].metadata["file_path"] == str(md_file.resolve())
    assert docs[0].metadata["file_type"] == "md"
    assert docs[0].metadata["page"] == 1


def test_load_unsupported_extension_raises(tmp_path):
    loader = DocumentLoader()
    bad_file = tmp_path / "script.exe"
    bad_file.write_text("binary content", encoding="utf-8")

    with pytest.raises(ValueError, match="Unsupported file format: '.exe'"):
        loader.load(bad_file)


def test_load_missing_file_raises(tmp_path):
    loader = DocumentLoader()
    missing_file = tmp_path / "non_existent.pdf"

    with pytest.raises(FileNotFoundError, match="File not found"):
        loader.load(missing_file)


def test_load_pdf_file_multipage(tmp_path):
    loader = DocumentLoader()
    pdf_path = tmp_path / "sample_multipage.pdf"

    # Create a 2-page PDF with pypdf
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    writer.add_blank_page(width=72, height=72)
    with open(pdf_path, "wb") as f:
        writer.write(f)

    docs = loader.load(pdf_path)

    assert len(docs) == 2
    assert docs[0].metadata["page"] == 1
    assert docs[1].metadata["page"] == 2
    assert docs[0].metadata["file_type"] == "pdf"
    assert docs[0].metadata["source"] == "sample_multipage.pdf"
    assert docs[0].metadata["file_path"] == str(pdf_path.resolve())
