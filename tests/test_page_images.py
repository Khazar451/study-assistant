from pypdf import PdfWriter

from src.ingestion.page_images import attach_page_images, render_pdf_page


def test_render_pdf_page_writes_jpeg(tmp_path, monkeypatch):
    monkeypatch.setattr("src.ingestion.page_images.library_roots", lambda: [tmp_path])
    monkeypatch.setattr("src.ingestion.page_images.CACHE_DIR", tmp_path / "cache")
    pdf_path = tmp_path / "notes.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    with open(pdf_path, "wb") as handle:
        writer.write(handle)

    image = render_pdf_page(str(pdf_path), 1)
    assert image is not None
    assert image.exists()
    assert image.stat().st_size > 100
    assert render_pdf_page(str(pdf_path), 1) == image


def test_attach_page_images_only_for_cited_pages(tmp_path, monkeypatch):
    monkeypatch.setattr("src.ingestion.page_images.library_roots", lambda: [tmp_path])
    monkeypatch.setattr("src.ingestion.page_images.CACHE_DIR", tmp_path / "cache")
    pdf_path = tmp_path / "notes.pdf"
    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    writer.add_blank_page(width=200, height=200)
    with open(pdf_path, "wb") as handle:
        writer.write(handle)

    references = [
        {"n": 1, "file_path": str(pdf_path), "page": 1},
        {"n": 2, "file_path": str(pdf_path), "page": 2},
    ]
    attach_page_images(references, "Only the first page is used [1].")
    assert references[0]["image_url"].startswith("/api/materials/page?")
    assert references[1]["image_url"] == ""
