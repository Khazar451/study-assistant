"""Render cited PDF pages so an explanation can show the book page itself."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional
from urllib.parse import quote

CACHE_DIR = Path("./data/page_cache")
MAX_PAGES_PER_ANSWER = 3


def library_roots() -> list[Path]:
    return [Path("./data/uploads").resolve(), Path("./sample_materials").resolve()]


def is_library_file(path: Path) -> bool:
    target = path.resolve()
    return target.is_file() and any(target == root or root in target.parents for root in library_roots())


def render_pdf_page(file_path: str, page: int) -> Optional[Path]:
    """Return a cached JPEG of a 1-indexed PDF page, or None when it cannot be drawn."""
    target = Path(file_path).resolve()
    if target.suffix.lower() != ".pdf" or not is_library_file(target):
        return None
    page_number = int(page or 1)
    if page_number < 1:
        return None

    stamp = int(target.stat().st_mtime)
    key = hashlib.sha256(f"{target}:{stamp}:{page_number}".encode()).hexdigest()[:20]
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    image_path = CACHE_DIR / f"{key}.jpg"
    if image_path.exists():
        return image_path

    import fitz

    document = fitz.open(target)
    try:
        if page_number > document.page_count:
            return None
        pixmap = document[page_number - 1].get_pixmap(matrix=fitz.Matrix(1.3, 1.3), alpha=False)
        pixmap.save(str(image_path), jpg_quality=82)
    finally:
        document.close()
    return image_path


def attach_page_images(references: list[dict], answer: str) -> None:
    """Add an image URL to the first few cited PDF pages."""
    shown: set[tuple[str, int]] = set()
    for ref in references:
        ref["image_url"] = ""
        number = ref.get("n")
        if not number or f"[{number}]" not in answer:
            continue
        if len(shown) >= MAX_PAGES_PER_ANSWER:
            continue
        file_path = str(ref.get("file_path") or "")
        try:
            page = int(ref.get("page") or 1)
        except (TypeError, ValueError):
            page = 1
        key = (file_path, page)
        if not file_path or key in shown:
            continue
        image = render_pdf_page(file_path, page)
        if image is None:
            continue
        shown.add(key)
        ref["image_url"] = f"/api/materials/page?path={quote(file_path)}&page={page}"
