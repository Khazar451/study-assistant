import os
from pathlib import Path
from typing import List, Union
from langchain_core.documents import Document
from pypdf import PdfReader


class DocumentLoader:
    """Document loader supporting PDF, TXT, and Markdown files with a uniform metadata schema."""

    SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md"}

    def load(self, file_path: Union[str, Path]) -> List[Document]:
        path = Path(file_path).resolve()

        if not path.is_file():
            raise FileNotFoundError(f"File not found: {path}")

        ext = path.suffix.lower()
        if ext not in self.SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file format: '{ext}'. Supported formats: {sorted(self.SUPPORTED_EXTENSIONS)}"
            )

        if ext == ".pdf":
            return self._load_pdf(path)
        return self._load_text(path, ext)

    def _load_pdf(self, path: Path) -> List[Document]:
        documents: List[Document] = []
        reader = PdfReader(str(path))
        for page_idx, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": path.name,
                        "file_path": str(path),
                        "file_type": "pdf",
                        "page": int(page_idx),
                    },
                )
            )
        return documents

    def _load_text(self, path: Path, ext: str) -> List[Document]:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
        return [
            Document(
                page_content=content,
                metadata={
                    "source": path.name,
                    "file_path": str(path),
                    "file_type": ext.lstrip("."),
                    "page": 1,
                },
            )
        ]
