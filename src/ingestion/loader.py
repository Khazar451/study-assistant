import os
from pathlib import Path
from typing import List, Optional, Union
from langchain_core.documents import Document
from pypdf import PdfReader

from src.ingestion.vision import VisionDescriber


class DocumentLoader:
    """Document loader supporting PDF, TXT, Markdown, and slide images with a uniform metadata schema."""

    SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".md", ".png", ".jpg", ".jpeg"}
    IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}

    def __init__(
        self,
        vision_describer: Optional[VisionDescriber] = None,
        extract_visuals: bool = True,
    ):
        self.extract_visuals = extract_visuals
        if vision_describer is not None:
            self.vision_describer = vision_describer
        elif extract_visuals:
            try:
                self.vision_describer = VisionDescriber()
            except Exception:
                self.vision_describer = None
        else:
            self.vision_describer = None

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
        if ext in self.IMAGE_EXTENSIONS:
            return self._load_image(path, ext)
        return self._load_text(path, ext)

    def _load_pdf(self, path: Path) -> List[Document]:
        documents: List[Document] = []
        reader = PdfReader(str(path))

        for page_idx, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            diagram_descriptions: List[str] = []

            # Extract visual diagrams and slide figures if enabled
            if self.extract_visuals and self.vision_describer:
                try:
                    for img_obj in page.images:
                        img_bytes = img_obj.data
                        if self.vision_describer.is_meaningful_diagram(img_bytes):
                            desc = self.vision_describer.describe_diagram(
                                img_bytes,
                                context_hint=f"{path.name} (Page {page_idx})",
                            )
                            if desc and desc not in diagram_descriptions:
                                diagram_descriptions.append(desc)
                except Exception:
                    # Robust error isolation: Never fail text extraction on image parsing hiccups
                    pass

            if diagram_descriptions:
                visual_blocks = "\n\n".join(
                    f"[Visual Diagram / Slide Figure on Page {page_idx}]:\n{d}"
                    for d in diagram_descriptions
                )
                page_content = f"{text}\n\n{visual_blocks}".strip() if text.strip() else visual_blocks
            else:
                page_content = text

            meta = {
                "source": path.name,
                "file_path": str(path),
                "file_type": "pdf",
                "page": int(page_idx),
            }
            if diagram_descriptions:
                meta["has_visuals"] = True
                meta["visual_count"] = len(diagram_descriptions)

            documents.append(
                Document(
                    page_content=page_content,
                    metadata=meta,
                )
            )
        return documents

    def _load_image(self, path: Path, ext: str) -> List[Document]:
        data = path.read_bytes()
        desc = ""
        if self.vision_describer:
            desc = self.vision_describer.describe_diagram(
                data, context_hint=f"Lecture Slide Image: {path.name}"
            )
        if not desc:
            desc = f"Academic diagram / slide image: {path.name}"

        content = f"[Lecture Slide / Diagram: {path.name}]:\n{desc}"
        return [
            Document(
                page_content=content,
                metadata={
                    "source": path.name,
                    "file_path": str(path),
                    "file_type": "image",
                    "page": 1,
                    "has_visuals": True,
                    "visual_count": 1,
                },
            )
        ]

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
