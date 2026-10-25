from typing import List, Optional
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

def get_text_splitter(chunk_size: int = 500, chunk_overlap: int = 50) -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap,
        length_function = len,
        add_start_index = True,
        separators = ["\n\n", "\n", ". ", " ", ""]
    )

def clean_text(text: str) -> str:
    """Drop characters that cannot be encoded as UTF-8.

    PDF extraction of math fonts (e.g. italic 𝑥) can yield lone surrogate halves such as '\ud835'.
    They crash every later step that encodes the text (embedding requests, ChromaDB), so a whole
    book fails part-way through. NUL bytes are removed for the same reason.
    """
    return text.encode("utf-8", "ignore").decode("utf-8").replace("\x00", "")


def chunk_documents(
    docs: List[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 50
) -> List[Document]:

    if not docs:
        return []

    for doc in docs:
        doc.page_content = clean_text(doc.page_content or "")
        # Titles are taken from page text too (e.g. slide_title), so they carry the same characters.
        doc.metadata = {k: clean_text(v) if isinstance(v, str) else v for k, v in (doc.metadata or {}).items()}

    splitter = get_text_splitter(chunk_size = chunk_size, chunk_overlap = chunk_overlap)
    return splitter.split_documents(docs)

def chunk_text(
    text: str,
    metadata: Optional[dict] = None,
    chunk_size: int = 500,
    chunk_overlap: int = 50
) -> List[Document]:

    if not text or not text.strip():
        return []

    doc = Document(page_content = text, metadata = metadata or {})
    return chunk_documents([doc], chunk_size = chunk_size, chunk_overlap = chunk_overlap)