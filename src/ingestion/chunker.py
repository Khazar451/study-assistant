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

def chunk_documents(
    docs: List[Document],
    chunk_size: int = 500,
    chunk_overlap: int = 50
) -> List[Document]:

    if not docs:
        return []

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