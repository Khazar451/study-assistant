"""Precomputed embeddings for the bundled sample.

A fresh or just-woken demo indexes the sample from this file instead of calling the embedding API,
so the library is never empty because the API was slow or unreachable at startup.

Rebuild after changing the sample text, the chunking settings or EMBEDDING_MODEL:
    python -m src.ingestion.sample_index
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

from src.ingestion.loader import DocumentLoader
from src.ingestion.pipeline import DocumentChunker, text_key

SAMPLE_PATH = Path("sample_materials/astronomy_kepler.txt")
logger = logging.getLogger(__name__)


def index_path(sample: Path) -> Path:
    return sample.with_name(f"{sample.stem}.index.json")


def sample_chunk_texts(sample: Path) -> List[str]:
    """The chunk texts IngestionPipeline produces for this file with its default settings."""
    chunks = DocumentChunker().split_documents(DocumentLoader(extract_visuals=False).load(sample))
    return [chunk.page_content for chunk in chunks if (chunk.page_content or "").strip()]


def load_embedding_cache(sample: Path, model: Optional[str]) -> Optional[Dict[str, List[float]]]:
    """Vectors by chunk text, or None when there is no file or it was built with another model."""
    path = index_path(sample)
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("model") != model:
        logger.info("Ignoring %s: built with %s, the embedder uses %s", path.name, data.get("model"), model)
        return None
    return data.get("embeddings") or None


def build(sample: Path = SAMPLE_PATH) -> Path:
    """Embed the sample's chunks once with the configured embedder and write the index file."""
    from src.ingestion.embedder import NvidiaEmbedder

    texts = sample_chunk_texts(sample)
    embedder = NvidiaEmbedder()
    # Same call as IngestionPipeline.ingest_file, so the vectors match a live ingest.
    vectors = embedder.embed_batch(
        texts, input_type="passage", batch_size=min(8, len(texts)), max_retries=5, truncate="END"
    )
    payload = {
        "model": embedder.model,
        "dimensions": len(vectors[0]),
        "chunks": len(texts),
        "embeddings": {text_key(text): vector for text, vector in zip(texts, vectors)},
    }
    target = index_path(sample)
    target.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description="Precompute embeddings for the bundled sample")
    parser.add_argument("--sample", type=Path, default=SAMPLE_PATH)
    args = parser.parse_args()
    target = build(args.sample)
    data = json.loads(target.read_text(encoding="utf-8"))
    print(f"Wrote {target}: {data['chunks']} chunks, {data['dimensions']} dimensions, model {data['model']}")


if __name__ == "__main__":
    main()
