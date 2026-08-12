"""Build normalized documents and retrieval chunks from raw contexts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from config import LegalQAConfig
from src.prepare_data.chunking import chunk_corpus, load_tokenizer
from src.prepare_data.preprocessing import load_raw_contexts, normalize_corpus


def build_corpus_from_contexts(
    contexts_dir: Path,
    config: LegalQAConfig,
    limit: int | None = None,
    tokenizer_model: str | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    raw_documents = load_raw_contexts(contexts_dir, limit=limit)
    documents = normalize_corpus(raw_documents)
    tokenizer = load_tokenizer(tokenizer_model or config.biencoder_model)
    chunks = chunk_corpus(
        documents,
        tokenizer,
        chunk_size=config.chunk_size,
        overlap=config.chunk_overlap,
    )
    return documents, chunks
