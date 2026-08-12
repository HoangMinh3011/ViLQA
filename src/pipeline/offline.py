"""Offline index building pipeline."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from config import LegalQAConfig
from src.prepare_data.build_corpus import build_corpus_from_contexts
from src.retrieval.biencoder import BiencoderRetriever
from src.retrieval.bm25s import BM25sRetriever

logger = logging.getLogger(__name__)


def build_indexes_from_contexts(
    contexts_dir: Path,
    config: LegalQAConfig,
    limit: int | None = None,
    build_dense: bool = True,
    tokenizer_model: str | None = None,
) -> dict[str, Any]:
    logger.info("Building normalized corpus and chunks...")
    documents, chunks = build_corpus_from_contexts(
        contexts_dir,
        config=config,
        limit=limit,
        tokenizer_model=tokenizer_model,
    )
    document_lookup = {document["document_id"]: document for document in documents}
    logger.info("Documents: %d | Chunks: %d", len(documents), len(chunks))

    logger.info("Building BM25 index...")
    bm25 = BM25sRetriever().build(chunks)

    artifacts: dict[str, Any] = {
        "documents": documents,
        "document_lookup": document_lookup,
        "chunks": chunks,
        "bm25": bm25,
    }

    if build_dense:
        logger.info("Building dense index with %s...", config.biencoder_model)
        dense = BiencoderRetriever(
            config.biencoder_model,
            batch_size=config.dense_batch_size,
            device=config.embedding_device,
        )
        embeddings = dense.encode_documents([chunk["text"] for chunk in chunks])
        artifacts["embeddings"] = embeddings
        artifacts["dense_index"] = dense.build_faiss_index(embeddings)

    return artifacts
