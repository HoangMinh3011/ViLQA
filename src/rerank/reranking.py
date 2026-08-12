"""Cross-encoder reranking."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    def __init__(self, model_name: str, max_length: int = 256) -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise ImportError("Install sentence-transformers to use cross-encoder reranking.") from exc

        self.model_name = model_name
        self.model = CrossEncoder(model_name, max_length=max_length)
        self._align_tokenizer_and_embeddings()

    def _align_tokenizer_and_embeddings(self) -> None:
        tokenizer = getattr(self.model, "tokenizer", None)
        transformer = getattr(self.model, "model", None)
        if tokenizer is None or transformer is None:
            return

        embeddings = transformer.get_input_embeddings()
        tokenizer_size = len(tokenizer)
        embedding_size = embeddings.num_embeddings
        if tokenizer_size > embedding_size:
            logger.warning(
                "Resizing reranker token embeddings from %d to %d to match tokenizer.",
                embedding_size,
                tokenizer_size,
            )
            transformer.resize_token_embeddings(tokenizer_size)

    def rerank(
        self,
        question: str,
        candidate_chunks: list[dict[str, Any]],
        batch_size: int = 16,
    ) -> list[dict[str, Any]]:
        if not candidate_chunks:
            return []
        pairs = [(question, item["text"]) for item in candidate_chunks]
        scores = self.model.predict(pairs, batch_size=batch_size)
        scores = np.asarray(scores).reshape(-1)

        reranked: list[dict[str, Any]] = []
        for item, score in zip(candidate_chunks, scores):
            new_item = dict(item)
            new_item["rerank_score"] = float(score)
            reranked.append(new_item)

        return sorted(reranked, key=lambda x: x["rerank_score"], reverse=True)


def select_top_k_evidence(ranked_chunks: list[dict[str, Any]], k: int = 5) -> list[dict[str, Any]]:
    return ranked_chunks[:k]
