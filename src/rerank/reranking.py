"""Cross-encoder reranking."""

from __future__ import annotations

from typing import Any


class CrossEncoderReranker:
    def __init__(self, model_name: str, max_length: int = 512) -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise ImportError("Install sentence-transformers to use cross-encoder reranking.") from exc

        self.model_name = model_name
        self.model = CrossEncoder(model_name, max_length=max_length)

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

        reranked: list[dict[str, Any]] = []
        for item, score in zip(candidate_chunks, scores):
            new_item = dict(item)
            new_item["rerank_score"] = float(score)
            reranked.append(new_item)

        return sorted(reranked, key=lambda x: x["rerank_score"], reverse=True)


def select_top_k_evidence(ranked_chunks: list[dict[str, Any]], k: int = 5) -> list[dict[str, Any]]:
    return ranked_chunks[:k]
