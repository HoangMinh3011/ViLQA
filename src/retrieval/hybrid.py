"""Hybrid retrieval fusion for BM25 and dense results."""

from __future__ import annotations

from typing import Any

import numpy as np


def minmax_normalize(values: list[float] | np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if len(arr) == 0:
        return arr
    lo, hi = arr.min(), arr.max()
    if hi - lo < 1e-12:
        if abs(hi) < 1e-12:
            return np.zeros_like(arr)
        return np.ones_like(arr)
    return (arr - lo) / (hi - lo)


def hybrid_fusion(
    bm25_results: list[dict[str, Any]],
    dense_results: list[dict[str, Any]],
    alpha: float = 0.5,
) -> list[dict[str, Any]]:
    candidates: dict[str, dict[str, Any]] = {}
    bm25_norm = minmax_normalize([r.get("bm25_score", 0.0) for r in bm25_results])
    dense_norm = minmax_normalize([r.get("dense_score", 0.0) for r in dense_results])

    for item, score in zip(bm25_results, bm25_norm):
        chunk_id = item["chunk_id"]
        candidate = candidates.setdefault(chunk_id, dict(item))
        candidate["bm25_norm"] = float(score)

    for item, score in zip(dense_results, dense_norm):
        chunk_id = item["chunk_id"]
        candidate = candidates.setdefault(chunk_id, dict(item))
        candidate["dense_norm"] = float(score)

    for item in candidates.values():
        item["hybrid_score"] = (
            alpha * item.get("bm25_norm", 0.0)
            + (1.0 - alpha) * item.get("dense_norm", 0.0)
        )

    return sorted(candidates.values(), key=lambda x: x.get("hybrid_score", 0.0), reverse=True)


def get_candidate_chunks(
    question: str,
    bm25_retriever,
    bm25_index,
    dense_retriever,
    dense_index,
    chunks: list[dict[str, Any]],
    top_k: int = 20,
    alpha: float = 0.5,
) -> list[dict[str, Any]]:
    bm25_results = bm25_retriever.retrieve(question, bm25_index, chunks, top_k=top_k)
    if dense_retriever is None or dense_index is None:
        bm25_norm = minmax_normalize([r.get("bm25_score", 0.0) for r in bm25_results])
        for item, score in zip(bm25_results, bm25_norm):
            item["bm25_norm"] = float(score)
            item["hybrid_score"] = float(score)
        return bm25_results[:top_k]

    dense_results = dense_retriever.retrieve(question, dense_index, chunks, top_k=top_k)
    return hybrid_fusion(bm25_results, dense_results, alpha=alpha)[:top_k]
