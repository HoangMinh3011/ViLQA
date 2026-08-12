"""Dense retrieval with SentenceTransformer + FAISS."""

from __future__ import annotations

from typing import Any

import faiss
import numpy as np


class BiencoderRetriever:
    def __init__(self, model_name: str, batch_size: int = 32) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError("Install sentence-transformers to use dense retrieval.") from exc

        self.model_name = model_name
        self.batch_size = batch_size
        self.model = SentenceTransformer(model_name)

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        embeddings = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=True,
        )
        return np.asarray(embeddings, dtype="float32")

    def encode_query(self, query: str) -> np.ndarray:
        embedding = self.model.encode(query, normalize_embeddings=True)
        return np.asarray(embedding, dtype="float32")

    @staticmethod
    def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
        embeddings = np.asarray(embeddings, dtype="float32")
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        return index

    def retrieve(
        self,
        query: str,
        index: faiss.Index,
        chunks: list[dict[str, Any]],
        top_k: int = 20,
    ) -> list[dict[str, Any]]:
        query_vec = self.encode_query(query).reshape(1, -1)
        scores, indices = index.search(query_vec, top_k)
        results: list[dict[str, Any]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            item = dict(chunks[int(idx)])
            item["dense_score"] = float(score)
            results.append(item)
        return results
