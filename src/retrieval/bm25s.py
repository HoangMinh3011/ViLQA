"""BM25 retrieval using bm25s."""

from __future__ import annotations

from pathlib import Path
from typing import Any


class BM25sRetriever:
    def __init__(self) -> None:
        try:
            import bm25s
        except ImportError as exc:
            raise ImportError("Install bm25s to use BM25 retrieval.") from exc

        self.bm25s = bm25s

    def build(self, chunks: list[dict[str, Any]]):
        corpus_texts = [chunk["text"] for chunk in chunks]
        corpus_tokens = self.bm25s.tokenize(corpus_texts, stopwords=None)
        retriever = self.bm25s.BM25()
        retriever.index(corpus_tokens)
        return retriever

    def retrieve(
        self,
        query: str,
        retriever,
        chunks: list[dict[str, Any]],
        top_k: int = 20,
    ) -> list[dict[str, Any]]:
        query_tokens = self.bm25s.tokenize([query], stopwords=None)
        results, scores = retriever.retrieve(query_tokens, k=top_k)
        output: list[dict[str, Any]] = []
        for idx, score in zip(results[0], scores[0]):
            item = dict(chunks[int(idx)])
            item["bm25_score"] = float(score)
            output.append(item)
        return output

    @staticmethod
    def save(retriever, path: Path) -> None:
        retriever.save(str(path), corpus=None)

    def load(self, path: Path):
        return self.bm25s.BM25.load(str(path))
