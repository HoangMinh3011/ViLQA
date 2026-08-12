"""Save and load pipeline artifacts."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from src.retrieval.bm25s import BM25sRetriever


def save_artifacts(artifacts: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "documents.json").write_text(
        json.dumps(artifacts["documents"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_dir / "chunks.json").write_text(
        json.dumps(artifacts["chunks"], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    with (out_dir / "document_lookup.pkl").open("wb") as f:
        pickle.dump(artifacts["document_lookup"], f)

    with (out_dir / "bm25.pkl").open("wb") as f:
        pickle.dump(artifacts["bm25"], f)

    if "dense_index" in artifacts:
        faiss.write_index(artifacts["dense_index"], str(out_dir / "dense.index"))
    if "embeddings" in artifacts:
        np.save(out_dir / "embeddings.npy", artifacts["embeddings"])


def load_artifacts(artifacts_dir: Path) -> dict[str, Any]:
    chunks = json.loads((artifacts_dir / "chunks.json").read_text(encoding="utf-8"))
    documents = json.loads((artifacts_dir / "documents.json").read_text(encoding="utf-8"))
    with (artifacts_dir / "document_lookup.pkl").open("rb") as f:
        document_lookup = pickle.load(f)

    bm25_pickle = artifacts_dir / "bm25.pkl"
    if bm25_pickle.exists():
        with bm25_pickle.open("rb") as f:
            bm25 = pickle.load(f)
    else:
        bm25 = BM25sRetriever().load(artifacts_dir / "bm25_index")

    artifacts: dict[str, Any] = {
        "documents": documents,
        "document_lookup": document_lookup,
        "chunks": chunks,
        "bm25": bm25,
    }

    dense_path = artifacts_dir / "dense.index"
    embeddings_path = artifacts_dir / "embeddings.npy"
    if dense_path.exists():
        artifacts["dense_index"] = faiss.read_index(str(dense_path))
    if embeddings_path.exists():
        artifacts["embeddings"] = np.load(embeddings_path)

    return artifacts
