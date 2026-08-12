#!/usr/bin/env python3
"""Build BM25/dense retrieval artifacts from selected-contexts corpus."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import LegalQAConfig
from src.io.artifacts import save_artifacts
from src.pipeline.offline import build_indexes_from_contexts


def main() -> None:
    parser = argparse.ArgumentParser(description="Build LegalQA retrieval artifacts")
    parser.add_argument("--contexts-dir", type=Path, default=Path("selected-contexts"))
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--skip-dense", action="store_true")
    parser.add_argument("--biencoder-model", default=LegalQAConfig.biencoder_model)
    parser.add_argument("--tokenizer-model", default=None)
    parser.add_argument("--chunk-size", type=int, default=LegalQAConfig.chunk_size)
    parser.add_argument("--chunk-overlap", type=int, default=LegalQAConfig.chunk_overlap)
    parser.add_argument("--dense-batch-size", type=int, default=LegalQAConfig.dense_batch_size)
    parser.add_argument("--device", default=LegalQAConfig.embedding_device)
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    config = LegalQAConfig(
        biencoder_model=args.biencoder_model,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        dense_batch_size=args.dense_batch_size,
        embedding_device=args.device,
    )
    artifacts = build_indexes_from_contexts(
        args.contexts_dir,
        config=config,
        limit=args.limit,
        build_dense=not args.skip_dense,
        tokenizer_model=args.tokenizer_model,
    )
    save_artifacts(artifacts, args.artifacts_dir)
    logging.info("Saved artifacts to %s", args.artifacts_dir)


if __name__ == "__main__":
    main()
