#!/usr/bin/env python3
"""Explicit, resumable GGUF downloader."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise SystemExit("Install huggingface-hub first: pip install huggingface-hub") from exc

    parser = argparse.ArgumentParser(description="Download one GGUF file from Hugging Face")
    parser.add_argument("--repo-id", default="unsloth/Qwen3.5-4B-GGUF")
    parser.add_argument("--filename", default="Qwen3.5-4B-UD-Q4_K_XL.gguf")
    parser.add_argument("--output", default="models/Qwen3.5-4B-UD-Q4_K_XL.gguf")
    parser.add_argument("--revision", default="main")
    args = parser.parse_args()

    target = Path(args.output).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    downloaded = hf_hub_download(
        repo_id=args.repo_id,
        filename=args.filename,
        revision=args.revision,
        local_dir=target.parent,
    )
    source = Path(downloaded)
    if source.resolve() != target:
        source.replace(target)
    print(f"Saved: {target}")


if __name__ == "__main__":
    main()
