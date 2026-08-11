#!/usr/bin/env python3
"""Explicit, resumable GGUF downloader; model selection remains the user's decision."""
import argparse
from pathlib import Path
from __future__ import annotations
from huggingface_hub import hf_hub_download


def main() -> None:
    parser = argparse.ArgumentParser(description="Download one GGUF file from Hugging Face")
    parser.add_argument("--repo-id", required=True, help="e.g. org/model-GGUF")
    parser.add_argument("--filename", required=True, help="exact .gguf filename in the repository")
    parser.add_argument("--output", required=True, help="destination .gguf path")
    parser.add_argument("--revision", default="main")
    
    args = parser.parse_args()
    target = Path(args.output).resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    downloaded = hf_hub_download(repo_id=args.repo_id, filename=args.filename, revision=args.revision, local_dir=target.parent)
    source = Path(downloaded)
    if source.resolve() != target:
        source.replace(target)
    print(f"Saved: {target}")


if __name__ == "__main__":
    main()