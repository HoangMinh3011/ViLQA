"""Token-overlap chunking for legal documents."""

from __future__ import annotations

from typing import Any, Protocol


class TokenizerLike(Protocol):
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]: ...

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str: ...


class WhitespaceTokenizer:
    """Fallback tokenizer for tests or environments without transformers."""

    def encode(self, text: str, add_special_tokens: bool = False) -> list[str]:
        return text.split()

    def decode(self, token_ids: list[str], skip_special_tokens: bool = True) -> str:
        return " ".join(token_ids)


def load_tokenizer(model_name: str) -> TokenizerLike:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        raise ImportError("Install transformers to load the chunking tokenizer.") from exc
    return AutoTokenizer.from_pretrained(model_name)


def chunk_document(
    document: dict[str, Any],
    tokenizer: TokenizerLike,
    chunk_size: int = 300,
    overlap: int = 50,
) -> list[dict[str, Any]]:
    text = document["text"]
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if not tokens:
        return []

    chunks: list[dict[str, Any]] = []
    step = max(1, chunk_size - overlap)
    start = 0
    chunk_index = 0

    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = tokenizer.decode(chunk_tokens, skip_special_tokens=True)
        chunks.append(
            {
                "chunk_id": f"{document['document_id']}_chunk_{chunk_index}",
                "document_id": document["document_id"],
                "document_title": document.get("title", ""),
                "chunk_index": chunk_index,
                "token_start": start,
                "token_end": end,
                "text": chunk_text.strip(),
                "metadata": document.get("metadata", {}),
            }
        )
        if end == len(tokens):
            break
        start += step
        chunk_index += 1

    return chunks


def chunk_corpus(
    documents: list[dict[str, Any]],
    tokenizer: TokenizerLike,
    chunk_size: int = 300,
    overlap: int = 50,
) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = []
    for document in documents:
        chunks.extend(chunk_document(document, tokenizer, chunk_size=chunk_size, overlap=overlap))
    return chunks
