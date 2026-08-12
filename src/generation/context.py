"""Context reconstruction from reranked evidence chunks."""

from __future__ import annotations

from typing import Any

from src.prepare_data.chunking import TokenizerLike


def reconstruct_context(
    evidence: list[dict[str, Any]],
    document_lookup: dict[str, dict[str, Any]],
    tokenizer: TokenizerLike,
    max_context_tokens: int = 5000,
    expand_window: int = 50,
) -> str:
    seen_chunks: set[str] = set()
    blocks: list[str] = []
    used_tokens = 0

    for item in evidence:
        chunk_id = item["chunk_id"]
        if chunk_id in seen_chunks:
            continue

        document = document_lookup.get(item["document_id"])
        if document is None:
            continue

        doc_tokens = tokenizer.encode(document["text"], add_special_tokens=False)
        start = max(0, int(item["token_start"]) - expand_window)
        end = min(len(doc_tokens), int(item["token_end"]) + expand_window)
        expanded_tokens = doc_tokens[start:end]
        n_tokens = len(expanded_tokens)

        if used_tokens + n_tokens > max_context_tokens:
            remaining = max_context_tokens - used_tokens
            if remaining <= 0:
                break
            expanded_tokens = expanded_tokens[:remaining]
            n_tokens = len(expanded_tokens)

        text = tokenizer.decode(expanded_tokens, skip_special_tokens=True).strip()
        if not text:
            continue
        header = f"[{item.get('document_title') or document.get('title') or item['document_id']}]"
        blocks.append(f"{header}\n{text}")
        seen_chunks.add(chunk_id)
        used_tokens += n_tokens

        if used_tokens >= max_context_tokens:
            break

    return "\n\n---\n\n".join(blocks)

