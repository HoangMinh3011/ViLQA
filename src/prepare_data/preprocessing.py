"""Document normalization for selected-contexts style legal corpus."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any


DOC_NUMBER_PATTERN = re.compile(r"\b(?:So|S)\s*:\s*([\w\d/\-.]+)", flags=re.IGNORECASE)
DOC_TITLE_PATTERN = re.compile(
    r"(?im)^\s*("
    r"THONG TU|THÔNG TƯ|"
    r"NGHI DINH|NGHỊ ĐỊNH|"
    r"QUYET DINH|QUYẾT ĐỊNH|"
    r"LUAT|LUẬT|"
    r"NGHI QUYET|NGHỊ QUYẾT|"
    r"CHI THI|CHỈ THỊ|"
    r"TCVN|QCVN"
    r")\b.*$"
)


def normalize_text(raw_text: str) -> str:
    text = raw_text.replace("\r\n", " ").replace("\r", " ").replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def title_from_link(link: str) -> str | None:
    if not link:
        return None
    slug = link.rstrip("/").split("/")[-1]
    slug = re.sub(r"-\d+\.aspx$", "", slug)
    slug = re.sub(r"\.aspx$", "", slug)
    return slug.replace("-", " ").strip() or None


def extract_document_metadata(raw_text: str, link: str = "") -> dict[str, Any]:
    number_match = DOC_NUMBER_PATTERN.search(raw_text)
    title_match = DOC_TITLE_PATTERN.search(raw_text)
    return {
        "document_number": number_match.group(1) if number_match else None,
        "title": title_match.group(0).strip() if title_match else title_from_link(link),
    }


def normalize_document(raw: dict[str, Any], fallback_id: str | None = None) -> dict[str, Any]:
    raw_text = raw.get("passage") or raw.get("text") or ""
    link = raw.get("link") or raw.get("source") or ""
    doc_id = str(raw.get("id") or raw.get("document_id") or fallback_id or "")
    meta = extract_document_metadata(raw_text, link=link)

    return {
        "document_id": doc_id,
        "title": raw.get("name") or raw.get("title") or meta["title"] or doc_id,
        "text": normalize_text(raw_text),
        "metadata": {
            "document_number": meta["document_number"],
            "source": link,
        },
    }


def load_raw_contexts(contexts_dir: Path, limit: int | None = None) -> list[dict[str, Any]]:
    import json

    paths = sorted(contexts_dir.glob("context_*.json"))
    if limit is not None:
        paths = paths[:limit]

    records: list[dict[str, Any]] = []
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        data.setdefault("id", path.stem.replace("context_", ""))
        records.append(data)
    return records


def normalize_corpus(raw_documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [normalize_document(raw) for raw in raw_documents]
