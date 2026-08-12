"""Online LegalQA pipeline."""

from __future__ import annotations

from typing import Any

from config import LegalQAConfig
from src.generation.context import reconstruct_context
from src.model.qwen3_5_4b_gguf import QwenGGUFBackend
from src.prepare_data.chunking import load_tokenizer
from src.rerank.reranking import CrossEncoderReranker, select_top_k_evidence
from src.retrieval.biencoder import BiencoderRetriever
from src.retrieval.bm25s import BM25sRetriever
from src.retrieval.hybrid import get_candidate_chunks


class MockLLMBackend:
    def generate_answer(self, question: str, context: str) -> str:
        return "[MOCK] Chua cau hinh GGUF model, hay truyen --llm-model-path de sinh cau tra loi that."


def retrieve_evidence(
    question: str,
    artifacts: dict[str, Any],
    config: LegalQAConfig,
    dense_retriever: BiencoderRetriever | None = None,
    reranker: CrossEncoderReranker | None = None,
    use_reranker: bool = True,
) -> dict[str, Any]:
    bm25_retriever = BM25sRetriever()
    dense_index = artifacts.get("dense_index")
    if dense_index is not None and dense_retriever is None:
        dense_retriever = BiencoderRetriever(config.biencoder_model, batch_size=config.dense_batch_size)

    candidate_chunks = get_candidate_chunks(
        question,
        bm25_retriever=bm25_retriever,
        bm25_index=artifacts["bm25"],
        dense_retriever=dense_retriever,
        dense_index=dense_index,
        chunks=artifacts["chunks"],
        top_k=config.retrieval_k,
        alpha=config.alpha_hybrid,
    )

    if use_reranker:
        if reranker is None:
            reranker = CrossEncoderReranker(config.reranker_model, max_length=config.reranker_max_length)
        ranked_chunks = reranker.rerank(
            question,
            candidate_chunks,
            batch_size=config.reranker_batch_size,
        )
    else:
        ranked_chunks = sorted(candidate_chunks, key=lambda x: x.get("hybrid_score", 0.0), reverse=True)

    evidence = select_top_k_evidence(ranked_chunks, k=config.evidence_k)
    return {
        "candidate_chunks": candidate_chunks,
        "ranked_chunks": ranked_chunks,
        "evidence": evidence,
    }


def answer_question(
    question: str,
    artifacts: dict[str, Any],
    config: LegalQAConfig,
    llm_model_path: str | None = None,
    dense_retriever: BiencoderRetriever | None = None,
    reranker: CrossEncoderReranker | None = None,
    tokenizer_model: str | None = None,
    use_reranker: bool = True,
) -> dict[str, Any]:
    stages = retrieve_evidence(
        question,
        artifacts,
        config=config,
        dense_retriever=dense_retriever,
        reranker=reranker,
        use_reranker=use_reranker,
    )
    tokenizer = load_tokenizer(tokenizer_model or config.biencoder_model)
    context = reconstruct_context(
        stages["evidence"],
        artifacts["document_lookup"],
        tokenizer=tokenizer,
        max_context_tokens=config.max_context_tokens,
        expand_window=config.expand_window,
    )

    llm = QwenGGUFBackend(llm_model_path, config) if llm_model_path else MockLLMBackend()
    answer = llm.generate_answer(question, context)
    return {
        "question": question,
        "evidence": stages["evidence"],
        "context": context,
        "answer": answer,
    }
