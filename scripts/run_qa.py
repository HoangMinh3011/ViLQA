#!/usr/bin/env python3
"""Run LegalQA inference for one question or a JSON question file."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import LegalQAConfig
from src.io.artifacts import load_artifacts
from src.model.qwen3_5_4b_gguf import QwenGGUFBackend
from src.pipeline.online import MockLLMBackend, answer_question, retrieve_evidence
from src.prepare_data.chunking import load_tokenizer
from src.rerank.reranking import CrossEncoderReranker
from src.retrieval.biencoder import BiencoderRetriever


def load_questions(path: Path) -> list[tuple[str, str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    questions: list[tuple[str, str]] = []
    for qid, item in payload.items():
        question = item["question"] if isinstance(item, dict) else str(item)
        questions.append((str(qid), question))
    return questions


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LegalQA pipeline")
    parser.add_argument("--question", default=None)
    parser.add_argument("--questions-file", type=Path, default=None)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--artifacts-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--llm-model-path", default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--output-format", choices=("full", "submission"), default="full")
    parser.add_argument("--zip-output", type=Path, default=None)
    parser.add_argument("--inspect", action="store_true")
    parser.add_argument("--skip-rerank", action="store_true")

    parser.add_argument("--biencoder-model", default=LegalQAConfig.biencoder_model)
    parser.add_argument("--reranker-model", default=LegalQAConfig.reranker_model)
    parser.add_argument("--tokenizer-model", default=None)
    parser.add_argument("--retrieval-k", type=int, default=LegalQAConfig.retrieval_k)
    parser.add_argument("--evidence-k", type=int, default=LegalQAConfig.evidence_k)
    parser.add_argument("--alpha-hybrid", type=float, default=LegalQAConfig.alpha_hybrid)
    parser.add_argument("--max-context-tokens", type=int, default=LegalQAConfig.max_context_tokens)
    parser.add_argument("--expand-window", type=int, default=LegalQAConfig.expand_window)
    parser.add_argument("--dense-batch-size", type=int, default=LegalQAConfig.dense_batch_size)
    parser.add_argument("--device", default=None)
    parser.add_argument("--embedding-device", default=LegalQAConfig.embedding_device)
    parser.add_argument("--reranker-batch-size", type=int, default=LegalQAConfig.reranker_batch_size)
    parser.add_argument("--reranker-max-length", type=int, default=LegalQAConfig.reranker_max_length)
    parser.add_argument("--reranker-device", default=LegalQAConfig.reranker_device)

    parser.add_argument("--llm-n-ctx", type=int, default=LegalQAConfig.llm_n_ctx)
    parser.add_argument("--llm-n-threads", type=int, default=LegalQAConfig.llm_n_threads)
    parser.add_argument("--llm-n-gpu-layers", type=int, default=LegalQAConfig.llm_n_gpu_layers)
    parser.add_argument("--llm-max-tokens", type=int, default=LegalQAConfig.llm_max_tokens)
    parser.add_argument("--llm-temperature", type=float, default=LegalQAConfig.llm_temperature)
    parser.add_argument("--llm-top-p", type=float, default=LegalQAConfig.llm_top_p)
    parser.add_argument("--llm-prompt-safety-tokens", type=int, default=LegalQAConfig.llm_prompt_safety_tokens)
    parser.add_argument("--llm-verbose", action="store_true")
    parser.add_argument("--timing", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    if not args.question and not args.questions_file:
        parser.error("Provide --question or --questions-file")
    if args.inspect and (args.output_format == "submission" or args.zip_output):
        parser.error("--inspect cannot be used with submission output")

    config = LegalQAConfig(
        biencoder_model=args.biencoder_model,
        reranker_model=args.reranker_model,
        retrieval_k=args.retrieval_k,
        evidence_k=args.evidence_k,
        alpha_hybrid=args.alpha_hybrid,
        max_context_tokens=args.max_context_tokens,
        expand_window=args.expand_window,
        dense_batch_size=args.dense_batch_size,
        embedding_device=args.embedding_device or args.device,
        reranker_batch_size=args.reranker_batch_size,
        reranker_max_length=args.reranker_max_length,
        reranker_device=args.reranker_device or args.device,
        llm_n_ctx=args.llm_n_ctx,
        llm_n_threads=args.llm_n_threads,
        llm_n_gpu_layers=args.llm_n_gpu_layers,
        llm_max_tokens=args.llm_max_tokens,
        llm_temperature=args.llm_temperature,
        llm_top_p=args.llm_top_p,
        llm_prompt_safety_tokens=args.llm_prompt_safety_tokens,
        llm_verbose=args.llm_verbose,
    )

    artifacts = load_artifacts(args.artifacts_dir)
    dense_retriever = (
        BiencoderRetriever(
            config.biencoder_model,
            batch_size=config.dense_batch_size,
            device=config.embedding_device,
        )
        if "dense_index" in artifacts
        else None
    )
    reranker = (
        None
        if args.skip_rerank
        else CrossEncoderReranker(
            config.reranker_model,
            config.reranker_max_length,
            device=config.reranker_device,
        )
    )

    questions: list[tuple[str, str]] = []
    if args.question:
        questions.append(("manual", args.question))
    if args.questions_file:
        questions.extend(load_questions(args.questions_file))
    if args.offset or args.limit is not None:
        start = max(0, args.offset)
        end = None if args.limit is None else start + max(0, args.limit)
        questions = questions[start:end]

    results = []
    submission: dict[str, dict[str, str]] = {}
    tokenizer_model = args.tokenizer_model or config.biencoder_model
    tokenizer = load_tokenizer(tokenizer_model)
    document_token_cache = {}
    llm_backend = None
    if not args.inspect:
        llm_backend = (
            QwenGGUFBackend(args.llm_model_path, config)
            if args.llm_model_path
            else MockLLMBackend()
        )

    for qid, question in questions:
        if args.inspect:
            stages = retrieve_evidence(
                question,
                artifacts,
                config=config,
                dense_retriever=dense_retriever,
                reranker=reranker,
                use_reranker=not args.skip_rerank,
            )
            result = {
                "id": qid,
                "question": question,
                **stages,
                "context_preview": "",
            }
            from src.generation.context import reconstruct_context

            result["context_preview"] = reconstruct_context(
                stages["evidence"],
                artifacts["document_lookup"],
                tokenizer,
                max_context_tokens=config.max_context_tokens,
                expand_window=config.expand_window,
                document_token_cache=document_token_cache,
            )
        else:
            qa = answer_question(
                question,
                artifacts,
                config=config,
                llm=llm_backend,
                dense_retriever=dense_retriever,
                reranker=reranker,
                tokenizer=tokenizer,
                document_token_cache=document_token_cache,
                use_reranker=not args.skip_rerank,
                log_timing=args.timing,
            )
            result = {"id": qid, **qa}
            submission[qid] = {"answer": str(qa["answer"])}

        results.append(result)
        print(json.dumps(result, ensure_ascii=False, indent=2))

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        payload = submission if args.output_format == "submission" else results
        args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.zip_output:
        args.zip_output.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(args.zip_output, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("submission.json", json.dumps(submission, ensure_ascii=False, indent=2).encode("utf-8"))


if __name__ == "__main__":
    main()
