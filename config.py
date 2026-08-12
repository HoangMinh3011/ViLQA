"""Central configuration for the LegalQA pipeline."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LegalQAConfig:
    # Models
    biencoder_model: str = "NghiemAbe/Vi-Legal-Bi-Encoder-v2"
    reranker_model: str = "itdainb/PhoRanker"
    llm_repo_id: str = "unsloth/Qwen3.5-4B-GGUF"
    llm_filename: str = "Qwen3.5-4B-UD-Q4_K_XL.gguf"

    # Chunking
    chunk_size: int = 300
    chunk_overlap: int = 50

    # Retrieval
    retrieval_k: int = 20
    evidence_k: int = 5
    alpha_hybrid: float = 0.5

    # Context reconstruction
    max_context_tokens: int = 5000
    expand_window: int = 50

    # Dense encoder
    dense_batch_size: int = 32

    # Cross-encoder reranker
    reranker_batch_size: int = 16
    reranker_max_length: int = 256

    # LLM
    llm_n_ctx: int = 8192
    llm_n_threads: int = 8
    llm_n_gpu_layers: int = -1
    llm_max_tokens: int = 1024
    llm_temperature: float = 0.1
    llm_top_p: float = 0.9
    llm_prompt_safety_tokens: int = 128
