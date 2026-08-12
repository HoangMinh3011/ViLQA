"""llama.cpp backend for Qwen GGUF generation."""

from __future__ import annotations

import logging

from config import LegalQAConfig
from src.prompt.prompt_utils import build_generation_prompt

logger = logging.getLogger(__name__)


class QwenGGUFBackend:
    def __init__(self, model_path: str, config: LegalQAConfig) -> None:
        try:
            from llama_cpp import Llama
        except ImportError as exc:
            raise ImportError("Install llama-cpp-python to use GGUF generation.") from exc

        self.config = config
        logger.info(
            "Loading GGUF with n_gpu_layers=%s, n_ctx=%s, n_threads=%s",
            config.llm_n_gpu_layers,
            config.llm_n_ctx,
            config.llm_n_threads,
        )
        self.llm = Llama(
            model_path=model_path,
            n_ctx=config.llm_n_ctx,
            n_threads=config.llm_n_threads,
            n_gpu_layers=config.llm_n_gpu_layers,
            verbose=config.llm_verbose,
        )

    def _tokenize(self, text: str, add_bos: bool = False) -> list[int]:
        return self.llm.tokenize(text.encode("utf-8"), add_bos=add_bos, special=False)

    def _detokenize(self, tokens: list[int]) -> str:
        return self.llm.detokenize(tokens).decode("utf-8", errors="ignore")

    def _fit_prompt(self, question: str, context: str) -> str:
        prompt = build_generation_prompt(question, context)
        max_prompt_tokens = max(
            1,
            self.config.llm_n_ctx
            - self.config.llm_max_tokens
            - self.config.llm_prompt_safety_tokens,
        )
        prompt_tokens = self._tokenize(prompt, add_bos=True)
        if len(prompt_tokens) <= max_prompt_tokens:
            return prompt

        empty_prompt = build_generation_prompt(question, "")
        overhead = len(self._tokenize(empty_prompt, add_bos=True))
        context_budget = max(0, max_prompt_tokens - overhead)
        context_tokens = self._tokenize(context, add_bos=False)
        trimmed_context = self._detokenize(context_tokens[:context_budget])
        logger.warning(
            "Truncated context from %d to %d tokens to fit n_ctx=%d",
            len(context_tokens),
            context_budget,
            self.config.llm_n_ctx,
        )
        return build_generation_prompt(question, trimmed_context)

    def generate_answer(self, question: str, context: str) -> str:
        prompt = self._fit_prompt(question, context)
        output = self.llm(
            prompt,
            max_tokens=self.config.llm_max_tokens,
            temperature=self.config.llm_temperature,
            top_p=self.config.llm_top_p,
            stop=["[END]"],
        )
        return output["choices"][0]["text"].strip()
