#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import atexit
from dataclasses import dataclass
from functools import lru_cache
import logging
import math
from typing import Any

from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.inputs import TokensPrompt

from llm_core.vllm_runtime_utils import (
    AUTO_FALLBACK_MODEL_LEN_STEP,
    resolve_fallback_max_model_len,
    should_try_stepdown_fallback,
)
from nl2sql.utils.vllm_runtime import build_local_llm, shutdown_registered_llms

from .config import load_runtime_tuning

REGISTERED_LLMS: list[object] = []
SHUTDOWN_LLM_IDS: set[int] = set()
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RerankRuntime:
    llm: Any
    tokenizer: Any
    true_token_id: int
    false_token_id: int
    suffix_token_ids: tuple[int, ...]
    requested_max_model_len: int
    effective_max_model_len: int


def clear_reranker_runtime() -> None:
    """Libera el reranker para escenarios con VRAM ajustada."""

    shutdown_registered_llms(
        REGISTERED_LLMS,
        SHUTDOWN_LLM_IDS,
        warning_prefix="No se pudo apagar el engine de rerank",
        logger=logger,
    )
    get_reranker_runtime.cache_clear()


def _resolve_binary_token_id(tokenizer: Any, candidates: tuple[str, ...]) -> int:
    for candidate in candidates:
        token_ids = tokenizer(candidate, add_special_tokens=False).input_ids
        if len(token_ids) == 1:
            return int(token_ids[0])
    raise ValueError(f"No se pudo resolver un token binario para {candidates!r}.")


def _build_runtime(
    model: str,
    *,
    dtype: str,
    max_model_len: int,
    gpu_memory_utilization: float,
    tensor_parallel_size: int,
    trust_remote_code: bool,
) -> RerankRuntime:
    runtime_tuning = load_runtime_tuning()
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=trust_remote_code)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # El reranker se usa en una ventana corta y luego se descarga. Eager evita
    # compilar/capturar grafos sin aportar valor material y reduce fallos de
    # arranque por memoria en vLLM 0.21 sobre GPUs de 16 GiB.
    llm = build_local_llm(
        LLM,
        model=model,
        dtype=dtype,
        max_model_len=max_model_len,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        trust_remote_code=trust_remote_code,
        enforce_eager=True,
        enable_prefix_caching=True,
    )
    REGISTERED_LLMS.append(llm)

    return RerankRuntime(
        llm=llm,
        tokenizer=tokenizer,
        true_token_id=_resolve_binary_token_id(tokenizer, runtime_tuning.rerank_positive_token_candidates),
        false_token_id=_resolve_binary_token_id(tokenizer, runtime_tuning.rerank_negative_token_candidates),
        suffix_token_ids=tuple(tokenizer.encode(runtime_tuning.rerank_suffix, add_special_tokens=False)),
        requested_max_model_len=max_model_len,
        effective_max_model_len=max_model_len,
    )


@lru_cache(maxsize=4)
def get_reranker_runtime(
    model: str,
    *,
    dtype: str,
    max_model_len: int,
    gpu_memory_utilization: float,
    tensor_parallel_size: int,
    trust_remote_code: bool,
) -> RerankRuntime:
    current_max_model_len = max_model_len
    last_error: Exception | None = None

    while current_max_model_len > AUTO_FALLBACK_MODEL_LEN_STEP:
        try:
            runtime = _build_runtime(
                model,
                dtype=dtype,
                max_model_len=current_max_model_len,
                gpu_memory_utilization=gpu_memory_utilization,
                tensor_parallel_size=tensor_parallel_size,
                trust_remote_code=trust_remote_code,
            )
            return RerankRuntime(
                llm=runtime.llm,
                tokenizer=runtime.tokenizer,
                true_token_id=runtime.true_token_id,
                false_token_id=runtime.false_token_id,
                suffix_token_ids=runtime.suffix_token_ids,
                requested_max_model_len=max_model_len,
                effective_max_model_len=current_max_model_len,
            )
        except Exception as error:
            last_error = error
            fallback_max_model_len = resolve_fallback_max_model_len(current_max_model_len, error)
            if fallback_max_model_len is None and should_try_stepdown_fallback(error):
                fallback_max_model_len = current_max_model_len - AUTO_FALLBACK_MODEL_LEN_STEP
            if fallback_max_model_len is None or fallback_max_model_len >= current_max_model_len:
                raise
            current_max_model_len = fallback_max_model_len

    if last_error is not None:
        raise last_error
    raise RuntimeError("No se pudo inicializar el engine de rerank.")


def _apply_chat_template(tokenizer: Any, messages: list[dict[str, str]]) -> list[int]:
    try:
        tokenized_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
            enable_thinking=False,
        )
    except TypeError:
        tokenized_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
        )

    # En esta version de transformers el chat template devuelve un BatchEncoding,
    # no una lista plana de ids. Aqui se normaliza siempre a list[int] para vLLM.
    if hasattr(tokenized_prompt, "get"):
        token_ids = tokenized_prompt.get("input_ids", tokenized_prompt)
    else:
        token_ids = tokenized_prompt

    if isinstance(token_ids, list) and token_ids and isinstance(token_ids[0], list):
        token_ids = token_ids[0]

    return [int(token_id) for token_id in token_ids]


def _wrap_prompt(prompt_token_ids: list[int]) -> Any:
    return TokensPrompt(prompt_token_ids=prompt_token_ids)


def _build_prompt_token_ids(
    runtime: RerankRuntime,
    *,
    query: str,
    document: str,
    instruction: str,
    system_prompt: str,
    user_prompt_template: str,
    prompt_token_margin: int,
) -> list[int]:
    user_prompt = user_prompt_template.format(
        instruction=instruction,
        query=query,
        document=document,
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    token_ids = _apply_chat_template(runtime.tokenizer, messages)
    max_prompt_tokens = max(
        1,
        runtime.effective_max_model_len - len(runtime.suffix_token_ids) - prompt_token_margin,
    )
    if len(token_ids) > max_prompt_tokens:
        token_ids = token_ids[:max_prompt_tokens]
    return token_ids + list(runtime.suffix_token_ids)


def _extract_logprob_value(logprobs: object, token_id: int, default: float = -10.0) -> float:
    if not isinstance(logprobs, dict) or token_id not in logprobs:
        return default
    token_logprob = logprobs[token_id]
    value = getattr(token_logprob, "logprob", token_logprob)
    if isinstance(value, (int, float)):
        return float(value)
    return default


def _compute_probability(yes_logprob: float, no_logprob: float) -> float:
    max_logprob = max(yes_logprob, no_logprob)
    yes_score = math.exp(yes_logprob - max_logprob)
    no_score = math.exp(no_logprob - max_logprob)
    return yes_score / (yes_score + no_score)


def rerank_candidates(
    runtime: RerankRuntime,
    *,
    query: str,
    instruction: str,
    system_prompt: str,
    user_prompt_template: str,
    candidates: list[tuple[object, str, float]],
    batch_size: int,
    max_document_chars: int,
    logprobs: int,
    prompt_token_margin: int,
) -> list[tuple[object, float, float]]:
    """Reranquea pares query-documento con la plantilla oficial del modelo.

    Se fuerza `max_tokens=1` y `temperature=0.0` porque el objetivo no es generar
    texto libre, sino leer la probabilidad relativa entre las salidas binarias yes/no.
    """

    if not candidates:
        return []

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=1,
        logprobs=logprobs,
        allowed_token_ids=[runtime.true_token_id, runtime.false_token_id],
    )

    reranked: list[tuple[object, float, float]] = []
    safe_batch_size = max(1, batch_size)

    for start_index in range(0, len(candidates), safe_batch_size):
        batch = candidates[start_index : start_index + safe_batch_size]
        prompts = []
        for _asset, document_text, _embedding_score in batch:
            prompt_token_ids = _build_prompt_token_ids(
                runtime,
                query=query,
                document=document_text[:max_document_chars],
                instruction=instruction,
                system_prompt=system_prompt,
                user_prompt_template=user_prompt_template,
                prompt_token_margin=prompt_token_margin,
            )
            prompts.append(_wrap_prompt(prompt_token_ids))

        outputs = runtime.llm.generate(prompts, sampling_params, use_tqdm=False)
        for (asset, _document_text, embedding_score), output in zip(batch, outputs):
            output_items = getattr(output, "outputs", [])
            if not output_items:
                reranked.append((asset, embedding_score, 0.0))
                continue

            item = output_items[0]
            logprob_steps = getattr(item, "logprobs", None) or []
            last_logprobs = logprob_steps[-1] if logprob_steps else {}
            yes_logprob = _extract_logprob_value(last_logprobs, runtime.true_token_id)
            no_logprob = _extract_logprob_value(last_logprobs, runtime.false_token_id)
            # Se normaliza solo entre yes y no para obtener un score calibrado 0-1.
            reranked.append((asset, embedding_score, _compute_probability(yes_logprob, no_logprob)))

    return reranked


atexit.register(clear_reranker_runtime)
