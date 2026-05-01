#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import atexit
from dataclasses import dataclass
from functools import lru_cache
import logging
from typing import Any

from vllm import LLM
from vllm.config import PoolerConfig

from llm_core.vllm_runtime_utils import (
    AUTO_FALLBACK_MODEL_LEN_STEP,
    resolve_fallback_max_model_len,
    should_try_stepdown_fallback,
)
from nl2sql.utils.vllm_runtime import (
    build_local_llm,
    release_local_llm,
    shutdown_registered_llms,
)

from .config import DEFAULT_POOLING_TYPE, SemanticSchemaPruningConfig

_REGISTERED_LLMS: list[object] = []
_SHUTDOWN_LLM_IDS: set[int] = set()
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class E2RankRuntime:
    llm: Any
    tokenizer: Any
    model: str
    requested_max_model_len: int
    effective_max_model_len: int
    gpu_memory_utilization: float


def _build_pooler_config() -> PoolerConfig:
    """Construye la configuracion de pooling compatible con el runtime actual."""
    return PoolerConfig(task="embed", pooling_type=DEFAULT_POOLING_TYPE)


def shutdown_cached_llm() -> None:
    """Libera el engine cacheado para evitar cierres ruidosos de EngineCore."""
    shutdown_registered_llms(
        _REGISTERED_LLMS,
        _SHUTDOWN_LLM_IDS,
        warning_prefix="No se pudo apagar el engine E2Rank",
        logger=logger,
    )
    _build_runtime.cache_clear()
    get_e2rank_runtime.cache_clear()


def clear_e2rank_runtime() -> None:
    """Libera runtimes cacheados y vacia la cache CUDA si esta disponible."""

    shutdown_cached_llm()
    release_local_llm()


@lru_cache(maxsize=4)
def _build_runtime(
    model: str,
    *,
    dtype: str,
    max_model_len: int,
    gpu_memory_utilization: float,
    tensor_parallel_size: int,
) -> E2RankRuntime:
    # El constructor publico de vLLM para este runtime sigue exponiendo
    # `runner=`/`convert=` para embeddings, mientras `PoolerConfig` no acepta
    # `normalize`. Ademas se fuerza eager para evitar el costo de
    # `torch.compile`/cudagraphs en un engine corto que arranca y se destruye
    # dentro del pipeline; en vLLM 0.20 eso estabiliza el budget de KV cache en
    # la RTX 3080 Ti Laptop de 16 GiB.
    llm = build_local_llm(
        LLM,
        model=model,
        runner="pooling",
        convert="embed",
        dtype=dtype,
        max_model_len=max_model_len,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=gpu_memory_utilization,
        trust_remote_code=True,
        enforce_eager=True,
        pooler_config=_build_pooler_config(),
    )
    tokenizer = llm.get_tokenizer()
    _REGISTERED_LLMS.append(llm)
    return E2RankRuntime(
        llm=llm,
        tokenizer=tokenizer,
        model=model,
        requested_max_model_len=max_model_len,
        effective_max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
    )


@lru_cache(maxsize=4)
def get_e2rank_runtime(
    model: str,
    *,
    dtype: str,
    max_model_len: int,
    gpu_memory_utilization: float,
    tensor_parallel_size: int,
) -> E2RankRuntime:
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
            )
            return E2RankRuntime(
                llm=runtime.llm,
                tokenizer=runtime.tokenizer,
                model=runtime.model,
                requested_max_model_len=max_model_len,
                effective_max_model_len=current_max_model_len,
                gpu_memory_utilization=runtime.gpu_memory_utilization,
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
    raise RuntimeError("No se pudo inicializar el engine E2Rank.")


def get_e2rank_llm(config: SemanticSchemaPruningConfig) -> Any:
    return get_e2rank_runtime(
        config.model,
        dtype=config.dtype,
        max_model_len=config.max_model_len,
        gpu_memory_utilization=config.gpu_memory_utilization,
        tensor_parallel_size=config.tensor_parallel_size,
    ).llm


def get_e2rank_tokenizer(config: SemanticSchemaPruningConfig) -> Any:
    return get_e2rank_runtime(
        config.model,
        dtype=config.dtype,
        max_model_len=config.max_model_len,
        gpu_memory_utilization=config.gpu_memory_utilization,
        tensor_parallel_size=config.tensor_parallel_size,
    ).tokenizer


def get_effective_max_model_len(config: SemanticSchemaPruningConfig) -> int:
    return get_e2rank_runtime(
        config.model,
        dtype=config.dtype,
        max_model_len=config.max_model_len,
        gpu_memory_utilization=config.gpu_memory_utilization,
        tensor_parallel_size=config.tensor_parallel_size,
    ).effective_max_model_len


atexit.register(shutdown_cached_llm)
