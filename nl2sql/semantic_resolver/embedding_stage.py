#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import atexit
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import logging
from pathlib import Path
from typing import Any

import numpy as np
from vllm import LLM
from vllm.config import PoolerConfig

from llm_core.vllm_runtime_utils import (
    AUTO_FALLBACK_MODEL_LEN_STEP,
    resolve_fallback_max_model_len,
    should_try_stepdown_fallback,
)
from nl2sql.utils.embedding_cache import (
    EmbeddingCacheStats,
    embed_with_cache,
)
from nl2sql.utils.vllm_runtime import build_local_llm, shutdown_registered_llms
from nl2sql.utils.vector_math import normalize_matrix

from .assets import SemanticAsset
from .text_formatting import format_query_for_embedding

REGISTERED_LLMS: list[object] = []
SHUTDOWN_LLM_IDS: set[int] = set()
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EmbeddingRuntime:
    llm: Any
    model: str
    requested_max_model_len: int
    effective_max_model_len: int


def _build_pooler_config() -> PoolerConfig:
    return PoolerConfig(task="embed", pooling_type="LAST")


def _asset_hash(asset: SemanticAsset, text: str) -> str:
    """Versiona el cache cuando cambia el activo o su texto indexable."""

    payload = f"resolver_asset_v1::{asset.asset_id}::{text}".encode("utf-8")
    return hashlib.sha1(payload).hexdigest()


def clear_embedding_runtime() -> None:
    """Libera el engine de embedding para poder ejecutar en modo secuencial."""

    shutdown_registered_llms(
        REGISTERED_LLMS,
        SHUTDOWN_LLM_IDS,
        warning_prefix="No se pudo apagar el engine de embeddings",
        logger=logger,
    )
    _build_runtime.cache_clear()
    get_embedding_runtime.cache_clear()


def _build_llm(
    model: str,
    *,
    dtype: str,
    max_model_len: int,
    gpu_memory_utilization: float,
    tensor_parallel_size: int,
    trust_remote_code: bool,
) -> Any:
    # Estos engines son efimeros y se cargan secuencialmente. Forzar eager evita
    # el overhead de `torch.compile`/cudagraphs que en vLLM 0.20 reduce el
    # budget efectivo de KV cache sobre 16 GiB.
    try:
        return build_local_llm(
            LLM,
            model=model,
            task="embed",
            dtype=dtype,
            max_model_len=max_model_len,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=trust_remote_code,
            enforce_eager=True,
        )
    except TypeError:
        return build_local_llm(
            LLM,
            model=model,
            runner="pooling",
            convert="embed",
            dtype=dtype,
            max_model_len=max_model_len,
            tensor_parallel_size=tensor_parallel_size,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=trust_remote_code,
            enforce_eager=True,
            pooler_config=_build_pooler_config(),
        )


@lru_cache(maxsize=4)
def _build_runtime(
    model: str,
    *,
    dtype: str,
    max_model_len: int,
    gpu_memory_utilization: float,
    tensor_parallel_size: int,
    trust_remote_code: bool,
) -> EmbeddingRuntime:
    llm = _build_llm(
        model,
        dtype=dtype,
        max_model_len=max_model_len,
        gpu_memory_utilization=gpu_memory_utilization,
        tensor_parallel_size=tensor_parallel_size,
        trust_remote_code=trust_remote_code,
    )
    REGISTERED_LLMS.append(llm)
    return EmbeddingRuntime(
        llm=llm,
        model=model,
        requested_max_model_len=max_model_len,
        effective_max_model_len=max_model_len,
    )


@lru_cache(maxsize=4)
def get_embedding_runtime(
    model: str,
    *,
    dtype: str,
    max_model_len: int,
    gpu_memory_utilization: float,
    tensor_parallel_size: int,
    trust_remote_code: bool,
) -> EmbeddingRuntime:
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
            return EmbeddingRuntime(
                llm=runtime.llm,
                model=runtime.model,
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
    raise RuntimeError("No se pudo inicializar el engine de embeddings.")


def embed_texts(llm: Any, texts: list[str]) -> np.ndarray:
    if not texts:
        return np.empty((0, 0), dtype=np.float32)

    outputs = llm.embed(texts)
    vectors = np.vstack([np.asarray(output.outputs.embedding, dtype=np.float32) for output in outputs])
    # Se normaliza una sola vez aqui para que coseno sea simplemente producto punto
    # y no recalcular normas en cada consulta posterior.
    return normalize_matrix(vectors)


def embed_query(llm: Any, query: str, instruction: str, query_template: str) -> np.ndarray:
    rendered_query = format_query_for_embedding(query, instruction, query_template)
    return embed_texts(llm, [rendered_query])[0]


def embed_assets(llm: Any, texts: list[str]) -> np.ndarray:
    return embed_texts(llm, texts)


def embed_assets_cached(
    llm: Any,
    assets: list[SemanticAsset],
    texts: list[str],
    *,
    model: str,
    cache_dir: Path,
    enable_cache: bool,
) -> tuple[np.ndarray, EmbeddingCacheStats]:
    """Embebe activos semanticos reusando embeddings persistidos en disco."""

    if len(assets) != len(texts):
        raise ValueError("embed_assets_cached requiere la misma cantidad de activos y textos")
    return embed_with_cache(
        model=model,
        cache_dir=cache_dir,
        enable_cache=enable_cache,
        texts=texts,
        item_hashes=[_asset_hash(asset, text) for asset, text in zip(assets, texts)],
        embed_fn=lambda missing_texts: embed_assets(llm, missing_texts),
    )


def retrieve_top_k(
    query_embedding: np.ndarray,
    asset_embeddings: np.ndarray,
    assets: list[SemanticAsset],
    top_k: int,
    min_score: float,
) -> list[tuple[SemanticAsset, float]]:
    if asset_embeddings.size == 0 or not assets:
        return []

    scores = asset_embeddings @ query_embedding
    order = np.argsort(-scores)[:top_k]
    # El threshold de embedding es un filtro blando: reduce ruido antes del reranker
    # sin perder el recall amplio que se busca en la primera pasada.
    return [(assets[index], float(scores[index])) for index in order if float(scores[index]) >= min_score]


atexit.register(clear_embedding_runtime)
