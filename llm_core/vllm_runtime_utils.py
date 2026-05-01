#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Utilidades compartidas para runtimes vLLM."""

from __future__ import annotations

import gc
import logging
import re

import torch

AUTO_FALLBACK_MODEL_LEN_STEP = 256
GPU_MEMORY_UTILIZATION_SAFETY_MARGIN_GIB = 0.25
KV_CACHE_MAX_MODEL_LEN_RE = re.compile(r"estimated maximum model length is\s+(\d+)", re.IGNORECASE)
logger = logging.getLogger(__name__)


def release_cuda_memory() -> None:
    """Ejecuta GC y vacia la cache CUDA si PyTorch puede verla."""

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def clamp_gpu_memory_utilization(value: float) -> float:
    """Acota el budget de VRAM a un rango valido para vLLM."""

    return min(max(value, 0.01), 0.99)


def resolve_gpu_utilization_from_available_memory(
    requested_gpu_memory_utilization: float,
    *,
    free_memory_gib: float,
    total_memory_gib: float,
    safety_margin_gib: float = GPU_MEMORY_UTILIZATION_SAFETY_MARGIN_GIB,
) -> float:
    """Reduce `gpu_memory_utilization` cuando la VRAM libre real no alcanza.

    vLLM valida este budget al inicio contra la memoria libre observada. En una
    GPU compartida o con memoria reservada por otros procesos, un valor fijo de
    `0.90` puede fallar aunque el modelo siga siendo viable con un budget menor.
    """

    if total_memory_gib <= 0.0 or free_memory_gib <= 0.0:
        return requested_gpu_memory_utilization

    usable_free_gib = max(free_memory_gib - safety_margin_gib, 0.01)
    allowed_gpu_memory_utilization = clamp_gpu_memory_utilization(usable_free_gib / total_memory_gib)
    if allowed_gpu_memory_utilization >= requested_gpu_memory_utilization:
        return requested_gpu_memory_utilization
    return allowed_gpu_memory_utilization


def iter_exception_messages(error: BaseException) -> list[str]:
    """Recorre la cadena de excepciones y devuelve mensajes renderizados útiles."""

    messages: list[str] = []
    visited_errors: set[int] = set()
    current_error: BaseException | None = error

    while current_error is not None and id(current_error) not in visited_errors:
        visited_errors.add(id(current_error))
        rendered_message = str(current_error).strip()
        if rendered_message:
            messages.append(rendered_message)
        current_error = current_error.__cause__ or current_error.__context__

    return messages


def resolve_fallback_max_model_len(requested_max_model_len: int, error: BaseException) -> int | None:
    """Recorta el contexto cuando vLLM informa un techo real de KV cache menor."""

    estimated_max_model_len: int | None = None
    for message in iter_exception_messages(error):
        match = KV_CACHE_MAX_MODEL_LEN_RE.search(message)
        if match is None:
            continue
        estimated_max_model_len = int(match.group(1))
        break

    if estimated_max_model_len is None or estimated_max_model_len >= requested_max_model_len:
        return None

    rounded_max_model_len = (estimated_max_model_len // AUTO_FALLBACK_MODEL_LEN_STEP) * AUTO_FALLBACK_MODEL_LEN_STEP
    if rounded_max_model_len <= 0:
        return estimated_max_model_len
    if rounded_max_model_len >= requested_max_model_len:
        return estimated_max_model_len
    return rounded_max_model_len


def should_try_stepdown_fallback(error: BaseException) -> bool:
    """Determina si conviene reintentar con menos contexto tras un error de engine."""

    combined_messages = "\n".join(iter_exception_messages(error)).lower()
    return "engine core initialization failed" in combined_messages or "kv cache" in combined_messages


def shutdown_vllm_engine_once(llm: object, shutdown_ids: set[int]) -> None:
    """Invoca `engine_core.shutdown()` una sola vez por instancia de LLM."""

    if id(llm) in shutdown_ids:
        return

    shutdown_ids.add(id(llm))
    shutdown = getattr(getattr(getattr(llm, "llm_engine", None), "engine_core", None), "shutdown", None)
    if callable(shutdown):
        shutdown()


def destroy_distributed_process_group() -> None:
    """Destruye el process group global solo si PyTorch lo inicializo."""

    if not torch.distributed.is_available():
        return
    if torch.distributed.is_initialized():
        logger.info("Destruyendo process group distribuido de PyTorch.")
        torch.distributed.destroy_process_group()
