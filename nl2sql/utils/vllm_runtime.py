#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Helpers compartidos para crear y liberar runtimes locales de vLLM."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from llm_core.vllm_runtime_utils import release_cuda_memory, shutdown_vllm_engine_once


def build_local_llm(llm_factory: Callable[..., Any], **kwargs: Any) -> Any:
    """Instancia un `vllm.LLM` o fake equivalente con kwargs ya normalizados."""

    return llm_factory(**kwargs)


def shutdown_registered_llms(
    registered_llms: list[object],
    shutdown_llm_ids: set[int],
    *,
    warning_prefix: str,
    logger: Any,
) -> None:
    """Apaga engines registrados de forma idempotente y limpia la lista."""

    for llm in reversed(registered_llms):
        try:
            shutdown_vllm_engine_once(llm, shutdown_llm_ids)
        except Exception as exc:
            logger.warning("%s: %s", warning_prefix, exc)
    registered_llms.clear()


def release_local_llm(llm: object | None = None) -> None:
    """Libera una referencia LLM puntual y vacia memoria CUDA si existe."""

    if llm is not None:
        del llm
    release_cuda_memory()
