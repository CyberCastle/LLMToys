#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Runtime local, preflight y sampling del solver SQL."""

from __future__ import annotations

from functools import lru_cache
import logging
from pathlib import Path
import re
import time
from typing import Any

import torch

from llm_core.tokenizer_utils import load_tokenizer as load_solver_tokenizer
from llm_core.vllm_runtime_utils import (
    clamp_gpu_memory_utilization,
    resolve_gpu_utilization_from_available_memory,
)
from nl2sql.config import (
    SQLGenerationTuningRules,
    env_float,
    load_sql_solver_generation_tuning_rules,
    resolve_nl2sql_config_path,
)
from nl2sql.utils.prompt_budget import (
    PromptTooLongError as SharedPromptTooLongError,
    assert_prompt_fits,
)

SOLVER_STARTUP_MEMORY_RE = re.compile(
    r"Free memory on device cuda:\d+ \((?P<free>[0-9.]+)/(?P<total>[0-9.]+) GiB\).*desired GPU memory utilization \((?P<requested>[0-9.]+), (?P<desired>[0-9.]+) GiB\)",
    re.IGNORECASE,
)
logger = logging.getLogger(__name__)


@lru_cache(maxsize=8)
def load_generation_tuning_rules(
    path: str | Path | None = None,
) -> SQLGenerationTuningRules:
    """Devuelve el tuning tipado del solver ya validado en `nl2sql.config`."""

    resolved_path = Path(path).expanduser().resolve() if path is not None else resolve_nl2sql_config_path()
    return load_sql_solver_generation_tuning_rules(resolved_path)


class PromptTooLongError(SharedPromptTooLongError):
    """Error de preflight cuando prompt + salida reservada exceden el contexto."""


def resolve_min_cpu_offload_gb() -> float:
    """Resuelve el piso de offload CPU efectivo del solver con override opcional."""

    tuning_rules = load_generation_tuning_rules()
    return env_float("SQL_SOLVER_MIN_CPU_OFFLOAD_GB", tuning_rules.min_cpu_offload_gb)


def require_solver_model_name(model_name: str) -> str:
    """Valida que el solver tenga un repo/model id utilizable antes de tocar HF/vLLM."""

    normalized = model_name.strip()
    if not normalized:
        raise ValueError("SQL_SOLVER_MODEL no puede estar vacio")
    return normalized


def resolve_initial_solver_runtime_settings(
    *,
    gpu_memory_utilization: float,
    enforce_eager: bool,
    cpu_offload_gb: float,
) -> tuple[float, bool, float]:
    """Ajusta el runtime inicial del solver segun la VRAM libre del momento."""

    min_cpu_offload_gb = resolve_min_cpu_offload_gb()
    effective_gpu_memory_utilization = gpu_memory_utilization
    if torch.cuda.is_available():
        free_memory_bytes, total_memory_bytes = torch.cuda.mem_get_info()
        effective_gpu_memory_utilization = resolve_gpu_utilization_from_available_memory(
            gpu_memory_utilization,
            free_memory_gib=free_memory_bytes / (1024**3),
            total_memory_gib=total_memory_bytes / (1024**3),
        )
        if effective_gpu_memory_utilization < gpu_memory_utilization:
            logger.warning(
                "sql_solver.runtime ajusta gpu_memory_utilization %.2f -> %.4f por VRAM libre actual",
                gpu_memory_utilization,
                effective_gpu_memory_utilization,
            )

    return (
        effective_gpu_memory_utilization,
        enforce_eager,
        max(cpu_offload_gb, min_cpu_offload_gb),
    )


def resolve_solver_runtime_retry(
    error: BaseException,
    *,
    current_gpu_memory_utilization: float,
    enforce_eager: bool,
    cpu_offload_gb: float,
) -> tuple[float, bool, float] | None:
    """Calcula un unico reintento cuando vLLM falla por budget de VRAM al iniciar."""

    min_cpu_offload_gb = resolve_min_cpu_offload_gb()
    match = SOLVER_STARTUP_MEMORY_RE.search(str(error))
    if match is None:
        return None

    retry_gpu_memory_utilization = resolve_gpu_utilization_from_available_memory(
        current_gpu_memory_utilization,
        free_memory_gib=float(match.group("free")),
        total_memory_gib=float(match.group("total")),
    )
    if retry_gpu_memory_utilization >= current_gpu_memory_utilization:
        retry_gpu_memory_utilization = clamp_gpu_memory_utilization(current_gpu_memory_utilization - 0.03)
    if retry_gpu_memory_utilization >= current_gpu_memory_utilization:
        return None

    return (
        retry_gpu_memory_utilization,
        True,
        max(cpu_offload_gb, min_cpu_offload_gb),
    )


def preflight_prompt_size(
    model_name: str,
    system_prompt: str,
    user_prompt: str,
    max_model_len: int,
    max_tokens: int,
) -> int:
    """Verifica que el prompt del solver quepa antes de invocar vLLM."""

    tuning_rules = load_generation_tuning_rules()
    tokenizer = load_solver_tokenizer(model_name)
    try:
        return assert_prompt_fits(
            tokenizer=tokenizer,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_model_len=max_model_len,
            max_tokens=max_tokens,
            safety_margin=tuning_rules.prompt_token_safety_margin,
            overflow_message=(
                "prompt_tokens={prompt_tokens} + max_tokens="
                f"{max_tokens} + safety={tuning_rules.prompt_token_safety_margin} > max_model_len={max_model_len}; "
                "reduce el payload o sube el contexto."
            ),
            error_message="El tokenizer del solver no soporta apply_chat_template",
        )
    except SharedPromptTooLongError as exc:
        raise PromptTooLongError(str(exc)) from exc


def build_sampling_kwargs(*, tokenizer: Any, temperature: float, max_tokens: int) -> dict[str, Any]:
    """Construye SamplingParams kwargs con stops declarativos del solver."""

    tuning_rules = load_generation_tuning_rules()
    stop_token_ids: list[int] | None = None
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if isinstance(eos_token_id, int):
        stop_token_ids = [eos_token_id]
    return {
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stop": [tuning_rules.end_of_output_marker],
        "stop_token_ids": stop_token_ids,
        "skip_special_tokens": True,
    }


def run_local_llm_chat(
    *,
    llm: Any,
    system_prompt: str,
    user_prompt: str,
    sampling: Any,
) -> tuple[str, dict[str, Any]]:
    """Ejecuta chat local y devuelve texto mas diagnosticos minimos."""

    started = time.perf_counter()
    outputs = llm.chat(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        sampling_params=sampling,
    )
    completion = outputs[0].outputs[0]
    return completion.text.strip(), {
        "finish_reason": str(getattr(completion, "finish_reason", "")),
        "generated_tokens": len(getattr(completion, "token_ids", []) or []),
        "wall_time_seconds": time.perf_counter() - started,
    }
