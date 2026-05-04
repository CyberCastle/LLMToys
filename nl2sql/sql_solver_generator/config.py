#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from nl2sql.config import (
    SolverFilterValueRules,
    SqlSolverPromptRules,
    SqlSolverRuntimeTuning,
    SqlSolverSettings,
    SQLGenerationTuningRules,
    env_bool,
    env_float,
    env_int,
    env_str,
    load_sql_solver_settings,
    resolve_nl2sql_config_path,
)

DEFAULT_MODEL = "XGenerationLab/XiYanSQL-QwenCoder-7B-2504"

SolverDialectName = Literal["tsql", "postgres"]


def resolve_settings_path() -> Path:
    """Resuelve la ruta del YAML unificado para la configuracion del solver."""

    return resolve_nl2sql_config_path()


def resolve_filter_value_rules_path() -> Path:
    """Resuelve el YAML de reglas lexicas para valores de filtros semanticos."""

    return resolve_settings_path()


@dataclass(frozen=True)
class SolverConfig:
    settings: SqlSolverSettings | None = None
    prompts: SqlSolverPromptRules | None = None
    filter_value_rules: SolverFilterValueRules | None = None
    generation_tuning: SQLGenerationTuningRules | None = None
    runtime_tuning: SqlSolverRuntimeTuning | None = None
    prompts_path: Path = field(default_factory=resolve_settings_path)
    filter_value_rules_path: Path = field(default_factory=resolve_filter_value_rules_path)

    dialect: SolverDialectName = field(default_factory=lambda: env_str("SQL_DIALECT", "tsql"))  # type: ignore
    model: str = field(default_factory=lambda: env_str("SQL_SOLVER_MODEL", DEFAULT_MODEL))
    max_retries: int | None = None

    llm_dtype: Literal["auto", "bfloat16", "float16", "float32"] | None = None
    max_model_len: int | None = None
    max_tokens: int | None = None
    temperature: float | None = None
    gpu_memory_utilization: float | None = None
    enforce_eager: bool | None = None
    cpu_offload_gb: float | None = None
    swap_space_gb: float | None = None
    fail_on_validation_error: bool | None = None

    def __post_init__(self) -> None:
        active_settings = self.settings or load_sql_solver_settings(self.prompts_path)
        if self.prompts is None:
            object.__setattr__(self, "prompts", active_settings.prompts)
        if self.filter_value_rules is None:
            object.__setattr__(self, "filter_value_rules", active_settings.filter_value_rules)
        if self.generation_tuning is None:
            object.__setattr__(self, "generation_tuning", active_settings.generation_tuning)
        active_runtime_tuning = self.runtime_tuning or active_settings.runtime_tuning
        if self.runtime_tuning is None:
            object.__setattr__(self, "runtime_tuning", active_runtime_tuning)
        runtime_env_resolvers = {
            "max_retries": lambda default: env_int("SQL_SOLVER_MAX_RETRIES", default),
            "llm_dtype": lambda default: env_str("SQL_SOLVER_LLM_DTYPE", default).lower(),
            "max_model_len": lambda default: env_int("SQL_SOLVER_MAX_MODEL_LEN", default),
            "max_tokens": lambda default: env_int("SQL_SOLVER_MAX_TOKENS", default),
            "temperature": lambda default: env_float("SQL_SOLVER_TEMPERATURE", default),
            "gpu_memory_utilization": lambda default: env_float("SQL_SOLVER_GPU_MEMORY_UTILIZATION", default),
            "enforce_eager": lambda default: env_bool("SQL_SOLVER_ENFORCE_EAGER", default),
            "cpu_offload_gb": lambda default: env_float("SQL_SOLVER_CPU_OFFLOAD_GB", default),
            "swap_space_gb": lambda default: env_float("SQL_SOLVER_SWAP_SPACE_GB", default),
            "fail_on_validation_error": lambda default: env_bool("SQL_SOLVER_FAIL_ON_VALIDATION_ERROR", default),
        }
        for field_name, resolve_from_env in runtime_env_resolvers.items():
            if getattr(self, field_name) is None:
                object.__setattr__(self, field_name, resolve_from_env(getattr(active_runtime_tuning, field_name)))
