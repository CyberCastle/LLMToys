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
        for field_name in (
            "max_retries",
            "llm_dtype",
            "max_model_len",
            "max_tokens",
            "temperature",
            "gpu_memory_utilization",
            "enforce_eager",
            "cpu_offload_gb",
            "swap_space_gb",
            "fail_on_validation_error",
        ):
            if getattr(self, field_name) is None:
                object.__setattr__(self, field_name, getattr(active_runtime_tuning, field_name))
