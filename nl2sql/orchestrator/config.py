#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Configuracion general del orquestador NL2SQL."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

from nl2sql.config import (
    NL2SQLRuntimeBundle,
    NarrativePromptRules,
    env_bool,
    env_int,
    load_nl2sql_runtime_bundle,
    load_orchestrator_settings,
    resolve_nl2sql_config_path,
    resolve_semantic_rules_path,
)

from .llm_manager import NarrativeLLMSettings


def resolve_narrative_prompt_path() -> str:
    """Resuelve la ruta del YAML unificado para el prompt narrativo."""

    return str(resolve_nl2sql_config_path())


@dataclass(frozen=True)
class NL2SQLConfig:
    """Configuración operativa del orquestador NL2SQL."""

    runtime_bundle: NL2SQLRuntimeBundle | None = None
    narrative_prompt: NarrativePromptRules | None = None
    narrative_prompt_path: str = field(default_factory=resolve_narrative_prompt_path)
    max_rows: int = field(default_factory=lambda: env_int("NL2SQL_MAX_ROWS", 1000))
    rows_preview_limit: int = field(default_factory=lambda: env_int("NL2SQL_ROWS_PREVIEW_LIMIT", 25))
    execution_sql_optimization_enabled: bool = field(default_factory=lambda: env_bool("NL2SQL_EXECUTION_SQL_OPTIMIZATION_ENABLED", False))
    narrative: NarrativeLLMSettings = field(default_factory=NarrativeLLMSettings)

    def __post_init__(self) -> None:
        if self.narrative_prompt is not None:
            return
        if self.runtime_bundle is not None:
            object.__setattr__(self, "narrative_prompt", self.runtime_bundle.settings.orchestrator.narrative_prompt)
            return
        settings = load_orchestrator_settings(self.narrative_prompt_path)
        object.__setattr__(self, "narrative_prompt", settings.narrative_prompt)


def ensure_runtime_bundle_loaded(
    config: NL2SQLConfig | None = None,
    *,
    semantic_rules_path: str | Path | None = None,
) -> NL2SQLConfig:
    """Devuelve una configuracion con `runtime_bundle` precargado y validado."""

    effective_config = config or NL2SQLConfig()
    expected_semantic_rules_path = (
        Path(semantic_rules_path).expanduser().resolve() if semantic_rules_path is not None else resolve_semantic_rules_path().resolve()
    )

    if effective_config.runtime_bundle is not None and effective_config.runtime_bundle.semantic_rules_path == expected_semantic_rules_path:
        return effective_config

    runtime_bundle = load_nl2sql_runtime_bundle(
        effective_config.narrative_prompt_path,
        expected_semantic_rules_path,
    )
    return replace(
        effective_config,
        runtime_bundle=runtime_bundle,
        narrative_prompt=runtime_bundle.settings.orchestrator.narrative_prompt,
    )
