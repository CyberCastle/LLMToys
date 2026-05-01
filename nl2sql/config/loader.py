#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Helpers centralizados para cargar la configuracion YAML interna de NL2SQL."""

from __future__ import annotations

from functools import lru_cache
import os
from pathlib import Path
from typing import Any, Mapping

from pydantic import ValidationError

from .models import (
    CompilerRules,
    NL2SQLRuntimeBundle,
    NL2SQLSettings,
    NL2SQLSettingsConfig,
    OrchestratorSettings,
    QueryFormRule,
    SemanticPruneSettings,
    SemanticResolverSettings,
    SemanticResolverVerificationRules,
    SolverFilterValueRules,
    SqlSolverPromptRules,
    SqlSolverSettings,
    SQLGenerationTuningRules,
)
from nl2sql.utils.semantic_contract import SemanticContract, load_semantic_contract
from nl2sql.utils.yaml_utils import load_yaml_mapping

NL2SQL_CONFIG_PATH_ENV_VAR = "NL2SQL_CONFIG_PATH"
DEFAULT_NL2SQL_CONFIG_PATH = Path(__file__).resolve().parent / "settings.yaml"
SEMANTIC_RULES_PATH_ENV_VAR = "SEMANTIC_RULES_PATH"
DEFAULT_SEMANTIC_RULES_PATH = Path("schema-docs/semantic_rules.yaml")


def resolve_nl2sql_config_path() -> Path:
    """Resuelve la ruta del YAML unificado de configuracion de NL2SQL.

    `NL2SQL_CONFIG_PATH` es la unica variable soportada para redirigir el
    archivo de configuracion unificado.
    """

    canonical_path = os.getenv(NL2SQL_CONFIG_PATH_ENV_VAR)
    if canonical_path is not None and canonical_path.strip():
        return Path(canonical_path.strip()).expanduser()
    return DEFAULT_NL2SQL_CONFIG_PATH


def resolve_semantic_rules_path() -> Path:
    """Resuelve la ruta del contrato semantico compartido del pipeline."""

    raw_value = os.getenv(SEMANTIC_RULES_PATH_ENV_VAR)
    if raw_value is not None and raw_value.strip():
        return Path(raw_value.strip()).expanduser()
    return DEFAULT_SEMANTIC_RULES_PATH


@lru_cache(maxsize=8)
def _load_nl2sql_config_cached(path: str) -> dict[str, Any]:
    """Carga y cachea el YAML unificado de configuracion de NL2SQL."""

    resolved_path = Path(path).expanduser().resolve()
    payload = load_yaml_mapping(resolved_path, artifact_name=str(resolved_path))
    return dict(payload)


def load_nl2sql_config(path: str | Path) -> dict[str, Any]:
    """Devuelve el documento YAML unificado de configuracion como mapping plano."""

    resolved_path = Path(path).expanduser().resolve()
    return dict(_load_nl2sql_config_cached(str(resolved_path)))


def _merge_config_mappings(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Fusiona un override parcial sobre el documento canonico por defecto."""

    merged: dict[str, Any] = dict(base)
    for key, override_value in override.items():
        base_value = merged.get(key)
        if isinstance(base_value, Mapping) and isinstance(override_value, Mapping):
            merged[key] = _merge_config_mappings(base_value, override_value)
            continue
        merged[key] = override_value
    return merged


def _load_effective_nl2sql_config(path: Path) -> dict[str, Any]:
    """Carga el payload efectivo, permitiendo overlays parciales sobre settings.yaml."""

    resolved_path = path.expanduser().resolve()
    payload = load_nl2sql_config(resolved_path)
    default_path = DEFAULT_NL2SQL_CONFIG_PATH.resolve()
    if resolved_path == default_path:
        return payload
    return _merge_config_mappings(load_nl2sql_config(default_path), payload)


def _validate_nl2sql_settings(path: Path) -> NL2SQLSettings:
    payload = _load_effective_nl2sql_config(path)
    try:
        raw_settings = NL2SQLSettingsConfig.model_validate(payload)
    except ValidationError as exc:
        raise ValueError(f"settings.yaml invalido en {path}: {exc}") from exc
    return raw_settings.to_runtime()


@lru_cache(maxsize=8)
def _load_nl2sql_settings_cached(path: str) -> NL2SQLSettings:
    resolved_path = Path(path).expanduser().resolve()
    return _validate_nl2sql_settings(resolved_path)


def load_nl2sql_settings(path: str | Path | None = None) -> NL2SQLSettings:
    """Carga y valida `settings.yaml` devolviendo modelos listos para usar."""

    resolved_path = Path(path).expanduser().resolve() if path is not None else resolve_nl2sql_config_path().resolve()
    return _load_nl2sql_settings_cached(str(resolved_path))


def _build_query_forms(contract: SemanticContract) -> tuple[QueryFormRule, ...]:
    query_forms: list[QueryFormRule] = []
    for raw_row in contract.retrieval_heuristics.query_forms:
        query_forms.append(QueryFormRule.model_validate(raw_row))
    return tuple(query_forms)


@lru_cache(maxsize=8)
def _load_nl2sql_runtime_bundle_cached(settings_path: str, semantic_rules_path: str) -> NL2SQLRuntimeBundle:
    resolved_settings_path = Path(settings_path).expanduser().resolve()
    resolved_semantic_rules_path = Path(semantic_rules_path).expanduser().resolve()

    try:
        semantic_contract = load_semantic_contract(resolved_semantic_rules_path)
    except ValidationError as exc:
        raise ValueError(f"semantic_rules.yaml invalido en {resolved_semantic_rules_path}: {exc}") from exc
    except ValueError as exc:
        raise ValueError(f"semantic_rules.yaml invalido en {resolved_semantic_rules_path}: {exc}") from exc

    payload = _load_effective_nl2sql_config(resolved_settings_path)
    try:
        raw_settings = NL2SQLSettingsConfig.model_validate(payload)
    except ValidationError as exc:
        raise ValueError(f"settings.yaml invalido en {resolved_settings_path}: {exc}") from exc

    settings = raw_settings.to_runtime(query_forms=_build_query_forms(semantic_contract))
    return NL2SQLRuntimeBundle(
        settings_path=resolved_settings_path,
        semantic_rules_path=resolved_semantic_rules_path,
        settings=settings,
        semantic_contract=semantic_contract,
    )


def load_nl2sql_runtime_bundle(
    config_path: str | Path | None = None,
    semantic_rules_path: str | Path | None = None,
) -> NL2SQLRuntimeBundle:
    """Carga y valida `settings.yaml` y `semantic_rules.yaml` en una sola pasada."""

    resolved_settings_path = Path(config_path).expanduser().resolve() if config_path is not None else resolve_nl2sql_config_path().resolve()
    resolved_semantic_rules_path = (
        Path(semantic_rules_path).expanduser().resolve() if semantic_rules_path is not None else resolve_semantic_rules_path().resolve()
    )
    return _load_nl2sql_runtime_bundle_cached(str(resolved_settings_path), str(resolved_semantic_rules_path))


def load_semantic_prune_settings(path: str | Path | None = None) -> SemanticPruneSettings:
    """Devuelve la seccion tipada de semantic prune."""

    return load_nl2sql_settings(path).semantic_prune


def load_semantic_resolver_settings(
    path: str | Path | None = None,
    semantic_rules_path: str | Path | None = None,
) -> SemanticResolverSettings:
    """Devuelve la seccion tipada del semantic resolver."""

    if semantic_rules_path is None:
        return load_nl2sql_settings(path).semantic_resolver
    return load_nl2sql_runtime_bundle(path, semantic_rules_path).settings.semantic_resolver


def load_sql_solver_settings(path: str | Path | None = None) -> SqlSolverSettings:
    """Devuelve la seccion tipada del solver SQL."""

    return load_nl2sql_settings(path).sql_solver


def load_orchestrator_settings(path: str | Path | None = None) -> OrchestratorSettings:
    """Devuelve la seccion tipada del orquestador."""

    return load_nl2sql_settings(path).orchestrator


def load_semantic_resolver_compiler_rules(
    path: str | Path | None = None,
    semantic_rules_path: str | Path | None = None,
) -> CompilerRules:
    """Devuelve las reglas tipadas del compilador semantico."""

    return load_semantic_resolver_settings(path, semantic_rules_path).compiler_rules


def load_semantic_resolver_verification_rules(path: str | Path | None = None) -> SemanticResolverVerificationRules:
    """Devuelve las reglas tipadas del verificador semantico."""

    return load_semantic_resolver_settings(path).verification


def load_sql_solver_prompt_rules(path: str | Path | None = None) -> SqlSolverPromptRules:
    """Devuelve los prompts tipados del solver SQL."""

    return load_sql_solver_settings(path).prompts


def load_sql_solver_filter_value_rules(path: str | Path | None = None) -> SolverFilterValueRules:
    """Devuelve las reglas lexicas tipadas del solver SQL."""

    return load_sql_solver_settings(path).filter_value_rules


def load_sql_solver_generation_tuning_rules(path: str | Path | None = None) -> SQLGenerationTuningRules:
    """Devuelve el tuning tipado del solver SQL."""

    return load_sql_solver_settings(path).generation_tuning
