#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Etapa LangChain que envuelve el semantic pruning basado en E2Rank."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Mapping

from etl.inspect_db import get_db_schema, save_schema_to_yaml_file
from langchain_core.runnables import RunnableLambda

from nl2sql.semantic_prune import SemanticSchemaPruningConfig, clear_e2rank_runtime, persist_pruned_schema, run_semantic_schema_pruning
from nl2sql.utils.schema_normalization import (
    normalize_column_descriptions,
    normalize_column_types,
    normalize_columns,
    normalize_foreign_keys,
)
from nl2sql.utils.yaml_utils import load_yaml_mapping

from ..config import NL2SQLConfig, ensure_runtime_bundle_loaded
from ..contracts import StageArtifact


def _normalize_runtime_schema(raw_schema: object) -> dict[str, object]:
    """Adapta el YAML de esquema al formato operativo que espera `semantic_prune`."""

    if not isinstance(raw_schema, Mapping):
        raise ValueError("db_schema.yaml invalido: la raiz debe ser un mapping de tablas.")

    normalized_schema: dict[str, object] = {}
    for raw_table_name, raw_table_info in raw_schema.items():
        if not isinstance(raw_table_name, str) or not isinstance(raw_table_info, Mapping):
            continue

        raw_columns = raw_table_info.get("columns", [])
        column_types = normalize_column_types(raw_table_info)
        columns = [(column_name, column_types.get(column_name, "")) for column_name in normalize_columns(raw_columns)]
        column_descriptions = normalize_column_descriptions(raw_columns)

        primary_keys = [str(column_name) for column_name in raw_table_info.get("primary_keys", []) or []]

        foreign_keys = normalize_foreign_keys(raw_table_info.get("foreign_keys", []), source_key="col")

        normalized_schema[raw_table_name] = {
            "description": str(raw_table_info.get("description", "") or ""),
            "columns": columns,
            "column_descriptions": column_descriptions,
            "primary_keys": primary_keys,
            "foreign_keys": foreign_keys,
        }

    return normalized_schema


def _load_runtime_schema(schema_path: str | Path) -> tuple[dict[str, object], str]:
    """Carga el esquema desde YAML o lo refleja desde la BD si aun no existe."""

    resolved_path = Path(schema_path)
    if resolved_path.exists():
        raw_schema = load_yaml_mapping(resolved_path, artifact_name=str(resolved_path))
        return _normalize_runtime_schema(raw_schema), "yaml"

    reflected_schema = get_db_schema()
    save_schema_to_yaml_file(reflected_schema, str(resolved_path))
    return reflected_schema, "database_reflection"


def build_prune_runnable(config: NL2SQLConfig | None = None) -> RunnableLambda:
    """Construye la etapa de pruning como `RunnableLambda` de LangChain."""

    effective_config = ensure_runtime_bundle_loaded(config)
    runtime_bundle = effective_config.runtime_bundle
    if runtime_bundle is None:
        raise ValueError("NL2SQLConfig.runtime_bundle debe estar precargado antes de construir la etapa de prune.")

    def _run(state: dict[str, Any]) -> dict[str, Any]:
        request = state["request"]
        schema, schema_source = _load_runtime_schema(request.db_schema_path)
        config = SemanticSchemaPruningConfig(
            query=request.query,
            settings=runtime_bundle.settings.semantic_prune,
            semantic_rules_path=str(runtime_bundle.semantic_rules_path),
        )
        started = time.perf_counter()
        try:
            result = run_semantic_schema_pruning(config, schema=schema)
            output_path = persist_pruned_schema(
                result,
                query=request.query,
                out_path=Path(request.out_dir) / "semantic_pruned_schema.yaml",
            )
        finally:
            clear_e2rank_runtime()

        payload = load_yaml_mapping(output_path, artifact_name=str(output_path))
        payload["schema_source"] = schema_source
        state["pruned_schema_path"] = output_path
        state.setdefault("artifacts", []).append(
            StageArtifact(
                name="prune",
                path=output_path,
                payload=payload,
                duration_seconds=time.perf_counter() - started,
            )
        )
        return state

    return RunnableLambda(_run, name="semantic_prune_stage")
