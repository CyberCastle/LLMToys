#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from nl2sql.utils.semantic_contract import SemanticContract, load_semantic_contract
from nl2sql.utils.schema_normalization import (
    extract_tables_root,
    normalize_column_descriptions,
    normalize_column_types,
    normalize_columns,
    normalize_foreign_keys,
)
from nl2sql.utils.yaml_utils import load_yaml_mapping


def load_pruned_schema(source: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    raw = _read(source)
    tables = extract_tables_root(raw)

    normalized: dict[str, Any] = {}
    for table_name, body in tables.items():
        if not isinstance(body, Mapping):
            continue
        normalized[str(table_name)] = {
            "columns": normalize_columns(body.get("columns", {})),
            "column_types": normalize_column_types(body),
            "column_descriptions": normalize_column_descriptions(body.get("columns", {})),
            "primary_keys": list(body.get("primary_keys", []) or []),
            "foreign_keys": normalize_foreign_keys(body.get("foreign_keys", [])),
            "description": body.get("description", ""),
            "role": body.get("role"),
        }
    return normalized


def load_semantic_rules(source: str | Path | Mapping[str, Any] | SemanticContract) -> SemanticContract:
    if isinstance(source, SemanticContract):
        return source
    return load_semantic_contract(source)


def _read(source: str | Path | Mapping[str, Any]) -> Mapping[str, Any]:
    if isinstance(source, Mapping):
        return source
    return load_yaml_mapping(Path(source), artifact_name=str(source))
