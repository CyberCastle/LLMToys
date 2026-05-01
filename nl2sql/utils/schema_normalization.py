#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Normalización compartida de variantes de schema usadas por NL2SQL."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def extract_tables_root(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    """Acepta roots `pruned_schema`, `tables` o mapping directo de tablas."""

    raw_pruned_schema = payload.get("pruned_schema")
    if isinstance(raw_pruned_schema, Mapping):
        return raw_pruned_schema
    raw_tables = payload.get("tables")
    if isinstance(raw_tables, Mapping):
        return raw_tables
    return payload


def normalize_columns(raw_columns: object) -> dict[str, dict[str, Any]]:
    """Normaliza columnas como mapping `{nombre: metadata}`."""

    if isinstance(raw_columns, Mapping):
        return {str(name): dict(meta) if isinstance(meta, Mapping) else {} for name, meta in raw_columns.items()}
    if not isinstance(raw_columns, list):
        return {}

    normalized: dict[str, dict[str, Any]] = {}
    for raw_column in raw_columns:
        if isinstance(raw_column, str):
            normalized[raw_column] = {}
            continue
        if isinstance(raw_column, Mapping):
            column_name = raw_column.get("name")
            if column_name:
                normalized[str(column_name)] = {str(key): value for key, value in raw_column.items() if key != "name"}
            continue
        if isinstance(raw_column, (list, tuple)) and raw_column:
            column_name = str(raw_column[0])
            column_type = raw_column[1] if len(raw_column) > 1 else ""
            normalized[column_name] = {"type": str(column_type or "")}
    return normalized


def normalize_column_types(table_body: Mapping[str, Any]) -> dict[str, str]:
    """Extrae tipos desde `column_types` explicito o metadata de columnas."""

    column_types: dict[str, str] = {}
    explicit = table_body.get("column_types")
    if isinstance(explicit, Mapping):
        column_types.update({str(name): str(type_name or "") for name, type_name in explicit.items()})

    raw_columns = table_body.get("columns", {})
    if isinstance(raw_columns, Mapping):
        for raw_column_name, raw_column_meta in raw_columns.items():
            if str(raw_column_name) in column_types:
                continue
            column_types[str(raw_column_name)] = _normalize_column_type(raw_column_meta)
        return column_types

    if isinstance(raw_columns, list):
        for raw_column in raw_columns:
            if isinstance(raw_column, str):
                column_types.setdefault(raw_column, "")
            elif isinstance(raw_column, Mapping) and raw_column.get("name"):
                column_name = str(raw_column["name"])
                column_types.setdefault(column_name, str(raw_column.get("type", "") or ""))
            elif isinstance(raw_column, (list, tuple)) and raw_column:
                column_name = str(raw_column[0])
                column_type = raw_column[1] if len(raw_column) > 1 else ""
                column_types.setdefault(column_name, str(column_type or ""))
    return column_types


def normalize_column_descriptions(raw_columns: object) -> dict[str, str]:
    """Extrae descripciones de columnas cuando el schema las provee."""

    descriptions: dict[str, str] = {}
    if isinstance(raw_columns, Mapping):
        for raw_column_name, raw_column_meta in raw_columns.items():
            if isinstance(raw_column_meta, Mapping):
                description = raw_column_meta.get("description")
                if isinstance(description, str) and description.strip():
                    descriptions[str(raw_column_name)] = description.strip()
        return descriptions
    if not isinstance(raw_columns, list):
        return descriptions
    for raw_column in raw_columns:
        if isinstance(raw_column, Mapping) and raw_column.get("name") and raw_column.get("description"):
            descriptions[str(raw_column["name"])] = str(raw_column["description"]).strip()
        elif isinstance(raw_column, (list, tuple)) and len(raw_column) > 2 and str(raw_column[2]).strip():
            descriptions[str(raw_column[0])] = str(raw_column[2]).strip()
    return descriptions


def normalize_foreign_keys(raw_foreign_keys: object, *, source_key: str = "column") -> list[dict[str, str]]:
    """Normaliza FKs tolerando aliases historicos de columnas/targets."""

    if not isinstance(raw_foreign_keys, list):
        return []

    normalized: list[dict[str, str]] = []
    for raw_foreign_key in raw_foreign_keys:
        if not isinstance(raw_foreign_key, Mapping):
            continue
        column = raw_foreign_key.get("col") or raw_foreign_key.get("column") or raw_foreign_key.get("source_column")
        ref_table = raw_foreign_key.get("ref_table") or raw_foreign_key.get("target_table")
        ref_col = raw_foreign_key.get("ref_col") or raw_foreign_key.get("target_column")
        if column and ref_table and ref_col:
            normalized.append({source_key: str(column), "ref_table": str(ref_table), "ref_col": str(ref_col)})
    return normalized


def _normalize_column_type(raw_column_meta: object) -> str:
    if isinstance(raw_column_meta, Mapping):
        return str(raw_column_meta.get("type", raw_column_meta.get("dtype", "")) or "")
    return str(raw_column_meta or "")
