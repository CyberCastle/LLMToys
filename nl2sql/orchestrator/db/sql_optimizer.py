#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Optimizacion SQL previa a ejecucion usando sqlglot y el esquema operativo."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sqlglot.optimizer import optimize as optimize_expression

from nl2sql.sql_solver_generator.dialects.base import SqlDialect
from nl2sql.sql_solver_generator.dialects.registry import get_dialect
from nl2sql.sql_solver_generator.sql_normalizer import render_sql_ast
from nl2sql.utils.schema_normalization import extract_tables_root, normalize_column_types
from nl2sql.utils.yaml_utils import load_yaml_value


@dataclass(frozen=True)
class SqlOptimizationResult:
    """Resultado de optimizar una sentencia junto con el resumen del esquema usado."""

    sql: str
    schema_tables: int
    schema_columns: int


def optimize_sql_for_execution(
    raw_sql: str,
    *,
    dialect: str | SqlDialect,
    schema_source: str | Path | Mapping[str, Any],
) -> SqlOptimizationResult:
    """Optimiza el SQL final con `sqlglot` usando el esquema disponible."""

    resolved_dialect = dialect if isinstance(dialect, SqlDialect) else get_dialect(dialect)
    sqlglot_schema = build_sqlglot_schema(schema_source)
    optimized_ast = optimize_expression(
        raw_sql,
        schema=sqlglot_schema,  # type: ignore
        dialect=resolved_dialect.sqlglot_dialect,
    )
    return SqlOptimizationResult(
        sql=render_sql_ast(optimized_ast, resolved_dialect),  # type: ignore
        schema_tables=len(sqlglot_schema),
        schema_columns=sum(len(columns) for columns in sqlglot_schema.values()),
    )


def build_sqlglot_schema(schema_source: str | Path | Mapping[str, Any]) -> dict[str, dict[str, str]]:
    """Convierte el YAML operativo del esquema al mapping esperado por `sqlglot`."""

    payload = _read_schema_source(schema_source)
    tables = extract_tables_root(payload)

    sqlglot_schema: dict[str, dict[str, str]] = {}
    for raw_table_name, raw_table_body in tables.items():
        if not isinstance(raw_table_name, str) or not isinstance(raw_table_body, Mapping):
            continue
        sqlglot_schema[raw_table_name] = normalize_column_types(raw_table_body)
    return sqlglot_schema


def _read_schema_source(schema_source: str | Path | Mapping[str, Any]) -> Mapping[str, Any]:
    """Lee el origen de esquema aceptando tanto rutas YAML como mappings ya cargados."""

    if isinstance(schema_source, Mapping):
        return schema_source

    payload = load_yaml_value(schema_source) or {}
    if not isinstance(payload, Mapping):
        raise ValueError("El esquema usado para optimizacion SQL debe tener una raiz mapping YAML.")
    return payload
