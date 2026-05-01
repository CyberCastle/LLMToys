#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Lectura normalizada de tablas, columnas y perfiles estructurales."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict

from nl2sql.config.models import HeuristicRules, QuerySignalRules
from nl2sql.utils.normalization import normalize_text_for_matching
from nl2sql.utils.schema_roles import LookupRoleProfile, is_lookup_like_profile

from .query_signals import extract_meaningful_terms


class SchemaForeignKey(TypedDict):
    """Referencia FK serializada en los schemas de entrada."""

    col: str
    ref_table: str
    ref_col: str


@dataclass(frozen=True)
class TableStructureProfile:
    """Perfil estructural de una tabla usado por scoring y pruning."""

    column_count: int
    foreign_key_count: int
    temporal_column_count: int
    numeric_column_count: int
    descriptor_column_count: int
    lookup_like: bool


def is_temporal_type_hint(column_type: str) -> bool:
    """Detecta tipos temporales comunes de SQL."""

    normalized_type = normalize_text_for_matching(column_type, keep_underscore=True)
    return any(type_hint in normalized_type for type_hint in ("date", "time", "datetime", "timestamp"))


def is_numeric_type_hint(column_type: str) -> bool:
    """Detecta tipos numericos comunes de SQL."""

    normalized_type = normalize_text_for_matching(column_type, keep_underscore=True)
    return any(
        type_hint in normalized_type
        for type_hint in (
            "bigint",
            "decimal",
            "double",
            "float",
            "int",
            "money",
            "numeric",
            "real",
            "smallint",
        )
    )


def get_table_description(table_info: dict[str, object]) -> str | None:
    """Obtiene la descripcion textual de una tabla si existe."""

    raw_description = table_info.get("description")
    if isinstance(raw_description, str) and raw_description.strip():
        return raw_description.strip()
    return None


def get_column_descriptions(table_info: dict[str, object]) -> dict[str, str]:
    """Obtiene descripciones de columnas en formato mapping simple."""

    raw_column_descriptions = table_info.get("column_descriptions", {})
    if not isinstance(raw_column_descriptions, dict):
        return {}

    column_descriptions: dict[str, str] = {}
    for column_name, column_description in raw_column_descriptions.items():
        if not isinstance(column_name, str):
            continue
        if isinstance(column_description, str) and column_description.strip():
            column_descriptions[column_name] = column_description.strip()

    return column_descriptions


def get_schema_columns(table_info: dict[str, object]) -> list[tuple[str, str]]:
    """Acepta columnas serializadas como tuple o como list al cargar YAML."""

    raw_columns = table_info.get("columns", [])
    columns: list[tuple[str, str]] = []
    if not isinstance(raw_columns, list):
        return columns

    for raw_column in raw_columns:
        if (
            isinstance(raw_column, (tuple, list))
            and len(raw_column) == 2
            and isinstance(raw_column[0], str)
            and isinstance(raw_column[1], str)
        ):
            columns.append((raw_column[0], raw_column[1]))

    return columns


def get_primary_keys(table_info: dict[str, object]) -> list[str]:
    """Obtiene la lista canonica de primary keys de una tabla."""

    raw_primary_keys = table_info.get("primary_keys", [])
    if not isinstance(raw_primary_keys, list):
        return []
    return [value for value in raw_primary_keys if isinstance(value, str)]


def get_foreign_keys(table_info: dict[str, object]) -> list[SchemaForeignKey]:
    """Obtiene y ordena FKs en el formato usado por semantic prune."""

    raw_foreign_keys = table_info.get("foreign_keys", [])
    if not isinstance(raw_foreign_keys, list):
        return []

    foreign_keys: list[SchemaForeignKey] = []
    for raw_foreign_key in raw_foreign_keys:
        if not isinstance(raw_foreign_key, dict):
            continue
        column_name = raw_foreign_key.get("col")
        ref_table = raw_foreign_key.get("ref_table")
        ref_column = raw_foreign_key.get("ref_col")
        if not isinstance(column_name, str) or not isinstance(ref_table, str) or not isinstance(ref_column, str):
            continue
        foreign_keys.append(
            {
                "col": column_name,
                "ref_table": ref_table,
                "ref_col": ref_column,
            }
        )

    foreign_keys.sort(key=lambda item: (item["col"], item["ref_table"], item["ref_col"]))
    return foreign_keys


def build_table_structure_profile(
    table_info: dict[str, object],
    signal_rules: QuerySignalRules,
    heuristic_rules: HeuristicRules,
) -> TableStructureProfile:
    """Calcula densidad, tipos y rol lookup de una tabla."""

    columns = get_schema_columns(table_info)
    foreign_keys = get_foreign_keys(table_info)
    descriptor_column_count = 0
    temporal_column_count = 0
    numeric_column_count = 0
    structure_rules = heuristic_rules.structure_profile

    for column_name, column_type in columns:
        column_terms = extract_meaningful_terms(column_name, signal_rules)
        if column_name.lower() == "id" or column_name.lower().endswith("_id") or column_terms & signal_rules.lookup_descriptor_terms:
            descriptor_column_count += 1
        if is_temporal_type_hint(column_type):
            temporal_column_count += 1
        if is_numeric_type_hint(column_type):
            numeric_column_count += 1

    column_count = len(columns)
    role_profile = LookupRoleProfile(
        column_count=column_count,
        foreign_key_count=len(foreign_keys),
        temporal_column_count=temporal_column_count,
        numeric_column_count=numeric_column_count,
        descriptor_column_count=descriptor_column_count,
    )
    lookup_like = is_lookup_like_profile(
        role_profile,
        lookup_column_count_max=structure_rules.lookup_column_count_max,
        lookup_max_numeric_columns=structure_rules.lookup_max_numeric_columns,
        lookup_min_descriptor_columns=structure_rules.lookup_min_descriptor_columns,
        lookup_descriptor_margin=structure_rules.lookup_descriptor_margin,
        require_descriptor=True,
    )
    return TableStructureProfile(
        column_count=column_count,
        foreign_key_count=len(foreign_keys),
        temporal_column_count=temporal_column_count,
        numeric_column_count=numeric_column_count,
        descriptor_column_count=descriptor_column_count,
        lookup_like=lookup_like,
    )
