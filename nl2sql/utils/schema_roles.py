#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Clasificacion estructural compartida de roles de tablas del schema."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from nl2sql.utils.normalization import normalize_text_for_matching
from nl2sql.utils.schema_normalization import (
    normalize_columns,
    normalize_column_types,
    normalize_foreign_keys,
)


@dataclass(frozen=True)
class LookupRoleProfile:
    """Conteos estructurales usados para decidir si una tabla es catalogo."""

    column_count: int
    foreign_key_count: int
    temporal_column_count: int
    numeric_column_count: int
    descriptor_column_count: int = 0


def type_matches_any(column_type: str, type_hints: frozenset[str] | set[str] | tuple[str, ...]) -> bool:
    """Evalua si un tipo de columna contiene alguno de los hints declarados."""

    normalized_type = normalize_text_for_matching(column_type, keep_underscore=True)
    return any(type_hint in normalized_type for type_hint in type_hints)


def is_lookup_like_profile(
    profile: LookupRoleProfile,
    *,
    lookup_column_count_max: int,
    lookup_max_numeric_columns: int,
    lookup_min_descriptor_columns: int = 0,
    lookup_descriptor_margin: int = 0,
    require_descriptor: bool = False,
) -> bool:
    """Aplica la politica comun de tabla lookup/catalogo sobre conteos ya calculados."""

    if profile.column_count == 0 or profile.column_count > lookup_column_count_max:
        return False
    if profile.foreign_key_count != 0 or profile.temporal_column_count != 0:
        return False
    if profile.numeric_column_count > lookup_max_numeric_columns:
        return False
    if not require_descriptor:
        return True
    required_descriptors = max(lookup_min_descriptor_columns, profile.column_count - lookup_descriptor_margin)
    return profile.descriptor_column_count >= required_descriptors


def detect_lookup_tables(
    pruned_schema: Mapping[str, object] | None,
    *,
    lookup_column_count_max: int,
    lookup_max_numeric_columns: int,
    temporal_type_hints: frozenset[str] | set[str] | tuple[str, ...],
    numeric_type_hints: frozenset[str] | set[str] | tuple[str, ...],
    identifier_column_names: frozenset[str] | set[str] | tuple[str, ...] = (),
    identifier_suffixes: tuple[str, ...] = (),
) -> frozenset[str]:
    """Detecta tablas lookup desde un `pruned_schema` normalizado o crudo."""

    if not isinstance(pruned_schema, Mapping):
        return frozenset()

    detected_tables: set[str] = set()
    for table_name, raw_table_info in pruned_schema.items():
        if not isinstance(table_name, str) or not isinstance(raw_table_info, Mapping):
            continue
        profile = build_lookup_role_profile(
            raw_table_info,
            temporal_type_hints=temporal_type_hints,
            numeric_type_hints=numeric_type_hints,
            identifier_column_names=identifier_column_names,
            identifier_suffixes=identifier_suffixes,
        )
        if is_lookup_like_profile(
            profile,
            lookup_column_count_max=lookup_column_count_max,
            lookup_max_numeric_columns=lookup_max_numeric_columns,
        ):
            detected_tables.add(table_name)
    return frozenset(detected_tables)


def build_lookup_role_profile(
    table_info: Mapping[str, Any],
    *,
    temporal_type_hints: frozenset[str] | set[str] | tuple[str, ...],
    numeric_type_hints: frozenset[str] | set[str] | tuple[str, ...],
    identifier_column_names: frozenset[str] | set[str] | tuple[str, ...] = (),
    identifier_suffixes: tuple[str, ...] = (),
) -> LookupRoleProfile:
    """Calcula conteos de lookup a partir de columnas y FKs en formatos tolerantes."""

    columns = normalize_columns(table_info.get("columns"))
    column_types = normalize_column_types(table_info)
    foreign_keys = normalize_foreign_keys(table_info.get("foreign_keys"))
    temporal_count = 0
    numeric_count = 0
    for column_name, column_meta in columns.items():
        normalized_column_name = column_name.lower()
        column_type = str(column_meta.get("type") or column_types.get(column_name, ""))
        if type_matches_any(column_type, temporal_type_hints):
            temporal_count += 1
        if (
            type_matches_any(column_type, numeric_type_hints)
            and normalized_column_name not in identifier_column_names
            and not normalized_column_name.endswith(identifier_suffixes)
        ):
            numeric_count += 1
    return LookupRoleProfile(
        column_count=len(columns),
        foreign_key_count=len(foreign_keys),
        temporal_column_count=temporal_count,
        numeric_column_count=numeric_count,
    )
