#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from nl2sql.utils.sql_identifiers import TABLE_REFERENCE_RE

from .assets import SemanticAsset


def build_direct_table_references(
    asset: SemanticAsset,
    *,
    entity_to_table: dict[str, str],
    model_to_tables: dict[str, set[str]],
) -> set[str]:
    """Extrae referencias directas a tablas desde el payload del activo."""

    payload = asset.payload
    tables: set[str] = set()

    source_table = payload.get("source_table")
    if isinstance(source_table, str) and source_table:
        tables.add(source_table)

    entity_name = payload.get("entity")
    if isinstance(entity_name, str) and entity_name in entity_to_table:
        tables.add(entity_to_table[entity_name])

    model_name = payload.get("model")
    if isinstance(model_name, str):
        tables.update(model_to_tables.get(model_name, set()))

    applies_to = payload.get("applies_to")
    if isinstance(applies_to, list):
        for item in applies_to:
            tables.update(model_to_tables.get(str(item), set()))

    for list_key in ("core_tables", "tables"):
        value = payload.get(list_key)
        if isinstance(value, list):
            tables.update(str(item) for item in value if str(item).strip())

    for scalar_key in ("field", "source", "formula", "key", "from", "to"):
        value = payload.get(scalar_key)
        if isinstance(value, str):
            tables.update(match.group(1) for match in TABLE_REFERENCE_RE.finditer(value))

    return tables


def score_compatibility(
    asset: SemanticAsset,
    *,
    entity_to_table: dict[str, str],
    model_to_tables: dict[str, set[str]],
    available_tables: set[str],
) -> tuple[float, tuple[str, ...], str | None]:
    """Mide si el activo es util con respecto al esquema podado disponible."""

    references = build_direct_table_references(
        asset,
        entity_to_table=entity_to_table,
        model_to_tables=model_to_tables,
    )
    if not references or not available_tables:
        return 1.0, tuple(), None

    compatible_tables = tuple(sorted(references & available_tables))
    if not compatible_tables:
        return 0.0, tuple(), "no_table_in_pruned_schema"

    score = len(compatible_tables) / len(references)
    return score, compatible_tables, None
