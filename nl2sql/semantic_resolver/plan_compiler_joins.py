#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Grafo de joins y busqueda de caminos para el compilador semantico."""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from typing import Any


def add_join_edge(graph: dict[str, list[tuple[str, str]]], left_ref: str, right_ref: str) -> None:
    """Agrega una arista bidireccional al grafo desde referencias tabla.col."""

    if "." not in left_ref or "." not in right_ref:
        return
    left_ref = left_ref.strip()
    right_ref = right_ref.strip()
    left_table = left_ref.split(".", 1)[0]
    right_table = right_ref.split(".", 1)[0]
    edge_label = f"{left_ref} = {right_ref}"
    graph.setdefault(left_table, []).append((right_table, edge_label))
    graph.setdefault(right_table, []).append((left_table, edge_label))


def split_join_edge(edge_label: str) -> tuple[str, str] | None:
    """Divide una igualdad de join tolerando espacios no canonicos."""

    if "=" not in edge_label:
        return None
    left_side, right_side = (side.strip() for side in edge_label.split("=", 1))
    if not left_side or not right_side:
        return None
    return left_side, right_side


def foreign_key_column(raw_foreign_key: Mapping[str, object]) -> str | None:
    """Obtiene la columna local de una FK en los formatos usados por las etapas."""

    column_name = raw_foreign_key.get("col") or raw_foreign_key.get("column") or raw_foreign_key.get("source_column")
    return column_name if isinstance(column_name, str) and column_name else None


def build_join_graph(
    relationship_assets: list[Any],
    pruned_schema: Mapping[str, object] | None = None,
) -> dict[str, list[tuple[str, str]]]:
    """Construye un grafo de joins desde relaciones semanticas y FKs visibles."""

    graph: dict[str, list[tuple[str, str]]] = {}

    for matched_asset in relationship_assets:
        payload = matched_asset.asset.payload
        left_ref = payload.get("from")
        right_ref = payload.get("to")
        if isinstance(left_ref, str) and isinstance(right_ref, str):
            add_join_edge(graph, left_ref, right_ref)
            continue

        expression = payload.get("on") or payload.get("key")
        if isinstance(expression, str) and "=" in expression:
            left_ref, right_ref = (side.strip() for side in expression.split("=", 1))
            add_join_edge(graph, left_ref, right_ref)

    if isinstance(pruned_schema, Mapping):
        for table_name, raw_table_info in pruned_schema.items():
            if not isinstance(table_name, str) or not isinstance(raw_table_info, Mapping):
                continue
            raw_foreign_keys = raw_table_info.get("foreign_keys")
            if not isinstance(raw_foreign_keys, list):
                continue
            for raw_foreign_key in raw_foreign_keys:
                if not isinstance(raw_foreign_key, Mapping):
                    continue
                column_name = foreign_key_column(raw_foreign_key)
                ref_table = raw_foreign_key.get("ref_table")
                ref_col = raw_foreign_key.get("ref_col")
                if not isinstance(column_name, str) or not isinstance(ref_table, str) or not isinstance(ref_col, str):
                    continue
                add_join_edge(graph, f"{table_name}.{column_name}", f"{ref_table}.{ref_col}")

    for table_name in list(graph):
        graph[table_name].sort(key=lambda item: (item[0], item[1]))
    return graph


def shortest_join_path(
    graph: dict[str, list[tuple[str, str]]],
    source_table: str,
    target_table: str,
    *,
    forbidden_transit: frozenset[str] | set[str] | None = None,
) -> list[str]:
    """Resuelve el camino FK mas corto evitando catalogos como puentes."""

    if source_table == target_table:
        return []

    forbidden = forbidden_transit or frozenset()
    queue: deque[tuple[str, list[str]]] = deque([(source_table, [])])
    seen_tables = {source_table}
    while queue:
        current_table, current_path = queue.popleft()
        for neighbor_table, edge_label in graph.get(current_table, []):
            if neighbor_table in seen_tables:
                continue
            next_path = current_path + [edge_label]
            if neighbor_table == target_table:
                return next_path
            if neighbor_table in forbidden and neighbor_table != source_table:
                continue
            seen_tables.add(neighbor_table)
            queue.append((neighbor_table, next_path))
    return []


def path_transit_tables(path: list[str], base_table: str, target_table: str) -> set[str]:
    """Extrae las tablas que aparecen como puente, excluyendo origen y destino."""

    transit_tables: set[str] = set()
    for edge_label in path:
        join_sides = split_join_edge(edge_label)
        if join_sides is None:
            continue
        left_side, right_side = join_sides
        for side in (left_side, right_side):
            if "." not in side:
                continue
            table_name = side.split(".", 1)[0]
            if table_name != base_table and table_name != target_table:
                transit_tables.add(table_name)
    return transit_tables
