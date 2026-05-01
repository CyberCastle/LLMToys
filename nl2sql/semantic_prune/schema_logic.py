#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Logica central de pruning semantico para reducir el schema fisico.

Este modulo combina scoring lexico/estructural con conectividad FK para
preservar solo las tablas, columnas y relaciones que una consulta NL2SQL
necesita realmente antes de pasar al resolver semantico.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from functools import lru_cache
import heapq
from pathlib import Path
import re
from typing import TYPE_CHECKING

import numpy as np

from nl2sql.config import load_semantic_prune_settings
from nl2sql.config.models import (
    FkPathHeuristics,
    HeuristicRules,
    QuerySignalRules,
    RelationshipExpansionHeuristics,
)
from nl2sql.utils.normalization import normalize_text_for_matching
from nl2sql.utils.semantic_filters import SemanticFilterSelection, resolve_semantic_filter_selections
from nl2sql.utils.sql_identifiers import TABLE_COLUMN_RE
from nl2sql.utils.text_utils import truncate_text
from nl2sql.utils.yaml_utils import load_yaml_mapping

from .query_signals import (
    QuerySignalProfile,
    extract_meaningful_terms,
    get_term_overlap_count,
    infer_query_signal_profile,
)
from .schema_tables import (
    SchemaForeignKey,
    TableStructureProfile,
    build_table_structure_profile,
    get_column_descriptions,
    get_foreign_keys,
    get_primary_keys,
    get_schema_columns,
    get_table_description,
    is_numeric_type_hint,
    is_temporal_type_hint,
)

if TYPE_CHECKING:
    from .config import SemanticSchemaPruningConfig


@lru_cache(maxsize=8)
def load_query_signal_rules(rules_path: str) -> QuerySignalRules:
    """Devuelve las reglas tipadas ya validadas del semantic prune."""

    return load_semantic_prune_settings(Path(rules_path).expanduser().resolve()).query_signal_rules


@lru_cache(maxsize=8)
def load_heuristic_rules(rules_path: str) -> HeuristicRules:
    """Devuelve las heuristicas tipadas ya validadas del semantic prune."""

    return load_semantic_prune_settings(Path(rules_path).expanduser().resolve()).heuristic_rules


def resolve_relationship_expansion_heuristics(
    config: SemanticSchemaPruningConfig,
    heuristic_rules: HeuristicRules,
) -> RelationshipExpansionHeuristics:
    """Combina defaults del YAML con overrides puntuales definidos en config."""

    defaults = heuristic_rules.relationship_expansion
    return RelationshipExpansionHeuristics(
        outbound_hops=(
            defaults.outbound_hops if config.relationship_expansion_outbound_hops is None else config.relationship_expansion_outbound_hops
        ),
        inbound_hops=(
            defaults.inbound_hops if config.relationship_expansion_inbound_hops is None else config.relationship_expansion_inbound_hops
        ),
        max_neighbors_per_table=(
            defaults.max_neighbors_per_table
            if config.relationship_expansion_max_neighbors_per_table is None
            else config.relationship_expansion_max_neighbors_per_table
        ),
        outbound_min_score=(
            defaults.outbound_min_score if config.relationship_expansion_min_score is None else config.relationship_expansion_min_score
        ),
        inbound_min_score=(
            defaults.inbound_min_score
            if config.relationship_expansion_inbound_min_score is None
            else config.relationship_expansion_inbound_min_score
        ),
        bridge_max_hops=(defaults.bridge_max_hops if config.relationship_bridge_max_hops is None else config.relationship_bridge_max_hops),
        bridge_table_min_score=(
            defaults.bridge_table_min_score
            if config.relationship_bridge_table_min_score is None
            else config.relationship_bridge_table_min_score
        ),
        unused_foreign_key_score_margin=defaults.unused_foreign_key_score_margin,
    )


def resolve_fk_path_heuristics(
    config: SemanticSchemaPruningConfig,
    heuristic_rules: HeuristicRules,
) -> FkPathHeuristics:
    """Resuelve controles de anclas y expansion FK con fallback al YAML."""

    defaults = heuristic_rules.fk_path
    return FkPathHeuristics(
        enabled=(defaults.enabled if config.enable_fk_path_expansion is None else config.enable_fk_path_expansion),
        max_hops=(defaults.max_hops if config.fk_path_max_hops is None else config.fk_path_max_hops),
        anchor_min_overlap=(
            defaults.anchor_min_overlap if config.fk_path_anchor_min_overlap is None else config.fk_path_anchor_min_overlap
        ),
        max_anchors_per_role=(
            defaults.max_anchors_per_role if config.fk_path_max_anchors_per_role is None else config.fk_path_max_anchors_per_role
        ),
    )


@dataclass(frozen=True)
class SchemaGraphEdge:
    current_table: str
    neighbor_table: str
    current_column: str
    neighbor_column: str
    direction: str


@dataclass(frozen=True)
class SchemaGraph:
    adjacency: dict[str, tuple[SchemaGraphEdge, ...]]


@dataclass(frozen=True)
class SchemaForeignKeyReference:
    table_name: str
    column_name: str
    ref_table: str
    ref_column: str


@dataclass
class SchemaSubgraphSelection:
    tables: list[str]
    required_columns_by_table: dict[str, set[str]]
    required_foreign_keys_by_table: dict[str, set[SchemaForeignKeyReference]]
    table_reasons: dict[str, str]
    metric_anchor_tables: tuple[str, ...]
    dimension_anchor_tables: tuple[str, ...]
    fk_path_edges: tuple[tuple[SchemaGraphEdge, ...], ...]


@dataclass(frozen=True)
class SemanticDependencySpec:
    """Dependencias fisicas declaradas por activos semanticos relevantes."""

    name: str
    tables: frozenset[str]
    required_columns_by_table: dict[str, frozenset[str]]
    relationship_edges: tuple[SchemaGraphEdge, ...]


@dataclass(frozen=True)
class SemanticScoreContext:
    table_doc_scores: dict[str, float]
    table_scores: dict[str, float]
    column_scores: dict[str, dict[str, float]]
    table_min_score: float
    column_min_score: float
    semantic_seed_tables: list[str]
    metric_anchor_tables: tuple[str, ...]
    dimension_anchor_tables: tuple[str, ...]
    query_profile: QuerySignalProfile


def as_float(value: object, default: float = 0.0) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return default
    return default


def compute_table_signal_adjustment(
    table_name: str,
    table_info: dict[str, object],
    query_profile: QuerySignalProfile,
    signal_rules: QuerySignalRules,
    heuristic_rules: HeuristicRules,
) -> float:
    structure_profile = build_table_structure_profile(table_info, signal_rules, heuristic_rules)
    table_signal = heuristic_rules.table_signal
    structure_rules = heuristic_rules.structure_profile
    lexical_overlap = get_term_overlap_count(
        f"{table_name} {get_table_description(table_info) or ''}",
        query_profile.query_terms,
        signal_rules,
    )
    lexical_bonus = min(
        table_signal.lexical_bonus_cap,
        lexical_overlap * table_signal.lexical_overlap_weight,
    )
    richness_bonus = min(
        table_signal.richness_bonus_cap,
        max(
            structure_profile.column_count - structure_rules.richness_extra_columns_start,
            0,
        )
        * table_signal.richness_extra_column_weight,
    )

    context_bonus = 0.0
    if query_profile.wants_temporal and structure_profile.temporal_column_count > 0:
        context_bonus += min(
            table_signal.temporal_bonus_cap,
            table_signal.temporal_bonus_base
            + table_signal.temporal_bonus_per_column
            * min(
                structure_profile.temporal_column_count,
                structure_rules.temporal_bonus_column_cap,
            ),
        )
    if query_profile.wants_grouping and structure_profile.foreign_key_count > 0:
        context_bonus += min(
            table_signal.grouping_bonus_cap,
            table_signal.grouping_bonus_base
            + table_signal.grouping_bonus_per_fk
            * min(
                structure_profile.foreign_key_count,
                structure_rules.grouping_bonus_fk_cap,
            ),
        )
    if query_profile.wants_aggregation and structure_profile.column_count >= structure_rules.aggregation_min_column_count:
        if structure_profile.foreign_key_count > 0 or structure_profile.temporal_column_count > 0:
            context_bonus += table_signal.aggregation_structural_bonus
        if (
            structure_profile.numeric_column_count >= structure_rules.aggregation_min_numeric_columns
            and structure_profile.foreign_key_count > 0
        ):
            context_bonus += table_signal.aggregation_numeric_fk_bonus

    lookup_penalty = table_signal.lookup_penalty if structure_profile.lookup_like else 0.0
    return max(
        table_signal.output_min,
        min(
            table_signal.output_max,
            lexical_bonus + richness_bonus + context_bonus + lookup_penalty,
        ),
    )


def compute_column_signal_adjustment(
    table_name: str,
    column_name: str,
    column_type: str,
    column_description: str | None,
    query_profile: QuerySignalProfile,
    signal_rules: QuerySignalRules,
    heuristic_rules: HeuristicRules,
) -> float:
    column_signal = heuristic_rules.column_signal
    lexical_overlap = get_term_overlap_count(
        f"{table_name} {column_name} {column_description or ''}",
        query_profile.query_terms,
        signal_rules,
    )
    lexical_bonus = min(
        column_signal.lexical_bonus_cap,
        lexical_overlap * column_signal.lexical_overlap_weight,
    )

    context_bonus = 0.0
    if query_profile.wants_temporal and is_temporal_type_hint(column_type):
        context_bonus += column_signal.temporal_type_bonus
        if extract_meaningful_terms(f"{column_name} {column_description or ''}", signal_rules) & signal_rules.temporal_column_name_terms:
            context_bonus += column_signal.temporal_name_bonus

    return max(
        column_signal.output_min,
        min(column_signal.output_max, lexical_bonus + context_bonus),
    )


def verbalize_join(
    local_table: str,
    local_description: str | None,
    ref_table: str,
    ref_description: str | None,
) -> str:
    left = local_description or local_table
    right = ref_description or ref_table
    return f"Relaciona {left} con {right}"


def build_foreign_key_metadata_lookup(
    table_name: str,
    table_info: dict[str, object],
    schema: dict[str, object],
) -> dict[str, list[dict[str, str]]]:
    lookup: dict[str, list[dict[str, str]]] = {}
    local_description = get_table_description(table_info)

    for foreign_key in get_foreign_keys(table_info):
        ref_table = foreign_key["ref_table"]
        ref_table_info = schema.get(ref_table)
        ref_description = get_table_description(ref_table_info) if isinstance(ref_table_info, dict) else None
        lookup.setdefault(foreign_key["col"], []).append(
            {
                "relation_text": f"{ref_table}.{foreign_key['ref_col']}",
                "verbalized_join": verbalize_join(
                    table_name,
                    local_description,
                    ref_table,
                    ref_description,
                ),
            }
        )

    return lookup


def build_foreign_key_reference(
    table_name: str,
    foreign_key: SchemaForeignKey,
) -> SchemaForeignKeyReference | None:
    column_name = foreign_key.get("col")
    referenced_table = foreign_key.get("ref_table")
    referenced_column = foreign_key.get("ref_col")
    if not isinstance(column_name, str) or not isinstance(referenced_table, str) or not isinstance(referenced_column, str):
        return None

    return SchemaForeignKeyReference(
        table_name=table_name,
        column_name=column_name,
        ref_table=referenced_table,
        ref_column=referenced_column,
    )


def build_schema_graph(schema: dict[str, object]) -> SchemaGraph:
    """Convierte las foreign keys en un grafo navegable entre tablas."""
    adjacency: dict[str, list[SchemaGraphEdge]] = {table_name: [] for table_name in schema if isinstance(table_name, str)}

    for table_name, raw_table_info in schema.items():
        if not isinstance(table_name, str) or not isinstance(raw_table_info, dict):
            continue

        for foreign_key in get_foreign_keys(raw_table_info):
            neighbor_table = foreign_key["ref_table"]
            if neighbor_table not in adjacency:
                continue

            adjacency[table_name].append(
                SchemaGraphEdge(
                    current_table=table_name,
                    neighbor_table=neighbor_table,
                    current_column=foreign_key["col"],
                    neighbor_column=foreign_key["ref_col"],
                    direction="outbound",
                )
            )
            adjacency[neighbor_table].append(
                SchemaGraphEdge(
                    current_table=neighbor_table,
                    neighbor_table=table_name,
                    current_column=foreign_key["ref_col"],
                    neighbor_column=foreign_key["col"],
                    direction="inbound",
                )
            )

    return SchemaGraph(adjacency={table_name: tuple(edges) for table_name, edges in adjacency.items()})


def build_foreign_key_reference_from_edge(
    relationship_edge: SchemaGraphEdge,
) -> SchemaForeignKeyReference:
    if relationship_edge.direction == "outbound":
        return SchemaForeignKeyReference(
            table_name=relationship_edge.current_table,
            column_name=relationship_edge.current_column,
            ref_table=relationship_edge.neighbor_table,
            ref_column=relationship_edge.neighbor_column,
        )

    return SchemaForeignKeyReference(
        table_name=relationship_edge.neighbor_table,
        column_name=relationship_edge.neighbor_column,
        ref_table=relationship_edge.current_table,
        ref_column=relationship_edge.current_column,
    )


def mark_graph_edge_usage(
    required_columns_by_table: dict[str, set[str]],
    required_foreign_keys_by_table: dict[str, set[SchemaForeignKeyReference]],
    relationship_edge: SchemaGraphEdge,
) -> None:
    required_columns_by_table.setdefault(relationship_edge.current_table, set()).add(relationship_edge.current_column)
    required_columns_by_table.setdefault(relationship_edge.neighbor_table, set()).add(relationship_edge.neighbor_column)
    foreign_key_reference = build_foreign_key_reference_from_edge(relationship_edge)
    required_foreign_keys_by_table.setdefault(foreign_key_reference.table_name, set()).add(foreign_key_reference)


def mark_selected_table_reason(
    table_reasons: dict[str, str],
    table_name: str,
    reason: str,
    heuristic_rules: HeuristicRules,
) -> None:
    current_reason = table_reasons.get(table_name)
    if current_reason is None:
        table_reasons[table_name] = reason
        return

    if heuristic_rules.table_selection_reason_priority.get(reason, 0) > heuristic_rules.table_selection_reason_priority.get(
        current_reason,
        0,
    ):
        table_reasons[table_name] = reason


def register_selected_table(selected_tables: list[str], selected_table_set: set[str], table_name: str) -> bool:
    if table_name in selected_table_set:
        return False

    selected_table_set.add(table_name)
    selected_tables.append(table_name)
    return True


def get_table_semantic_score(table_scores: dict[str, float], table_name: str) -> float:
    return table_scores.get(table_name, float("-inf"))


def get_column_semantic_score(column_scores: dict[str, dict[str, float]], table_name: str, column_name: str) -> float:
    return column_scores.get(table_name, {}).get(column_name, float("-inf"))


def get_relationship_semantic_score(
    relationship_edge: SchemaGraphEdge,
    table_scores: dict[str, float],
    column_scores: dict[str, dict[str, float]],
    heuristic_rules: HeuristicRules,
) -> float:
    """Resume en un score la evidencia semantica disponible para una FK."""
    evidence_scores = [
        get_table_semantic_score(table_scores, relationship_edge.neighbor_table),
        get_column_semantic_score(
            column_scores,
            relationship_edge.current_table,
            relationship_edge.current_column,
        ),
        get_column_semantic_score(
            column_scores,
            relationship_edge.neighbor_table,
            relationship_edge.neighbor_column,
        ),
    ]
    valid_scores = [score for score in evidence_scores if score != float("-inf")]
    if not valid_scores:
        return float("-inf")

    valid_scores.sort(reverse=True)
    primary_score = valid_scores[0]
    supporting_score = valid_scores[1] if len(valid_scores) > 1 else primary_score
    relationship_rules = heuristic_rules.relationship_scoring
    return relationship_rules.primary_evidence_weight * primary_score + relationship_rules.supporting_evidence_weight * supporting_score


def compute_relationship_edge_cost(
    relationship_edge: SchemaGraphEdge,
    table_scores: dict[str, float],
    column_scores: dict[str, dict[str, float]],
    heuristic_rules: HeuristicRules,
    lookup_tables: frozenset[str] | set[str] | None = None,
) -> float:
    score = get_relationship_semantic_score(relationship_edge, table_scores, column_scores, heuristic_rules)
    normalized_score = max(-1.0, min(1.0, score))
    base_cost = max(1e-3, 1.0 - normalized_score)
    # Penalizacion dura cuando la arista apunta a una tabla catalogo/lookup.
    # Esto evita que Dijkstra use catalogos como puente entre un ancla metrica
    # y un ancla dimensional cuando existe una ruta alternativa por FKs reales
    # (ej. preferir una ruta FK directa entre tablas raiz frente a una ruta
    # via tablas de configuracion o puentes debiles, que es no-deterministico).
    if lookup_tables and relationship_edge.neighbor_table in lookup_tables:
        base_cost += heuristic_rules.relationship_scoring.lookup_edge_penalty
    return base_cost


def build_lookup_tables_set(
    schema: dict[str, object],
    signal_rules: QuerySignalRules,
    heuristic_rules: HeuristicRules,
) -> frozenset[str]:
    """Identifica tablas catalogo/lookup puras del esquema.

    Se usa para penalizar su uso como nodo intermedio en busquedas de rutas
    estructurales: son tablas de referencia N:1 que NO sirven como puente
    deterministico entre dos entidades de negocio distintas.
    """
    detected_tables: set[str] = set()
    for table_name, raw_table_info in schema.items():
        if not isinstance(table_name, str) or not isinstance(raw_table_info, dict):
            continue
        if build_table_structure_profile(raw_table_info, signal_rules, heuristic_rules).lookup_like:
            detected_tables.add(table_name)
    return frozenset(detected_tables)


def build_allowed_bridge_tables(
    seed_tables: list[str],
    schema_graph: SchemaGraph,
    table_scores: dict[str, float],
    min_score: float,
    column_scores: dict[str, dict[str, float]] | None = None,
) -> set[str]:
    """Limita los bridges semanticos a tablas con alguna señal relevante.

    Esta restriccion mantiene conservador el pruning normal, pero tambien es la
    razon por la que tablas puramente estructurales pueden quedar fuera. Para
    esos casos existe la expansion dura `fk_path`, que no usa este filtro.
    """
    allowed_tables = set(seed_tables)
    resolved_column_scores = column_scores or {}
    for table_name in schema_graph.adjacency:
        strongest_column_score = max(resolved_column_scores.get(table_name, {}).values(), default=float("-inf"))
        if (
            max(
                get_table_semantic_score(table_scores, table_name),
                strongest_column_score,
            )
            >= min_score
        ):
            allowed_tables.add(table_name)
    return allowed_tables


def get_anchor_overlap_scores(
    table_name: str,
    table_info: dict[str, object],
    candidate_terms: frozenset[str],
    signal_rules: QuerySignalRules,
) -> tuple[int, int, int]:
    table_name_overlap = get_term_overlap_count(table_name, candidate_terms, signal_rules)
    lexical_overlap = get_term_overlap_count(
        f"{table_name} {get_table_description(table_info) or ''}",
        candidate_terms,
        signal_rules,
    )
    column_descriptions = get_column_descriptions(table_info)
    column_overlap = 0
    for column_name, _column_type in get_schema_columns(table_info):
        column_overlap = max(
            column_overlap,
            get_term_overlap_count(
                f"{column_name} {column_descriptions.get(column_name, '')}",
                candidate_terms,
                signal_rules,
            ),
        )
    return table_name_overlap, lexical_overlap, column_overlap


def is_documental_table_name(table_name: str, signal_rules: QuerySignalRules) -> bool:
    """Detecta si una tabla representa documentacion/archivos/carpetas.

    Se basa en coincidencia de tokens del nombre de la tabla con la lista
    `documental_terms` del YAML lexico. La intencion es identificar
    tablas que vinculan carpetas o archivos externos (Box, SharePoint) o que
    son metadata documental, y evitar que se promocionen como anclas de
    negocio cuando la consulta es operacional.
    """

    terms = signal_rules.documental_terms
    if not terms:
        return False
    normalized = table_name.lower()
    # Comparacion por tokens separados por "_" para no confundir, por ejemplo,
    # una tabla documental con otra cuyo nombre solo comparte un substring.
    tokens = {token for token in normalized.split("_") if token}
    return bool(tokens & terms)


# Sufijos de tablas "derivadas" de otra entidad raiz. Cuando la raiz esta
# presente en el schema, estas tablas rara vez son el ancla correcta para una
# consulta operacional (p.ej. la tabla raiz frente a su tabla de revisiones).
def get_derived_table_root(table_name: str, schema: dict[str, object], suffixes: tuple[str, ...]) -> str | None:
    """Si `table_name` termina en un sufijo derivado y la tabla raiz existe,
    devuelve el nombre de la raiz. En cualquier otro caso devuelve None.
    """
    normalized = table_name.lower()
    for suffix in suffixes:
        if normalized.endswith(suffix):
            root = normalized[: -len(suffix)]
            if root and root in schema:
                return root
    return None


def pick_anchor_tables(
    candidate_terms: frozenset[str],
    schema: dict[str, object],
    table_scores: dict[str, float],
    signal_rules: QuerySignalRules,
    heuristic_rules: HeuristicRules,
    *,
    max_anchors: int,
    min_overlap: int,
    excluded_tables: set[str] | None = None,
) -> tuple[str, ...]:
    if not candidate_terms or max_anchors <= 0:
        return ()

    # Si la consulta no pide documentacion/carpetas/archivos, las tablas
    # documentales no deben actuar como anclas de metrica ni de dimension
    # aunque compartan tokens con la entidad principal. Esto evita rutas
    # espurias via metadata documental o catalogos auxiliares.
    query_mentions_documents = bool(candidate_terms & signal_rules.documental_terms)

    ranked_candidates: list[tuple[int, int, int, float, str]] = []
    fallback_candidates: list[tuple[int, int, int, float, str]] = []
    resolved_excluded_tables = excluded_tables or set()
    for table_name, raw_table_info in schema.items():
        if table_name in resolved_excluded_tables or not isinstance(table_name, str) or not isinstance(raw_table_info, dict):
            continue

        if not query_mentions_documents and is_documental_table_name(table_name, signal_rules):
            continue

        # Tablas lookup-like (solo `id` + descriptores, sin FKs) no son ancla
        # de negocio: son catalogos. Las descartamos aunque compartan tokens
        # con la consulta o con una tabla principal del dominio.
        if build_table_structure_profile(raw_table_info, signal_rules, heuristic_rules).lookup_like:
            continue

        # Si existe la tabla raiz y la candidata es su derivada de revisiones,
        # excluimos la derivada como ancla. La raiz es la entidad de negocio.
        derived_root = get_derived_table_root(table_name, schema, heuristic_rules.structure_profile.derived_table_suffixes)
        if derived_root is not None and derived_root != table_name:
            continue

        table_name_overlap, lexical_overlap, column_overlap = get_anchor_overlap_scores(
            table_name,
            raw_table_info,
            candidate_terms,
            signal_rules,
        )
        # Regla estricta: el nombre y/o la descripcion de la tabla DEBEN
        # contener al menos un termino del rol (metric/dimension). Dejar
        # pasar con solo `column_overlap` filtra por descripciones de FK
        # (p.ej. un termino del rol embebido en una FK como `bridge.entity_id`) y
        # convierte en anclas tablas que solo referencian a la entidad.
        if lexical_overlap < min_overlap:
            continue
        strongest_overlap = max(lexical_overlap, column_overlap)
        if strongest_overlap < min_overlap:
            continue

        candidate_tuple = (
            table_name_overlap,
            lexical_overlap,
            column_overlap,
            table_scores.get(table_name, float("-inf")),
            table_name,
        )
        # Preferimos candidatos cuyo NOMBRE de tabla contiene el termino del rol
        # (p.ej. un token simple que coincide con el nombre de la tabla o un
        # token compuesto cuyas piezas aparecen en el nombre de la tabla).
        # Si una tabla solo aparece por descripcion (no por nombre), la dejamos
        # como fallback: solo entra si no hay mejores opciones.
        if table_name_overlap > 0:
            ranked_candidates.append(candidate_tuple)
        else:
            fallback_candidates.append(candidate_tuple)

    ranked_candidates.sort(reverse=True)
    # Si hay AL MENOS una tabla cuyo nombre coincide con el rol, no degradamos
    # el resultado con candidatos basados solo en descripcion. Esto evita que
    # anclas fuertes sean acompanadas por tablas que solo mencionan la entidad
    # en descripciones de FKs o notas auxiliares.
    if ranked_candidates:
        return tuple(table_name for *_scores, table_name in ranked_candidates[:max_anchors])
    # Solo cuando nadie matchea por nombre permitimos fallback lexico puro.
    fallback_candidates.sort(reverse=True)
    return tuple(table_name for *_scores, table_name in fallback_candidates[:max_anchors])


def detect_query_anchor_tables(
    schema: dict[str, object],
    table_scores: dict[str, float],
    query_profile: QuerySignalProfile,
    signal_rules: QuerySignalRules,
    heuristic_rules: HeuristicRules,
    *,
    max_anchors_per_role: int,
    min_overlap: int,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    dimension_anchors = pick_anchor_tables(
        query_profile.dimension_terms,
        schema,
        table_scores,
        signal_rules,
        heuristic_rules,
        max_anchors=max_anchors_per_role,
        min_overlap=min_overlap,
    )
    metric_anchors = pick_anchor_tables(
        query_profile.metric_terms,
        schema,
        table_scores,
        signal_rules,
        heuristic_rules,
        max_anchors=max_anchors_per_role,
        min_overlap=min_overlap,
        excluded_tables=set(dimension_anchors),
    )

    if not metric_anchors and query_profile.wants_aggregation:
        metric_anchors = tuple(
            table_name
            for table_name, _score in sorted(table_scores.items(), key=lambda item: item[1], reverse=True)
            if table_name not in set(dimension_anchors)
        )[:max_anchors_per_role]

    if not dimension_anchors and query_profile.wants_grouping:
        fallback_dimension_terms = frozenset(
            term
            for term in query_profile.query_terms
            if term not in query_profile.metric_terms
            and term not in signal_rules.query_aggregation_terms
            and term not in signal_rules.query_temporal_terms
            and term not in signal_rules.query_intent_noise_terms
        )
        dimension_anchors = pick_anchor_tables(
            fallback_dimension_terms,
            schema,
            table_scores,
            signal_rules,
            heuristic_rules,
            max_anchors=max_anchors_per_role,
            min_overlap=min_overlap,
            excluded_tables=set(metric_anchors),
        )

    return metric_anchors, dimension_anchors


def merge_seed_tables(seed_tables: list[str], *forced_groups: tuple[str, ...]) -> list[str]:
    merged_seed_tables = list(seed_tables)
    seen_tables = set(seed_tables)
    for forced_group in forced_groups:
        for table_name in forced_group:
            if table_name in seen_tables:
                continue
            seen_tables.add(table_name)
            merged_seed_tables.append(table_name)
    return merged_seed_tables


def find_cheapest_relationship_path(
    start_table: str,
    goal_table: str,
    schema_graph: SchemaGraph,
    max_hops: int,
    *,
    table_scores: dict[str, float],
    column_scores: dict[str, dict[str, float]],
    heuristic_rules: HeuristicRules,
    allowed_tables: set[str] | None = None,
    lookup_tables: frozenset[str] | set[str] | None = None,
) -> list[SchemaGraphEdge] | None:
    """Busca el camino relacional mas barato segun evidencia semantica."""
    if start_table == goal_table:
        return []

    heap: list[tuple[float, int, str]] = [(0.0, 0, start_table)]
    best_cost_by_state: dict[tuple[str, int], float] = {(start_table, 0): 0.0}
    previous_step: dict[tuple[str, int], tuple[tuple[str, int], SchemaGraphEdge]] = {}
    goal_state: tuple[str, int] | None = None

    while heap:
        current_cost, current_hops, current_table = heapq.heappop(heap)
        current_state = (current_table, current_hops)
        if current_cost > best_cost_by_state.get(current_state, float("inf")):
            continue

        if current_table == goal_table:
            goal_state = current_state
            break

        if current_hops >= max_hops:
            continue

        for relationship_edge in schema_graph.adjacency.get(current_table, ()):  # pragma: no branch
            neighbor_table = relationship_edge.neighbor_table
            if allowed_tables is not None and neighbor_table not in allowed_tables and neighbor_table != goal_table:
                continue

            # El cost function penaliza las aristas que apuntan a lookups, excepto
            # cuando el destino es el goal (un ancla dimensional real puede no ser
            # lookup pero eventualmente hay que llegar).
            edge_lookup_tables = lookup_tables
            if edge_lookup_tables and neighbor_table == goal_table:
                edge_lookup_tables = None

            next_state = (neighbor_table, current_hops + 1)
            next_cost = current_cost + compute_relationship_edge_cost(
                relationship_edge,
                table_scores,
                column_scores,
                heuristic_rules,
                lookup_tables=edge_lookup_tables,
            )
            if next_cost >= best_cost_by_state.get(next_state, float("inf")):
                continue

            best_cost_by_state[next_state] = next_cost
            previous_step[next_state] = (current_state, relationship_edge)
            heapq.heappush(heap, (next_cost, current_hops + 1, neighbor_table))

    if goal_state is None:
        return None

    path_edges: list[SchemaGraphEdge] = []
    current_state = goal_state
    while current_state[0] != start_table or current_state[1] != 0:
        previous_state, previous_edge = previous_step[current_state]
        path_edges.append(previous_edge)
        current_state = previous_state

    path_edges.reverse()
    return path_edges


def steiner_approx_paths(
    seed_tables: list[str],
    schema_graph: SchemaGraph,
    *,
    max_hops: int,
    table_scores: dict[str, float],
    column_scores: dict[str, dict[str, float]],
    heuristic_rules: HeuristicRules,
    allowed_tables: set[str] | None,
    lookup_tables: frozenset[str] | set[str] | None = None,
) -> list[list[SchemaGraphEdge]]:
    """Conecta semillas via MST sobre el grafo completo de terminales."""
    if len(seed_tables) < 2:
        return []

    pair_paths: dict[tuple[str, str], list[SchemaGraphEdge]] = {}
    pair_costs: list[tuple[float, str, str]] = []

    for start_table_index, start_table in enumerate(seed_tables):
        for goal_table in seed_tables[start_table_index + 1 :]:
            path = find_cheapest_relationship_path(
                start_table,
                goal_table,
                schema_graph,
                max_hops,
                table_scores=table_scores,
                column_scores=column_scores,
                heuristic_rules=heuristic_rules,
                allowed_tables=allowed_tables,
                lookup_tables=lookup_tables,
            )
            if path is None:
                continue

            pair_paths[(start_table, goal_table)] = path
            pair_costs.append(
                (
                    sum(
                        compute_relationship_edge_cost(
                            edge,
                            table_scores,
                            column_scores,
                            heuristic_rules,
                            lookup_tables=lookup_tables,
                        )
                        for edge in path
                    ),
                    start_table,
                    goal_table,
                )
            )

    parent = {table_name: table_name for table_name in seed_tables}

    def find_root(table_name: str) -> str:
        current = table_name
        while parent[current] != current:
            parent[current] = parent[parent[current]]
            current = parent[current]
        return current

    selected_paths: list[list[SchemaGraphEdge]] = []
    for _, start_table, goal_table in sorted(pair_costs, key=lambda item: item[0]):
        start_root = find_root(start_table)
        goal_root = find_root(goal_table)
        if start_root == goal_root:
            continue

        parent[start_root] = goal_root
        selected_paths.append(pair_paths[(start_table, goal_table)])

    return selected_paths


def connect_anchor_pairs(
    metric_anchor_tables: tuple[str, ...],
    dimension_anchor_tables: tuple[str, ...],
    schema_graph: SchemaGraph,
    *,
    max_hops: int,
    table_scores: dict[str, float],
    column_scores: dict[str, dict[str, float]],
    heuristic_rules: HeuristicRules,
    lookup_tables: frozenset[str] | set[str] | None = None,
    semantic_join_path_specs: tuple[SemanticJoinPathSpec, ...] = (),
) -> tuple[tuple[SchemaGraphEdge, ...], ...]:
    selected_paths: list[tuple[SchemaGraphEdge, ...]] = []
    seen_signatures: set[tuple[tuple[str, str, str, str, str], ...]] = set()

    for start_table in metric_anchor_tables:
        for goal_table in dimension_anchor_tables:
            if start_table == goal_table:
                continue

            # Prioridad absoluta a rutas canonicas del YAML cuando conectan este
            # par de anclas. Son la fuente de verdad manual y ganan sobre cualquier
            # camino FK generico propuesto por Dijkstra.
            semantic_path = find_semantic_join_path_for_anchors(
                semantic_join_path_specs,
                start_table,
                goal_table,
            )
            if semantic_path:
                signature = tuple(
                    (
                        edge.current_table,
                        edge.neighbor_table,
                        edge.current_column,
                        edge.neighbor_column,
                        edge.direction,
                    )
                    for edge in semantic_path
                )
                if signature not in seen_signatures:
                    seen_signatures.add(signature)
                    selected_paths.append(semantic_path)
                continue

            path = find_cheapest_relationship_path(
                start_table,
                goal_table,
                schema_graph,
                max_hops,
                table_scores=table_scores,
                column_scores=column_scores,
                heuristic_rules=heuristic_rules,
                allowed_tables=None,
                lookup_tables=lookup_tables,
            )
            if not path:
                continue

            signature = tuple(
                (
                    relationship_edge.current_table,
                    relationship_edge.neighbor_table,
                    relationship_edge.current_column,
                    relationship_edge.neighbor_column,
                    relationship_edge.direction,
                )
                for relationship_edge in path
            )
            if signature in seen_signatures:
                continue

            seen_signatures.add(signature)
            selected_paths.append(tuple(path))

    selected_paths.sort(
        key=lambda path_edges: (
            len(path_edges),
            sum(
                compute_relationship_edge_cost(
                    edge,
                    table_scores,
                    column_scores,
                    heuristic_rules,
                    lookup_tables=lookup_tables,
                )
                for edge in path_edges
            ),
        )
    )
    return tuple(selected_paths)


def serialize_schema_graph_path(
    path_edges: tuple[SchemaGraphEdge, ...] | list[SchemaGraphEdge],
) -> list[dict[str, str]]:
    return [
        {
            "from_table": relationship_edge.current_table,
            "from_column": relationship_edge.current_column,
            "to_table": relationship_edge.neighbor_table,
            "to_column": relationship_edge.neighbor_column,
            "direction": relationship_edge.direction,
        }
        for relationship_edge in path_edges
    ]


def select_schema_subgraph(
    semantic_tables: list[str],
    schema_graph: SchemaGraph,
    heuristic_rules: HeuristicRules,
    *,
    outbound_hops: int,
    inbound_hops: int,
    outbound_max_neighbors_per_table: int,
    outbound_min_score: float,
    inbound_min_score: float,
    bridge_max_hops: int,
    bridge_table_min_score: float,
    metric_anchor_tables: tuple[str, ...] = (),
    dimension_anchor_tables: tuple[str, ...] = (),
    enable_fk_path_expansion: bool = True,
    fk_path_max_hops: int = 6,
    table_scores: dict[str, float] | None = None,
    column_scores: dict[str, dict[str, float]] | None = None,
    lookup_tables: frozenset[str] | set[str] | None = None,
    semantic_join_path_specs: tuple[SemanticJoinPathSpec, ...] = (),
    semantic_dependency_specs: tuple[SemanticDependencySpec, ...] = (),
) -> SchemaSubgraphSelection:
    """Selecciona el subgrafo final que se conservara en el schema podado.

    El bloque de bridges semanticos mantiene el comportamiento conservador ya
    existente. Encima de eso, `fk_path` fuerza el camino minimo entre anclas de
    metrica y dimension para no perder joins estructurales con score bajo.
    """
    resolved_table_scores = table_scores or {}
    resolved_column_scores = column_scores or {}
    resolved_metric_anchor_tables = tuple(table_name for table_name in metric_anchor_tables if table_name in schema_graph.adjacency)
    resolved_dimension_anchor_tables = tuple(table_name for table_name in dimension_anchor_tables if table_name in schema_graph.adjacency)
    seed_tables: list[str] = []
    seen_seed_tables: set[str] = set()
    for table_name in semantic_tables:
        if table_name not in schema_graph.adjacency or table_name in seen_seed_tables:
            continue
        seen_seed_tables.add(table_name)
        seed_tables.append(table_name)

    selected_tables = list(seed_tables)
    selected_table_set = set(seed_tables)
    required_columns_by_table: dict[str, set[str]] = {table_name: set() for table_name in seed_tables}
    required_foreign_keys_by_table: dict[str, set[SchemaForeignKeyReference]] = {}
    table_reasons = {table_name: "semantic" for table_name in seed_tables}

    if outbound_hops > 0 or inbound_hops > 0:
        frontier: deque[tuple[str, int, int]] = deque()
        seen_states: set[tuple[str, int, int]] = set()
        for table_name in seed_tables:
            state = (table_name, outbound_hops, inbound_hops)
            frontier.append(state)
            seen_states.add(state)

        while frontier:
            current_table, remaining_outbound_hops, remaining_inbound_hops = frontier.popleft()
            relationship_candidates: list[tuple[float, SchemaGraphEdge, tuple[str, int, int], str]] = []

            for relationship_edge in schema_graph.adjacency.get(current_table, ()):  # pragma: no branch
                if relationship_edge.direction == "outbound":
                    if remaining_outbound_hops <= 0:
                        continue
                    minimum_score = outbound_min_score
                    next_state = (
                        relationship_edge.neighbor_table,
                        remaining_outbound_hops - 1,
                        remaining_inbound_hops,
                    )
                    selection_reason = "outbound"
                else:
                    if remaining_inbound_hops <= 0:
                        continue
                    minimum_score = inbound_min_score
                    next_state = (
                        relationship_edge.neighbor_table,
                        remaining_outbound_hops,
                        remaining_inbound_hops - 1,
                    )
                    selection_reason = "inbound"

                relationship_score = get_relationship_semantic_score(
                    relationship_edge,
                    resolved_table_scores,
                    resolved_column_scores,
                    heuristic_rules,
                )
                if relationship_score < minimum_score:
                    continue

                relationship_candidates.append(
                    (
                        relationship_score,
                        relationship_edge,
                        next_state,
                        selection_reason,
                    )
                )

            relationship_candidates.sort(key=lambda item: item[0], reverse=True)

            selected_relationships: list[tuple[SchemaGraphEdge, tuple[str, int, int], str]] = []
            seen_neighbor_tables: set[str] = set()
            for (
                _,
                relationship_edge,
                next_state,
                selection_reason,
            ) in relationship_candidates:
                if relationship_edge.neighbor_table in seen_neighbor_tables:
                    continue
                seen_neighbor_tables.add(relationship_edge.neighbor_table)
                selected_relationships.append((relationship_edge, next_state, selection_reason))
                if outbound_max_neighbors_per_table > 0 and len(selected_relationships) >= outbound_max_neighbors_per_table:
                    break

            for (
                relationship_edge,
                next_state,
                selection_reason,
            ) in selected_relationships:
                mark_graph_edge_usage(
                    required_columns_by_table,
                    required_foreign_keys_by_table,
                    relationship_edge,
                )
                register_selected_table(
                    selected_tables,
                    selected_table_set,
                    relationship_edge.neighbor_table,
                )
                mark_selected_table_reason(
                    table_reasons,
                    relationship_edge.neighbor_table,
                    selection_reason,
                    heuristic_rules,
                )
                if next_state not in seen_states:
                    seen_states.add(next_state)
                    frontier.append(next_state)

    allowed_bridge_tables = build_allowed_bridge_tables(
        seed_tables,
        schema_graph,
        resolved_table_scores,
        bridge_table_min_score,
        resolved_column_scores,
    )
    for path_edges in steiner_approx_paths(
        seed_tables,
        schema_graph,
        max_hops=bridge_max_hops,
        table_scores=resolved_table_scores,
        column_scores=resolved_column_scores,
        heuristic_rules=heuristic_rules,
        allowed_tables=allowed_bridge_tables,
        lookup_tables=lookup_tables,
    ):
        for relationship_edge in path_edges:
            mark_graph_edge_usage(
                required_columns_by_table,
                required_foreign_keys_by_table,
                relationship_edge,
            )
            register_selected_table(selected_tables, selected_table_set, relationship_edge.neighbor_table)
            mark_selected_table_reason(
                table_reasons,
                relationship_edge.neighbor_table,
                "bridge",
                heuristic_rules,
            )

    fk_path_edges: list[tuple[SchemaGraphEdge, ...]] = []
    if enable_fk_path_expansion and resolved_metric_anchor_tables and resolved_dimension_anchor_tables:
        fk_path_edges = list(
            connect_anchor_pairs(
                resolved_metric_anchor_tables,
                resolved_dimension_anchor_tables,
                schema_graph,
                max_hops=fk_path_max_hops,
                table_scores=resolved_table_scores,
                column_scores=resolved_column_scores,
                heuristic_rules=heuristic_rules,
                lookup_tables=lookup_tables,
                semantic_join_path_specs=semantic_join_path_specs,
            )
        )
        for path_edges in fk_path_edges:
            # Si la ruta coincide con una declarada en semantic_join_paths, la
            # marcamos con el motivo mas fuerte ("semantic_join_path") para que
            # prevalezca sobre fk_path/bridge en priorizaciones posteriores.
            path_came_from_spec = False
            if semantic_join_path_specs:
                path_signature = tuple(
                    (
                        edge.current_table,
                        edge.neighbor_table,
                        edge.current_column,
                        edge.neighbor_column,
                    )
                    for edge in path_edges
                )
                for spec in semantic_join_path_specs:
                    spec_signature_forward = tuple(
                        (
                            e.current_table,
                            e.neighbor_table,
                            e.current_column,
                            e.neighbor_column,
                        )
                        for e in spec.edges
                    )
                    spec_signature_reverse = tuple(
                        (
                            e.neighbor_table,
                            e.current_table,
                            e.neighbor_column,
                            e.current_column,
                        )
                        for e in reversed(spec.edges)
                    )
                    if path_signature in (
                        spec_signature_forward,
                        spec_signature_reverse,
                    ):
                        path_came_from_spec = True
                        break
            reason = "semantic_join_path" if path_came_from_spec else "fk_path"
            for relationship_edge in path_edges:
                register_selected_table(selected_tables, selected_table_set, relationship_edge.current_table)
                register_selected_table(
                    selected_tables,
                    selected_table_set,
                    relationship_edge.neighbor_table,
                )
                mark_graph_edge_usage(
                    required_columns_by_table,
                    required_foreign_keys_by_table,
                    relationship_edge,
                )
                mark_selected_table_reason(
                    table_reasons,
                    relationship_edge.current_table,
                    reason,
                    heuristic_rules,
                )
                mark_selected_table_reason(
                    table_reasons,
                    relationship_edge.neighbor_table,
                    reason,
                    heuristic_rules,
                )

    for dependency_spec in semantic_dependency_specs:
        for table_name in dependency_spec.tables:
            register_selected_table(selected_tables, selected_table_set, table_name)
            mark_selected_table_reason(table_reasons, table_name, "semantic_dependency", heuristic_rules)
        for (
            table_name,
            column_names,
        ) in dependency_spec.required_columns_by_table.items():
            register_selected_table(selected_tables, selected_table_set, table_name)
            mark_selected_table_reason(table_reasons, table_name, "semantic_dependency", heuristic_rules)
            required_columns_by_table.setdefault(table_name, set()).update(column_names)
        for relationship_edge in dependency_spec.relationship_edges:
            register_selected_table(selected_tables, selected_table_set, relationship_edge.current_table)
            register_selected_table(selected_tables, selected_table_set, relationship_edge.neighbor_table)
            mark_graph_edge_usage(
                required_columns_by_table,
                required_foreign_keys_by_table,
                relationship_edge,
            )
            mark_selected_table_reason(
                table_reasons,
                relationship_edge.current_table,
                "semantic_dependency",
                heuristic_rules,
            )
            mark_selected_table_reason(
                table_reasons,
                relationship_edge.neighbor_table,
                "semantic_dependency",
                heuristic_rules,
            )

    return SchemaSubgraphSelection(
        tables=selected_tables,
        required_columns_by_table=required_columns_by_table,
        required_foreign_keys_by_table=required_foreign_keys_by_table,
        table_reasons=table_reasons,
        metric_anchor_tables=resolved_metric_anchor_tables,
        dimension_anchor_tables=resolved_dimension_anchor_tables,
        fk_path_edges=tuple(fk_path_edges),
    )


@dataclass(frozen=True)
class SemanticJoinPathSpec:
    """Ruta canonica declarada en semantic_rules.yaml (aplicable al pruning).

    Se construye a partir del YAML de reglas y se resuelve contra el schema
    real para producir edges navegables del grafo. Cuando esta disponible, el
    pruning la usa como bridge autoritativo entre un ancla metrica y un ancla
    dimension, evitando que Dijkstra elija rutas mas cortas pero semanticamente
    invalidas (p.ej. atravesar una tabla de configuracion entre dos entidades
    principales no relacionadas funcionalmente).
    """

    name: str
    from_entity: str
    to_entity: str
    edges: tuple[SchemaGraphEdge, ...]


def _parse_semantic_path_edge(raw_edge: str, schema: dict[str, object]) -> SchemaGraphEdge | None:
    """Convierte 'tabla.col = tabla.col' en un SchemaGraphEdge alineado al schema.

    La direccion (outbound/inbound) se infiere desde las FKs declaradas en el
    schema: si el lado izquierdo corresponde a una FK que apunta al derecho es
    outbound, y al reves es inbound. Si la igualdad no matchea ninguna FK se
    descarta para no introducir edges sinteticos que rompan el grafo.
    """
    if "=" not in raw_edge:
        return None
    left_side, right_side = (part.strip() for part in raw_edge.split("=", 1))
    if "." not in left_side or "." not in right_side:
        return None
    left_table, left_column = left_side.split(".", 1)
    right_table, right_column = right_side.split(".", 1)

    left_info = schema.get(left_table) if isinstance(schema.get(left_table), dict) else None
    right_info = schema.get(right_table) if isinstance(schema.get(right_table), dict) else None
    if not isinstance(left_info, dict) and not isinstance(right_info, dict):
        return None

    if isinstance(left_info, dict):
        for foreign_key in get_foreign_keys(left_info):
            if foreign_key["col"] == left_column and foreign_key["ref_table"] == right_table and foreign_key["ref_col"] == right_column:
                return SchemaGraphEdge(
                    current_table=left_table,
                    neighbor_table=right_table,
                    current_column=left_column,
                    neighbor_column=right_column,
                    direction="outbound",
                )
    if isinstance(right_info, dict):
        for foreign_key in get_foreign_keys(right_info):
            if foreign_key["col"] == right_column and foreign_key["ref_table"] == left_table and foreign_key["ref_col"] == left_column:
                return SchemaGraphEdge(
                    current_table=left_table,
                    neighbor_table=right_table,
                    current_column=left_column,
                    neighbor_column=right_column,
                    direction="inbound",
                )
    return None


def load_semantic_join_path_specs(
    rules_path: str | None,
    schema: dict[str, object],
) -> tuple[SemanticJoinPathSpec, ...]:
    """Carga y materializa las rutas canonicas declaradas para el pruning.

    Si `rules_path` es None o el archivo no existe, devuelve vacio para que el
    pruning mantenga su comportamiento actual. Solo se conservan rutas cuyos
    edges existen como FK real en el schema vigente: el YAML puede versionar
    rutas historicas que ya no aplican y no se quiere introducir edges irreales.
    """
    if not rules_path:
        return ()
    resolved_path = Path(rules_path).expanduser()
    if not resolved_path.exists():
        return ()

    raw_data = load_yaml_mapping(resolved_path, artifact_name=str(resolved_path))
    raw_section = raw_data.get("semantic_join_paths")
    if not isinstance(raw_section, list):
        return ()

    specs: list[SemanticJoinPathSpec] = []
    for raw_row in raw_section:
        if not isinstance(raw_row, dict):
            continue
        name = raw_row.get("name")
        from_entity = raw_row.get("from_entity")
        to_entity = raw_row.get("to_entity")
        raw_path = raw_row.get("path")
        if not (isinstance(name, str) and isinstance(from_entity, str) and isinstance(to_entity, str) and isinstance(raw_path, list)):
            continue

        edges: list[SchemaGraphEdge] = []
        valid_path = True
        for raw_edge in raw_path:
            if not isinstance(raw_edge, str):
                valid_path = False
                break
            edge = _parse_semantic_path_edge(raw_edge, schema)
            if edge is None:
                valid_path = False
                break
            edges.append(edge)

        if not valid_path or not edges:
            continue
        specs.append(
            SemanticJoinPathSpec(
                name=name.strip(),
                from_entity=from_entity.strip(),
                to_entity=to_entity.strip(),
                edges=tuple(edges),
            )
        )
    return tuple(specs)


def load_semantic_dependency_specs(
    rules_path: str | None,
    schema: dict[str, object],
    query: str,
    query_profile: QuerySignalProfile,
    signal_rules: QuerySignalRules,
    heuristic_rules: HeuristicRules,
    schema_graph: SchemaGraph,
    semantic_seed_tables: tuple[str, ...] | list[str] = (),
) -> tuple[SemanticDependencySpec, ...]:
    """Carga dependencias fisicas de activos semanticos relevantes para la query."""

    if not rules_path:
        return ()
    resolved_path = Path(rules_path).expanduser()
    if not resolved_path.exists():
        return ()

    raw_data = load_yaml_mapping(resolved_path, artifact_name=str(resolved_path))

    raw_metrics = raw_data.get("semantic_metrics")
    raw_entities = raw_data.get("semantic_entities")
    raw_filters = raw_data.get("semantic_filters")
    metric_rows = raw_metrics if isinstance(raw_metrics, list) else []
    entities_by_name: dict[str, dict[str, object]] = {}
    if isinstance(raw_entities, list):
        for raw_entity in raw_entities:
            if not isinstance(raw_entity, dict):
                continue
            entity_name = raw_entity.get("name")
            if isinstance(entity_name, str) and entity_name.strip():
                entities_by_name[entity_name.strip()] = raw_entity

    specs: list[SemanticDependencySpec] = []
    preserved_entity_names: set[str] = set()
    matched_metric_source_tables: set[str] = set()
    normalized_seed_tables = {table_name for table_name in semantic_seed_tables if isinstance(table_name, str) and table_name in schema}
    lookup_tables = build_lookup_tables_set(schema, signal_rules, heuristic_rules)
    for raw_metric in metric_rows:
        if not isinstance(raw_metric, dict) or not _semantic_metric_matches_query(raw_metric, query_profile, signal_rules):
            continue
        dependency_spec = _build_semantic_metric_dependency_spec(raw_metric, schema)
        if dependency_spec is not None:
            specs.append(dependency_spec)
        entity_name = raw_metric.get("entity")
        if not isinstance(entity_name, str) or not entity_name.strip() or entity_name in preserved_entity_names:
            continue
        entity_source_table = _resolve_semantic_entity_source_table(entities_by_name.get(entity_name.strip()), schema)
        if entity_source_table is not None:
            matched_metric_source_tables.add(entity_source_table)
        entity_dependency_spec = _build_semantic_entity_dependency_spec(
            entities_by_name.get(entity_name.strip()),
            schema,
            schema_graph=schema_graph,
            heuristic_rules=heuristic_rules,
        )
        if entity_dependency_spec is None:
            continue
        specs.append(entity_dependency_spec)
        preserved_entity_names.add(entity_name.strip())

    if query_profile.wants_temporal:
        for raw_entity in entities_by_name.values():
            source_table = raw_entity.get("source_table")
            entity_name = raw_entity.get("name")
            if not isinstance(source_table, str) or source_table not in normalized_seed_tables:
                continue
            if not isinstance(entity_name, str) or entity_name in preserved_entity_names:
                continue
            entity_dependency_spec = _build_semantic_entity_dependency_spec(
                raw_entity,
                schema,
                schema_graph=schema_graph,
                heuristic_rules=heuristic_rules,
            )
            if entity_dependency_spec is None:
                continue
            specs.append(entity_dependency_spec)
            preserved_entity_names.add(entity_name)

    candidate_source_tables = normalized_seed_tables | matched_metric_source_tables
    for selected_filter in _resolve_selected_semantic_filters(query, raw_filters):
        filter_dependency_spec = _build_semantic_filter_dependency_spec(
            selected_filter,
            schema,
            candidate_source_tables=candidate_source_tables,
            schema_graph=schema_graph,
            heuristic_rules=heuristic_rules,
            lookup_tables=lookup_tables,
        )
        if filter_dependency_spec is not None:
            specs.append(filter_dependency_spec)
    return tuple(specs)


def _resolve_semantic_entity_source_table(
    raw_entity: dict[str, object] | None,
    schema: dict[str, object],
) -> str | None:
    """Resuelve la tabla fuente de una entidad semantica declarada."""

    if not isinstance(raw_entity, dict):
        return None
    source_table = raw_entity.get("source_table")
    if not isinstance(source_table, str) or source_table not in schema:
        return None
    return source_table


def _resolve_selected_semantic_filters(
    query: str,
    raw_filters: object,
) -> list[SemanticFilterSelection]:
    """Materializa filtros semanticos con valor explicito dentro de la query."""

    if not isinstance(raw_filters, list):
        return []

    filter_rows: list[tuple[str, dict[str, object]]] = []
    for raw_filter in raw_filters:
        if not isinstance(raw_filter, dict):
            continue
        filter_name = raw_filter.get("name")
        if not isinstance(filter_name, str) or not filter_name.strip():
            continue
        filter_rows.append((filter_name.strip(), raw_filter))
    return resolve_semantic_filter_selections(query, filter_rows)


def _semantic_metric_matches_query(
    raw_metric: dict[str, object],
    query_profile: QuerySignalProfile,
    signal_rules: QuerySignalRules,
) -> bool:
    """Decide si una metrica YAML es suficientemente relevante para preservar dependencias."""

    metric_name = str(raw_metric.get("name") or "")
    candidate_phrases = [metric_name.replace("_", " ")]
    raw_synonyms = raw_metric.get("synonyms")
    if isinstance(raw_synonyms, list):
        candidate_phrases.extend(str(value) for value in raw_synonyms if isinstance(value, str) and value.strip())

    for candidate_phrase in candidate_phrases:
        normalized_phrase = normalize_text_for_matching(candidate_phrase, keep_underscore=True)
        if normalized_phrase and normalized_phrase in query_profile.normalized_query:
            return True

    raw_examples = raw_metric.get("examples")
    if isinstance(raw_examples, list):
        for raw_example in raw_examples:
            if not isinstance(raw_example, dict):
                continue
            question = raw_example.get("question")
            if isinstance(question, str) and normalize_text_for_matching(question, keep_underscore=True) == query_profile.normalized_query:
                return True

    metric_text = " ".join(
        str(value)
        for value in (
            raw_metric.get("name"),
            raw_metric.get("description"),
            raw_metric.get("business_definition"),
        )
        if isinstance(value, str)
    )
    metric_terms = extract_meaningful_terms(metric_text, signal_rules)
    return len(metric_terms & set(query_profile.metric_terms)) >= 2


def _build_semantic_metric_dependency_spec(
    raw_metric: dict[str, object],
    schema: dict[str, object],
) -> SemanticDependencySpec | None:
    """Convierte una metrica relevante en tablas/columnas/FKs que el prune debe conservar."""

    required_columns_by_table: dict[str, set[str]] = {}
    required_tables: set[str] = set()
    relationship_edges: list[SchemaGraphEdge] = []

    def _add_column(table_name: str, column_name: str) -> None:
        if _schema_has_column(schema, table_name, column_name):
            required_tables.add(table_name)
            required_columns_by_table.setdefault(table_name, set()).add(column_name)

    formula = raw_metric.get("formula")
    if isinstance(formula, str):
        for match in TABLE_COLUMN_RE.finditer(formula):
            _add_column(match.group(1), match.group(2))

    source_catalog = raw_metric.get("source_catalog")
    if isinstance(source_catalog, dict):
        source_table = source_catalog.get("table")
        if isinstance(source_table, str) and source_table in schema:
            required_tables.add(source_table)
            key_column = source_catalog.get("key_column")
            value_column = source_catalog.get("value_column")
            if isinstance(key_column, str):
                _add_column(source_table, key_column)
            if isinstance(value_column, str):
                _add_column(source_table, value_column)

    raw_relationships = raw_metric.get("required_relationships")
    if isinstance(raw_relationships, list):
        for raw_relationship in raw_relationships:
            if not isinstance(raw_relationship, dict):
                continue
            left_reference = raw_relationship.get("from")
            right_reference = raw_relationship.get("to")
            if not isinstance(left_reference, str) or not isinstance(right_reference, str):
                continue
            edge = _parse_semantic_path_edge(f"{left_reference} = {right_reference}", schema)
            if edge is None:
                _add_reference_columns(left_reference, required_tables, required_columns_by_table, schema)
                _add_reference_columns(right_reference, required_tables, required_columns_by_table, schema)
                continue
            relationship_edges.append(edge)

    if not required_tables and not required_columns_by_table and not relationship_edges:
        return None

    return SemanticDependencySpec(
        name=str(raw_metric.get("name") or "semantic_metric"),
        tables=frozenset(required_tables),
        required_columns_by_table={table: frozenset(columns) for table, columns in required_columns_by_table.items()},
        relationship_edges=tuple(relationship_edges),
    )


def _build_semantic_entity_dependency_spec(
    raw_entity: dict[str, object] | None,
    schema: dict[str, object],
    *,
    schema_graph: SchemaGraph,
    heuristic_rules: HeuristicRules,
) -> SemanticDependencySpec | None:
    """Preserva la infraestructura fisica requerida por el time_field de una entidad."""

    if not isinstance(raw_entity, dict):
        return None

    source_table = raw_entity.get("source_table")
    if not isinstance(source_table, str) or source_table not in schema:
        return None

    time_reference = None
    for field_name in ("time_field", "date_field"):
        candidate = raw_entity.get(field_name)
        if isinstance(candidate, str) and candidate.strip():
            time_reference = candidate.strip()
            break
    if time_reference is None or "." not in time_reference:
        return None

    required_columns_by_table: dict[str, set[str]] = {}
    required_tables: set[str] = set()
    relationship_edges: list[SchemaGraphEdge] = []
    _add_reference_columns(time_reference, required_tables, required_columns_by_table, schema)

    time_table = time_reference.split(".", 1)[0]
    if time_table != source_table and time_table in schema:
        path_edges = find_cheapest_relationship_path(
            source_table,
            time_table,
            schema_graph,
            max_hops=max(1, len(schema)),
            table_scores={},
            column_scores={},
            heuristic_rules=heuristic_rules,
            allowed_tables=None,
            lookup_tables=None,
        )
        if path_edges is not None:
            relationship_edges.extend(path_edges)

    if not required_tables and not relationship_edges:
        return None

    return SemanticDependencySpec(
        name=str(raw_entity.get("name") or source_table),
        tables=frozenset(required_tables),
        required_columns_by_table={table: frozenset(columns) for table, columns in required_columns_by_table.items()},
        relationship_edges=tuple(relationship_edges),
    )


def _build_semantic_filter_dependency_spec(
    selected_filter: SemanticFilterSelection,
    schema: dict[str, object],
    *,
    candidate_source_tables: set[str],
    schema_graph: SchemaGraph,
    heuristic_rules: HeuristicRules,
    lookup_tables: frozenset[str],
) -> SemanticDependencySpec | None:
    """Preserva columnas y ruta FK minima para filtros semanticos explicitamente seleccionados."""

    required_columns_by_table: dict[str, set[str]] = {}
    required_tables: set[str] = set()
    relationship_edges: list[SchemaGraphEdge] = []

    # Algunos filtros usan expresiones SQL simples tipo COALESCE(...). El regex
    # compartido ya extrae todas las referencias `tabla.col` relevantes.
    for table_name, column_name in TABLE_COLUMN_RE.findall(selected_filter.field):
        _add_reference_columns(f"{table_name}.{column_name}", required_tables, required_columns_by_table, schema)

    resolved_source_tables = {table_name for table_name in candidate_source_tables if table_name in schema}
    for target_table in sorted(required_tables):
        # Elegimos la mejor ruta desde cualquier tabla seed relevante para no
        # atar el filtro a una sola entidad cuando el prune detecto varias.
        best_path = _find_best_filter_dependency_path(
            resolved_source_tables,
            target_table,
            schema_graph,
            heuristic_rules=heuristic_rules,
            lookup_tables=lookup_tables,
        )
        if best_path is None:
            continue
        for relationship_edge in best_path:
            if relationship_edge not in relationship_edges:
                relationship_edges.append(relationship_edge)

    if not required_tables and not relationship_edges:
        return None

    return SemanticDependencySpec(
        name=selected_filter.name,
        tables=frozenset(required_tables),
        required_columns_by_table={table: frozenset(columns) for table, columns in required_columns_by_table.items()},
        relationship_edges=tuple(relationship_edges),
    )


def _find_best_filter_dependency_path(
    candidate_source_tables: set[str],
    target_table: str,
    schema_graph: SchemaGraph,
    *,
    heuristic_rules: HeuristicRules,
    lookup_tables: frozenset[str],
) -> list[SchemaGraphEdge] | None:
    """Selecciona la ruta relacional mas barata desde una tabla relevante hasta el filtro."""

    if not candidate_source_tables:
        return None
    if target_table in candidate_source_tables:
        return []

    best_path: list[SchemaGraphEdge] | None = None
    best_cost = float("inf")
    max_hops = max(1, len(schema_graph.adjacency))
    for source_table in sorted(candidate_source_tables):
        if source_table == target_table:
            return []
        candidate_path = find_cheapest_relationship_path(
            source_table,
            target_table,
            schema_graph,
            max_hops=max_hops,
            table_scores={},
            column_scores={},
            heuristic_rules=heuristic_rules,
            allowed_tables=None,
            lookup_tables=lookup_tables,
        )
        if candidate_path is None:
            continue
        candidate_cost = _compute_relationship_path_cost(
            candidate_path,
            goal_table=target_table,
            heuristic_rules=heuristic_rules,
            lookup_tables=lookup_tables,
        )
        if best_path is None or candidate_cost < best_cost or (candidate_cost == best_cost and len(candidate_path) < len(best_path)):
            best_path = candidate_path
            best_cost = candidate_cost
    return best_path


def _compute_relationship_path_cost(
    path_edges: list[SchemaGraphEdge],
    *,
    goal_table: str,
    heuristic_rules: HeuristicRules,
    lookup_tables: frozenset[str],
) -> float:
    """Suma el costo semantico de una ruta usando la misma regla del pathfinder."""

    total_cost = 0.0
    for relationship_edge in path_edges:
        edge_lookup_tables = None if relationship_edge.neighbor_table == goal_table else lookup_tables
        total_cost += compute_relationship_edge_cost(
            relationship_edge,
            table_scores={},
            column_scores={},
            heuristic_rules=heuristic_rules,
            lookup_tables=edge_lookup_tables,
        )
    return total_cost


def _add_reference_columns(
    reference: str,
    required_tables: set[str],
    required_columns_by_table: dict[str, set[str]],
    schema: dict[str, object],
) -> None:
    """Agrega tabla/columna si la referencia existe en el schema real."""

    if "." not in reference:
        return
    table_name, column_name = reference.split(".", 1)
    if not _schema_has_column(schema, table_name, column_name):
        return
    required_tables.add(table_name)
    required_columns_by_table.setdefault(table_name, set()).add(column_name)


def _schema_has_column(schema: dict[str, object], table_name: str, column_name: str) -> bool:
    """Comprueba si una columna existe en una tabla del schema bruto."""

    table_info = schema.get(table_name)
    if not isinstance(table_info, dict):
        return False
    return any(candidate_column == column_name for candidate_column, _column_type in get_schema_columns(table_info))


def _orient_semantic_path_edges(
    edges: tuple[SchemaGraphEdge, ...],
    start_table: str,
) -> tuple[SchemaGraphEdge, ...]:
    """Reorienta los edges para que la caminata empiece en start_table.

    Se invierte el orden y tambien la direccion de cada edge para preservar la
    semantica de navegacion. No se toca `current_column`/`neighbor_column`
    porque el par ya es simetrico; solo se flipean current/neighbor.
    """
    if not edges:
        return edges
    first_edge = edges[0]
    if first_edge.current_table == start_table:
        return edges

    reversed_edges: list[SchemaGraphEdge] = []
    for edge in reversed(edges):
        reversed_edges.append(
            SchemaGraphEdge(
                current_table=edge.neighbor_table,
                neighbor_table=edge.current_table,
                current_column=edge.neighbor_column,
                neighbor_column=edge.current_column,
                direction="inbound" if edge.direction == "outbound" else "outbound",
            )
        )
    return tuple(reversed_edges)


def find_semantic_join_path_for_anchors(
    join_path_specs: tuple[SemanticJoinPathSpec, ...],
    start_table: str,
    goal_table: str,
) -> tuple[SchemaGraphEdge, ...] | None:
    """Busca una ruta canonica declarada que conecte dos anclas.

    El matching usa directamente los nombres de tabla porque en el proyecto
    las entidades canonicas comparten nombre con su source_table. Si el par
    coincide pero en sentido inverso, se reorientan los edges antes de
    devolverlos para que partan desde start_table.
    """
    if not join_path_specs:
        return None
    for spec in join_path_specs:
        endpoints = {spec.from_entity, spec.to_entity}
        if {start_table, goal_table} != endpoints:
            continue
        return _orient_semantic_path_edges(spec.edges, start_table)
    return None


def build_table_document(
    table_name: str,
    table_info: dict[str, object],
    schema: dict[str, object],
) -> dict[str, str]:
    """Serializa una tabla completa en texto para comparacion semantica."""
    columns = get_schema_columns(table_info)
    primary_keys = get_primary_keys(table_info)
    foreign_key_metadata = build_foreign_key_metadata_lookup(table_name, table_info, schema)
    column_descriptions_lookup = get_column_descriptions(table_info)

    column_descriptions = []
    for column_name, column_type in columns:
        column_description = column_descriptions_lookup.get(column_name)
        rendered_column = f"{column_name} ({column_type})"
        if column_description:
            rendered_column = f"{rendered_column}: {column_description}"
        column_descriptions.append(rendered_column)

    foreign_key_descriptions = []
    for column_name, relations in foreign_key_metadata.items():
        rendered_relations = []
        for relation in relations:
            rendered_relations.append(f"{relation['relation_text']} ({relation['verbalized_join']})")
        foreign_key_descriptions.append(f"{column_name} -> {', '.join(rendered_relations)}")

    table_description = get_table_description(table_info)

    lines = [f"Tabla: {table_name}"]
    if table_description:
        lines.append(f"Descripcion: {table_description}")
    if column_descriptions:
        lines.append(f"Columnas: {'; '.join(column_descriptions)}")
    if primary_keys:
        lines.append(f"Primary keys: {', '.join(primary_keys)}")
    if foreign_key_descriptions:
        lines.append(f"Foreign keys: {'; '.join(foreign_key_descriptions)}")

    return {
        "id": f"table::{table_name}",
        "kind": "table",
        "table": table_name,
        "column": "",
        "text": "\n".join(lines),
    }


def build_column_documents(
    table_name: str,
    table_info: dict[str, object],
    schema: dict[str, object],
) -> list[dict[str, str]]:
    """Genera un documento por columna para captar matches mas finos."""
    columns = get_schema_columns(table_info)
    foreign_key_metadata = build_foreign_key_metadata_lookup(table_name, table_info, schema)
    table_description = get_table_description(table_info)
    column_descriptions_lookup = get_column_descriptions(table_info)

    documents: list[dict[str, str]] = []
    for column_name, column_type in columns:
        column_description = column_descriptions_lookup.get(column_name)
        foreign_key_relations = foreign_key_metadata.get(column_name, [])

        lines = [
            f"Tabla: {table_name}",
            f"Columna: {column_name}",
            f"Tipo: {column_type}",
        ]
        if table_description:
            lines.append(f"Descripcion tabla: {table_description}")
        if column_description:
            lines.append(f"Descripcion columna: {column_description}")
        if foreign_key_relations:
            relation_texts = [relation["relation_text"] for relation in foreign_key_relations]
            verbalized_relations = list(dict.fromkeys(relation["verbalized_join"] for relation in foreign_key_relations))
            lines.append(f"Foreign key: {', '.join(relation_texts)}")
            lines.append(f"Relacion de negocio: {'; '.join(verbalized_relations)}")

        documents.append(
            {
                "id": f"column::{table_name}.{column_name}",
                "kind": "column",
                "table": table_name,
                "column": column_name,
                "text": "\n".join(lines),
            }
        )

    return documents


def compute_table_score(
    table_doc_score: float | None,
    column_scores_for_table: dict[str, float],
    doc_weight: float,
    column_topn: int,
) -> float:
    top_column_scores = sorted(column_scores_for_table.values(), reverse=True)[: max(1, column_topn)]
    mean_columns = sum(top_column_scores) / len(top_column_scores) if top_column_scores else 0.0
    resolved_table_doc_score = table_doc_score if table_doc_score is not None else mean_columns
    return doc_weight * resolved_table_doc_score + (1.0 - doc_weight) * mean_columns


def adaptive_threshold(scores: np.ndarray, *, floor: float = 0.20, k_sigma: float = 1.0) -> float:
    if scores.size == 0:
        return floor
    mean_score = float(np.mean(scores))
    std_score = float(np.std(scores)) if scores.size > 1 else 0.0
    return max(floor, mean_score + k_sigma * std_score)


def select_semantic_seed_tables(
    schema: dict[str, object],
    table_scores: dict[str, float],
    table_min_score: float,
    top_k_tables: int,
    query_profile: QuerySignalProfile,
    signal_rules: QuerySignalRules,
    heuristic_rules: HeuristicRules,
) -> list[str]:
    """Elige semillas con diversidad por tabla antes de expandir relaciones."""
    if top_k_tables <= 0:
        return []

    selected_tables: list[str] = []
    selected_table_set: set[str] = set()
    sorted_tables = sorted(table_scores.items(), key=lambda item: item[1], reverse=True)
    anchor_min_score = max(
        heuristic_rules.seed_selection.anchor_min_floor,
        table_min_score - heuristic_rules.seed_selection.anchor_min_margin,
    )

    # Si la consulta no menciona documentos/carpetas/archivos, excluimos
    # tablas documentales de las semillas. Entran despues solo si son
    # alcanzadas como bridge o anchor fk_path de manera justificada.
    query_mentions_documents = bool(query_profile.query_terms & signal_rules.documental_terms)

    for table_name, table_score in sorted_tables:
        if table_name in selected_table_set or table_score < anchor_min_score:
            continue
        raw_table_info = schema.get(table_name)
        if not isinstance(raw_table_info, dict):
            continue
        if not query_mentions_documents and is_documental_table_name(table_name, signal_rules):
            continue
        if (
            get_term_overlap_count(
                f"{table_name} {get_table_description(raw_table_info) or ''}",
                query_profile.query_terms,
                signal_rules,
            )
            <= 0
        ):
            continue

        selected_table_set.add(table_name)
        selected_tables.append(table_name)
        if len(selected_tables) >= top_k_tables:
            return selected_tables

    for table_name, table_score in sorted_tables:
        if table_name in selected_table_set or table_score < table_min_score:
            continue
        if not query_mentions_documents and is_documental_table_name(table_name, signal_rules):
            continue
        selected_table_set.add(table_name)
        selected_tables.append(table_name)
        if len(selected_tables) >= top_k_tables:
            return selected_tables

    for table_name, _ in sorted_tables:
        if table_name in selected_table_set:
            continue
        if not query_mentions_documents and is_documental_table_name(table_name, signal_rules):
            continue
        selected_table_set.add(table_name)
        selected_tables.append(table_name)
        if len(selected_tables) >= top_k_tables:
            break

    return selected_tables


def build_semantic_score_context(
    ranked_documents: list[dict[str, object]],
    schema: dict[str, object],
    config: SemanticSchemaPruningConfig,
) -> SemanticScoreContext:
    table_score_doc_weight = config.table_score_doc_weight
    table_score_column_topn = config.table_score_column_topn
    min_score = config.min_score
    adaptive_threshold_k_sigma = config.adaptive_threshold_k_sigma
    adaptive_threshold_k_sigma_columns = config.adaptive_threshold_k_sigma_columns
    top_k_tables = config.top_k_tables
    if (
        table_score_doc_weight is None
        or table_score_column_topn is None
        or min_score is None
        or adaptive_threshold_k_sigma is None
        or adaptive_threshold_k_sigma_columns is None
        or top_k_tables is None
    ):
        raise ValueError("SemanticSchemaPruningConfig incompleta: faltan tunables obligatorios del runtime de prune.")

    signal_rules = config.query_signal_rules or load_query_signal_rules(str(config.signal_rules_path))
    heuristic_rules = config.heuristic_rules or load_heuristic_rules(str(config.heuristic_rules_path))
    fk_path_heuristics = resolve_fk_path_heuristics(config, heuristic_rules)
    query_profile = infer_query_signal_profile(config.query, signal_rules)
    table_doc_scores: dict[str, float] = {}
    column_scores: dict[str, dict[str, float]] = {}
    threshold_table_doc_scores: dict[str, float] = {}
    threshold_column_scores: dict[str, dict[str, float]] = {}

    for document in ranked_documents:
        table_name = document.get("table")
        if not isinstance(table_name, str) or table_name not in schema:
            continue

        score = as_float(document.get("effective_score", document.get("score", 0.0)))
        threshold_score = as_float(document.get("score", 0.0))
        document_kind = document.get("kind")
        if document_kind == "table":
            table_doc_scores[table_name] = max(table_doc_scores.get(table_name, float("-inf")), score)
            threshold_table_doc_scores[table_name] = max(
                threshold_table_doc_scores.get(table_name, float("-inf")),
                threshold_score,
            )
            continue

        if document_kind != "column":
            continue

        column_name = document.get("column")
        if not isinstance(column_name, str) or not column_name:
            continue

        table_column_scores = column_scores.setdefault(table_name, {})
        table_column_scores[column_name] = max(table_column_scores.get(column_name, float("-inf")), score)
        threshold_table_column_scores = threshold_column_scores.setdefault(table_name, {})
        threshold_table_column_scores[column_name] = max(
            threshold_table_column_scores.get(column_name, float("-inf")),
            threshold_score,
        )

    table_scores: dict[str, float] = {}
    threshold_table_scores: dict[str, float] = {}
    for table_name, raw_table_info in schema.items():
        if not isinstance(table_name, str) or not isinstance(raw_table_info, dict):
            continue

        column_type_lookup = dict(get_schema_columns(raw_table_info))
        column_descriptions_lookup = get_column_descriptions(raw_table_info)
        adjusted_column_scores: dict[str, float] = {}
        adjusted_threshold_column_scores: dict[str, float] = {}
        for column_name, raw_column_score in column_scores.get(table_name, {}).items():
            adjusted_column_scores[column_name] = min(
                1.0,
                raw_column_score
                + compute_column_signal_adjustment(
                    table_name,
                    column_name,
                    column_type_lookup.get(column_name, ""),
                    column_descriptions_lookup.get(column_name),
                    query_profile,
                    signal_rules,
                    heuristic_rules,
                ),
            )
        if adjusted_column_scores:
            column_scores[table_name] = adjusted_column_scores

        for column_name, raw_column_score in threshold_column_scores.get(table_name, {}).items():
            adjusted_threshold_column_scores[column_name] = min(
                1.0,
                raw_column_score
                + compute_column_signal_adjustment(
                    table_name,
                    column_name,
                    column_type_lookup.get(column_name, ""),
                    column_descriptions_lookup.get(column_name),
                    query_profile,
                    signal_rules,
                    heuristic_rules,
                ),
            )
        if adjusted_threshold_column_scores:
            threshold_column_scores[table_name] = adjusted_threshold_column_scores

        raw_table_doc_score = table_doc_scores.get(table_name)
        base_table_score = compute_table_score(
            (None if raw_table_doc_score is None or raw_table_doc_score == float("-inf") else raw_table_doc_score),
            column_scores.get(table_name, {}),
            table_score_doc_weight,
            table_score_column_topn,
        )
        table_scores[table_name] = min(
            1.0,
            base_table_score + compute_table_signal_adjustment(table_name, raw_table_info, query_profile, signal_rules, heuristic_rules),
        )

        raw_threshold_table_doc_score = threshold_table_doc_scores.get(table_name)
        threshold_base_table_score = compute_table_score(
            (
                None
                if raw_threshold_table_doc_score is None or raw_threshold_table_doc_score == float("-inf")
                else raw_threshold_table_doc_score
            ),
            threshold_column_scores.get(table_name, {}),
            table_score_doc_weight,
            table_score_column_topn,
        )
        threshold_table_scores[table_name] = min(
            1.0,
            threshold_base_table_score
            + compute_table_signal_adjustment(table_name, raw_table_info, query_profile, signal_rules, heuristic_rules),
        )

    table_score_values = (
        np.asarray(list(threshold_table_scores.values()), dtype=np.float32) if threshold_table_scores else np.empty(0, dtype=np.float32)
    )
    column_score_values = (
        np.asarray(
            [score for per_table_scores in threshold_column_scores.values() for score in per_table_scores.values()],
            dtype=np.float32,
        )
        if threshold_column_scores
        else np.empty(0, dtype=np.float32)
    )

    table_min_score = adaptive_threshold(
        table_score_values,
        floor=min_score,
        k_sigma=adaptive_threshold_k_sigma,
    )
    column_min_score = adaptive_threshold(
        column_score_values,
        floor=min_score,
        k_sigma=adaptive_threshold_k_sigma_columns,
    )
    metric_anchor_tables, dimension_anchor_tables = detect_query_anchor_tables(
        schema,
        table_scores,
        query_profile,
        signal_rules,
        heuristic_rules,
        max_anchors_per_role=fk_path_heuristics.max_anchors_per_role,
        min_overlap=fk_path_heuristics.anchor_min_overlap,
    )
    semantic_seed_tables = merge_seed_tables(
        select_semantic_seed_tables(
            schema,
            table_scores,
            table_min_score,
            top_k_tables,
            query_profile,
            signal_rules,
            heuristic_rules,
        ),
        metric_anchor_tables,
        dimension_anchor_tables,
    )

    return SemanticScoreContext(
        table_doc_scores=table_doc_scores,
        table_scores=table_scores,
        column_scores=column_scores,
        table_min_score=table_min_score,
        column_min_score=column_min_score,
        semantic_seed_tables=semantic_seed_tables,
        metric_anchor_tables=metric_anchor_tables,
        dimension_anchor_tables=dimension_anchor_tables,
        query_profile=query_profile,
    )


def build_pruned_schema(
    score_context: SemanticScoreContext,
    schema: dict[str, object],
    config: SemanticSchemaPruningConfig,
) -> tuple[dict[str, object], SchemaSubgraphSelection]:
    """Combina score semantico con conectividad relacional para podar el esquema."""
    top_k_columns_per_table = config.top_k_columns_per_table
    if top_k_columns_per_table is None:
        raise ValueError("SemanticSchemaPruningConfig incompleta: falta top_k_columns_per_table para construir el schema podado.")

    heuristic_rules = config.heuristic_rules or load_heuristic_rules(str(config.heuristic_rules_path))
    signal_rules = config.query_signal_rules or load_query_signal_rules(str(config.signal_rules_path))
    relationship_expansion = resolve_relationship_expansion_heuristics(config, heuristic_rules)
    fk_path = resolve_fk_path_heuristics(config, heuristic_rules)
    # Detectar tablas catalogo antes de enrutar. Las usamos como peaje
    # logico para evitar que Dijkstra elija payment_terms, moneda u otros
    # catalogos como bridge entre entidades operacionales distintas.
    lookup_tables = build_lookup_tables_set(schema, signal_rules, heuristic_rules)
    # Cargar rutas canonicas del YAML de reglas semanticas. Cuando existan, el
    # pruning usa estas rutas como bridge autoritativo en vez de confiar solo
    # en la topologia FK, impidiendo que rutas mas cortas via lookups ganen.
    schema_graph = build_schema_graph(schema)
    semantic_join_path_specs = load_semantic_join_path_specs(config.semantic_rules_path, schema)
    semantic_dependency_specs = load_semantic_dependency_specs(
        config.semantic_rules_path,
        schema,
        config.query,
        score_context.query_profile,
        signal_rules,
        heuristic_rules,
        schema_graph,
        score_context.semantic_seed_tables,
    )
    selection = select_schema_subgraph(
        score_context.semantic_seed_tables,
        schema_graph,
        heuristic_rules,
        outbound_hops=relationship_expansion.outbound_hops,
        inbound_hops=relationship_expansion.inbound_hops,
        outbound_max_neighbors_per_table=relationship_expansion.max_neighbors_per_table,
        outbound_min_score=relationship_expansion.outbound_min_score,
        inbound_min_score=relationship_expansion.inbound_min_score,
        bridge_max_hops=relationship_expansion.bridge_max_hops,
        bridge_table_min_score=relationship_expansion.bridge_table_min_score,
        metric_anchor_tables=score_context.metric_anchor_tables,
        dimension_anchor_tables=score_context.dimension_anchor_tables,
        enable_fk_path_expansion=fk_path.enabled,
        fk_path_max_hops=fk_path.max_hops,
        table_scores=score_context.table_scores,
        column_scores=score_context.column_scores,
        lookup_tables=lookup_tables,
        semantic_join_path_specs=semantic_join_path_specs,
        semantic_dependency_specs=semantic_dependency_specs,
    )
    selected_tables = selection.tables
    required_columns_by_table = selection.required_columns_by_table

    pruned_schema: dict[str, object] = {}
    for table_name in selected_tables:
        raw_table_info = schema.get(table_name)
        if not isinstance(raw_table_info, dict):
            continue

        columns = get_schema_columns(raw_table_info)
        primary_keys = get_primary_keys(raw_table_info)
        available_foreign_keys = get_foreign_keys(raw_table_info)
        column_descriptions_lookup = get_column_descriptions(raw_table_info)
        required_foreign_keys = selection.required_foreign_keys_by_table.get(table_name, set())
        available_foreign_key_columns = {foreign_key["col"] for foreign_key in available_foreign_keys}

        foreign_keys: list[SchemaForeignKey] = []
        for foreign_key in available_foreign_keys:
            foreign_key_reference = build_foreign_key_reference(table_name, foreign_key)
            if foreign_key_reference is None or foreign_key_reference not in required_foreign_keys:
                continue
            foreign_keys.append(foreign_key)

        retained_foreign_key_columns = {foreign_key["col"] for foreign_key in foreign_keys}
        ranked_table_columns = sorted(
            score_context.column_scores.get(table_name, {}).items(),
            key=lambda item: item[1],
            reverse=True,
        )

        unused_foreign_key_min_score = max(
            score_context.column_min_score,
            relationship_expansion.outbound_min_score + relationship_expansion.unused_foreign_key_score_margin,
        )
        selected_columns: list[str] = []
        for column_name, score in ranked_table_columns:
            if score < score_context.column_min_score:
                continue
            if (
                column_name in available_foreign_key_columns
                and column_name not in retained_foreign_key_columns
                and score < unused_foreign_key_min_score
            ):
                continue

            selected_columns.append(column_name)
            if len(selected_columns) >= top_k_columns_per_table:
                break

        selected_column_names = set(selected_columns)
        selected_column_names.update(primary_keys)
        selected_column_names.update(required_columns_by_table.get(table_name, set()))

        rendered_columns = []
        for column_name, column_type in columns:
            if column_name not in selected_column_names:
                continue

            rendered_column: dict[str, object] = {
                "name": column_name,
                "type": column_type,
                "match_score": round(
                    score_context.column_scores.get(table_name, {}).get(column_name, 0.0),
                    6,
                ),
            }
            column_description = column_descriptions_lookup.get(column_name)
            if column_description:
                rendered_column["description"] = column_description
            rendered_columns.append(rendered_column)

        pruned_schema[table_name] = {
            "match_score": round(score_context.table_scores.get(table_name, 0.0), 6),
            "selection_reason": selection.table_reasons.get(table_name, "semantic"),
            "description": get_table_description(raw_table_info),
            "columns": rendered_columns,
            "primary_keys": primary_keys,
            "foreign_keys": foreign_keys,
        }

    return pruned_schema, selection
