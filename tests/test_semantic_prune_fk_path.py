#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import tempfile
from pathlib import Path

import yaml

from nl2sql.semantic_prune import SemanticSchemaPruningConfig
from nl2sql.semantic_prune.schema_logic import (
    build_schema_graph,
    connect_anchor_pairs,
    infer_query_signal_profile,
    is_documental_table_name,
    load_heuristic_rules,
    load_query_signal_rules,
    load_semantic_dependency_specs,
    load_semantic_join_path_specs,
    pick_anchor_tables,
)

QUERY = "cual es el promedio de entity_a por entity_c en el ultimo ano?"

SCHEMA_FIXTURE: dict[str, object] = {
    "entity_c": {
        "description": "Entidad agrupadora del dominio generico.",
        "columns": [("id", "BIGINT"), ("display_name", "VARCHAR")],
        "primary_keys": ["id"],
        "foreign_keys": [],
    },
    "entity_c_site": {
        "description": "Sitio asociado a la entidad agrupadora.",
        "columns": [("id", "BIGINT"), ("entity_c_id", "BIGINT")],
        "primary_keys": ["id"],
        "foreign_keys": [{"col": "entity_c_id", "ref_table": "entity_c", "ref_col": "id"}],
    },
    "bridge_contact": {
        "description": "Tabla puente entre el registro comercial y el sitio agrupador.",
        "columns": [("id", "BIGINT"), ("entity_c_site_id", "BIGINT")],
        "primary_keys": ["id"],
        "foreign_keys": [{"col": "entity_c_site_id", "ref_table": "entity_c_site", "ref_col": "id"}],
    },
    "entity_b": {
        "description": "Registro comercial generico asociado a la entidad agrupadora.",
        "columns": [("id", "BIGINT"), ("bridge_contact_id", "BIGINT"), ("status_b_id", "BIGINT"), ("requested_at", "DATE")],
        "primary_keys": ["id"],
        "foreign_keys": [
            {"col": "bridge_contact_id", "ref_table": "bridge_contact", "ref_col": "id"},
            {"col": "status_b_id", "ref_table": "status_b", "ref_col": "id"},
        ],
    },
    "entity_b_version": {
        "description": "Version intermedia asociada al registro comercial.",
        "columns": [("id", "BIGINT"), ("entity_b_id", "BIGINT")],
        "primary_keys": ["id"],
        "foreign_keys": [{"col": "entity_b_id", "ref_table": "entity_b", "ref_col": "id"}],
    },
    "entity_a": {
        "description": "Registro operativo principal sobre el que se agregan metricas.",
        "columns": [("id", "BIGINT"), ("entity_b_version_id", "BIGINT"), ("created_at", "DATETIME")],
        "primary_keys": ["id"],
        "foreign_keys": [{"col": "entity_b_version_id", "ref_table": "entity_b_version", "ref_col": "id"}],
    },
    "status_b": {
        "description": "Catalogo de estados comerciales.",
        "columns": [("id", "BIGINT"), ("name", "VARCHAR")],
        "primary_keys": ["id"],
        "foreign_keys": [],
    },
}


def _write_rules_file(directory: str, payload: dict[str, object]) -> Path:
    rules_path = Path(directory) / "semantic_rules.yaml"
    rules_path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return rules_path


def _load_signal_rules() -> object:
    config = SemanticSchemaPruningConfig(query=QUERY)
    return load_query_signal_rules(str(config.signal_rules_path))


def _load_heuristic_rules() -> object:
    config = SemanticSchemaPruningConfig(query=QUERY)
    return load_heuristic_rules(str(config.heuristic_rules_path))


def test_connect_anchor_pairs_forces_minimum_fk_path_between_generic_anchors() -> None:
    schema_graph = build_schema_graph(SCHEMA_FIXTURE)
    heuristic_rules = _load_heuristic_rules()

    paths = connect_anchor_pairs(
        ("entity_a",),
        ("entity_c",),
        schema_graph,
        max_hops=6,
        table_scores={},
        column_scores={},
        heuristic_rules=heuristic_rules,
        lookup_tables=frozenset(),
    )

    assert len(paths) == 1
    assert [(edge.current_table, edge.neighbor_table) for edge in paths[0]] == [
        ("entity_a", "entity_b_version"),
        ("entity_b_version", "entity_b"),
        ("entity_b", "bridge_contact"),
        ("bridge_contact", "entity_c_site"),
        ("entity_c_site", "entity_c"),
    ]


def test_load_semantic_join_path_specs_uses_canonical_generic_path() -> None:
    payload = {
        "semantic_join_paths": [
            {
                "name": "entity_c_to_a_via_b",
                "from_entity": "entity_c",
                "to_entity": "entity_a",
                "path": [
                    "entity_c.id = entity_c_site.entity_c_id",
                    "entity_c_site.id = bridge_contact.entity_c_site_id",
                    "bridge_contact.id = entity_b.bridge_contact_id",
                    "entity_b.id = entity_b_version.entity_b_id",
                    "entity_b_version.id = entity_a.entity_b_version_id",
                ],
            }
        ]
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        rules_path = _write_rules_file(tmpdir, payload)
        specs = load_semantic_join_path_specs(str(rules_path), SCHEMA_FIXTURE)

    assert len(specs) == 1
    assert specs[0].name == "entity_c_to_a_via_b"

    schema_graph = build_schema_graph(SCHEMA_FIXTURE)
    heuristic_rules = _load_heuristic_rules()
    paths = connect_anchor_pairs(
        ("entity_a",),
        ("entity_c",),
        schema_graph,
        max_hops=6,
        table_scores={},
        column_scores={},
        heuristic_rules=heuristic_rules,
        lookup_tables=frozenset(),
        semantic_join_path_specs=specs,
    )

    assert len(paths) == 1
    assert [(edge.current_table, edge.neighbor_table) for edge in paths[0]] == [
        ("entity_a", "entity_b_version"),
        ("entity_b_version", "entity_b"),
        ("entity_b", "bridge_contact"),
        ("bridge_contact", "entity_c_site"),
        ("entity_c_site", "entity_c"),
    ]


def test_load_semantic_dependency_specs_keeps_required_relationships_for_generic_metric() -> None:
    query = "cual es el promedio de entity_b archivadas por entity_c en el ultimo ano?"
    payload = {
        "semantic_metrics": [
            {
                "name": "metric_count_b_lost",
                "description": "Cantidad de entity_b archivadas.",
                "formula": "count_distinct(case when status_b.name = 'Archived' then entity_b.id end)",
                "synonyms": ["entity_b archivadas"],
                "source_catalog": {
                    "table": "status_b",
                    "key_column": "id",
                    "value_column": "name",
                },
                "required_relationships": [
                    {
                        "from": "entity_b.status_b_id",
                        "to": "status_b.id",
                    }
                ],
            }
        ]
    }

    signal_rules = _load_signal_rules()
    heuristic_rules = _load_heuristic_rules()
    query_profile = infer_query_signal_profile(query, signal_rules)
    schema_graph = build_schema_graph(SCHEMA_FIXTURE)

    with tempfile.TemporaryDirectory() as tmpdir:
        rules_path = _write_rules_file(tmpdir, payload)
        specs = load_semantic_dependency_specs(
            str(rules_path),
            SCHEMA_FIXTURE,
            query,
            query_profile,
            signal_rules,
            heuristic_rules,
            schema_graph,
        )

    assert len(specs) == 1
    spec = specs[0]
    assert spec.name == "metric_count_b_lost"
    assert spec.tables == frozenset({"entity_b", "status_b"})
    assert spec.required_columns_by_table["entity_b"] >= frozenset({"id"})
    assert spec.required_columns_by_table["status_b"] >= frozenset({"id", "name"})
    assert any(edge.current_table == "entity_b" and edge.neighbor_table == "status_b" for edge in spec.relationship_edges)


def test_load_semantic_dependency_specs_keeps_entity_time_field_path_for_relevant_metric() -> None:
    query = "cual es el total de entity_a en el ultimo ano?"
    payload = {
        "semantic_entities": [
            {
                "name": "entity_a",
                "source_table": "entity_a",
                "key": "entity_a.id",
                "time_field": "entity_b.requested_at",
            }
        ],
        "semantic_metrics": [
            {
                "name": "metric_count_a",
                "description": "Cantidad de entity_a.",
                "entity": "entity_a",
                "formula": "count_distinct(entity_a.id)",
                "synonyms": ["total de entity_a"],
            }
        ],
    }

    signal_rules = _load_signal_rules()
    heuristic_rules = _load_heuristic_rules()
    query_profile = infer_query_signal_profile(query, signal_rules)
    schema_graph = build_schema_graph(SCHEMA_FIXTURE)

    with tempfile.TemporaryDirectory() as tmpdir:
        rules_path = _write_rules_file(tmpdir, payload)
        specs = load_semantic_dependency_specs(
            str(rules_path),
            SCHEMA_FIXTURE,
            query,
            query_profile,
            signal_rules,
            heuristic_rules,
            schema_graph,
            ["entity_a"],
        )

    assert len(specs) == 2
    entity_spec = next(spec for spec in specs if spec.name == "entity_a")
    assert entity_spec.tables == frozenset({"entity_b"})
    assert entity_spec.required_columns_by_table["entity_b"] >= frozenset({"requested_at"})
    assert [(edge.current_table, edge.neighbor_table) for edge in entity_spec.relationship_edges] == [
        ("entity_a", "entity_b_version"),
        ("entity_b_version", "entity_b"),
    ]


def test_load_semantic_dependency_specs_keeps_selected_filter_path_for_external_table() -> None:
    query = "cual es el total de entity_a con status b = 7?"
    payload = {
        "semantic_entities": [
            {
                "name": "entity_a",
                "source_table": "entity_a",
                "key": "entity_a.id",
            }
        ],
        "semantic_metrics": [
            {
                "name": "metric_count_a",
                "description": "Cantidad de entity_a.",
                "entity": "entity_a",
                "formula": "count_distinct(entity_a.id)",
                "synonyms": ["total de entity_a"],
            }
        ],
        "semantic_filters": [
            {
                "name": "by_status_b",
                "field": "status_b.id",
                "operator": "equals",
                "synonyms": ["status b"],
            }
        ],
    }

    signal_rules = _load_signal_rules()
    heuristic_rules = _load_heuristic_rules()
    query_profile = infer_query_signal_profile(query, signal_rules)
    schema_graph = build_schema_graph(SCHEMA_FIXTURE)

    with tempfile.TemporaryDirectory() as tmpdir:
        rules_path = _write_rules_file(tmpdir, payload)
        specs = load_semantic_dependency_specs(
            str(rules_path),
            SCHEMA_FIXTURE,
            query,
            query_profile,
            signal_rules,
            heuristic_rules,
            schema_graph,
            ["entity_a"],
        )

    assert {spec.name for spec in specs} >= {"metric_count_a", "by_status_b"}
    filter_spec = next(spec for spec in specs if spec.name == "by_status_b")
    assert filter_spec.tables == frozenset({"status_b"})
    assert filter_spec.required_columns_by_table["status_b"] >= frozenset({"id"})
    assert [(edge.current_table, edge.neighbor_table) for edge in filter_spec.relationship_edges] == [
        ("entity_a", "entity_b_version"),
        ("entity_b_version", "entity_b"),
        ("entity_b", "status_b"),
    ]


def test_load_semantic_dependency_specs_keeps_filter_dependencies_without_metrics_section() -> None:
    query = "entity_a con status b = 7"
    payload = {
        "semantic_filters": [
            {
                "name": "by_status_b",
                "field": "status_b.id",
                "operator": "equals",
                "synonyms": ["status b"],
            }
        ]
    }

    signal_rules = _load_signal_rules()
    heuristic_rules = _load_heuristic_rules()
    query_profile = infer_query_signal_profile(query, signal_rules)
    schema_graph = build_schema_graph(SCHEMA_FIXTURE)

    with tempfile.TemporaryDirectory() as tmpdir:
        rules_path = _write_rules_file(tmpdir, payload)
        specs = load_semantic_dependency_specs(
            str(rules_path),
            SCHEMA_FIXTURE,
            query,
            query_profile,
            signal_rules,
            heuristic_rules,
            schema_graph,
            ["entity_a"],
        )

    assert len(specs) == 1
    filter_spec = specs[0]
    assert filter_spec.name == "by_status_b"
    assert filter_spec.tables == frozenset({"status_b"})
    assert [(edge.current_table, edge.neighbor_table) for edge in filter_spec.relationship_edges] == [
        ("entity_a", "entity_b_version"),
        ("entity_b_version", "entity_b"),
        ("entity_b", "status_b"),
    ]


def test_pick_anchor_tables_excludes_documental_generic_tables() -> None:
    schema_with_folders: dict[str, object] = {
        **SCHEMA_FIXTURE,
        "carpeta_entity_a": {
            "description": "Vinculo documental para entity_a.",
            "columns": [("id", "BIGINT"), ("entity_a_id", "BIGINT")],
            "primary_keys": ["id"],
            "foreign_keys": [{"col": "entity_a_id", "ref_table": "entity_a", "ref_col": "id"}],
        },
        "carpeta_entity_c": {
            "description": "Vinculo documental para entity_c.",
            "columns": [("id", "BIGINT"), ("entity_c_id", "BIGINT")],
            "primary_keys": ["id"],
            "foreign_keys": [{"col": "entity_c_id", "ref_table": "entity_c", "ref_col": "id"}],
        },
    }
    table_scores = {
        "entity_a": 0.82,
        "entity_c": 0.77,
        "carpeta_entity_a": 0.60,
        "carpeta_entity_c": 0.55,
    }

    signal_rules = _load_signal_rules()
    heuristic_rules = _load_heuristic_rules()
    query_profile = infer_query_signal_profile(QUERY, signal_rules)
    metric_anchors = pick_anchor_tables(
        query_profile.metric_terms,
        schema_with_folders,
        table_scores,
        signal_rules,
        heuristic_rules,
        max_anchors=2,
        min_overlap=1,
    )
    dimension_anchors = pick_anchor_tables(
        query_profile.dimension_terms,
        schema_with_folders,
        table_scores,
        signal_rules,
        heuristic_rules,
        max_anchors=2,
        min_overlap=1,
        excluded_tables=set(metric_anchors),
    )

    assert is_documental_table_name("carpeta_entity_a", signal_rules) is True
    assert is_documental_table_name("carpeta_entity_c", signal_rules) is True
    assert "carpeta_entity_a" not in metric_anchors
    assert "carpeta_entity_c" not in dimension_anchors
