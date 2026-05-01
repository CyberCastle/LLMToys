#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from nl2sql.sql_solver_generator.contracts import SolverInput
from nl2sql.sql_solver_generator.stages.plan_normalization_stage import (
    run_plan_normalization_stage,
)

MINIMAL_SEMANTIC_RULES = {
    "semantic_contract": {
        "business_invariants": {},
        "retrieval_heuristics": {},
        "sql_safety": {},
    }
}


def test_plan_normalization_preserves_pruned_schema_contract_without_repairing() -> None:
    semantic_plan = {
        "compiled_plan": {
            "query": "promedio de registros_b archivados por entidad_c",
            "intent": "post_aggregated_metric",
            "base_entity": "entity_b",
            "grain": "entity_b.id",
            "measure": {
                "name": "metric_count_b_lost",
                "formula": "count_distinct(case when status_b.name = 'Archived' then entity_b.id end)",
                "source_table": "entity_b",
            },
            "group_by": ["entity_c.id"],
            "time_filter": {
                "field": "entity_b.requested_at",
                "operator": ">=",
                "value": "today - 1 year",
            },
            "join_path": [
                "entity_b.bridge_contact_id = bridge_contact.id",
                "bridge_contact.entity_c_site_id = entity_c_site.id",
                "entity_c_site.entity_c_id = entity_c.id",
                "entity_b.status_b_id = status_b.id",
            ],
            "required_tables": [
                "entity_b",
                "bridge_contact",
                "entity_c_site",
                "entity_c",
                "status_b",
            ],
        }
    }
    pruned_schema = {
        "pruned_schema": {
            "entity_b": {
                "columns": [
                    {"name": "id", "type": "BIGINT"},
                    {"name": "requested_at", "type": "DATE"},
                    {"name": "bridge_contact_id", "type": "BIGINT"},
                ],
                "primary_keys": ["id"],
                "foreign_keys": [
                    {
                        "col": "bridge_contact_id",
                        "ref_table": "bridge_contact",
                        "ref_col": "id",
                    }
                ],
            },
            "bridge_contact": {
                "columns": [
                    {"name": "id", "type": "BIGINT"},
                    {"name": "entity_c_site_id", "type": "BIGINT"},
                ],
                "primary_keys": ["id"],
                "foreign_keys": [{"col": "entity_c_site_id", "ref_table": "entity_c_site", "ref_col": "id"}],
            },
            "entity_c_site": {
                "columns": [
                    {"name": "id", "type": "BIGINT"},
                    {"name": "entity_c_id", "type": "BIGINT"},
                ],
                "primary_keys": ["id"],
                "foreign_keys": [{"col": "entity_c_id", "ref_table": "entity_c", "ref_col": "id"}],
            },
            "entity_c": {
                "columns": [{"name": "id", "type": "BIGINT"}],
                "primary_keys": ["id"],
                "foreign_keys": [],
            },
        }
    }

    normalized = run_plan_normalization_stage(
        SolverInput(
            semantic_plan=semantic_plan,
            pruned_schema=pruned_schema,
            semantic_rules=MINIMAL_SEMANTIC_RULES,
        )
    )

    assert "status_b" not in normalized.pruned_schema
    assert "status_b_id" not in normalized.pruned_schema["entity_b"]["columns"]
    assert normalized.semantic_plan["compiled_plan"] == semantic_plan["compiled_plan"]
    assert normalized.semantic_plan["retrieved_candidates"] == {}
