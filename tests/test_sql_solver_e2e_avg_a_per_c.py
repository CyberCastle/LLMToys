#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from unittest.mock import patch

from nl2sql.sql_solver_generator import SolverConfig, SolverInput, run_sql_solver
from nl2sql.sql_solver_generator.spec_model import SQLJoinEdge, SQLJoinPlan, SQLMetric, SQLQuerySpec, SQLTimeFilter

FAKE_SQL = """
WITH base AS (
    SELECT entity_c.id, COUNT(DISTINCT entity_a.id) AS grouped_measure
    FROM entity_c
    INNER JOIN entity_a ON entity_c.id = entity_a.entity_c_id
    WHERE entity_a.created_at >= DATEADD(year, -1, CAST(GETDATE() AS date))
    GROUP BY entity_c.id
)
SELECT AVG(CAST(grouped_measure AS float)) AS avg_value FROM base
""".strip()


def test_avg_a_per_c() -> None:
    def fake(*, model_name, **_):
        return (
            SQLQuerySpec(
                query_type="derived_metric",
                dialect="tsql",
                base_entity="entity_c",
                base_table="entity_c",
                selected_metrics=[
                    SQLMetric(
                        name="metric_count_a",
                        formula="COUNT(DISTINCT entity_a.id)",
                        entity="entity_a",
                        aggregation_level="derived_two_level",
                    )
                ],
                time_filter=SQLTimeFilter(
                    field="entity_a.created_at",
                    operator=">=",
                    value="ultimo_ano",
                    resolved_expression="DATEADD(year, -1, CAST(GETDATE() AS date))",
                ),
                join_plan=[
                    SQLJoinPlan(
                        path_name="entity_c_to_a_via_b",
                        joins=[
                            SQLJoinEdge(
                                left_table="entity_c",
                                left_column="id",
                                right_table="entity_a",
                                right_column="entity_c_id",
                                join_type="inner",
                            )
                        ],
                    )
                ],
                base_group_by=["entity_c.id"],
                final_group_by=[],
                post_aggregation="avg",
            ),
            FAKE_SQL,
            {"finish_reason": "stop", "prompt_tokens": 1200, "generated_tokens": 220, "wall_time_seconds": 3.5},
        )

    plan = {
        "semantic_plan": {
            "compiled_plan": {
                "query": "q",
                "intent": "post_aggregated_metric",
                "base_entity": "entity_c",
                "grain": "entity_c",
                "measure": {},
                "group_by": ["entity_c.id"],
                "join_path": [],
                "required_tables": ["entity_a"],
            }
        }
    }
    schema = {
        "entity_c": {"columns": {"id": {}}},
        "entity_a": {"columns": {"id": {}, "entity_c_id": {}, "created_at": {}}},
    }
    rules = {
        "semantic_contract": {
            "business_invariants": {
                "semantic_models": [],
                "semantic_entities": [],
                "semantic_dimensions": [],
                "semantic_metrics": [],
                "semantic_filters": [],
                "semantic_business_rules": [],
                "semantic_relationships": [],
                "semantic_constraints": [],
                "semantic_join_paths": [],
                "semantic_derived_metrics": [],
            },
            "retrieval_heuristics": {
                "semantic_synonyms": {},
                "semantic_examples": [],
            },
            "sql_safety": {
                "execution_safety": {"forbidden_keywords": []},
                "hard_join_blacklist": [],
                "semantic_sql_business_rules": [],
            },
        }
    }

    with patch("nl2sql.sql_solver_generator.llm_router.generate_spec_and_sql", side_effect=fake):
        result = run_sql_solver(
            SolverInput(semantic_plan=plan, pruned_schema=schema, semantic_rules=rules),
            SolverConfig(dialect="tsql"),
        )

    assert result.metadata.model_used == SolverConfig().model
    assert "entity_c_to_a_via_b" in result.metadata.join_paths_used
    assert result.metadata.finish_reason == "stop"
    assert result.metadata.prompt_tokens == 1200
    assert result.sql_query_spec.query_type == "derived_metric"
    assert result.sql_query_spec.post_aggregation == "avg"
    upper = result.sql_final.upper()
    assert "WITH BASE" in upper or "SELECT AVG" in upper
    assert "DATEADD(YEAR, -1" in upper
    assert result.issues == []
