#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from nl2sql.sql_solver_generator.business_rules import BusinessRule, apply_business_rules
from nl2sql.sql_solver_generator.rules_loader import load_business_rules
from nl2sql.sql_solver_generator.spec_model import SQLDimension, SQLMetric, SQLQuerySpec


def _semantic_contract(sql_business_rules: list[dict[str, object]]) -> dict[str, object]:
    return {
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
                "semantic_sql_business_rules": sql_business_rules,
            },
        }
    }


def test_forbid_column_reporta_error() -> None:
    rules = [BusinessRule(id="r1", type="forbid_column", params={"column": "entity_b.entity_c_id"})]
    spec = SQLQuerySpec(
        query_type="scalar_metric",
        dialect="tsql",
        base_entity="entity_b",
        base_table="entity_b",
        selected_dimensions=[SQLDimension(name="group_key", source="entity_b.entity_c_id", entity="entity_b", type="categorical")],
    )
    result = apply_business_rules(rules=rules, spec=spec, pruned_schema={})
    assert any(issue.code == "forbidden_column" for issue in result.errors)


def test_inject_filter_if_column_present() -> None:
    rules = [
        BusinessRule(
            id="r2",
            type="inject_filter_if_column_present",
            params={
                "apply_when_table_has_column": "id",
                "apply_to_tables_with_role": ["entity"],
                "filter": {"operator": ">=", "value": 0},
            },
        )
    ]
    schema = {"entity_c": {"columns": {"id": {}}, "role": "entity"}}
    spec = SQLQuerySpec(query_type="scalar_metric", dialect="tsql", base_entity="entity_c", base_table="entity_c")
    result = apply_business_rules(rules=rules, spec=spec, pruned_schema=schema)
    assert any(filter_obj.field == "entity_c.id" for filter_obj in result.filters_to_inject)


def test_requires_normalization_before_sum_warns() -> None:
    rules = [
        BusinessRule(
            id="r3",
            type="require_normalization_before_sum",
            params={
                "affected_columns": ["entity_b_version.total_amount"],
                "required_filter_matches": ["*.moneda*"],
            },
        )
    ]
    spec = SQLQuerySpec(
        query_type="grouped_metric",
        dialect="tsql",
        base_entity="entity_b_version",
        base_table="entity_b_version",
        selected_metrics=[
            SQLMetric(
                name="amount_total",
                formula="SUM(entity_b_version.total_amount)",
                entity="entity_b_version",
                aggregation_level="grouped",
            )
        ],
    )
    result = apply_business_rules(rules=rules, spec=spec, pruned_schema={})
    assert any(issue.code == "multicurrency_sum_requires_normalization" for issue in result.warnings)


def test_load_business_rules_desde_semantic_rules() -> None:
    rules = load_business_rules(
        _semantic_contract(
            [
                {
                    "id": "forbid_custom_column",
                    "type": "forbid_column",
                    "column": "entity_b.entity_c_id",
                }
            ]
        )
    )

    assert any(rule.id == "forbid_custom_column" and rule.type == "forbid_column" for rule in rules)
