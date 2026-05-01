#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from nl2sql.sql_solver_generator import get_dialect
from nl2sql.sql_solver_generator.spec_model import SQLFilter, SQLQuerySpec
from nl2sql.sql_solver_generator.stages.validation_stage import ValidationStage


def _mk_stage() -> ValidationStage:
    return ValidationStage(
        dialect=get_dialect("tsql"),
        schema={"entity_c": {"columns": {"id": {}}}},
        business_rules=[],
        forbidden_keywords=frozenset({"DROP", "DELETE", "UPDATE"}),
    )


def test_bloquea_update() -> None:
    issues = _mk_stage().validate_full(
        SQLQuerySpec(
            query_type="scalar_metric",
            dialect="tsql",
            base_entity="entity_c",
            base_table="entity_c",
        ),
        "UPDATE entity_c SET display_name='x'",
    )
    assert any(issue.code == "forbidden_keyword" and issue.context.get("keyword") == "UPDATE" for issue in issues)


def test_bloquea_cross_join() -> None:
    issues = _mk_stage().validate_full(
        SQLQuerySpec(
            query_type="scalar_metric",
            dialect="tsql",
            base_entity="entity_c",
            base_table="entity_c",
        ),
        "SELECT 1 FROM entity_c CROSS JOIN entity_c c2",
    )
    assert any(issue.code == "cross_join_forbidden" for issue in issues)


def test_acepta_select_simple() -> None:
    issues = _mk_stage().validate_full(
        SQLQuerySpec(
            query_type="scalar_metric",
            dialect="tsql",
            base_entity="entity_c",
            base_table="entity_c",
        ),
        "SELECT TOP 10 id FROM entity_c",
    )
    assert issues == []


def test_bloquea_where_no_declarado_en_cte() -> None:
    issues = _mk_stage().validate_full(
        SQLQuerySpec(
            query_type="derived_metric",
            dialect="tsql",
            base_entity="entity_c",
            base_table="entity_c",
        ),
        "WITH base AS (SELECT id FROM entity_c) SELECT COUNT(*) FROM base WHERE id = 0",
    )
    assert any(issue.code == "undeclared_where_column" and issue.context.get("column") == "id" for issue in issues)


def test_acepta_where_declarado_en_selected_filters() -> None:
    issues = _mk_stage().validate_full(
        SQLQuerySpec(
            query_type="scalar_metric",
            dialect="tsql",
            base_entity="entity_c",
            base_table="entity_c",
            selected_filters=[SQLFilter(name="by_entity_c", field="entity_c.id", operator="=", value=1)],
        ),
        "SELECT c.id FROM entity_c AS c WHERE c.id = 1",
    )
    assert issues == []


def test_detail_listing_requiere_limit() -> None:
    issues = _mk_stage().validate_full(
        SQLQuerySpec(
            query_type="detail_listing",
            dialect="tsql",
            base_entity="entity_c",
            base_table="entity_c",
        ),
        "SELECT id FROM entity_c",
    )
    assert any(issue.code == "detail_listing_missing_limit" for issue in issues)


def test_no_valida_estructura_del_spec_contra_schema() -> None:
    issues = _mk_stage().validate_full(
        SQLQuerySpec(
            query_type="scalar_metric",
            dialect="tsql",
            base_entity="entity_b",
            base_table="entity_b",
        ),
        "SELECT TOP 10 id FROM entity_c",
    )
    assert "missing_base_table:entity_b" not in issues
    assert "missing_table:entity_b" not in issues


def test_bloquea_columnas_desconocidas_en_select() -> None:
    stage = ValidationStage(
        dialect=get_dialect("tsql"),
        schema={"entity_b": {"columns": {"id": {}, "requested_at": {}}}},
        business_rules=[],
        forbidden_keywords=frozenset({"DROP", "DELETE", "UPDATE"}),
    )

    issues = stage.validate_full(
        SQLQuerySpec(
            query_type="derived_metric",
            dialect="tsql",
            base_entity="entity_b",
            base_table="entity_b",
        ),
        "WITH base AS (SELECT metric_alpha / NULLIF(metric_beta, 0) AS ratio_value FROM entity_b) SELECT AVG(CAST(base.ratio_value AS DECIMAL(18,2))) FROM base",
    )

    assert any(issue.code == "unknown_bare_column" and issue.context.get("column") == "metric_alpha" for issue in issues)
    assert any(issue.code == "unknown_bare_column" and issue.context.get("column") == "metric_beta" for issue in issues)


def test_derived_metric_rechaza_subquery_sin_group_by_base() -> None:
    stage = ValidationStage(
        dialect=get_dialect("tsql"),
        schema={
            "entity_b": {"columns": {"id": {}, "entity_c_id": {}}},
            "entity_c": {"columns": {"id": {}}},
        },
        business_rules=[],
        forbidden_keywords=frozenset({"DROP", "DELETE", "UPDATE"}),
    )

    issues = stage.validate_full(
        SQLQuerySpec(
            query_type="derived_metric",
            dialect="tsql",
            base_entity="entity_b",
            base_table="entity_b",
            base_group_by=["entity_c.id"],
            post_aggregation="avg",
        ),
        "SELECT AVG(CAST(base.metric AS DECIMAL(18,2))) AS avg_value FROM (SELECT COUNT(DISTINCT entity_b.id) AS metric FROM entity_b JOIN entity_c ON entity_b.entity_c_id = entity_c.id) AS base",
    )

    assert any(issue.code == "derived_metric_missing_grouped_subquery" for issue in issues)


def test_derived_metric_acepta_subquery_con_group_by_base() -> None:
    stage = ValidationStage(
        dialect=get_dialect("tsql"),
        schema={
            "entity_b": {"columns": {"id": {}, "entity_c_id": {}}},
            "entity_c": {"columns": {"id": {}}},
        },
        business_rules=[],
        forbidden_keywords=frozenset({"DROP", "DELETE", "UPDATE"}),
    )

    issues = stage.validate_full(
        SQLQuerySpec(
            query_type="derived_metric",
            dialect="tsql",
            base_entity="entity_b",
            base_table="entity_b",
            base_group_by=["entity_c.id"],
            post_aggregation="avg",
        ),
        "SELECT AVG(CAST(base.metric AS DECIMAL(18,2))) AS avg_value FROM (SELECT COUNT(DISTINCT entity_b.id) AS metric FROM entity_b JOIN entity_c ON entity_b.entity_c_id = entity_c.id GROUP BY entity_c.id) AS base",
    )

    assert not any(issue.code in {"derived_metric_missing_two_level_query", "derived_metric_missing_grouped_subquery"} for issue in issues)


def test_bloquea_select_con_agregado_y_columna_fuera_de_group_by() -> None:
    stage = ValidationStage(
        dialect=get_dialect("tsql"),
        schema={"ordercounts": {"columns": {"cliente_id": {}, "ordenes_trabajo_totales": {}}}},
        business_rules=[],
        forbidden_keywords=frozenset({"DROP", "DELETE", "UPDATE"}),
    )

    issues = stage.validate_full(
        SQLQuerySpec(
            query_type="ranking",
            dialect="tsql",
            base_entity="orden_trabajo",
            base_table="orden_trabajo",
            final_group_by=["cliente.id"],
            limit=5,
        ),
        "WITH OrderCounts AS (SELECT cliente.id AS cliente_id, COUNT(*) AS ordenes_trabajo_totales FROM cliente GROUP BY cliente.id) "
        "SELECT TOP 5 cliente_id, AVG(CAST(ordenes_trabajo_totales AS DECIMAL(18,2))) AS avg_ordenes_trabajo_totales FROM OrderCounts ORDER BY avg_ordenes_trabajo_totales DESC",
    )

    assert any(issue.code == "select_expression_not_grouped" and issue.context.get("expression") == "cliente_id" for issue in issues)
