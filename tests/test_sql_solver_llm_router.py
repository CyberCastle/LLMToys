#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from unittest.mock import patch

from nl2sql.utils.decision_models import DecisionIssue

from nl2sql.sql_solver_generator.config import SolverConfig
from nl2sql.sql_solver_generator.llm_router import LlmRouter
from nl2sql.sql_solver_generator.spec_model import SQLQuerySpec

MINIMAL_SEMANTIC_RULES = {
    "semantic_contract": {
        "business_invariants": {},
        "retrieval_heuristics": {},
        "sql_safety": {},
    }
}


class _ValidationAlwaysFail:
    def __init__(self) -> None:
        self.calls = 0

    def validate_full(self, spec, sql):
        self.calls += 1
        del spec, sql
        return [] if self.calls > 3 else [DecisionIssue(stage="sql_validation", code="issue", severity="error", message="issue")]


class _Dialect:
    name = "tsql"
    sqlglot_dialect = "tsql"

    def render_row_limit(self, sql: str, limit: int | None) -> str:
        if not limit:
            return sql
        return sql.replace("SELECT ", f"SELECT TOP {int(limit)} ", 1)


class _ValidationAlwaysPass:
    def __init__(self) -> None:
        self.last_sql = ""

    def validate_full(self, spec, sql):
        self.last_sql = sql
        return []


class _ValidationDerivedMetricRepairOnce:
    def __init__(self) -> None:
        self.calls = 0

    def validate_full(self, spec, sql):
        self.calls += 1
        del spec, sql
        if self.calls == 1:
            return [
                DecisionIssue(
                    stage="sql_validation",
                    code="derived_metric_missing_grouped_subquery",
                    severity="error",
                    message="missing inner group by",
                )
            ]
        return []


def test_reintentos_devuelven_ultimo_intento_del_modelo() -> None:
    cfg = SolverConfig(max_retries=2)
    calls: list[str] = []

    def fake(*, model_name, **_):
        calls.append(model_name)
        return (
            SQLQuerySpec(query_type="scalar_metric", dialect="tsql", base_entity="t", base_table="t"),
            "SELECT 1",
            {"finish_reason": "stop", "prompt_tokens": 10, "generated_tokens": 5, "wall_time_seconds": 0.1},
        )

    with patch("nl2sql.sql_solver_generator.llm_router.generate_spec_and_sql", side_effect=fake):
        attempt = LlmRouter(cfg).run(
            semantic_plan={},
            pruned_schema={},
            semantic_rules=MINIMAL_SEMANTIC_RULES,
            business_rules_summary="",
            prompts={},
            dialect=_Dialect(),
            validator=_ValidationAlwaysFail(),
        )

    assert calls.count(cfg.model) == cfg.max_retries + 1
    assert calls == [cfg.model, cfg.model, cfg.model]
    assert attempt.model_name == cfg.model
    assert attempt.attempts == cfg.max_retries + 1
    assert [issue.code for issue in attempt.issues] == ["issue"]


def test_error_de_validacion_del_request_no_reintenta() -> None:
    cfg = SolverConfig(max_retries=2)
    calls: list[str] = []

    class VLLMValidationError(ValueError):
        pass

    def fake(*, model_name, **_):
        calls.append(model_name)
        raise VLLMValidationError("maximum context length exceeded")

    with patch("nl2sql.sql_solver_generator.llm_router.generate_spec_and_sql", side_effect=fake):
        attempt = LlmRouter(cfg).run(
            semantic_plan={},
            pruned_schema={},
            semantic_rules=MINIMAL_SEMANTIC_RULES,
            business_rules_summary="",
            prompts={},
            dialect=_Dialect(),
            validator=_ValidationAlwaysFail(),
        )

    assert calls == [cfg.model]
    assert attempt.model_name == cfg.model
    assert attempt.attempts == 1
    assert [issue.code for issue in attempt.issues] == ["model_raised"]
    assert attempt.issues[0].message == "maximum context length exceeded"


def test_detail_listing_aplica_limit_en_runtime() -> None:
    cfg = SolverConfig(max_retries=0)
    validator = _ValidationAlwaysPass()

    def fake(*, model_name, **_):
        del model_name
        return (
            SQLQuerySpec(query_type="detail_listing", dialect="tsql", base_entity="entity_c", base_table="entity_c", limit=25),
            "SELECT id FROM entity_c",
            {"finish_reason": "stop", "prompt_tokens": 10, "generated_tokens": 5, "wall_time_seconds": 0.1},
        )

    with patch("nl2sql.sql_solver_generator.llm_router.generate_spec_and_sql", side_effect=fake):
        attempt = LlmRouter(cfg).run(
            semantic_plan={},
            pruned_schema={},
            semantic_rules=MINIMAL_SEMANTIC_RULES,
            business_rules_summary="",
            prompts={},
            dialect=_Dialect(),
            validator=validator,
        )

    assert attempt.sql == "SELECT TOP 25 id FROM entity_c"
    assert validator.last_sql == "SELECT TOP 25 id FROM entity_c"


def test_reintento_refuerza_prompt_para_derived_metric_sin_group_by_interno() -> None:
    cfg = SolverConfig(max_retries=1)
    validator = _ValidationDerivedMetricRepairOnce()
    business_rule_summaries: list[str] = []

    def fake(*, business_rules_summary, model_name, **_):
        business_rule_summaries.append(business_rules_summary)
        del model_name
        return (
            SQLQuerySpec(
                query_type="derived_metric",
                dialect="tsql",
                base_entity="entity_b",
                base_table="entity_b",
                base_group_by=["entity_c.id"],
                post_aggregation="avg",
            ),
            "SELECT 1",
            {"finish_reason": "stop", "prompt_tokens": 10, "generated_tokens": 5, "wall_time_seconds": 0.1},
        )

    with patch("nl2sql.sql_solver_generator.llm_router.generate_spec_and_sql", side_effect=fake):
        attempt = LlmRouter(cfg).run(
            semantic_plan={},
            pruned_schema={},
            semantic_rules=MINIMAL_SEMANTIC_RULES,
            business_rules_summary="- regla_base: tipo",
            prompts={},
            dialect=_Dialect(),
            validator=validator,
        )

    assert attempt.attempts == 2
    assert business_rule_summaries[0] == "- regla_base: tipo"
    assert "Additional retry rules:" in business_rule_summaries[1]
    assert "base_group_by" in business_rule_summaries[1]
    assert "mandatory GROUP BY" in business_rule_summaries[1]
