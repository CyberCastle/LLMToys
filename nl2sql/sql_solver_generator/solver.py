#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Any, Mapping

import sqlglot

from nl2sql.utils.collections import dedupe_preserve_order
from nl2sql.utils.decision_models import DecisionIssue, dedupe_decision_issues

from .config import SolverConfig
from .contracts import SolverInput, SolverMetadata, SolverOutput
from .dialects.registry import get_dialect
from .llm_router import LlmRouter
from .rules_loader import load_business_rules, load_solver_prompts, load_solver_rules
from .spec_model import SQLQuerySpec
from .stages import ValidationStage, run_generation, run_plan_normalization_stage, run_sql_normalization


def run_sql_solver(solver_input: SolverInput, config: SolverConfig | None = None) -> SolverOutput:
    cfg = config or SolverConfig()
    dialect = get_dialect(cfg.dialect)

    normalized = run_plan_normalization_stage(solver_input)
    solver_rules = load_solver_rules(normalized.semantic_rules)
    business_rules = load_business_rules(normalized.semantic_rules)
    prompts = cfg.prompts.model_dump(mode="python") if cfg.prompts is not None else load_solver_prompts(str(cfg.prompts_path))

    validator = ValidationStage(
        dialect=dialect,
        schema=normalized.pruned_schema,
        business_rules=business_rules,
        forbidden_keywords=solver_rules.forbidden_keywords,
    )
    router = LlmRouter(cfg)
    attempt = run_generation(
        router,
        semantic_plan=normalized.semantic_plan,
        pruned_schema=normalized.pruned_schema,
        semantic_rules=normalized.semantic_rules,
        business_rules_summary=_summarize_business_rules(solver_rules, business_rules),
        prompts=prompts,
        dialect=dialect,
        validator=validator,
    ).attempt

    spec = attempt.spec or _empty_spec(cfg.dialect)
    issues = list(attempt.issues)
    sql_final = ""
    if attempt.sql:
        try:
            sql_final = run_sql_normalization(attempt.sql, dialect)
            issues = validator.validate_full(spec, sql_final)
        except sqlglot.errors.ParseError as exc:
            issues.append(
                DecisionIssue(
                    stage="sql_normalization",
                    code="sql_normalization_error",
                    severity="error",
                    message="El SQL generado no se pudo normalizar con sqlglot.",
                    context={"detail": str(exc)},
                )
            )

    if cfg.fail_on_validation_error and issues:
        raise RuntimeError("Solver detuvo la ejecucion por issues: " f"{[issue.model_dump(mode='python') for issue in issues]}")

    metadata = _build_metadata(spec, attempt.generation_diagnostics, cfg, attempt.model_name, attempt.attempts, issues)
    warnings = list(spec.warnings)

    return SolverOutput(
        sql_final=sql_final,
        sql_query_spec=spec,
        metadata=metadata,
        warnings=dedupe_preserve_order(warnings),
        issues=dedupe_decision_issues(issues),
    )


def _build_metadata(
    spec: SQLQuerySpec,
    generation_diagnostics: Mapping[str, Any],
    config: SolverConfig,
    model_name: str,
    attempts: int,
    issues: list[DecisionIssue],
) -> SolverMetadata:
    return SolverMetadata(
        tables_used=_tables_from_spec(spec),
        columns_used=_columns_from_spec(spec),
        join_paths_used=[plan.path_name for plan in spec.join_plan if plan.path_name],
        dialect=config.dialect,
        model_used=model_name,
        attempts=attempts,
        finish_reason=str(generation_diagnostics.get("finish_reason", "")),
        prompt_tokens=int(generation_diagnostics.get("prompt_tokens", 0) or 0),
        generated_tokens=int(generation_diagnostics.get("generated_tokens", 0) or 0),
        wall_time_seconds=float(generation_diagnostics.get("wall_time_seconds", 0.0) or 0.0),
        validator_trace=dedupe_decision_issues(issues),
    )


def _summarize_business_rules(solver_rules, rules) -> str:
    lines = [f"- {rule.id}: {rule.type}" for rule in rules]
    for hard_join_rule in solver_rules.hard_join_blacklist:
        reason = hard_join_rule.reason or "an explicit compatibility rule"
        lines.append(f"- hard_join_blacklist: do not use {hard_join_rule.table} as a hard join without {reason}")
    return "\n".join(lines) or "(sin reglas)"


def _empty_spec(dialect: str) -> SQLQuerySpec:
    return SQLQuerySpec(query_type="scalar_metric", dialect=dialect, base_entity="", base_table="")


def _tables_from_spec(spec: SQLQuerySpec) -> list[str]:
    tables = {spec.base_table} if spec.base_table else set()
    for join_plan in spec.join_plan:
        for edge in join_plan.joins:
            if edge.left_table:
                tables.add(edge.left_table)
            if edge.right_table:
                tables.add(edge.right_table)
    return sorted(tables)


def _columns_from_spec(spec: SQLQuerySpec) -> list[str]:
    columns = [dimension.source for dimension in spec.selected_dimensions if dimension.source]
    columns.extend(filter_obj.field for filter_obj in spec.selected_filters if filter_obj.field)
    if spec.time_filter is not None and spec.time_filter.field:
        columns.append(spec.time_filter.field)
    return dedupe_preserve_order(columns)
