#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import fnmatch
from dataclasses import dataclass
from typing import Any, Callable, Literal, Mapping

from pydantic import Field

from nl2sql.utils.decision_models import DecisionIssue, StrictModel

from .spec_model import SQLFilter, SQLQuerySpec


class BusinessRule(StrictModel):
    """Regla declarativa de negocio consumida por la etapa de validacion SQL."""

    id: str
    type: str
    params: Mapping[str, Any]


class BusinessRuleResult(StrictModel):
    """Resultado agregado de aplicar reglas declarativas del dominio."""

    filters_to_inject: list[SQLFilter] = Field(default_factory=list)
    warnings: list[DecisionIssue] = Field(default_factory=list)
    errors: list[DecisionIssue] = Field(default_factory=list)


@dataclass(frozen=True)
class BusinessRuleEvaluationContext:
    """Valores derivados del spec que se comparten entre handlers."""

    tables_used: frozenset[str]
    columns_used: frozenset[str]
    present_filters: frozenset[tuple[str, str, Any]]


RuleHandler = Callable[
    [BusinessRule, SQLQuerySpec, Mapping[str, Mapping[str, Any]], BusinessRuleResult, BusinessRuleEvaluationContext],
    None,
]

RULE_HANDLERS: dict[str, RuleHandler] = {}


def _rule_issue(
    *,
    rule: BusinessRule,
    severity: Literal["warning", "error"],
    code: str,
    message: str,
    context: dict[str, Any] | None = None,
) -> DecisionIssue:
    return DecisionIssue(
        stage="sql_business_rules",
        code=code,
        severity=severity,
        message=message,
        context={"rule_id": rule.id, **(context or {})},
    )


def register(type_name: str) -> Callable[[RuleHandler], RuleHandler]:
    def _wrap(fn: RuleHandler) -> RuleHandler:
        RULE_HANDLERS[type_name] = fn
        return fn

    return _wrap


@register("inject_filter_if_column_present")
def _inject_filter(
    rule: BusinessRule,
    spec: SQLQuerySpec,
    schema: Mapping[str, Mapping[str, Any]],
    result: BusinessRuleResult,
    context: BusinessRuleEvaluationContext,
) -> None:
    del spec
    del context
    column = rule.params.get("apply_when_table_has_column")
    roles = set(rule.params.get("apply_to_tables_with_role", []))
    filter_data = rule.params.get("filter", {})
    if not isinstance(column, str):
        return

    for table_name, metadata in schema.items():
        columns = metadata.get("columns", {}) or {}
        role = metadata.get("role")
        if column in columns and (not roles or role in roles):
            result.filters_to_inject.append(
                SQLFilter(
                    name=f"{rule.id}:{table_name}",
                    field=f"{table_name}.{column}",
                    operator=str(filter_data.get("operator", ">=")),
                    value=filter_data.get("value", 0),
                    source=f"business_rule:{rule.id}",
                )
            )


@register("forbid_column")
def _forbid_column(
    rule: BusinessRule,
    spec: SQLQuerySpec,
    schema: Mapping[str, Mapping[str, Any]],
    result: BusinessRuleResult,
    context: BusinessRuleEvaluationContext,
) -> None:
    del spec
    del schema
    forbidden_column = str(rule.params.get("column", ""))
    if forbidden_column and forbidden_column in context.columns_used:
        result.errors.append(
            _rule_issue(
                rule=rule,
                severity="error",
                code="forbidden_column",
                message="La consulta usa una columna prohibida por regla de negocio.",
                context={"column": forbidden_column},
            )
        )


@register("require_filter_when_table_used")
def _require_filter_when_table_used(
    rule: BusinessRule,
    spec: SQLQuerySpec,
    schema: Mapping[str, Mapping[str, Any]],
    result: BusinessRuleResult,
    context: BusinessRuleEvaluationContext,
) -> None:
    del spec
    del schema
    table_name = rule.params.get("table")
    if not isinstance(table_name, str) or table_name not in context.tables_used:
        return

    required_options = rule.params.get("any_of", [])
    satisfied = False
    if isinstance(required_options, list):
        for option in required_options:
            if not isinstance(option, Mapping):
                continue
            signature = (option.get("field"), option.get("operator"), option.get("value"))
            if signature in context.present_filters:
                satisfied = True
                break

    if not satisfied:
        result.errors.append(
            _rule_issue(
                rule=rule,
                severity="error",
                code="missing_safe_filter",
                message="La tabla requiere un filtro de seguridad declarado para poder usarse.",
                context={"table": table_name},
            )
        )


@register("require_table_when_table_used")
def _require_table_when_table_used(
    rule: BusinessRule,
    spec: SQLQuerySpec,
    schema: Mapping[str, Mapping[str, Any]],
    result: BusinessRuleResult,
    context: BusinessRuleEvaluationContext,
) -> None:
    del spec
    del schema
    when_table = rule.params.get("when_table_used")
    required_table = rule.params.get("required_table")
    if isinstance(when_table, str) and isinstance(required_table, str):
        if when_table in context.tables_used and required_table not in context.tables_used:
            result.errors.append(
                _rule_issue(
                    rule=rule,
                    severity="error",
                    code="missing_required_table",
                    message="La consulta usa una tabla sin incluir la tabla dependiente requerida.",
                    context={"when_table_used": when_table, "required_table": required_table},
                )
            )


@register("require_normalization_before_sum")
def _require_normalization_before_sum(
    rule: BusinessRule,
    spec: SQLQuerySpec,
    schema: Mapping[str, Mapping[str, Any]],
    result: BusinessRuleResult,
    context: BusinessRuleEvaluationContext,
) -> None:
    del schema
    del context
    affected_columns = set(rule.params.get("affected_columns", []))
    if not affected_columns:
        return

    uses_affected_metric = any(any(column in (metric.formula or "") for column in affected_columns) for metric in spec.selected_metrics)
    if not uses_affected_metric:
        return

    required_patterns = rule.params.get("required_filter_matches", [])
    if any(_match_patterns(flt.field, required_patterns) for flt in spec.selected_filters):
        return
    result.warnings.append(
        _rule_issue(
            rule=rule,
            severity="warning",
            code="multicurrency_sum_requires_normalization",
            message="La metrica suma columnas monetarias sin una normalizacion explicita.",
            context={"affected_columns": sorted(affected_columns)},
        )
    )


def apply_business_rules(
    *,
    rules: list[BusinessRule],
    spec: SQLQuerySpec,
    pruned_schema: Mapping[str, Mapping[str, Any]],
) -> BusinessRuleResult:
    result = BusinessRuleResult()
    context = BusinessRuleEvaluationContext(
        tables_used=frozenset(_tables_used(spec)),
        columns_used=frozenset(_columns_used(spec)),
        present_filters=frozenset((flt.field, flt.operator, flt.value) for flt in spec.selected_filters),
    )
    for rule in rules:
        handler = RULE_HANDLERS.get(rule.type)
        if handler is None:
            result.warnings.append(
                _rule_issue(
                    rule=rule,
                    severity="warning",
                    code="unknown_rule_type",
                    message="Se ignoro una regla declarativa porque no existe un handler registrado.",
                    context={"rule_type": rule.type},
                )
            )
            continue
        handler(rule, spec, pruned_schema, result, context)
    return result


def _columns_used(spec: SQLQuerySpec) -> set[str]:
    used = {dimension.source for dimension in spec.selected_dimensions}
    used.update(flt.field for flt in spec.selected_filters)
    if spec.time_filter is not None:
        used.add(spec.time_filter.field)
    return used


def _tables_used(spec: SQLQuerySpec) -> set[str]:
    used_tables = {spec.base_table}
    for join_plan in spec.join_plan:
        for edge in join_plan.joins:
            used_tables.add(edge.left_table)
            used_tables.add(edge.right_table)
    return used_tables


def _match_patterns(value: str, patterns: object) -> bool:
    if not isinstance(patterns, list):
        return False
    return any(isinstance(pattern, str) and fnmatch.fnmatch(value, pattern) for pattern in patterns)
