#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import re
from typing import Any, Literal, Mapping

import sqlglot
from sqlglot import exp

from nl2sql.utils.decision_models import DecisionIssue, dedupe_decision_issues

from ..business_rules import BusinessRule, apply_business_rules
from ..dialects.base import SqlDialect
from ..spec_model import SQLQuerySpec


class ValidationStage:
    def __init__(
        self,
        *,
        dialect: SqlDialect,
        schema: Mapping[str, Mapping[str, Any]],
        business_rules: list[BusinessRule],
        forbidden_keywords: frozenset[str],
    ):
        self.dialect = dialect
        self.schema = schema
        self.business_rules = business_rules
        self.forbidden_keywords = forbidden_keywords

    def validate_full(self, spec: SQLQuerySpec, sql: str) -> list[DecisionIssue]:
        issues: list[DecisionIssue] = []
        issues.extend(self._validate_query_shape(spec))
        issues.extend(self._validate_business_rules(spec))
        issues.extend(self._validate_lexical(sql))
        parsed_statements, parse_issues = self._parse_sql(sql)
        issues.extend(parse_issues)
        if not parse_issues:
            issues.extend(self._validate_sql_ast(spec, parsed_statements))
        return dedupe_decision_issues(issues)

    def _validate_query_shape(self, spec: SQLQuerySpec) -> list[DecisionIssue]:
        """Valida restricciones semanticas de consulta, no estructura del schema."""

        issues: list[DecisionIssue] = []
        if spec.query_type == "derived_metric":
            if not spec.base_group_by:
                issues.append(_issue("derived_metric_missing_base_group_by", "El spec derivado no declara base_group_by."))
            if spec.post_aggregation == "none":
                issues.append(_issue("derived_metric_missing_post_aggregation", "El spec derivado no declara post_aggregation."))

        if spec.query_type == "detail_listing" and not spec.limit:
            issues.append(_issue("detail_listing_missing_limit", "Las consultas detail_listing deben declarar limit."))

        return issues

    def _validate_business_rules(self, spec: SQLQuerySpec) -> list[DecisionIssue]:
        result = apply_business_rules(rules=self.business_rules, spec=spec, pruned_schema=self.schema)
        return [*result.errors, *result.warnings]

    def _validate_lexical(self, sql: str) -> list[DecisionIssue]:
        upper_sql = sql.upper()
        return [
            _issue(
                "forbidden_keyword",
                "El SQL contiene una palabra reservada prohibida.",
                context={"keyword": keyword},
            )
            for keyword in self.forbidden_keywords
            if re.search(rf"\b{keyword}\b", upper_sql)
        ]

    def _parse_sql(self, sql: str) -> tuple[list[exp.Expression], list[DecisionIssue]]:
        try:
            statements = sqlglot.parse(sql, dialect=self.dialect.sqlglot_dialect)
        except sqlglot.errors.ParseError as exc:
            return [], [_issue("unparseable_sql", "No se pudo parsear el SQL generado.", context={"detail": str(exc)})]
        return [statement for statement in statements if statement is not None], []

    def _validate_sql_ast(self, spec: SQLQuerySpec, statements: list[exp.Expression]) -> list[DecisionIssue]:
        issues: list[DecisionIssue] = []
        allowed_filter_fields, allowed_filter_columns = _build_allowed_where_fields(spec)
        physical_columns = _build_known_physical_columns(self.schema)
        if len(statements) != 1:
            issues.append(_issue("multiple_statements", "El SQL generado contiene mas de una sentencia."))

        for statement in statements:
            if statement is None:
                continue
            for node_cls, code in (
                (exp.Insert, "no_dml:insert"),
                (exp.Update, "no_dml:update"),
                (exp.Delete, "no_dml:delete"),
                (exp.Merge, "no_dml:merge"),
                (exp.Create, "no_ddl:create"),
                (exp.Drop, "no_ddl:drop"),
                (exp.Alter, "no_ddl:alter"),
            ):
                if statement.find(node_cls) is not None:
                    issues.append(_issue_from_token(code))
            for join_node in statement.find_all(exp.Join):
                kind = str(join_node.args.get("kind") or "").upper()
                if kind == "CROSS":
                    issues.append(_issue("cross_join_forbidden", "El SQL generado usa CROSS JOIN, que esta prohibido."))
                elif join_node.args.get("on") is None:
                    issues.append(_issue("cartesian_join_detected", "Se detecto un join cartesiano sin clausula ON."))
            for where_node in statement.find_all(exp.Where):
                issues.extend(
                    _validate_where_predicates(
                        where_node,
                        allowed_filter_fields=allowed_filter_fields,
                        allowed_filter_columns=allowed_filter_columns,
                        sqlglot_dialect=self.dialect.sqlglot_dialect,
                    )
                )
            issues.extend(
                _validate_unknown_bare_columns(
                    statement,
                    physical_columns=physical_columns,
                )
            )
            for select_node in statement.find_all(exp.Select):
                issues.extend(
                    _validate_mixed_aggregate_projection(
                        select_node,
                        sqlglot_dialect=self.dialect.sqlglot_dialect,
                    )
                )
            issues.extend(_validate_derived_metric_query_structure(spec, statement))
        return issues


def _issue(
    code: str,
    message: str,
    *,
    severity: Literal["warning", "error"] = "error",
    context: dict[str, Any] | None = None,
) -> DecisionIssue:
    return DecisionIssue(
        stage="sql_validation",
        code=code,
        severity=severity,
        message=message,
        context=context or {},
    )


def _issue_from_token(token: str) -> DecisionIssue:
    code, _separator, detail = token.partition(":")
    message = detail or code.replace("_", " ")
    context = {"detail": detail} if detail else {}
    return _issue(code, message, context=context)


def _build_allowed_where_fields(spec: SQLQuerySpec) -> tuple[set[str], set[str]]:
    fields: set[str] = set()
    columns: set[str] = set()

    def _add(field_name: str) -> None:
        normalized_field = field_name.strip().lower()
        if not normalized_field:
            return
        fields.add(normalized_field)
        columns.add(normalized_field.split(".", 1)[1] if "." in normalized_field else normalized_field)

    for filter_obj in spec.selected_filters:
        _add(filter_obj.field)
    if spec.time_filter is not None:
        _add(spec.time_filter.field)
    return fields, columns


def _build_known_physical_columns(schema: Mapping[str, Mapping[str, Any]]) -> set[str]:
    """Construye el conjunto plano de nombres de columnas fisicas del schema."""

    known_columns: set[str] = set()
    for table_meta in schema.values():
        columns = table_meta.get("columns", {}) or {}
        if isinstance(columns, Mapping):
            known_columns.update(str(column_name).strip().lower() for column_name in columns)
            continue
        known_columns.update(str(column_name).strip().lower() for column_name in columns if str(column_name).strip())
    return known_columns


def _collect_select_aliases(statement: exp.Expr) -> set[str]:
    """Recolecta aliases explicitos para no confundirlos con columnas fisicas."""

    aliases: set[str] = set()
    for alias_node in statement.find_all(exp.Alias):
        alias_name = str(alias_node.alias or "").strip().lower()
        if alias_name:
            aliases.add(alias_name)
    return aliases


def _iter_where_predicates(node: exp.Expr) -> list[exp.Expr]:
    if isinstance(node, exp.Where):
        return _iter_where_predicates(node.this)
    if isinstance(node, exp.Paren) and node.this is not None:
        return _iter_where_predicates(node.this)
    if isinstance(node, exp.And):
        return [*_iter_where_predicates(node.left), *_iter_where_predicates(node.right)]
    return [node]


def _column_reference(column: exp.Column) -> str:
    table_name = str(column.table or "").strip()
    column_name = str(column.name or "").strip()
    if table_name and column_name:
        return f"{table_name}.{column_name}".lower()
    return column_name.lower()


def _validate_unknown_bare_columns(
    statement: exp.Expr,
    *,
    physical_columns: set[str],
) -> list[DecisionIssue]:
    """Marca identificadores desnudos que no existen en el schema ni como alias."""

    issues: list[DecisionIssue] = []
    known_aliases = _collect_select_aliases(statement)
    for column in statement.find_all(exp.Column):
        if str(column.table or "").strip():
            continue
        column_name = str(column.name or "").strip().lower()
        if not column_name or column_name == "*":
            continue
        if column_name in physical_columns or column_name in known_aliases:
            continue
        issues.append(
            _issue(
                "unknown_bare_column",
                "El SQL usa un identificador desnudo que no existe en el schema ni como alias valido.",
                context={"column": column_name},
            )
        )
    return issues


def _validate_derived_metric_query_structure(spec: SQLQuerySpec, statement: exp.Expr) -> list[DecisionIssue]:
    """Exige el nivel intermedio agrupado para metricas derivadas con agregacion externa."""

    if spec.query_type != "derived_metric" or spec.post_aggregation == "none" or not spec.base_group_by:
        return []

    select_nodes = [select_node for select_node in statement.find_all(exp.Select)]
    if len(select_nodes) < 2:
        return [
            _issue(
                "derived_metric_missing_two_level_query",
                "El SQL derivado debe usar una CTE o subquery interna antes de aplicar la agregacion externa.",
                context={"expected_base_group_by": list(spec.base_group_by)},
            )
        ]

    inner_selects = select_nodes[1:]
    if any(_select_groups_by_expected_base(select_node, expected_base_group_by=spec.base_group_by) for select_node in inner_selects):
        return []

    return [
        _issue(
            "derived_metric_missing_grouped_subquery",
            "El SQL derivado debe agrupar la metrica base por base_group_by dentro de la CTE o subquery interna.",
            context={"expected_base_group_by": list(spec.base_group_by)},
        )
    ]


def _validate_mixed_aggregate_projection(
    select_node: exp.Select,
    *,
    sqlglot_dialect: str,
) -> list[DecisionIssue]:
    """Bloquea SELECTs que mezclan agregados con expresiones no agrupadas."""

    projections = list(select_node.expressions or [])
    if not projections:
        return []
    if not any(_expression_contains_aggregate(projection) for projection in projections):
        return []

    group_node = select_node.args.get("group")
    group_expressions = list(group_node.expressions or []) if isinstance(group_node, exp.Group) else []
    normalized_group_expressions = {_normalize_expression_sql(expression, sqlglot_dialect) for expression in group_expressions}

    issues: list[DecisionIssue] = []
    for projection in projections:
        projected_expression = projection.this if isinstance(projection, exp.Alias) else projection
        if _expression_contains_aggregate(projected_expression):
            continue
        if isinstance(projected_expression, (exp.Literal, exp.Star)):
            continue
        rendered_expression = _normalize_expression_sql(projected_expression, sqlglot_dialect)
        if rendered_expression in normalized_group_expressions:
            continue
        issues.append(
            _issue(
                "select_expression_not_grouped",
                "El SQL mezcla agregaciones con una expresion proyectada que no esta en GROUP BY.",
                context={"expression": rendered_expression},
            )
        )
    return issues


def _expression_contains_aggregate(expression: exp.Expression) -> bool:
    return expression.find(exp.AggFunc) is not None


def _normalize_expression_sql(expression: exp.Expression, sqlglot_dialect: str) -> str:
    return expression.sql(dialect=sqlglot_dialect).strip().lower()


def _select_groups_by_expected_base(select_node: exp.Select, *, expected_base_group_by: list[str]) -> bool:
    group_node = select_node.args.get("group")
    if not isinstance(group_node, exp.Group):
        return False

    expected_refs = {reference.strip().lower() for reference in expected_base_group_by if reference and reference.strip()}
    expected_columns = {reference.rsplit(".", 1)[-1].strip().lower() for reference in expected_refs if reference.rsplit(".", 1)[-1].strip()}
    actual_refs: set[str] = set()
    actual_columns: set[str] = set()

    for expression in group_node.expressions:
        column_nodes = [expression] if isinstance(expression, exp.Column) else list(expression.find_all(exp.Column))
        for column_node in column_nodes:
            reference = _column_reference(column_node)
            if not reference:
                continue
            actual_refs.add(reference)
            actual_columns.add(reference.rsplit(".", 1)[-1])

    if expected_refs and expected_refs.issubset(actual_refs):
        return True
    return bool(expected_columns) and expected_columns.issubset(actual_columns)


def _validate_where_predicates(
    where_node: exp.Where,
    *,
    allowed_filter_fields: set[str],
    allowed_filter_columns: set[str],
    sqlglot_dialect: str,
) -> list[DecisionIssue]:
    issues: list[DecisionIssue] = []
    for predicate in _iter_where_predicates(where_node):
        predicate_columns = [_column_reference(column) for column in predicate.find_all(exp.Column)]
        if not predicate_columns:
            issues.append(
                _issue(
                    "undeclared_where_predicate",
                    "El WHERE contiene un predicado sin columnas declaradas en el spec.",
                    context={"predicate": predicate.sql(dialect=sqlglot_dialect)},
                )
            )
            continue

        for column_ref in predicate_columns:
            bare_column = column_ref.split(".", 1)[1] if "." in column_ref else column_ref
            if column_ref in allowed_filter_fields or bare_column in allowed_filter_columns:
                continue
            issues.append(
                _issue(
                    "undeclared_where_column",
                    "El SQL usa una columna en WHERE que no fue declarada en los filtros del spec.",
                    context={"column": column_ref},
                )
            )
    return issues
