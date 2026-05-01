#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import sqlglot
from sqlglot import exp

from .dialects.base import SqlDialect


def normalize_sql_via_ast(raw_sql: str, dialect: SqlDialect) -> str:
    """Parsea SQL crudo y lo renderiza con una forma estable por dialecto."""

    ast = sqlglot.parse_one(raw_sql, dialect=dialect.sqlglot_dialect)
    return render_sql_ast(ast, dialect)


def render_sql_ast(ast: exp.Expr, dialect: SqlDialect) -> str:
    """Renderiza un AST SQL ya parseado aplicando ajustes dialectales estables."""

    rendered_ast = ast.copy()
    if dialect.name == "tsql":
        _cast_avg_count_aliases_to_decimal(rendered_ast)
    normalized = rendered_ast.sql(dialect=dialect.sqlglot_dialect, pretty=True)
    if dialect.name == "tsql":
        normalized = normalized.replace(" AS NUMERIC(18, 2)", " AS DECIMAL(18,2)")
    return normalized


def _cast_avg_count_aliases_to_decimal(ast: exp.Expr) -> None:
    count_aliases = _count_aliases(ast)
    if not count_aliases:
        return

    decimal_type = exp.DataType.build("DECIMAL(18,2)")
    for avg_expr in ast.find_all(exp.Avg):
        avg_arg = avg_expr.this
        if not isinstance(avg_arg, exp.Column):
            continue
        if avg_arg.name.lower() not in count_aliases:
            continue
        avg_expr.set("this", exp.Cast(this=avg_arg.copy(), to=decimal_type.copy()))


def _count_aliases(ast: exp.Expr) -> set[str]:
    aliases: set[str] = set()
    for alias_expr in ast.find_all(exp.Alias):
        if isinstance(alias_expr.this, exp.Count) and alias_expr.alias:
            aliases.add(alias_expr.alias.lower())
    return aliases
