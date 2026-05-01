#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import re

from .base import SqlDialect

_HAS_ROW_LIMIT_RE = re.compile(
    r"\bTOP\s*\(?\d+\)?\b|\bOFFSET\s+\d+\s+ROWS\b|\bFETCH\s+(?:FIRST|NEXT)\s+\d+\s+ROWS\s+ONLY\b",
    re.IGNORECASE,
)
_SELECT_PREFIX_RE = re.compile(r"^(?P<prefix>\s*SELECT\s+)(?P<distinct>DISTINCT\s+)?", re.IGNORECASE)


class TsqlDialect(SqlDialect):
    name = "tsql"
    sqlglot_dialect = "tsql"

    def render_row_limit(self, sql: str, limit: int | None) -> str:
        if not limit:
            return sql
        if _HAS_ROW_LIMIT_RE.search(sql):
            return sql
        match = _SELECT_PREFIX_RE.match(sql)
        if match is None:
            return sql
        distinct_clause = match.group("distinct") or ""
        return f"{match.group('prefix')}{distinct_clause}TOP {int(limit)} {sql[match.end():]}"

    def normalize_semantic_expression(self, expression: str) -> str:
        return expression.replace("count_distinct(", "COUNT(DISTINCT ")
