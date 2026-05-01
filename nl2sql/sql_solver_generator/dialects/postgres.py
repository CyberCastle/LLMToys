#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import re

from .base import SqlDialect

_HAS_ROW_LIMIT_RE = re.compile(r"\bLIMIT\s+\d+\b|\bFETCH\s+(?:FIRST|NEXT)\s+\d+\s+ROWS\s+ONLY\b", re.IGNORECASE)


class PostgresDialect(SqlDialect):
    name = "postgres"
    sqlglot_dialect = "postgres"

    def render_row_limit(self, sql: str, limit: int | None) -> str:
        if not limit:
            return sql
        if _HAS_ROW_LIMIT_RE.search(sql):
            return sql
        return f"{sql.rstrip(';').rstrip()}\nLIMIT {int(limit)}"

    def normalize_semantic_expression(self, expression: str) -> str:
        normalized = expression.replace("GETDATE()", "NOW()")
        return normalized
