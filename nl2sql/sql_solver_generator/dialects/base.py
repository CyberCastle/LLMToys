#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from abc import ABC, abstractmethod

import sqlglot


class SqlDialect(ABC):
    name: str
    sqlglot_dialect: str

    @abstractmethod
    def render_row_limit(self, sql: str, limit: int | None) -> str: ...

    @abstractmethod
    def normalize_semantic_expression(self, expression: str) -> str: ...

    def validate_syntax(self, sql: str) -> list[str]:
        try:
            sqlglot.parse(sql, dialect=self.sqlglot_dialect)
            return []
        except sqlglot.errors.ParseError as exc:
            return [f"unparseable_sql:{exc}"]

    def parse_ast(self, sql: str):
        return sqlglot.parse_one(sql, dialect=self.sqlglot_dialect)
