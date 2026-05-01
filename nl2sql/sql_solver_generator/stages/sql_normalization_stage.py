#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from ..dialects.base import SqlDialect
from ..sql_normalizer import normalize_sql_via_ast


def run_sql_normalization(raw_sql: str | None, dialect: SqlDialect) -> str:
    return "" if not raw_sql else normalize_sql_via_ast(raw_sql, dialect)
