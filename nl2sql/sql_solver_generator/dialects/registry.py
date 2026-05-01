#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from nl2sql.utils.dialect_registry import canonical_dialect_name

from .base import SqlDialect
from .postgres import PostgresDialect
from .tsql import TsqlDialect

_DIALECTS: dict[str, type[SqlDialect]] = {
    "postgres": PostgresDialect,
    "tsql": TsqlDialect,
}


def get_dialect(name: str) -> SqlDialect:
    key = canonical_dialect_name(name)
    try:
        return _DIALECTS[key]()
    except KeyError as exc:
        raise ValueError(f"Dialecto no soportado: {name}") from exc
