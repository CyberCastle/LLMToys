#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Alias canónicos compartidos para dialectos NL2SQL."""

from __future__ import annotations

_DIALECT_ALIASES: dict[str, str] = {
    "tsql": "tsql",
    "sqlserver": "tsql",
    "mssql": "tsql",
    "postgres": "postgres",
    "postgresql": "postgres",
    "pg": "postgres",
}


def canonical_dialect_name(name: str) -> str:
    """Normaliza aliases de dialecto a los nombres canónicos internos."""

    key = (name or "").strip().lower().replace("_", "")
    try:
        return _DIALECT_ALIASES[key]
    except KeyError as exc:
        raise ValueError(f"Dialecto NL2SQL desconocido: {name!r}. Soportados: {sorted(_DIALECT_ALIASES)}") from exc
