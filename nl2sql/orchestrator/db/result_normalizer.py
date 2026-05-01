#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Conversion segura de filas SQLAlchemy a dicts serializables."""

from __future__ import annotations

from datetime import date, datetime
from decimal import Decimal
from typing import Any, Sequence


def _coerce(value: Any) -> Any:
    """Normaliza tipos comunes de base de datos a valores YAML-safe."""

    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, memoryview):
        return f"<bytes:{len(value.tobytes())}>"
    if isinstance(value, (bytes, bytearray)):
        return f"<bytes:{len(value)}>"
    return value


def rows_to_dicts(rows: Sequence[Sequence[Any]], columns: Sequence[str]) -> list[dict[str, Any]]:
    """Convierte una matriz de filas a una lista de dicts por columna."""

    return [{str(column_name): _coerce(column_value) for column_name, column_value in zip(columns, row)} for row in rows]
