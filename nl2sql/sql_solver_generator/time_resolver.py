#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Mapping

from .dialects.base import SqlDialect
from .spec_model import SQLTimeFilter


def resolve_time_filter(
    time_filter_dict: Mapping[str, object] | None,
    time_patterns: Mapping[str, str],
    dialect: SqlDialect | None = None,
) -> SQLTimeFilter | None:
    if not time_filter_dict:
        return None

    field_name = str(time_filter_dict["field"])
    operator = str(time_filter_dict["operator"])
    value = str(time_filter_dict["value"])

    # El plan emite ``resolved_expressions`` indexado por nombre de dialecto.
    # Sin dialecto activo no hay forma deterministica de elegir una expresion;
    # se cae al mapeo canonico ``time_patterns`` o se levanta error.
    resolved_expressions = time_filter_dict.get("resolved_expressions")
    if isinstance(resolved_expressions, Mapping) and dialect is not None:
        dialect_value = resolved_expressions.get(dialect.name)
        if isinstance(dialect_value, str) and dialect_value.strip():
            return SQLTimeFilter(
                field=field_name,
                operator=operator,
                value=value,
                resolved_expression=dialect_value.strip(),
            )

    if value in time_patterns:
        return SQLTimeFilter(
            field=field_name,
            operator=operator,
            value=value,
            resolved_expression=time_patterns[value],
        )
    raise ValueError(f"No se puede resolver time_filter.value={value!r}")
