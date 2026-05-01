#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Factoría de dialectos para ``semantic_resolver``.

Expone :func:`get_resolver_dialect`, que mapea un nombre lógico (insensible a
mayúsculas y a guiones bajos) a una instancia concreta de
:class:`ResolverDialect`.
"""

from __future__ import annotations

from nl2sql.utils.dialect_registry import canonical_dialect_name

from .base import ResolverDialect
from .postgres import PostgresResolverDialect
from .tsql import TsqlResolverDialect

_DIALECTS: dict[str, type[ResolverDialect]] = {
    "tsql": TsqlResolverDialect,
    "postgres": PostgresResolverDialect,
}


def get_resolver_dialect(name: str) -> ResolverDialect:
    """Instancia el :class:`ResolverDialect` asociado a ``name``.

    :raises ValueError: Si ``name`` no corresponde a ningún dialecto soportado.
    """

    key = canonical_dialect_name(name)
    if key not in _DIALECTS:
        raise ValueError(f"Dialecto desconocido para semantic_resolver: {name!r}. Soportados: {sorted(_DIALECTS)}")
    return _DIALECTS[key]()
