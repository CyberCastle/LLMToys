#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Submódulo de dialectos para ``semantic_resolver``.

Encapsula cualquier especificidad SQL (expresiones temporales, sintaxis de
funciones, etc.) detrás de la interfaz :class:`ResolverDialect`, de manera
análoga a lo que hace ``sql_solver_generator/dialects``. Los módulos
``plan_compiler`` y ``resolver`` deben permanecer agnósticos al motor SQL y
delegar todo lo dialect-específico en la implementación concreta inyectada.
"""

from __future__ import annotations

from .base import ResolverDialect
from .postgres import PostgresResolverDialect
from .registry import get_resolver_dialect
from .tsql import TsqlResolverDialect

__all__ = [
    "ResolverDialect",
    "TsqlResolverDialect",
    "PostgresResolverDialect",
    "get_resolver_dialect",
]
