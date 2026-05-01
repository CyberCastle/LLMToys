#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Dialecto PostgreSQL para ``semantic_resolver``.

La traducción de expresiones temporales canónicas se resuelve desde
``semantic_resolver.compiler_rules``; este módulo solo declara el nombre del
dialecto para usar la clave ``value_postgres`` correspondiente.
"""

from __future__ import annotations

from .base import ResolverDialect


class PostgresResolverDialect(ResolverDialect):
    """Implementación de :class:`ResolverDialect` para PostgreSQL."""

    name = "postgres"
