#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Dialecto T-SQL (SQL Server) para ``semantic_resolver``.

La traducción de expresiones temporales canónicas se resuelve desde
``semantic_resolver.compiler_rules``; este módulo solo declara el nombre del
dialecto para usar la clave ``value_tsql`` correspondiente.
"""

from __future__ import annotations

from .base import ResolverDialect


class TsqlResolverDialect(ResolverDialect):
    """Implementación de :class:`ResolverDialect` para SQL Server (T-SQL)."""

    name = "tsql"
