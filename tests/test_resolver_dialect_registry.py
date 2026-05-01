#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Tests del registry de dialectos de ``semantic_resolver``."""

from __future__ import annotations

import pytest

from nl2sql.semantic_resolver.dialects import (
    PostgresResolverDialect,
    ResolverDialect,
    TsqlResolverDialect,
    get_resolver_dialect,
)
from nl2sql.semantic_resolver.config import resolve_compiler_rules_path
from nl2sql.semantic_resolver.rules_loader import load_compiler_rules


def test_get_resolver_dialect_devuelve_instancias_correctas() -> None:
    """El registry instancia el dialecto correcto para cada alias soportado."""

    tsql = get_resolver_dialect("tsql")
    postgres = get_resolver_dialect("postgres")
    assert isinstance(tsql, TsqlResolverDialect)
    assert isinstance(postgres, PostgresResolverDialect)
    assert tsql.name == "tsql"
    assert postgres.name == "postgres"


@pytest.mark.parametrize("alias", ["TSQL", "sqlserver", "MSSQL"])
def test_get_resolver_dialect_acepta_alias_tsql(alias: str) -> None:
    """Los alias comunes deben mapear a T-SQL sin distinguir mayusculas."""

    dialect = get_resolver_dialect(alias)
    assert isinstance(dialect, TsqlResolverDialect)


@pytest.mark.parametrize("alias", ["PostgreSQL", "pg"])
def test_get_resolver_dialect_acepta_alias_postgres(alias: str) -> None:
    """Los alias comunes deben mapear a PostgreSQL."""

    dialect = get_resolver_dialect(alias)
    assert isinstance(dialect, PostgresResolverDialect)


def test_get_resolver_dialect_dialecto_desconocido_falla() -> None:
    """Cualquier dialecto no registrado debe levantar ValueError."""

    with pytest.raises(ValueError):
        get_resolver_dialect("oracle")


def test_resolver_dialects_implementan_render_time_expression() -> None:
    """Tanto T-SQL como PostgreSQL deben mapear las expresiones canonicas base."""

    tsql: ResolverDialect = TsqlResolverDialect()
    postgres: ResolverDialect = PostgresResolverDialect()
    rules = load_compiler_rules(str(resolve_compiler_rules_path()))
    assert tsql.render_time_expression("today - 1 year", compiler_rules=rules) is not None
    assert postgres.render_time_expression("today - 1 year", compiler_rules=rules) is not None
    assert tsql.render_time_expression("expresion_inexistente", compiler_rules=rules) is None
    assert postgres.render_time_expression("expresion_inexistente", compiler_rules=rules) is None
