#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import pytest

from nl2sql.sql_solver_generator import get_dialect


def test_tsql_uses_top_not_limit() -> None:
    sql = get_dialect("tsql").render_row_limit("SELECT a FROM t", 10)
    assert "TOP 10" in sql
    assert "LIMIT" not in sql


def test_tsql_preserva_top_existente() -> None:
    sql = get_dialect("tsql").render_row_limit("SELECT TOP 5 a FROM t", 10)
    assert sql == "SELECT TOP 5 a FROM t"


def test_tsql_inserta_top_despues_de_distinct() -> None:
    sql = get_dialect("tsql").render_row_limit("SELECT DISTINCT a FROM t", 10)
    assert sql == "SELECT DISTINCT TOP 10 a FROM t"


def test_postgres_uses_limit_not_top() -> None:
    sql = get_dialect("postgres").render_row_limit("SELECT a FROM t", 10)
    assert "LIMIT 10" in sql
    assert "TOP" not in sql


def test_postgres_preserva_limit_existente() -> None:
    sql = get_dialect("postgres").render_row_limit("SELECT a FROM t LIMIT 5", 10)
    assert sql == "SELECT a FROM t LIMIT 5"


def test_postgres_valida_select_basico() -> None:
    assert get_dialect("postgres").validate_syntax("SELECT 1") == []


def test_dialecto_desconocido_falla() -> None:
    with pytest.raises(ValueError):
        get_dialect("mysql")
