#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import pytest
import sqlglot

from nl2sql.sql_solver_generator import get_dialect
from nl2sql.sql_solver_generator.sql_normalizer import normalize_sql_via_ast


def test_normaliza_select_tsql() -> None:
    output = normalize_sql_via_ast("select top 5 id from entity_c", get_dialect("tsql"))
    assert "SELECT" in output.upper()
    assert "ENTITY_C" in output.upper()


def test_falla_sql_invalido() -> None:
    with pytest.raises(sqlglot.errors.ParseError):
        normalize_sql_via_ast("SELECT FROM WHERE", get_dialect("tsql"))


def test_tsql_promedio_de_alias_count_se_castea_a_decimal() -> None:
    raw_sql = """
        SELECT AVG(n_a) AS avg_a_per_c
        FROM (
            SELECT c.id AS entity_c_id, COUNT(DISTINCT a.id) AS n_a
            FROM entity_c AS c
            JOIN entity_a AS a ON c.id = a.entity_c_id
            GROUP BY c.id
        ) AS t
        """

    output = normalize_sql_via_ast(raw_sql, get_dialect("tsql"))

    assert "AVG(CAST(n_a AS DECIMAL(18,2)))" in output


def test_tsql_promedio_de_columna_no_count_no_se_castea() -> None:
    raw_sql = """
        SELECT AVG(monto) AS promedio_monto
        FROM (
            SELECT c.id AS entity_c_id, SUM(v.amount_total) AS monto
            FROM entity_c AS c
            JOIN fact_sale AS v ON c.id = v.entity_c_id
            GROUP BY c.id
        ) AS t
        """

    output = normalize_sql_via_ast(raw_sql, get_dialect("tsql"))

    assert "AVG(monto)" in output
    assert "AVG(CAST(monto" not in output
