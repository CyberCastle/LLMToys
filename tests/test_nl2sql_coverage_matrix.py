#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from tests.support_coverage_matrix import COVERAGE_MATRIX, render_coverage_matrix_report


def test_coverage_matrix_no_tiene_huecos_criticos() -> None:
    critical_cells = {
        ("metrica_derivada_promedio", "tsql"),
        ("filtro_temporal", "postgres"),
        ("verificacion_semantica", "tsql"),
        ("guardrail_orquestador", "tsql"),
    }

    missing = [
        (query_type, dialect)
        for query_type, dialect in critical_cells
        if not COVERAGE_MATRIX.get(query_type, {}).get(dialect)
    ]

    assert missing == []


def test_coverage_matrix_renderiza_reporte_markdown() -> None:
    report = render_coverage_matrix_report()

    assert "COBERTURA NL2SQL" in report
    assert "metrica_derivada_promedio" in report
    assert "verificacion_semantica" in report