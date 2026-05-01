#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

COVERAGE_MATRIX: dict[str, dict[str, str]] = {
    "metrica_derivada_promedio": {
        "tsql": "tests/test_sql_solver_e2e_avg_a_per_c.py::test_avg_a_per_c",
        "postgres": "tests/test_semantic_plan_compiler.py::test_compile_semantic_plan_builds_two_level_aggregation",
    },
    "filtro_temporal": {
        "tsql": "tests/test_semantic_plan_compiler.py::test_compile_semantic_plan_builds_two_level_aggregation",
        "postgres": "tests/test_resolver_time_filter_dialect.py::test_extract_time_filter_con_entity_time_field_postgres",
    },
    "filtro_estado_semantico": {
        "tsql": "tests/test_semantic_plan_compiler.py::test_compile_semantic_plan_uses_status_a_for_active_entity_a",
        "postgres": "tests/test_semantic_plan_compiler.py::test_compile_semantic_plan_uses_status_a_for_active_entity_a",
    },
    "joins_largos": {
        "tsql": "tests/test_semantic_plan_compiler.py::test_compile_semantic_plan_builds_two_level_aggregation",
        "postgres": "tests/test_semantic_plan_compiler.py::test_compile_semantic_plan_builds_two_level_aggregation",
    },
    "verificacion_semantica": {
        "tsql": "tests/test_semantic_resolver_verifier.py::test_verify_compiled_plan_inyecta_ejemplos_curados_en_prompt",
        "postgres": "tests/test_semantic_resolver_verifier.py::test_verify_compiled_plan_inyecta_ejemplos_curados_en_prompt",
    },
    "guardrail_orquestador": {
        "tsql": "tests/test_nl2sql_orchestrator_e2e.py::test_solver_stage_detiene_ejecucion_si_verificacion_semantica_falla",
        "postgres": "tests/test_nl2sql_orchestrator_e2e.py::test_solver_stage_detiene_ejecucion_si_verificacion_semantica_falla",
    },
}


def render_coverage_matrix_report() -> str:
    """Renderiza la matriz de cobertura como markdown simple para CI."""

    lines = ["# COBERTURA NL2SQL", "", "| tipo_consulta | tsql | postgres |", "| --- | --- | --- |"]
    for query_type, dialect_rows in COVERAGE_MATRIX.items():
        lines.append(f"| {query_type} | {dialect_rows.get('tsql', '')} | {dialect_rows.get('postgres', '')} |")

    gaps = [
        f"- {query_type}/{dialect}"
        for query_type, dialect_rows in COVERAGE_MATRIX.items()
        for dialect, ref in dialect_rows.items()
        if not ref
    ]
    lines.extend(["", "## Huecos", *(gaps or ["- sin huecos declarados"])])
    return "\n".join(lines)
