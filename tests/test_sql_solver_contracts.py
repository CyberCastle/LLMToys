#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from nl2sql.sql_solver_generator import SolverConfig, SolverInput, SolverMetadata, SolverOutput, run_sql_solver
from nl2sql.sql_solver_generator.llm_router import LlmAttempt
from nl2sql.sql_solver_generator.stages.generation_stage import GenerationResult
from nl2sql.sql_solver_generator.spec_model import SQLQuerySpec
from nl2sql.utils.decision_models import DecisionIssue

MINIMAL_SEMANTIC_RULES = {
    "semantic_contract": {
        "business_invariants": {},
        "retrieval_heuristics": {},
        "sql_safety": {},
    }
}


def test_public_contracts_are_constructible() -> None:
    spec = SQLQuerySpec(query_type="scalar_metric", dialect="tsql", base_entity="entity_c", base_table="entity_c")
    output = SolverOutput(sql_final="SELECT 1", sql_query_spec=spec, metadata=SolverMetadata())
    input_payload = SolverInput(semantic_plan={}, pruned_schema={}, semantic_rules=MINIMAL_SEMANTIC_RULES)
    config = SolverConfig()

    assert output.sql_final == "SELECT 1"
    assert input_payload.semantic_plan == {}
    assert config.model == "XGenerationLab/XiYanSQL-QwenCoder-7B-2504"
    assert config.max_model_len == 2048
    assert config.max_tokens == 384
    assert config.gpu_memory_utilization == 0.90
    assert config.enforce_eager is True
    assert config.cpu_offload_gb == 3.0


def test_solver_config_admite_overrides_por_entorno(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SQL_SOLVER_LLM_DTYPE", "auto")
    monkeypatch.setenv("SQL_SOLVER_GPU_MEMORY_UTILIZATION", "0.82")
    monkeypatch.setenv("SQL_SOLVER_ENFORCE_EAGER", "false")
    monkeypatch.setenv("SQL_SOLVER_CPU_OFFLOAD_GB", "0.0")
    monkeypatch.setenv("SQL_SOLVER_SWAP_SPACE_GB", "6.0")

    config = SolverConfig()

    assert config.llm_dtype == "auto"
    assert config.gpu_memory_utilization == pytest.approx(0.82)
    assert config.enforce_eager is False
    assert config.cpu_offload_gb == pytest.approx(0.0)
    assert config.swap_space_gb == pytest.approx(6.0)


def test_run_sql_solver_revalida_sql_normalizado_y_descarta_issues_obsoletos() -> None:
    solver_input = SolverInput(semantic_plan={}, pruned_schema={}, semantic_rules=MINIMAL_SEMANTIC_RULES)
    spec = SQLQuerySpec(
        query_type="ranking",
        dialect="tsql",
        base_entity="cliente",
        base_table="cliente",
    )
    stale_issue = DecisionIssue(
        stage="sql_validation",
        code="unknown_bare_column",
        severity="error",
        message="El SQL usa un identificador desnudo que no existe en el schema ni como alias valido.",
        context={"column": "nombre_fantasia"},
    )
    validator = Mock()
    validator.validate_full.return_value = []

    with (
        patch(
            "nl2sql.sql_solver_generator.solver.run_plan_normalization_stage",
            return_value=SimpleNamespace(
                semantic_plan={},
                pruned_schema={"cliente": {"columns": {"nombre_fantasia": {}, "id": {}}}},
                semantic_rules=MINIMAL_SEMANTIC_RULES,
            ),
        ),
        patch(
            "nl2sql.sql_solver_generator.solver.load_solver_rules",
            return_value=SimpleNamespace(forbidden_keywords=frozenset(), hard_join_blacklist=[]),
        ),
        patch("nl2sql.sql_solver_generator.solver.load_business_rules", return_value=[]),
        patch("nl2sql.sql_solver_generator.solver.load_solver_prompts", return_value={}),
        patch("nl2sql.sql_solver_generator.solver.ValidationStage", return_value=validator),
        patch(
            "nl2sql.sql_solver_generator.solver.run_generation",
            return_value=GenerationResult(
                attempt=LlmAttempt(
                    model_name="fake-solver",
                    spec=spec,
                    sql="SELECT nombre_fantasia FROM OrdenesTrabajoCount",
                    issues=[stale_issue],
                    attempts=2,
                )
            ),
        ),
        patch(
            "nl2sql.sql_solver_generator.solver.run_sql_normalization",
            return_value=(
                "WITH OrdenesTrabajoCount AS (SELECT cliente.nombre_fantasia AS nombre_fantasia FROM cliente) "
                "SELECT nombre_fantasia FROM OrdenesTrabajoCount"
            ),
        ),
    ):
        result = run_sql_solver(solver_input, SolverConfig())

    assert result.issues == []
    assert result.metadata.validator_trace == []
    validator.validate_full.assert_called_once_with(
        spec,
        "WITH OrdenesTrabajoCount AS (SELECT cliente.nombre_fantasia AS nombre_fantasia FROM cliente) "
        "SELECT nombre_fantasia FROM OrdenesTrabajoCount",
    )
