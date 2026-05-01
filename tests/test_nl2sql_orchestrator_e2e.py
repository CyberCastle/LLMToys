#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Test end-to-end del orquestador NL2SQL con LLMs y BD mockeados."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from llm_core.prompt_optimizer import PromptTokenStats
from nl2sql.utils.decision_models import DecisionIssue

from nl2sql.orchestrator import NL2SQLConfig, NL2SQLRequest, run_nl2sql
from nl2sql.orchestrator.stages.prune_stage import build_prune_runnable
from nl2sql.orchestrator.stages.narrative_stage import build_narrative_runnable
from nl2sql.orchestrator.stages.resolver_stage import build_resolver_runnable
from nl2sql.orchestrator.stages.solver_stage import build_solver_runnable
from nl2sql.orchestrator.llm_manager import LLMManager
from nl2sql.sql_solver_generator.contracts import SolverMetadata, SolverOutput
from nl2sql.sql_solver_generator.spec_model import SQLQuerySpec


@pytest.fixture
def fake_assets(tmp_path: Path) -> dict[str, Path]:
    """Crea esquemas y assets de prompt minimos para la prueba e2e."""

    db_schema = tmp_path / "db_schema.yaml"
    db_schema.write_text(
        yaml.safe_dump(
            {
                "entity_c": {
                    "description": "Entidad agrupadora.",
                    "columns": [{"name": "id", "type": "BIGINT"}],
                    "primary_keys": ["id"],
                    "foreign_keys": [],
                }
            },
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    rules = tmp_path / "semantic_rules.yaml"
    rules.write_text(
        yaml.safe_dump(
            {
                "semantic_contract": {
                    "business_invariants": {
                        "semantic_models": [],
                        "semantic_entities": [],
                        "semantic_dimensions": [],
                        "semantic_metrics": [],
                        "semantic_filters": [],
                        "semantic_business_rules": [],
                        "semantic_relationships": [],
                        "semantic_constraints": [],
                        "semantic_join_paths": [],
                        "semantic_derived_metrics": [],
                    },
                    "retrieval_heuristics": {
                        "semantic_synonyms": {},
                        "semantic_examples": [],
                    },
                    "sql_safety": {
                        "execution_safety": {"forbidden_keywords": []},
                        "semantic_sql_business_rules": [],
                    },
                }
            },
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    config = tmp_path / "nl2sql_config.yaml"
    config.write_text(
        yaml.safe_dump(
            {
                "orchestrator": {
                    "narrative_prompt": {
                        "system": "eres un analista",
                        "user_template": "Q={query}\nSQL={sql}\nN={row_count}{truncated_marker}\n{rows_preview}",
                    }
                }
            },
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    return {
        "db_schema": db_schema,
        "rules": rules,
        "config": config,
        "out": tmp_path / "out",
    }


def _fake_engine_factory(*_args, **_kwargs) -> MagicMock:
    """Entrega un engine fake con una sola fila de resultado."""

    fake_conn = MagicMock()
    fake_result = MagicMock()
    fake_result.fetchmany.return_value = [(1, "Group A", 42)]
    fake_result.keys.return_value = ["id", "entity_c", "record_count"]
    fake_conn.execute.return_value = fake_result
    engine = MagicMock()
    engine.connect.return_value.__enter__.return_value = fake_conn
    return engine


def _fake_persist_pruned_schema(_result: object, *, query: str, out_path: str | Path) -> Path:
    """Persiste un YAML minimo de pruned schema para el flujo de prueba."""

    path = Path(out_path)
    payload = {
        "query": query,
        "pruned_schema": {
            "entity_c": {
                "columns": [{"name": "id", "type": "BIGINT"}],
                "primary_keys": ["id"],
                "foreign_keys": [],
            }
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path


def _fake_persist_semantic_plan(
    _semantic_plan: object,
    *,
    out_path: str | Path,
    pruned_schema_path: str | Path,
    rules_path: str | Path,
) -> Path:
    """Persiste un YAML minimo de semantic plan para el flujo de prueba."""

    path = Path(out_path)
    payload = {
        "semantic_plan": {
            "retrieved_candidates": {"query": "cual es el promedio de registros por entidad_c"},
            "compiled_plan": {
                "query": "cual es el promedio de registros por entidad_c",
                "intent": "scalar_metric",
                "base_entity": "entity_c",
                "grain": "entity_c.id",
                "measure": {"name": "record_count", "formula": "count(*)", "source_table": "entity_c"},
                "group_by": [],
                "join_path": [],
                "required_tables": ["entity_c"],
            },
        },
        "source_pruned_schema_path": str(pruned_schema_path),
        "source_rules_path": str(rules_path),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return path


def _fake_narrative_prompt_stats(
    *,
    final_prompt_tokens: int = 120,
    max_model_len: int = 2048,
    safety_margin_tokens: int = 256,
) -> PromptTokenStats:
    return PromptTokenStats(
        model_name="fake-narrative-model",
        tokenizer_name="fake-tokenizer",
        user_prompt_tokens=max(1, final_prompt_tokens - 12),
        final_prompt_tokens=final_prompt_tokens,
        max_model_len=max_model_len,
        safety_margin_tokens=safety_margin_tokens,
    )


def test_nl2sql_pipeline_end_to_end(fake_assets: dict[str, Path]) -> None:
    """Valida el flujo completo sin cargar modelos reales ni tocar una BD real."""

    solver_output = SolverOutput(
        sql_final="SELECT 1 AS id, 'Group A' AS entity_c, 42 AS record_count",
        sql_query_spec=SQLQuerySpec(
            query_type="scalar_metric",
            dialect="tsql",
            base_entity="entity_c",
            base_table="entity_c",
        ),
        metadata=SolverMetadata(model_used="fake-solver", dialect="tsql"),
    )

    with (
        patch("nl2sql.orchestrator.stages.prune_stage.run_semantic_schema_pruning") as prune_mock,
        patch("nl2sql.orchestrator.stages.prune_stage.persist_pruned_schema", side_effect=_fake_persist_pruned_schema),
        patch("nl2sql.orchestrator.stages.resolver_stage.run_semantic_resolver") as resolver_mock,
        patch("nl2sql.orchestrator.stages.resolver_stage.persist_semantic_plan", side_effect=_fake_persist_semantic_plan),
        patch("nl2sql.orchestrator.stages.solver_stage.run_sql_solver", return_value=solver_output) as solver_mock,
        patch("nl2sql.orchestrator.stages.execution_stage.build_engine", side_effect=_fake_engine_factory),
        patch("nl2sql.orchestrator.llm_manager.build_generic_runner") as runner_factory,
        patch(
            "nl2sql.orchestrator.stages.narrative_stage.count_prompt_tokens",
            return_value=_fake_narrative_prompt_stats(),
        ),
    ):
        prune_mock.return_value = MagicMock()
        resolver_mock.return_value = MagicMock()

        gemma_runner = MagicMock()
        gemma_runner.run.return_value = ["El promedio es 42 registros por entidad_c."]
        runner_factory.return_value = gemma_runner

        request = NL2SQLRequest(
            query="cual es el promedio de registros por entidad_c",
            db_schema_path=fake_assets["db_schema"],
            semantic_rules_path=fake_assets["rules"],
            out_dir=fake_assets["out"],
            dialect="tsql",
        )
        config = NL2SQLConfig(narrative_prompt_path=str(fake_assets["config"]))

        response = run_nl2sql(request, config)

    assert response.final_sql == "SELECT 1 AS id, 'Group A' AS entity_c, 42 AS record_count"
    assert response.status == "ok"
    assert response.row_count == 1
    assert response.rows[0]["entity_c"] == "Group A"
    assert "promedio" in response.narrative.lower()
    assert {artifact.name for artifact in response.artifacts} >= {
        "prune",
        "resolver",
        "solver",
        "execution",
        "narrative",
    }
    assert (fake_assets["out"] / "semantic_pruned_schema.yaml").exists()
    assert (fake_assets["out"] / "semantic_plan.yaml").exists()
    assert (fake_assets["out"] / "solver_result.sql").exists()
    assert (fake_assets["out"] / "solver_result.yaml").exists()
    assert not (fake_assets["out"] / "sql_execution_optimized.sql").exists()
    assert (fake_assets["out"] / "sql_execution_result.yaml").exists()
    assert (fake_assets["out"] / "narrative_response.yaml").exists()
    execution_artifact = next(artifact for artifact in response.artifacts if artifact.name == "execution")
    assert execution_artifact.payload["optimization"]["applied"] is False
    narrative_artifact = next(artifact for artifact in response.artifacts if artifact.name == "narrative")
    assert narrative_artifact.payload["prompt_diagnostics"]["prompt_variant"] == "full_sql_full_rows"
    prune_mock.assert_called_once()
    resolver_mock.assert_called_once()
    solver_mock.assert_called_once()
    runner_factory.assert_called_once_with("gemma4_e4b")
    gemma_runner.run.assert_called_once()


def test_nl2sql_pipeline_devuelve_fallo_estructurado_si_el_solver_no_produce_sql(fake_assets: dict[str, Path]) -> None:
    """Devuelve status estructurado y evita ejecutar SQL cuando el solver falla."""

    solver_output = SolverOutput(
        sql_final="",
        sql_query_spec=SQLQuerySpec(
            query_type="scalar_metric",
            dialect="tsql",
            base_entity="entity_c",
            base_table="entity_c",
        ),
        metadata=SolverMetadata(model_used="fake-solver", dialect="tsql"),
        issues=[
            DecisionIssue(
                stage="sql_generation",
                code="model_raised",
                severity="error",
                message="insufficient_vram",
            )
        ],
    )

    with (
        patch("nl2sql.orchestrator.stages.prune_stage.run_semantic_schema_pruning") as prune_mock,
        patch("nl2sql.orchestrator.stages.prune_stage.persist_pruned_schema", side_effect=_fake_persist_pruned_schema),
        patch("nl2sql.orchestrator.stages.resolver_stage.run_semantic_resolver") as resolver_mock,
        patch("nl2sql.orchestrator.stages.resolver_stage.persist_semantic_plan", side_effect=_fake_persist_semantic_plan),
        patch("nl2sql.orchestrator.stages.solver_stage.run_sql_solver", return_value=solver_output) as solver_mock,
        patch("nl2sql.orchestrator.stages.execution_stage.build_engine") as engine_factory,
        patch("nl2sql.orchestrator.llm_manager.build_generic_runner") as runner_factory,
    ):
        prune_mock.return_value = MagicMock()
        resolver_mock.return_value = MagicMock()

        request = NL2SQLRequest(
            query="cual es el promedio de registros por entidad_c",
            db_schema_path=fake_assets["db_schema"],
            semantic_rules_path=fake_assets["rules"],
            out_dir=fake_assets["out"],
            dialect="tsql",
        )
        config = NL2SQLConfig(narrative_prompt_path=str(fake_assets["config"]))

        response = run_nl2sql(request, config)

    assert response.status == "failed_runtime"
    assert response.final_sql == ""
    assert any(issue.code == "model_raised" for issue in response.issues)
    assert (fake_assets["out"] / "solver_result.yaml").exists()
    assert not (fake_assets["out"] / "solver_result.sql").exists()
    engine_factory.assert_not_called()
    runner_factory.assert_not_called()
    solver_mock.assert_called_once()


def test_prune_stage_libera_runtime_e2rank(fake_assets: dict[str, Path]) -> None:
    """El orquestador debe liberar E2Rank al terminar pruning para no contaminar la VRAM del solver."""

    request = NL2SQLRequest(
        query="cual es el promedio de registros por entidad_c",
        db_schema_path=fake_assets["db_schema"],
        semantic_rules_path=fake_assets["rules"],
        out_dir=fake_assets["out"],
        dialect="tsql",
    )

    with (
        patch("nl2sql.orchestrator.stages.prune_stage.run_semantic_schema_pruning", return_value=MagicMock()),
        patch("nl2sql.orchestrator.stages.prune_stage.persist_pruned_schema", side_effect=_fake_persist_pruned_schema),
        patch("nl2sql.orchestrator.stages.prune_stage.clear_e2rank_runtime") as clear_mock,
    ):
        state = build_prune_runnable().invoke({"request": request, "artifacts": []})

    assert "pruned_schema_path" in state
    clear_mock.assert_called_once()


def test_resolver_stage_libera_runtimes_del_resolver(fake_assets: dict[str, Path]) -> None:
    """El resolver debe dejar la GPU limpia al salir, incluso cuando el orquestador lo llama inline."""

    pruned_schema_path = _fake_persist_pruned_schema(
        MagicMock(),
        query="cual es el promedio de registros por entidad_c",
        out_path=fake_assets["out"] / "semantic_pruned_schema.yaml",
    )
    request = NL2SQLRequest(
        query="cual es el promedio de registros por entidad_c",
        db_schema_path=fake_assets["db_schema"],
        semantic_rules_path=fake_assets["rules"],
        out_dir=fake_assets["out"],
        dialect="tsql",
    )

    with (
        patch("nl2sql.orchestrator.stages.resolver_stage.run_semantic_resolver", return_value=MagicMock()),
        patch("nl2sql.orchestrator.stages.resolver_stage.persist_semantic_plan", side_effect=_fake_persist_semantic_plan),
        patch("nl2sql.orchestrator.stages.resolver_stage.release_semantic_resolver_runtimes") as release_mock,
    ):
        state = build_resolver_runnable().invoke(
            {
                "request": request,
                "artifacts": [],
                "pruned_schema_path": pruned_schema_path,
            }
        )

    assert "semantic_plan_path" in state
    release_mock.assert_called_once()


def test_resolver_stage_bloquea_issue_semantico_determinista(fake_assets: dict[str, Path]) -> None:
    """La etapa debe propagar fallos semanticos deterministas antes del solver."""

    def _persist_semantic_plan_with_blocking_issue(
        _semantic_plan: object,
        *,
        out_path: str | Path,
        pruned_schema_path: str | Path,
        rules_path: str | Path,
    ) -> Path:
        path = Path(out_path)
        payload = {
            "semantic_plan": {
                "retrieved_candidates": {"query": "q"},
                "compiled_plan": {
                    "query": "q",
                    "intent": "simple_metric",
                    "base_entity": "entity_c",
                    "grain": "entity_c.id",
                    "measure": {"name": "record_count", "formula": "count(*)", "source_table": "entity_c"},
                    "group_by": [],
                    "join_path": [],
                    "required_tables": ["entity_c"],
                    "issues": [
                        DecisionIssue(
                            stage="semantic_compilation",
                            code="time_filter_unresolved",
                            severity="error",
                            message="falta filtro temporal operativo",
                            context={"base_entity": "entity_c"},
                        ).model_dump(mode="python")
                    ],
                },
            },
            "source_pruned_schema_path": str(pruned_schema_path),
            "source_rules_path": str(rules_path),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")
        return path

    pruned_schema_path = _fake_persist_pruned_schema(
        MagicMock(),
        query="q",
        out_path=fake_assets["out"] / "semantic_pruned_schema.yaml",
    )
    request = NL2SQLRequest(
        query="q",
        db_schema_path=fake_assets["db_schema"],
        semantic_rules_path=fake_assets["rules"],
        out_dir=fake_assets["out"],
        dialect="tsql",
    )

    with (
        patch("nl2sql.orchestrator.stages.resolver_stage.run_semantic_resolver", return_value=MagicMock()),
        patch(
            "nl2sql.orchestrator.stages.resolver_stage.persist_semantic_plan",
            side_effect=_persist_semantic_plan_with_blocking_issue,
        ),
    ):
        state = build_resolver_runnable().invoke(
            {
                "request": request,
                "artifacts": [],
                "issues": [],
                "warnings": [],
                "status": "ok",
                "pruned_schema_path": pruned_schema_path,
            }
        )

    assert state["status"] == "failed_semantic_alignment"
    assert any(issue.code == "time_filter_unresolved" for issue in state["issues"])


def test_solver_stage_detiene_ejecucion_si_verificacion_semantica_falla(fake_assets: dict[str, Path]) -> None:
    semantic_plan_path = fake_assets["out"] / "semantic_plan.yaml"
    semantic_plan_path.parent.mkdir(parents=True, exist_ok=True)
    semantic_plan_path.write_text(
        yaml.safe_dump(
            {
                "semantic_plan": {
                    "retrieved_candidates": {"query": "q"},
                    "compiled_plan": {
                        "query": "q",
                        "intent": "scalar_metric",
                        "base_entity": "entity_c",
                        "grain": "entity_c.id",
                        "measure": {"name": "record_count", "formula": "count(*)", "source_table": "entity_c"},
                        "group_by": [],
                        "join_path": [],
                        "required_tables": ["entity_c"],
                        "verification": {
                            "is_semantically_aligned": False,
                            "missing_filters": ["status_a"],
                            "confidence": 0.2,
                            "rationale": "falta filtro",
                        },
                    },
                }
            },
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    pruned_schema_path = _fake_persist_pruned_schema(
        MagicMock(),
        query="q",
        out_path=fake_assets["out"] / "semantic_pruned_schema.yaml",
    )
    request = NL2SQLRequest(
        query="q",
        db_schema_path=fake_assets["db_schema"],
        semantic_rules_path=fake_assets["rules"],
        out_dir=fake_assets["out"],
        dialect="tsql",
    )

    with patch("nl2sql.orchestrator.stages.solver_stage.run_sql_solver") as solver_mock:
        state = build_solver_runnable().invoke(
            {
                "request": request,
                "artifacts": [],
                "status": "failed_semantic_alignment",
                "semantic_plan_path": semantic_plan_path,
                "pruned_schema_path": pruned_schema_path,
            }
        )

    assert state["status"] == "failed_semantic_alignment"
    solver_mock.assert_not_called()


def test_narrative_stage_reduce_prompt_si_full_preview_excede_contexto(fake_assets: dict[str, Path]) -> None:
    request = NL2SQLRequest(
        query="cual es el promedio de registros por entidad_c",
        db_schema_path=fake_assets["db_schema"],
        semantic_rules_path=fake_assets["rules"],
        out_dir=fake_assets["out"],
        dialect="tsql",
    )
    rows = [
        {"entity_c": f"Group {index}", "record_count": index, "comment": "fila extensa para empujar el prompt narrativo"}
        for index in range(1, 6)
    ]
    prompt_stats = [
        _fake_narrative_prompt_stats(final_prompt_tokens=2200),
        _fake_narrative_prompt_stats(final_prompt_tokens=1200),
    ]

    with (
        patch(
            "nl2sql.orchestrator.stages.narrative_stage.count_prompt_tokens",
            side_effect=prompt_stats,
        ),
        patch("nl2sql.orchestrator.llm_manager.build_generic_runner") as runner_factory,
    ):
        gemma_runner = MagicMock()
        gemma_runner.run.return_value = ["Respuesta narrativa compacta."]
        runner_factory.return_value = gemma_runner

        config = NL2SQLConfig(narrative_prompt_path=str(fake_assets["config"]), rows_preview_limit=5)
        state = build_narrative_runnable(LLMManager(config.narrative), config, fake_assets["config"]).invoke(
            {
                "request": request,
                "artifacts": [],
                "final_sql": "SELECT entity_c, AVG(record_count) FROM wide_table GROUP BY entity_c ORDER BY entity_c",
                "row_count": len(rows),
                "truncated": False,
                "rows": rows,
            }
        )

    assert state["narrative"] == "Respuesta narrativa compacta."
    assert state["narrative_prompt_diagnostics"]["prompt_variant"] == "full_sql_trimmed_rows"
    assert state["narrative_prompt_diagnostics"]["prompt_tokens"] == 1200
    narrative_artifact = next(artifact for artifact in state["artifacts"] if artifact.name == "narrative")
    assert narrative_artifact.payload["prompt_diagnostics"]["rows_preview_limit"] == 1
    gemma_runner.run.assert_called_once()
