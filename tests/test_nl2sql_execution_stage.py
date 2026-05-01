#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Pruebas para la optimizacion SQL previa a la ejecucion del pipeline NL2SQL."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import yaml

from nl2sql.orchestrator import NL2SQLConfig, NL2SQLRequest
from nl2sql.orchestrator.db.sql_optimizer import optimize_sql_for_execution
from nl2sql.orchestrator.stages.execution_stage import build_execution_runnable


def _write_db_schema(path: Path) -> Path:
    """Persiste un esquema minimo compatible con el optimizador de ejecucion."""

    schema_path = path / "db_schema.yaml"
    schema_path.write_text(
        yaml.safe_dump(
            {
                "entity_c": {
                    "description": "Entidad agrupadora.",
                    "columns": [
                        {"name": "id", "type": "BIGINT"},
                        {"name": "nombre", "type": "VARCHAR"},
                    ],
                    "primary_keys": ["id"],
                    "foreign_keys": [],
                }
            },
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    return schema_path


def test_optimize_sql_for_execution_califica_columnas_con_schema(tmp_path: Path) -> None:
    """La optimizacion debe usar el esquema para calificar identificadores desnudos."""

    schema_path = _write_db_schema(tmp_path)

    result = optimize_sql_for_execution(
        "SELECT id FROM entity_c WHERE id = 1",
        dialect="tsql",
        schema_source=schema_path,
    )

    assert "[entity_c].[id]" in result.sql
    assert "FROM [entity_c] AS [entity_c]" in result.sql
    assert result.schema_tables == 1
    assert result.schema_columns == 2


def test_execution_stage_ejecuta_sql_optimizada_y_persiste_trazabilidad(tmp_path: Path) -> None:
    """La etapa de ejecucion debe correr la SQL optimizada y guardar ambas versiones."""

    schema_path = _write_db_schema(tmp_path)
    out_dir = tmp_path / "out"
    request = NL2SQLRequest(
        query="cuantas entidades_c existen?",
        db_schema_path=schema_path,
        semantic_rules_path=tmp_path / "semantic_rules.yaml",
        out_dir=out_dir,
        dialect="tsql",
    )

    fake_conn = MagicMock()
    fake_result = MagicMock()
    fake_result.fetchmany.return_value = [(1,)]
    fake_result.keys.return_value = ["id"]
    fake_conn.execute.return_value = fake_result
    fake_engine = MagicMock()
    fake_engine.connect.return_value.__enter__.return_value = fake_conn

    state = {
        "request": request,
        "artifacts": [],
        "final_sql": "SELECT id FROM entity_c WHERE id = 1",
    }

    with patch("nl2sql.orchestrator.stages.execution_stage.build_engine", return_value=fake_engine):
        output = build_execution_runnable(NL2SQLConfig(execution_sql_optimization_enabled=True)).invoke(state)

    assert output["generated_sql"] == "SELECT id FROM entity_c WHERE id = 1"
    assert output["final_sql"] != output["generated_sql"]
    assert "[entity_c].[id]" in output["final_sql"]

    executed_clause = fake_conn.execute.call_args.args[0]
    assert executed_clause.text == output["final_sql"]

    execution_artifact = next(artifact for artifact in output["artifacts"] if artifact.name == "execution")
    assert execution_artifact.payload["original_sql"] == output["generated_sql"]
    assert execution_artifact.payload["sql"] == output["final_sql"]
    assert execution_artifact.payload["optimization"]["schema_tables"] == 1
    assert execution_artifact.payload["optimization"]["schema_columns"] == 2

    optimized_sql_path = out_dir / "sql_execution_optimized.sql"
    assert optimized_sql_path.exists()
    assert optimized_sql_path.read_text(encoding="utf-8") == output["final_sql"]

    execution_yaml = yaml.safe_load((out_dir / "sql_execution_result.yaml").read_text(encoding="utf-8"))
    assert execution_yaml["original_sql"] == output["generated_sql"]
    assert execution_yaml["sql"] == output["final_sql"]


def test_execution_stage_no_optimiza_si_el_flag_esta_desactivado(tmp_path: Path) -> None:
    """La etapa debe ejecutar el SQL original cuando la optimizacion esta apagada."""

    schema_path = _write_db_schema(tmp_path)
    out_dir = tmp_path / "out"
    original_sql = "SELECT id FROM entity_c WHERE id = 1"
    request = NL2SQLRequest(
        query="cuantas entidades_c existen?",
        db_schema_path=schema_path,
        semantic_rules_path=tmp_path / "semantic_rules.yaml",
        out_dir=out_dir,
        dialect="tsql",
    )

    fake_conn = MagicMock()
    fake_result = MagicMock()
    fake_result.fetchmany.return_value = [(1,)]
    fake_result.keys.return_value = ["id"]
    fake_conn.execute.return_value = fake_result
    fake_engine = MagicMock()
    fake_engine.connect.return_value.__enter__.return_value = fake_conn

    state = {
        "request": request,
        "artifacts": [],
        "final_sql": original_sql,
    }

    with patch("nl2sql.orchestrator.stages.execution_stage.build_engine", return_value=fake_engine):
        output = build_execution_runnable(NL2SQLConfig()).invoke(state)

    assert output["generated_sql"] == original_sql
    assert output["final_sql"] == original_sql
    assert "optimized_sql_path" not in output

    executed_clause = fake_conn.execute.call_args.args[0]
    assert executed_clause.text == original_sql

    execution_artifact = next(artifact for artifact in output["artifacts"] if artifact.name == "execution")
    assert execution_artifact.payload["optimization"]["applied"] is False
    assert execution_artifact.payload["optimization"]["reason"] == "disabled_via_config"
    assert execution_artifact.payload["optimization_seconds"] == 0.0

    assert not (out_dir / "sql_execution_optimized.sql").exists()

    execution_yaml = yaml.safe_load((out_dir / "sql_execution_result.yaml").read_text(encoding="utf-8"))
    assert execution_yaml["sql"] == original_sql
    assert execution_yaml["optimization"]["applied"] is False


def test_execution_stage_devuelve_failed_runtime_estructurado_si_falla_la_base(tmp_path: Path) -> None:
    schema_path = _write_db_schema(tmp_path)
    out_dir = tmp_path / "out"
    original_sql = "SELECT id FROM entity_c WHERE id = 1"
    request = NL2SQLRequest(
        query="cuantas entidades_c existen?",
        db_schema_path=schema_path,
        semantic_rules_path=tmp_path / "semantic_rules.yaml",
        out_dir=out_dir,
        dialect="tsql",
    )

    fake_conn = MagicMock()
    fake_conn.execute.side_effect = RuntimeError("db exploded")
    fake_engine = MagicMock()
    fake_engine.connect.return_value.__enter__.return_value = fake_conn

    state = {
        "request": request,
        "artifacts": [],
        "issues": [],
        "warnings": [],
        "status": "ok",
        "final_sql": original_sql,
    }

    with patch("nl2sql.orchestrator.stages.execution_stage.build_engine", return_value=fake_engine):
        output = build_execution_runnable(NL2SQLConfig()).invoke(state)

    assert output["status"] == "failed_runtime"
    assert output["rows"] == []
    assert any(issue.code == "execution_stage_failed" for issue in output["issues"])

    execution_yaml = yaml.safe_load((out_dir / "sql_execution_result.yaml").read_text(encoding="utf-8"))
    assert execution_yaml["error"]["code"] == "execution_stage_failed"
    assert execution_yaml["sql"] == original_sql
