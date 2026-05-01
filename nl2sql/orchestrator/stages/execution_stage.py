#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Etapa LangChain que ejecuta el SQL generado via SQLAlchemy."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from langchain_core.runnables import RunnableLambda
import sqlglot
from sqlalchemy import text
import yaml

from nl2sql.utils.decision_models import DecisionIssue, dedupe_decision_issues
from nl2sql.utils.yaml_utils import normalize_for_yaml

from ..config import NL2SQLConfig
from ..contracts import StageArtifact
from ..db.engine_factory import build_engine
from ..db.result_normalizer import rows_to_dicts
from ..db.sql_optimizer import optimize_sql_for_execution


def build_execution_runnable(config: NL2SQLConfig) -> RunnableLambda:
    """Construye la etapa de ejecucion SQL como `RunnableLambda`."""

    def _run(state: dict[str, Any]) -> dict[str, Any]:
        if state.get("status", "ok") != "ok":
            return state
        request = state["request"]
        sql = state.get("final_sql", "")
        if not isinstance(sql, str) or not sql.strip():
            solver_issues = list(getattr(state.get("solver_result"), "issues", []) or [])
            if solver_issues:
                raise RuntimeError(f"No existe SQL final para ejecutar. Issues del solver: {'; '.join(solver_issues)}")
            raise ValueError("No existe SQL final para ejecutar")

        stage_started = time.perf_counter()
        original_sql = sql
        state["generated_sql"] = original_sql
        optimization_elapsed = 0.0
        optimization_payload: dict[str, Any] = {
            "applied": False,
            "reason": "disabled_via_config",
        }

        if config.execution_sql_optimization_enabled:
            try:
                optimization_started = time.perf_counter()
                optimization_result = optimize_sql_for_execution(
                    original_sql,
                    dialect=request.dialect,
                    schema_source=request.db_schema_path,
                )
                optimization_elapsed = time.perf_counter() - optimization_started
            except (sqlglot.errors.SqlglotError, ValueError, FileNotFoundError) as exc:
                raise RuntimeError("No se pudo optimizar el SQL final con sqlglot antes de su ejecucion. " f"Detalle: {exc}") from exc

            sql = optimization_result.sql
            optimized_sql_path = Path(request.out_dir) / "sql_execution_optimized.sql"
            optimized_sql_path.parent.mkdir(parents=True, exist_ok=True)
            optimized_sql_path.write_text(sql, encoding="utf-8")
            state["optimized_sql_path"] = optimized_sql_path
            optimization_payload = {
                "applied": True,
                "schema_path": str(Path(request.db_schema_path).expanduser().resolve()),
                "schema_tables": optimization_result.schema_tables,
                "schema_columns": optimization_result.schema_columns,
                "optimized_sql_path": str(optimized_sql_path),
            }
        else:
            state.pop("optimized_sql_path", None)

        state["final_sql"] = sql

        engine = build_engine(request.dialect)
        execution_started = time.perf_counter()
        try:
            with engine.connect() as connection:
                result = connection.execute(text(sql))
                fetched_rows = result.fetchmany(config.max_rows + 1)
                columns = result.keys()
        except Exception as exc:  # noqa: BLE001
            execution_elapsed = time.perf_counter() - execution_started
            stage_elapsed = time.perf_counter() - stage_started
            issue = DecisionIssue(
                stage="sql_execution",
                code="execution_stage_failed",
                severity="error",
                message=str(exc),
                context={"error_type": exc.__class__.__name__},
            )
            payload = {
                "query": request.query,
                "sql": sql,
                "original_sql": original_sql,
                "optimization_seconds": optimization_elapsed,
                "optimization": optimization_payload,
                "execution_seconds": execution_elapsed,
                "error": issue.model_dump(mode="python"),
            }
            artifact_path = Path(request.out_dir) / "sql_execution_result.yaml"
            artifact_path.parent.mkdir(parents=True, exist_ok=True)
            artifact_path.write_text(
                yaml.safe_dump(normalize_for_yaml(payload), sort_keys=False, allow_unicode=True),
                encoding="utf-8",
            )
            state.setdefault("artifacts", []).append(
                StageArtifact(
                    name="execution",
                    path=artifact_path,
                    payload=payload,
                    duration_seconds=stage_elapsed,
                )
            )
            state["issues"] = dedupe_decision_issues([*list(state.get("issues", []) or []), issue])
            state["status"] = "failed_runtime"
            state["rows"] = []
            state["row_count"] = 0
            state["truncated"] = False
            state["execution_seconds"] = execution_elapsed
            return state
        finally:
            engine.dispose()

        truncated = len(fetched_rows) > config.max_rows
        serialized_rows = rows_to_dicts(fetched_rows[: config.max_rows], columns)
        execution_elapsed = time.perf_counter() - execution_started
        stage_elapsed = time.perf_counter() - stage_started

        payload = {
            "query": request.query,
            "sql": sql,
            "original_sql": original_sql,
            "row_count": len(serialized_rows),
            "truncated": truncated,
            "execution_seconds": execution_elapsed,
            "optimization_seconds": optimization_elapsed,
            "optimization": optimization_payload,
            "rows": serialized_rows,
        }
        artifact_path = Path(request.out_dir) / "sql_execution_result.yaml"
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_path.write_text(
            yaml.safe_dump(normalize_for_yaml(payload), sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )

        state["rows"] = serialized_rows
        state["row_count"] = len(serialized_rows)
        state["truncated"] = truncated
        state["execution_seconds"] = execution_elapsed
        state.setdefault("artifacts", []).append(
            StageArtifact(
                name="execution",
                path=artifact_path,
                payload=payload,
                duration_seconds=stage_elapsed,
            )
        )
        return state

    return RunnableLambda(_run, name="sql_execution_stage")
