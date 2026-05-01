#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Etapa LangChain que genera el SQL final a partir del semantic plan."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from langchain_core.runnables import RunnableLambda
import yaml

from nl2sql.sql_solver_generator import SolverConfig, SolverInput, run_sql_solver
from nl2sql.utils.collections import dedupe_preserve_order
from nl2sql.utils.decision_models import DecisionIssue, dedupe_decision_issues
from nl2sql.utils.yaml_utils import load_yaml_mapping, normalize_for_yaml

from ..config import NL2SQLConfig, ensure_runtime_bundle_loaded
from ..contracts import StageArtifact


def _write_yaml(path: Path, payload: object) -> dict[str, Any]:
    """Persiste un payload YAML-safe y devuelve el mapping normalizado."""

    normalized = normalize_for_yaml(payload)
    if not isinstance(normalized, dict):
        raise ValueError(f"El artefacto {path} no se pudo serializar como mapping YAML")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(normalized, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return normalized


def _solver_failure_status(issues: list[DecisionIssue]) -> str:
    if any(issue.stage in {"sql_business_rules", "sql_normalization", "sql_validation"} for issue in issues):
        return "failed_sql_validation"
    return "failed_runtime"


def _has_error_issues(issues: list[DecisionIssue]) -> bool:
    return any(issue.severity == "error" for issue in issues)


def build_solver_runnable(config: NL2SQLConfig | None = None) -> RunnableLambda:
    """Construye la etapa de solver SQL como `RunnableLambda`."""

    effective_config = ensure_runtime_bundle_loaded(config)
    runtime_bundle = effective_config.runtime_bundle
    if runtime_bundle is None:
        raise ValueError("NL2SQLConfig.runtime_bundle debe estar precargado antes de construir la etapa de solver.")

    def _run(state: dict[str, Any]) -> dict[str, Any]:
        request = state["request"]
        if state.get("status", "ok") != "ok":
            return state
        solver_input = SolverInput(
            semantic_plan=state["semantic_plan_path"],
            pruned_schema=state["pruned_schema_path"],
            semantic_rules=runtime_bundle.semantic_contract,
        )
        config = SolverConfig(
            settings=runtime_bundle.settings.sql_solver,
            prompts=runtime_bundle.settings.sql_solver.prompts,
            filter_value_rules=runtime_bundle.settings.sql_solver.filter_value_rules,
            generation_tuning=runtime_bundle.settings.sql_solver.generation_tuning,
            dialect=request.dialect,
        )
        started = time.perf_counter()
        metadata_path = Path(request.out_dir) / "solver_result.yaml"
        try:
            result = run_sql_solver(solver_input, config)
        except Exception as exc:  # noqa: BLE001
            issue = DecisionIssue(
                stage="sql_generation",
                code="solver_stage_failed",
                severity="error",
                message=str(exc),
                context={"error_type": exc.__class__.__name__},
            )
            payload = _write_yaml(metadata_path, {"sql_final": "", "issues": [issue.model_dump(mode="python")]})
            state["solver_result_path"] = metadata_path
            state.setdefault("artifacts", []).append(
                StageArtifact(
                    name="solver",
                    path=metadata_path,
                    payload=payload,
                    duration_seconds=time.perf_counter() - started,
                )
            )
            state["issues"] = dedupe_decision_issues([*list(state.get("issues", []) or []), issue])
            state["status"] = "failed_runtime"
            return state

        payload = _write_yaml(metadata_path, result)
        state["solver_result"] = result
        state["solver_result_path"] = metadata_path
        state.setdefault("artifacts", []).append(
            StageArtifact(
                name="solver",
                path=metadata_path,
                payload=payload,
                duration_seconds=time.perf_counter() - started,
            )
        )
        state["warnings"] = dedupe_preserve_order([*list(state.get("warnings", []) or []), *list(result.warnings)])
        state["issues"] = dedupe_decision_issues([*list(state.get("issues", []) or []), *list(result.issues)])

        if _has_error_issues(list(result.issues)) or not result.sql_final.strip():
            state["status"] = _solver_failure_status(list(result.issues))
            return state

        sql_path = Path(request.out_dir) / "solver_result.sql"
        sql_path.parent.mkdir(parents=True, exist_ok=True)
        sql_path.write_text(result.sql_final, encoding="utf-8")

        payload["sql_output_path"] = str(sql_path)
        metadata_path.write_text(
            yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )

        state["final_sql"] = result.sql_final
        state["solver_sql_path"] = sql_path
        return state

    return RunnableLambda(_run, name="sql_solver_stage")
