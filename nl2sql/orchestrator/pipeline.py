#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Composicion LangChain del flujo NL2SQL completo.

Etapas LLM: schema pruning, semantic resolver, verificacion semantica, solver
SQL y narrativa final.
Etapas deterministas: persistencia de artefactos, contratos tipados,
validacion AST/dialecto, reglas de negocio y ejecucion SQL segura. El
orquestador solo encadena ambas fronteras; no compensa fallos del LLM con
renderers SQL locales ni con bypass silenciosos.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

from langchain_core.runnables import Runnable, RunnableLambda, RunnableSequence

from nl2sql.utils.collections import dedupe_preserve_order
from nl2sql.utils.decision_models import DecisionIssue, dedupe_decision_issues

from .config import NL2SQLConfig, ensure_runtime_bundle_loaded
from .contracts import NL2SQLRequest, NL2SQLResponse
from .llm_manager import LLMManager
from .stages.execution_stage import build_execution_runnable
from .stages.narrative_stage import build_narrative_runnable
from .stages.prune_stage import build_prune_runnable
from .stages.resolver_stage import build_resolver_runnable
from .stages.solver_stage import build_solver_runnable


def _state_to_response(state: dict[str, Any]) -> NL2SQLResponse:
    """Convierte el estado interno de LangChain al contrato de salida publico."""

    solver_result = state.get("solver_result")
    warnings = list(state.get("warnings", []) or [])
    warnings.extend(list(getattr(solver_result, "warnings", []) or []))
    if state.get("truncated"):
        warnings.append("sql_result_truncated")

    state_issues = list(state.get("issues", []) or [])
    solver_issues = list(getattr(solver_result, "issues", []) or [])

    return NL2SQLResponse(
        status=state.get("status", "ok"),
        query=state["request"].query,
        final_sql=state.get("final_sql", ""),
        rows=state.get("rows", []),
        row_count=state.get("row_count", 0),
        truncated=state.get("truncated", False),
        narrative=state.get("narrative", ""),
        artifacts=state.get("artifacts", []),
        warnings=dedupe_preserve_order(warnings),
        issues=dedupe_decision_issues([*state_issues, *solver_issues]),
    )


def build_nl2sql_pipeline(config: NL2SQLConfig | None = None) -> Runnable:
    """Construye la secuencia Prune → Resolver → Solver → SQL → Narrative."""

    effective_config = ensure_runtime_bundle_loaded(config)
    if effective_config.runtime_bundle is None:
        raise ValueError("NL2SQLConfig.runtime_bundle debe cargarse y validarse antes de construir el pipeline.")

    llm_manager = LLMManager(effective_config.narrative)

    init_state = RunnableLambda(
        lambda request: {"request": request, "artifacts": [], "issues": [], "warnings": [], "status": "ok"},
        name="init_nl2sql_state",
    )
    finalize_state = RunnableLambda(_state_to_response, name="finalize_nl2sql_state")

    return RunnableSequence(
        init_state,
        build_prune_runnable(effective_config),
        build_resolver_runnable(effective_config),
        build_solver_runnable(effective_config),
        build_execution_runnable(effective_config),
        build_narrative_runnable(llm_manager, effective_config, Path(effective_config.narrative_prompt_path)),
        finalize_state,
    )


def run_nl2sql(request: NL2SQLRequest, config: NL2SQLConfig | None = None) -> NL2SQLResponse:
    """Ejecuta el pipeline completo de forma sincronica."""

    effective_config = ensure_runtime_bundle_loaded(config, semantic_rules_path=request.semantic_rules_path)
    return build_nl2sql_pipeline(effective_config).invoke(request)


def _init_state(request: NL2SQLRequest) -> dict[str, Any]:
    """Construye el estado inicial del pipeline para una sola peticion."""

    return {"request": request, "artifacts": [], "issues": [], "warnings": [], "status": "ok"}


def _build_stage_runnables(config: NL2SQLConfig) -> list[tuple[str, RunnableLambda]]:
    """Instancia las etapas del pipeline una sola vez para reusarlas en batch."""

    llm_manager = LLMManager(config.narrative)
    return [
        ("prune", build_prune_runnable(config)),
        ("resolver", build_resolver_runnable(config)),
        ("solver", build_solver_runnable(config)),
        ("execution", build_execution_runnable(config)),
        ("narrative", build_narrative_runnable(llm_manager, config, Path(config.narrative_prompt_path))),
    ]


def run_nl2sql_batch(
    requests: Sequence[NL2SQLRequest],
    config: NL2SQLConfig | None = None,
) -> list[NL2SQLResponse]:
    """Procesa varias peticiones agrupando las invocaciones por etapa.

    Por cada etapa del pipeline se itera primero sobre todas las peticiones
    activas antes de avanzar a la siguiente, evitando recargar los runtimes
    LLM una vez por peticion. Si una peticion falla en una etapa, se registra
    un `DecisionIssue` con `stage="orchestrator_batch"`, su estado se marca
    como `failed_runtime` y las etapas posteriores la saltan, pero el resto
    del batch sigue procesandose.

    Todas las peticiones del batch deben compartir el mismo
    `semantic_rules_path` para garantizar que el `runtime_bundle` cargado es
    el mismo para todas las etapas.
    """

    if not requests:
        return []

    semantic_rules_paths = {str(req.semantic_rules_path) for req in requests}
    if len(semantic_rules_paths) > 1:
        raise ValueError("run_nl2sql_batch requiere que todas las peticiones compartan el mismo semantic_rules_path")

    effective_config = ensure_runtime_bundle_loaded(config, semantic_rules_path=requests[0].semantic_rules_path)
    if effective_config.runtime_bundle is None:
        raise ValueError("NL2SQLConfig.runtime_bundle debe cargarse y validarse antes de ejecutar el batch.")

    stages = _build_stage_runnables(effective_config)
    states: list[dict[str, Any]] = [_init_state(req) for req in requests]

    for stage_name, runnable in stages:
        for state in states:
            # Saltar peticiones que ya hayan fallado en una etapa previa.
            if state.get("status", "ok") != "ok":
                continue
            try:
                runnable.invoke(state)
            except Exception as exc:  # noqa: BLE001
                # Capturar la excepcion para no abortar el batch y registrarla
                # como issue trazable en el response final de esa peticion.
                issue = DecisionIssue(
                    stage="orchestrator_batch",
                    code=f"{stage_name}_stage_exception",
                    severity="error",
                    message=str(exc),
                    context={"error_type": exc.__class__.__name__, "stage": stage_name},
                )
                state["issues"] = dedupe_decision_issues(
                    [
                        *list(state.get("issues", []) or []),
                        issue,
                    ]
                )
                state["status"] = "failed_runtime"

    return [_state_to_response(state) for state in states]
