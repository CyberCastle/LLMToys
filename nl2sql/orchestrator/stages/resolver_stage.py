#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Etapa LangChain que envuelve retrieval, rerank y compilacion semantica."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from langchain_core.runnables import RunnableLambda

from nl2sql.semantic_resolver import (
    SemanticResolverConfig,
    persist_semantic_plan,
    release_semantic_resolver_runtimes,
    run_semantic_resolver,
)
from nl2sql.utils.collections import dedupe_preserve_order
from nl2sql.utils.decision_models import DecisionIssue, dedupe_decision_issues
from nl2sql.utils.yaml_utils import load_yaml_mapping

from ..config import NL2SQLConfig, ensure_runtime_bundle_loaded
from ..contracts import StageArtifact


def _extract_compiled_plan_feedback(payload: dict[str, Any]) -> tuple[list[DecisionIssue], list[str], bool]:
    semantic_plan = payload.get("semantic_plan")
    if not isinstance(semantic_plan, dict):
        return [], [], False
    compiled_plan = semantic_plan.get("compiled_plan")
    if not isinstance(compiled_plan, dict):
        return [], [], False

    issues: list[DecisionIssue] = []
    blocking_semantic_issue = False
    raw_issues = compiled_plan.get("issues")
    if isinstance(raw_issues, list):
        for raw_issue in raw_issues:
            if not isinstance(raw_issue, dict):
                continue
            issue = DecisionIssue.model_validate(raw_issue)
            issues.append(issue)
            if issue.severity == "error":
                blocking_semantic_issue = True

    warnings = [str(item) for item in compiled_plan.get("warnings", []) or [] if str(item).strip()]
    return issues, warnings, blocking_semantic_issue


def build_resolver_runnable(config: NL2SQLConfig | None = None) -> RunnableLambda:
    """Construye la etapa de semantic resolver como `RunnableLambda`."""

    effective_config = ensure_runtime_bundle_loaded(config)
    runtime_bundle = effective_config.runtime_bundle
    if runtime_bundle is None:
        raise ValueError("NL2SQLConfig.runtime_bundle debe estar precargado antes de construir la etapa de resolver.")

    def _run(state: dict[str, Any]) -> dict[str, Any]:
        request = state["request"]
        pruned_schema = load_yaml_mapping(
            Path(state["pruned_schema_path"]),
            artifact_name=str(state["pruned_schema_path"]),
        )
        config = SemanticResolverConfig(
            settings=runtime_bundle.settings.semantic_resolver,
            semantic_contract=runtime_bundle.semantic_contract,
            compiler_rules=runtime_bundle.settings.semantic_resolver.compiler_rules,
            verification_rules=runtime_bundle.settings.semantic_resolver.verification,
            rules_path=str(runtime_bundle.semantic_rules_path),
            dialect=request.dialect,
        )
        started = time.perf_counter()
        try:
            semantic_plan = run_semantic_resolver(
                query=request.query,
                pruned_schema=pruned_schema,
                config=config,
            )
            output_path = persist_semantic_plan(
                semantic_plan,
                out_path=Path(request.out_dir) / "semantic_plan.yaml",
                pruned_schema_path=state["pruned_schema_path"],
                rules_path=runtime_bundle.semantic_rules_path,
            )
        finally:
            release_semantic_resolver_runtimes()

        payload = load_yaml_mapping(output_path, artifact_name=str(output_path))
        issues, warnings, blocking_semantic_issue = _extract_compiled_plan_feedback(payload)
        state["semantic_plan_path"] = output_path
        state.setdefault("artifacts", []).append(
            StageArtifact(
                name="resolver",
                path=output_path,
                payload=payload,
                duration_seconds=time.perf_counter() - started,
            )
        )
        state["issues"] = dedupe_decision_issues([*list(state.get("issues", []) or []), *issues])
        state["warnings"] = dedupe_preserve_order([*list(state.get("warnings", []) or []), *warnings])
        if blocking_semantic_issue:
            state["status"] = "failed_semantic_alignment"
        return state

    return RunnableLambda(_run, name="semantic_resolver_stage")
