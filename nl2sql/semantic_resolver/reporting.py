#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Mapping

import yaml

from nl2sql.utils.yaml_utils import normalize_for_yaml

from .assets import MatchedAsset, SemanticPlan


def _format_tables(tables: tuple[str, ...]) -> str:
    return ", ".join(tables) if tables else "none"


def _format_scores(asset: MatchedAsset) -> str:
    return f"emb={asset.embedding_score:.4f} | rerank={asset.rerank_score:.4f} | compat={asset.compatibility_score:.4f}"


def _render_compiled_plan(plan: SemanticPlan) -> list[str]:
    compiled_plan = plan.compiled_plan
    lines = ["-" * 80, "PLAN COMPILADO", "-" * 80]
    if compiled_plan is None:
        lines.append("Compilador deshabilitado.")
        return lines

    lines.extend(
        [
            f"intent              : {compiled_plan.intent}",
            f"semantic_model      : {compiled_plan.semantic_model or 'none'}",
            f"base_entity         : {compiled_plan.base_entity}",
            f"grain               : {compiled_plan.grain}",
        ]
    )

    if compiled_plan.measure is None:
        lines.append("measure             : none")
    else:
        lines.append(f"measure             : {compiled_plan.measure.name} -> {compiled_plan.measure.formula}")

    lines.append(f"group_by            : {', '.join(compiled_plan.group_by) or 'none'}")
    lines.append(f"final_group_by      : {', '.join(compiled_plan.final_group_by) or 'none'}")
    if compiled_plan.selected_filters:
        for selected_filter in compiled_plan.selected_filters:
            lines.append("selected_filter     : " f"{selected_filter.field} {selected_filter.operator} {selected_filter.value}")
    else:
        lines.append("selected_filter     : none")
    if compiled_plan.time_filter is None:
        lines.append("time_filter         : none")
    else:
        lines.append(
            "time_filter         : "
            f"{compiled_plan.time_filter.field} {compiled_plan.time_filter.operator} {compiled_plan.time_filter.value}"
        )
        if compiled_plan.time_filter.resolved_expressions:
            for dialect_name in sorted(compiled_plan.time_filter.resolved_expressions):
                lines.append("  resolved_expression : " f"[{dialect_name}] {compiled_plan.time_filter.resolved_expressions[dialect_name]}")

    if compiled_plan.post_aggregation is None:
        lines.append("post_aggregation    : none")
    else:
        lines.append(f"post_aggregation    : {compiled_plan.post_aggregation.function}({compiled_plan.post_aggregation.over})")

    if compiled_plan.ranking is None:
        lines.append("ranking             : none")
    else:
        lines.append(f"ranking             : direction={compiled_plan.ranking.direction} limit={compiled_plan.ranking.limit}")

    if compiled_plan.join_path:
        for join_edge in compiled_plan.join_path:
            lines.append(f"join_path           : {join_edge}")
    else:
        lines.append("join_path           : none")

    lines.append(f"join_path_hint      : {compiled_plan.join_path_hint or 'none'}")
    lines.append(f"derived_metric_ref  : {compiled_plan.derived_metric_ref or 'none'}")
    lines.append(f"population_scope    : {compiled_plan.population_scope or 'none'}")
    lines.append(f"base_group_by       : {', '.join(compiled_plan.base_group_by) or 'none'}")
    lines.append(f"intermediate_alias  : {compiled_plan.intermediate_alias or 'none'}")
    lines.append(f"required_tables     : {', '.join(compiled_plan.required_tables) or 'none'}")

    lines.append(f"confidence          : {compiled_plan.confidence:.2f}")
    if compiled_plan.verification is None:
        lines.append("verification        : none")
    else:
        lines.append(
            "verification        : "
            f"aligned={compiled_plan.verification.is_semantically_aligned} "
            f"confidence={compiled_plan.verification.confidence:.2f}"
        )
        if compiled_plan.verification.missing_filters:
            lines.append(f"verification_missing_filters : {', '.join(compiled_plan.verification.missing_filters)}")
        if compiled_plan.verification.wrong_metric:
            lines.append(f"verification_wrong_metric    : {compiled_plan.verification.wrong_metric}")
        if compiled_plan.verification.suggested_measure:
            lines.append(f"verification_suggested_measure: {compiled_plan.verification.suggested_measure}")
    if compiled_plan.warnings:
        for warning in compiled_plan.warnings:
            lines.append(f"warning             : {warning}")
    else:
        lines.append("warning             : none")

    if compiled_plan.issues:
        for issue in compiled_plan.issues:
            lines.append(f"issue               : {issue.code} -> {issue.message}")
    else:
        lines.append("issue               : none")

    if compiled_plan.metric_score_trace:
        for trace in compiled_plan.metric_score_trace:
            lines.append(
                "metric_trace        : "
                f"{trace.metric_name} score={trace.total_score:.2f} selected={trace.selected} reason={trace.rejected_reason or 'selected'}"
            )

    if compiled_plan.candidate_plan_set is not None and compiled_plan.candidate_plan_set.candidates:
        lines.append(
            "candidate_plan_set  : "
            f"selected_index={compiled_plan.candidate_plan_set.selected_index} "
            f"rationale={compiled_plan.candidate_plan_set.selection_rationale or 'n/a'}"
        )
        for index, candidate in enumerate(compiled_plan.candidate_plan_set.candidates):
            measure_name = candidate.measure.name if candidate.measure is not None else "none"
            lines.append(
                "candidate_plan      : "
                f"[{index}] measure={measure_name} score={candidate.score:.2f} confidence={candidate.confidence:.2f} "
                f"group_by={', '.join(candidate.group_by) or 'none'} issues={', '.join(candidate.issues) or 'none'}"
            )

    return lines


def build_semantic_plan_report(plan: SemanticPlan) -> str:
    lines = [
        "=" * 80,
        "SEMANTIC RESOLVER REPORT",
        "=" * 80,
        f"Query                : {plan.query}",
    ]

    retrieval_query = plan.diagnostics.get("retrieval_query")
    if isinstance(retrieval_query, str) and retrieval_query and retrieval_query != plan.query:
        lines.append(f"Retrieval Query      : {retrieval_query}")

    synonym_entities = plan.diagnostics.get("synonym_entities_detected")
    if isinstance(synonym_entities, (list, tuple)) and synonym_entities:
        lines.append(f"Synonym Entities     : {', '.join(map(str, synonym_entities))}")

    lines.extend(
        [
            f"Pruned Tables        : {_format_tables(plan.pruned_tables)}",
            "-" * 80,
            "ACTIVOS ACEPTADOS",
            "-" * 80,
        ]
    )

    if not plan.assets_by_kind:
        lines.append("No hay activos aceptados.")
    else:
        for kind in sorted(plan.assets_by_kind):
            lines.append(kind)
            for matched_asset in plan.assets_by_kind[kind]:
                lines.append(
                    f"  - {matched_asset.asset.name} -> {_format_scores(matched_asset)} | "
                    f"tables={_format_tables(matched_asset.compatible_tables)}"
                )

    lines.extend(_render_compiled_plan(plan))

    rejected_by_reason: dict[str, list[MatchedAsset]] = defaultdict(list)
    for matched_asset in plan.all_assets:
        if matched_asset.rejected_reason is None:
            continue
        rejected_by_reason[matched_asset.rejected_reason].append(matched_asset)

    lines.extend(["-" * 80, "ACTIVOS RECHAZADOS", "-" * 80])
    if not rejected_by_reason:
        lines.append("No hay activos rechazados.")
    else:
        for reason in sorted(rejected_by_reason):
            lines.append(reason)
            for matched_asset in rejected_by_reason[reason]:
                lines.append(f"  - {matched_asset.asset.kind} :: {matched_asset.asset.name}")

    lines.extend(["-" * 80, "DIAGNOSTICOS", "-" * 80])
    for key in sorted(plan.diagnostics):
        lines.append(f"{key}: {plan.diagnostics[key]}")

    return "\n".join(lines)


def render_semantic_plan(plan: SemanticPlan) -> None:
    print(build_semantic_plan_report(plan))


def persist_semantic_plan(
    semantic_plan: object,
    *,
    out_path: str | Path,
    pruned_schema_path: str | Path,
    rules_path: str | Path,
) -> Path:
    """Persiste el artefacto `semantic_plan.yaml` con el contrato esperado por el solver."""

    normalized_plan = normalize_for_yaml(semantic_plan)
    if not isinstance(normalized_plan, dict):
        raise ValueError("SemanticPlan invalido: no se pudo serializar a mapping YAML-safe")

    compiled_plan = normalized_plan.get("compiled_plan")
    if not isinstance(compiled_plan, Mapping):
        raise RuntimeError("Semantic resolver no produjo compiled_plan; verifica enable_plan_compiler y las reglas del compilador.")

    retrieved_candidates = {key: value for key, value in normalized_plan.items() if key != "compiled_plan"}
    payload = {
        "semantic_plan": {
            "retrieved_candidates": retrieved_candidates,
            "compiled_plan": compiled_plan,
        },
        "source_pruned_schema_path": str(pruned_schema_path),
        "source_rules_path": str(rules_path),
    }

    output_path = Path(out_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return output_path
