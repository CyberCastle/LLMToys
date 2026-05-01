#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import replace
from typing import Mapping

from .plan_model import CandidatePlan, CandidatePlanSet, CompiledSemanticPlan, build_compiled_semantic_plan
from .verification import SemanticVerificationResult, load_verification_rules


def _dedupe_preserve(values: list[str]) -> list[str]:
    seen_values: set[str] = set()
    deduped_values: list[str] = []
    for value in values:
        if value in seen_values:
            continue
        seen_values.add(value)
        deduped_values.append(value)
    return deduped_values


def _normalize_suggested_delta(
    verification: SemanticVerificationResult,
    suggested_delta: Mapping[str, object] | None,
) -> dict[str, object]:
    delta = dict(suggested_delta or {})
    if verification.suggested_measure and "measure_name" not in delta:
        delta["measure_name"] = verification.suggested_measure
    if verification.suggested_join_tables and "join_tables" not in delta:
        delta["join_tables"] = list(verification.suggested_join_tables)
    if verification.missing_filters and "group_by" not in delta:
        group_by_fields = [field for field in verification.missing_filters if isinstance(field, str) and "." in field]
        if group_by_fields:
            delta["group_by"] = group_by_fields
    return delta


def _candidate_repair_score(candidate: CandidatePlan, delta: Mapping[str, object], current_plan: CompiledSemanticPlan) -> float:
    repair_rules = load_verification_rules()
    score = float(candidate.score or candidate.confidence)

    measure_name = delta.get("measure_name")
    if isinstance(measure_name, str) and candidate.measure is not None and candidate.measure.name == measure_name:
        score += repair_rules.repair_measure_match_bonus

    raw_group_by = delta.get("group_by")
    if isinstance(raw_group_by, list):
        requested_group_by = {str(value).strip() for value in raw_group_by if str(value).strip()}
        score += len(requested_group_by & set(candidate.group_by)) * repair_rules.repair_group_by_overlap_weight

    raw_join_tables = delta.get("join_tables")
    if isinstance(raw_join_tables, list):
        requested_join_tables = {str(value).strip() for value in raw_join_tables if str(value).strip()}
        score += len(requested_join_tables & set(candidate.required_tables)) * repair_rules.repair_join_table_overlap_weight

    if candidate.intent == current_plan.intent:
        score += repair_rules.repair_same_intent_bonus
    if candidate.measure is not None and current_plan.measure is not None and candidate.measure.name == current_plan.measure.name:
        score += repair_rules.repair_same_measure_bonus

    return score


def _materialize_candidate(
    *,
    query: str,
    candidate: CandidatePlan,
    template: CompiledSemanticPlan,
    candidate_plan_set: CandidatePlanSet,
    selected_index: int,
    selection_rationale: str,
) -> CompiledSemanticPlan:
    return build_compiled_semantic_plan(
        query=query,
        semantic_model=candidate.semantic_model,
        intent=candidate.intent,
        base_entity=candidate.base_entity,
        grain=candidate.grain,
        measure=candidate.measure,
        group_by=list(candidate.group_by),
        final_group_by=list(candidate.final_group_by),
        time_filter=candidate.time_filter,
        ranking=candidate.ranking,
        post_aggregation=candidate.post_aggregation,
        join_path=list(candidate.join_path),
        required_tables=list(candidate.required_tables),
        warnings=_dedupe_preserve(
            [
                *list(candidate.warnings),
                "semantic_repair_applied",
            ]
        ),
        confidence=max(candidate.confidence, template.confidence),
        join_path_hint=candidate.join_path_hint,
        derived_metric_ref=candidate.derived_metric_ref,
        population_scope=candidate.population_scope,
        base_group_by=list(candidate.base_group_by),
        intermediate_alias=candidate.intermediate_alias,
        metric_score_trace=list(template.metric_score_trace),
        candidate_plan_set=replace(
            candidate_plan_set,
            selected_index=selected_index,
            selection_rationale=selection_rationale,
        ),
        query_form_name=template.query_form_name,
        issues=list(template.issues),
    )


def repair_compiled_plan(
    *,
    compiled_plan: CompiledSemanticPlan,
    verification: SemanticVerificationResult,
    suggested_delta: Mapping[str, object] | None = None,
) -> CompiledSemanticPlan:
    """Selecciona el mejor plan candidato segun el delta sugerido por el verificador."""

    candidate_plan_set = compiled_plan.candidate_plan_set
    if candidate_plan_set is None or not candidate_plan_set.candidates:
        return compiled_plan

    normalized_delta = _normalize_suggested_delta(verification, suggested_delta)
    best_index = candidate_plan_set.selected_index
    best_score = float("-inf")
    for index, candidate in enumerate(candidate_plan_set.candidates):
        candidate_score = _candidate_repair_score(candidate, normalized_delta, compiled_plan)
        if candidate_score <= best_score:
            continue
        best_index = index
        best_score = candidate_score

    if best_index == candidate_plan_set.selected_index:
        return compiled_plan

    selected_candidate = candidate_plan_set.candidates[best_index]
    return _materialize_candidate(
        query=compiled_plan.query,
        candidate=selected_candidate,
        template=compiled_plan,
        candidate_plan_set=candidate_plan_set,
        selected_index=best_index,
        selection_rationale="repaired_from_semantic_verifier_delta",
    )
