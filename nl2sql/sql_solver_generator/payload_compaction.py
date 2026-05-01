#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Compactacion de payloads semanticos para prompts del solver SQL."""

from __future__ import annotations

from typing import Any, Mapping

from .runtime import load_generation_tuning_rules


def drop_empty(value: object) -> Any:
    """Elimina valores vacios de mappings/listas preservando estructura util."""

    if isinstance(value, Mapping):
        compacted = {str(key): drop_empty(child) for key, child in value.items()}
        return {key: child for key, child in compacted.items() if child not in (None, "", [], {}, ())}
    if isinstance(value, list):
        compacted_list = [drop_empty(child) for child in value]
        return [child for child in compacted_list if child not in (None, "", [], {}, ())]
    return value


def compact_verification_context(raw_verification: object, *, minimal: bool) -> dict[str, Any] | None:
    """Compacta diagnosticos del verificador para el contexto del solver."""

    if not isinstance(raw_verification, Mapping):
        return None
    return drop_empty(
        {
            "is_semantically_aligned": raw_verification.get("is_semantically_aligned"),
            "failure_class": raw_verification.get("failure_class"),
            "repairability": raw_verification.get("repairability"),
            "wrong_metric": raw_verification.get("wrong_metric"),
            "suggested_measure": raw_verification.get("suggested_measure"),
            "suggested_join_tables": raw_verification.get("suggested_join_tables"),
            "suggested_plan_delta": (None if minimal else raw_verification.get("suggested_plan_delta")),
            "blocking_reason": raw_verification.get("blocking_reason"),
            "confidence": raw_verification.get("confidence"),
        }
    )


def compact_candidate_plan_set(raw_candidate_plan_set: object, *, minimal: bool) -> dict[str, Any] | None:
    """Compacta candidatos alternativos cuando el plan no es confiable."""

    if not isinstance(raw_candidate_plan_set, Mapping):
        return None
    raw_candidates = raw_candidate_plan_set.get("candidates")
    if not isinstance(raw_candidates, list) or not raw_candidates:
        return None

    candidate_payloads: list[dict[str, Any]] = []
    tuning_rules = load_generation_tuning_rules()
    candidate_limit = tuning_rules.minimal_candidate_plan_limit if minimal else tuning_rules.rich_candidate_plan_limit
    for raw_candidate in raw_candidates[:candidate_limit]:
        if not isinstance(raw_candidate, Mapping):
            continue
        candidate_payload = drop_empty(
            {
                "intent": raw_candidate.get("intent"),
                "measure": raw_candidate.get("measure"),
                "group_by": raw_candidate.get("group_by"),
                "ranking": raw_candidate.get("ranking"),
                "confidence": raw_candidate.get("confidence"),
                "issues": raw_candidate.get("issues"),
                "required_tables": (None if minimal else raw_candidate.get("required_tables")),
            }
        )
        if candidate_payload:
            candidate_payloads.append(candidate_payload)

    if not candidate_payloads:
        return None

    return drop_empty(
        {
            "selected_index": raw_candidate_plan_set.get("selected_index"),
            "selection_rationale": (None if minimal else raw_candidate_plan_set.get("selection_rationale")),
            "candidates": candidate_payloads,
        }
    )


def compact_solver_semantic_context(compiled_plan: Mapping[str, Any], *, minimal: bool) -> dict[str, Any] | None:
    """Incluye contexto semantico solo cuando aporta senal de reparacion."""

    tuning_rules = load_generation_tuning_rules()
    confidence = float(compiled_plan.get("confidence") or 0.0)
    verification = compact_verification_context(compiled_plan.get("verification"), minimal=minimal)
    candidate_plan_set = compact_candidate_plan_set(compiled_plan.get("candidate_plan_set"), minimal=minimal)
    has_semantic_mismatch = bool(verification and verification.get("is_semantically_aligned") is False)
    is_uncertain = confidence < tuning_rules.semantic_context_confidence_threshold
    should_include = is_uncertain or has_semantic_mismatch
    if not should_include:
        return None
    return drop_empty(
        {
            "selected_plan_confidence": confidence,
            "verification": verification,
            "candidate_plan_set": (candidate_plan_set if (is_uncertain or has_semantic_mismatch) else None),
        }
    )
