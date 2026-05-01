#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from nl2sql.utils.semantic_contract import SemanticContract

from ..contracts import SolverInput
from ..plan_loader import load_semantic_plan
from ..schema_loader import load_pruned_schema, load_semantic_rules


@dataclass(frozen=True)
class PlanNormalizationResult:
    semantic_plan: dict[str, Any]
    pruned_schema: dict[str, Any]
    semantic_rules: SemanticContract


def run_plan_normalization_stage(solver_input: SolverInput) -> PlanNormalizationResult:
    """Carga artefactos del solver sin reconstruir estructura del schema."""

    return PlanNormalizationResult(
        semantic_plan=load_semantic_plan(solver_input.semantic_plan),
        pruned_schema=load_pruned_schema(solver_input.pruned_schema),
        semantic_rules=load_semantic_rules(solver_input.semantic_rules),
    )
