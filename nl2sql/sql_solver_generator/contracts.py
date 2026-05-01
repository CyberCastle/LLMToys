#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from pydantic import Field

from nl2sql.utils.decision_models import DecisionIssue, StrictModel
from nl2sql.utils.semantic_contract import SemanticContract

from .spec_model import SQLQuerySpec

PathOrMapping = str | Path | Mapping[str, Any]
SemanticRulesSource = str | Path | Mapping[str, Any] | SemanticContract


class SolverInput(StrictModel):
    """Entradas del solver SQL como rutas o payloads ya cargados."""

    semantic_plan: PathOrMapping
    pruned_schema: PathOrMapping
    semantic_rules: SemanticRulesSource


class SolverMetadata(StrictModel):
    """Metadatos operativos y de validacion del solver."""

    tables_used: list[str] = Field(default_factory=list)
    columns_used: list[str] = Field(default_factory=list)
    join_paths_used: list[str] = Field(default_factory=list)
    dialect: str = "tsql"
    model_used: str = ""
    attempts: int = 0
    finish_reason: str = ""
    prompt_tokens: int = 0
    generated_tokens: int = 0
    wall_time_seconds: float = 0.0
    validator_trace: list[DecisionIssue] = Field(default_factory=list)


class SolverOutput(StrictModel):
    """Salida publica del solver SQL con contratos estrictos."""

    sql_final: str
    sql_query_spec: SQLQuerySpec
    metadata: SolverMetadata
    warnings: list[str] = Field(default_factory=list)
    issues: list[DecisionIssue] = Field(default_factory=list)
