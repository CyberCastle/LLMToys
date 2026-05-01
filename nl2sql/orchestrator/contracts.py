#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Contratos y utilidades de serializacion del orquestador NL2SQL."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from nl2sql.utils.decision_models import DecisionIssue
from nl2sql.utils.yaml_utils import normalize_for_yaml

DialectName = Literal["tsql", "postgres"]
ResponseStatus = Literal["ok", "failed_semantic_alignment", "failed_sql_validation", "failed_runtime"]


@dataclass(frozen=True)
class NL2SQLRequest:
    """Entrada del pipeline: pregunta natural y paths de soporte."""

    query: str
    db_schema_path: str | Path
    semantic_rules_path: str | Path
    catalogos_path: str | Path | None = None
    out_dir: str | Path = "out"
    dialect: DialectName = "tsql"


@dataclass(frozen=True)
class StageArtifact:
    """Artefacto persistido por una etapa del pipeline."""

    name: str
    path: Path
    payload: dict[str, Any]
    duration_seconds: float


@dataclass(frozen=True)
class NL2SQLResponse:
    """Salida consolidada del pipeline NL2SQL."""

    status: ResponseStatus
    query: str
    final_sql: str
    rows: list[dict[str, Any]] = field(default_factory=list)
    row_count: int = 0
    truncated: bool = False
    narrative: str = ""
    artifacts: list[StageArtifact] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    issues: list[DecisionIssue] = field(default_factory=list)
