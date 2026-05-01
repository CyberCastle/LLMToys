#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .plan_model import CompiledSemanticPlan


@dataclass(frozen=True)
class SemanticAsset:
    """Activo semantico normalizado a partir de una seccion del YAML."""

    asset_id: str
    kind: str
    name: str
    payload: dict[str, Any]


@dataclass(frozen=True)
class MatchedAsset:
    """Activo candidato ya puntuado por retrieval, rerank y compatibilidad."""

    asset: SemanticAsset
    embedding_score: float
    rerank_score: float
    compatibility_score: float
    compatible_tables: tuple[str, ...]
    rejected_reason: str | None = None


@dataclass(frozen=True)
class SemanticPlan:
    """Plan final que resume los activos utiles para responder una consulta."""

    query: str
    assets_by_kind: dict[str, list[MatchedAsset]]
    all_assets: list[MatchedAsset]
    pruned_tables: tuple[str, ...]
    diagnostics: dict[str, Any] = field(default_factory=dict)
    compiled_plan: CompiledSemanticPlan | None = None
