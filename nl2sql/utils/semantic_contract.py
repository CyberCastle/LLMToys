#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from pydantic import Field

from .decision_models import StrictModel
from .yaml_utils import load_yaml_mapping

SemanticRows = list[dict[str, Any]]
SemanticMapping = dict[str, Any]


class SemanticBusinessInvariants(StrictModel):
    """Secciones declarativas de negocio compartidas por las etapas NL2SQL."""

    semantic_models: SemanticRows = Field(default_factory=list)
    semantic_entities: SemanticRows = Field(default_factory=list)
    semantic_dimensions: SemanticRows = Field(default_factory=list)
    semantic_metrics: SemanticRows = Field(default_factory=list)
    semantic_filters: SemanticRows = Field(default_factory=list)
    semantic_business_rules: SemanticRows = Field(default_factory=list)
    semantic_relationships: SemanticRows = Field(default_factory=list)
    semantic_constraints: SemanticRows = Field(default_factory=list)
    semantic_join_paths: SemanticRows = Field(default_factory=list)
    semantic_derived_metrics: SemanticRows = Field(default_factory=list)


class SemanticRetrievalHeuristics(StrictModel):
    """Heuristicas declarativas usadas por prune y semantic resolver."""

    semantic_synonyms: SemanticMapping = Field(default_factory=dict)
    semantic_examples: SemanticRows = Field(default_factory=list)
    query_forms: SemanticRows = Field(default_factory=list)


class SemanticSqlSafety(StrictModel):
    """Reglas de seguridad y negocio especificas del solver SQL."""

    execution_safety: SemanticMapping = Field(default_factory=dict)
    hard_join_blacklist: SemanticRows = Field(default_factory=list)
    semantic_sql_business_rules: SemanticRows = Field(default_factory=list)


class SemanticContract(StrictModel):
    """Contrato raiz de reglas semanticas compartidas por NL2SQL."""

    business_invariants: SemanticBusinessInvariants
    retrieval_heuristics: SemanticRetrievalHeuristics
    sql_safety: SemanticSqlSafety


_SECTION_GROUPS: dict[str, tuple[str, str]] = {
    "semantic_models": ("business_invariants", "semantic_models"),
    "semantic_entities": ("business_invariants", "semantic_entities"),
    "semantic_dimensions": ("business_invariants", "semantic_dimensions"),
    "semantic_metrics": ("business_invariants", "semantic_metrics"),
    "semantic_filters": ("business_invariants", "semantic_filters"),
    "semantic_business_rules": ("business_invariants", "semantic_business_rules"),
    "semantic_relationships": ("business_invariants", "semantic_relationships"),
    "semantic_constraints": ("business_invariants", "semantic_constraints"),
    "semantic_join_paths": ("business_invariants", "semantic_join_paths"),
    "semantic_derived_metrics": ("business_invariants", "semantic_derived_metrics"),
    "semantic_synonyms": ("retrieval_heuristics", "semantic_synonyms"),
    "semantic_examples": ("retrieval_heuristics", "semantic_examples"),
    "execution_safety": ("sql_safety", "execution_safety"),
    "hard_join_blacklist": ("sql_safety", "hard_join_blacklist"),
    "semantic_sql_business_rules": ("sql_safety", "semantic_sql_business_rules"),
}


def load_semantic_contract(source: str | Path | Mapping[str, Any] | SemanticContract) -> SemanticContract:
    """Carga y valida `semantic_contract` desde YAML o desde memoria."""

    if isinstance(source, SemanticContract):
        return source

    if isinstance(source, Mapping):
        raw_payload = dict(source)
    else:
        raw_payload = load_yaml_mapping(Path(source), artifact_name=str(source))

    raw_contract = raw_payload.get("semantic_contract")
    if not isinstance(raw_contract, Mapping):
        raise ValueError("semantic_rules.yaml invalido: falta la raiz 'semantic_contract'")
    return SemanticContract.model_validate(raw_contract)


def select_semantic_sections(contract: SemanticContract, sections: tuple[str, ...]) -> dict[str, Any]:
    """Extrae secciones planas desde el contrato para consumidores internos."""

    selected: dict[str, Any] = {}
    for section_name in sections:
        group_name, field_name = _SECTION_GROUPS[section_name]
        selected[section_name] = getattr(getattr(contract, group_name), field_name)
    return selected
