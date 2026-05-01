#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from nl2sql.utils.yaml_utils import load_yaml_mapping

REQUIRED_COMPILED_FIELDS = (
    "query",
    "intent",
    "base_entity",
    "grain",
    "measure",
    "group_by",
    "join_path",
    "required_tables",
)


def load_semantic_plan(source: str | Path | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(source, Mapping):
        raw = dict(source)
    else:
        raw = load_yaml_mapping(Path(source), artifact_name=str(source))

    root = raw.get("semantic_plan", raw)
    if not isinstance(root, Mapping):
        raise ValueError("SemanticPlan invalido: la raiz debe ser un mapping")

    compiled_plan = root.get("compiled_plan") or {}
    if not isinstance(compiled_plan, Mapping):
        raise ValueError("SemanticPlan invalido: compiled_plan debe ser un mapping")

    missing = [field for field in REQUIRED_COMPILED_FIELDS if field not in compiled_plan]
    if missing:
        raise ValueError(f"SemanticPlan incompleto: faltan {missing}")

    return {
        "compiled_plan": dict(compiled_plan),
        "retrieved_candidates": dict(root.get("retrieved_candidates", {}) or {}),
    }
