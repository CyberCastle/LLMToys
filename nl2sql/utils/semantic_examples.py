#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from nl2sql.config import SemanticExampleSelectionRules, load_semantic_resolver_settings, resolve_nl2sql_config_path
from nl2sql.utils.normalization import normalize_text_for_matching
from nl2sql.utils.text_utils import truncate_text

from .semantic_contract import SemanticContract


@lru_cache(maxsize=8)
def load_semantic_example_selection_rules(path: str | Path | None = None) -> SemanticExampleSelectionRules:
    """Devuelve las reglas tipadas de seleccion de ejemplos semanticos."""

    resolved_path = Path(path).expanduser().resolve() if path is not None else resolve_nl2sql_config_path()
    return load_semantic_resolver_settings(resolved_path).compiler_rules.semantic_example_selection


def _tokenize_semantic_text(value: str) -> tuple[str, ...]:
    normalized = normalize_text_for_matching(value, keep_underscore=False, ascii_fallback=True)
    if not normalized:
        return ()
    return tuple(token for token in normalized.split(" ") if token)


def select_relevant_semantic_examples(
    contract: SemanticContract,
    query: str,
    *,
    limit: int = 3,
    config_path: str | Path | None = None,
    selection_rules: SemanticExampleSelectionRules | None = None,
) -> list[dict[str, Any]]:
    """Selecciona ejemplos del contrato con mayor solapamiento lexical."""

    query_norm = normalize_text_for_matching(query, keep_underscore=False, ascii_fallback=True)
    query_tokens = set(token for token in query_norm.split(" ") if token)
    active_selection_rules = selection_rules or load_semantic_example_selection_rules(config_path)
    ranked_examples: list[tuple[float, dict[str, Any]]] = []

    for raw_example in contract.retrieval_heuristics.semantic_examples:
        if not isinstance(raw_example, dict):
            continue
        question = raw_example.get("question")
        if not isinstance(question, str) or not question.strip():
            continue

        example_norm = normalize_text_for_matching(question, keep_underscore=False, ascii_fallback=True)
        example_tokens = set(token for token in example_norm.split(" ") if token)
        overlap = len(query_tokens & example_tokens)
        exact_bonus = active_selection_rules.exact_match_bonus if example_norm == query_norm else 0.0
        containment_bonus = (
            active_selection_rules.containment_bonus if example_norm and (example_norm in query_norm or query_norm in example_norm) else 0.0
        )
        score = exact_bonus + containment_bonus + float(overlap)
        if score <= 0.0:
            continue
        ranked_examples.append((score, dict(raw_example)))

    ranked_examples.sort(key=lambda item: item[0], reverse=True)
    return [example for _score, example in ranked_examples[:limit]]


def compact_semantic_examples_for_prompt(
    examples: list[dict[str, Any]],
    *,
    question_chars: int,
    metric_limit: int,
    dimension_limit: int,
) -> list[dict[str, Any]]:
    """Reduce ejemplos semánticos al contrato compacto usado por prompts locales."""

    compact_examples: list[dict[str, Any]] = []
    for raw_example in examples:
        compact_example: dict[str, Any] = {}
        question = raw_example.get("question")
        if isinstance(question, str) and question.strip():
            compact_example["question"] = truncate_text(question, question_chars)
        model_name = raw_example.get("model")
        if isinstance(model_name, str) and model_name.strip():
            compact_example["model"] = model_name.strip()
        metrics = raw_example.get("metrics")
        if isinstance(metrics, list):
            compact_example["metrics"] = [str(item) for item in metrics[:metric_limit] if str(item).strip()]
        dimensions = raw_example.get("dimensions")
        if isinstance(dimensions, list):
            compact_example["dimensions"] = [str(item) for item in dimensions[:dimension_limit] if str(item).strip()]
        if compact_example:
            compact_examples.append(compact_example)
    return compact_examples


def render_semantic_examples_for_prompt(examples: list[dict[str, Any]]) -> str:
    """Serializa ejemplos curados a YAML compacto para prompts locales."""

    if not examples:
        return "[]"
    return yaml.safe_dump(examples, sort_keys=False, allow_unicode=True).strip()
