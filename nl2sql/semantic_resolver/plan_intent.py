#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass

from nl2sql.utils.normalization import normalize_text_for_matching

from .config import resolve_compiler_rules_path, resolve_rules_path
from .plan_model import PLAN_INTENT_SET, PlanIntent
from .rules_loader import CompilerRules, load_compiler_rules


@dataclass(frozen=True)
class QueryFormMatch:
    """Coincidencia declarativa de una forma de consulta."""

    name: str
    intent: str
    groups: dict[str, str]
    output: dict[str, object]


def match_query_forms(query_norm: str, rules: CompilerRules) -> QueryFormMatch | None:
    """Busca la primera forma declarativa que aplica sobre la query normalizada."""

    for query_form in rules.query_forms:
        for pattern in query_form.patterns:
            match = pattern.search(query_norm)
            if match is None:
                continue
            groups = {key: value.strip() for key, value in match.groupdict().items() if isinstance(value, str) and value.strip()}
            return QueryFormMatch(
                name=query_form.name,
                intent=query_form.intent,
                groups=groups,
                output=dict(query_form.output),
            )
    return None


def detect_intent(query: str, rules: CompilerRules | None = None) -> PlanIntent:
    """Clasifica la intencion semantica con heuristicas conservadoras.

    Los patrones de deteccion se cargan desde `semantic_resolver.compiler_rules`
    en lugar de estar codificados en el modulo, lo que permite ajustarlos sin
    cambiar Python.
    Si no se pasa `rules` explicitamente, se cargan desde la ruta por defecto
    del YAML unificado de configuracion.

    El objetivo no es inferir todos los matices de negocio, sino distinguir si la
    consulta pide una medida simple, una medida con post-agregacion, un ranking o
    solo una exploracion/lookup. Esto permite compilar estructuras distintas sin
    depender del SQL final.
    """

    if rules is None:
        rules = load_compiler_rules(str(resolve_compiler_rules_path()), resolve_rules_path())

    query_norm = normalize_text_for_matching(query, keep_underscore=False)
    query_form_match = match_query_forms(query_norm, rules)
    if query_form_match is not None and query_form_match.intent in PLAN_INTENT_SET:
        return query_form_match.intent  # type: ignore[return-value]
    if any(pattern.search(query_norm) for pattern in rules.intent_patterns_post_agg):
        return "post_aggregated_metric"
    if any(pattern.search(query_norm) for pattern in rules.intent_patterns_ranking):
        return "ranking"
    if any(pattern.search(query_norm) for pattern in rules.intent_patterns_metric):
        return "simple_metric"
    if any(pattern.search(query_norm) for pattern in rules.intent_patterns_filter):
        return "filtered_metric"
    return "lookup"
