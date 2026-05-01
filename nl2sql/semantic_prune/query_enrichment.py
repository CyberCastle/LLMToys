#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path

from nl2sql.config import QuerySignalRules, resolve_nl2sql_config_path
from nl2sql.utils.normalization import normalize_text_for_matching

from .schema_logic import load_query_signal_rules


def enrich_query_for_retrieval(
    query: str,
    config_path: str | Path | None = None,
    signal_rules: QuerySignalRules | None = None,
) -> str:
    """Expande pistas de retrieval sin alterar la pregunta original de negocio."""

    normalized_query = normalize_text_for_matching(query)
    hints: list[str] = []
    rules = signal_rules
    if rules is None:
        rules_path = Path(config_path).expanduser().resolve() if config_path is not None else resolve_nl2sql_config_path()
        rules = load_query_signal_rules(str(rules_path))

    for pattern, hint in rules.query_enrichment_temporal_hints:
        if pattern.search(normalized_query):
            hints.append(hint)

    for token, hint in rules.query_enrichment_aggregation_hints:
        if token in normalized_query:
            hints.append(hint)

    unique_hints = list(dict.fromkeys(hints))
    if not unique_hints:
        return query

    return f"{query} | pistas: {'; '.join(unique_hints)}"
