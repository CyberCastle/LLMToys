#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Extraccion de señales léxicas desde la pregunta natural."""

from __future__ import annotations

from dataclasses import dataclass
import re

from nl2sql.config.models import QuerySignalRules
from nl2sql.utils.normalization import normalize_text_for_matching
from nl2sql.utils.spanish_morphology import singularize_token

_TERM_RE = re.compile(r"[a-z0-9_]+")


@dataclass(frozen=True)
class QuerySignalProfile:
    """Perfil compacto de intencion semantica inferido desde la query."""

    normalized_query: str
    query_terms: frozenset[str]
    metric_terms: frozenset[str]
    dimension_terms: frozenset[str]
    wants_temporal: bool
    wants_aggregation: bool
    wants_grouping: bool


def extract_meaningful_terms(text: str, signal_rules: QuerySignalRules) -> set[str]:
    """Extrae tokens utiles aplicando stopwords y singularizacion declarativa."""

    terms: set[str] = set()
    normalized_text = normalize_text_for_matching(text, keep_underscore=True)
    for raw_term in _TERM_RE.findall(normalized_text):
        for candidate_term in raw_term.split("_"):
            if len(candidate_term) <= 2 or candidate_term in signal_rules.query_term_stopwords:
                continue
            terms.add(candidate_term)
            singular_term = singularize_token(candidate_term, signal_rules.singular_suffix_rules)
            if len(singular_term) > 2 and singular_term not in signal_rules.query_term_stopwords:
                terms.add(singular_term)
    return terms


def infer_query_signal_profile(query: str, signal_rules: QuerySignalRules) -> QuerySignalProfile:
    """Construye el perfil semantico de una pregunta en lenguaje natural."""

    normalized_query = normalize_text_for_matching(query, keep_underscore=True)
    query_terms = frozenset(extract_meaningful_terms(query, signal_rules))

    dimension_terms: set[str] = set()
    for pattern in signal_rules.groupby_dimension_patterns:
        for match in pattern.finditer(normalized_query):
            dimension_terms.update(
                term
                for term in extract_meaningful_terms(match.group(1), signal_rules)
                if term not in signal_rules.query_temporal_terms and term not in signal_rules.query_intent_noise_terms
            )

    metric_terms = frozenset(
        term
        for term in query_terms
        if term not in dimension_terms
        and term not in signal_rules.query_aggregation_terms
        and term not in signal_rules.query_temporal_terms
        and term not in signal_rules.query_intent_noise_terms
    )
    if not metric_terms:
        metric_terms = frozenset(
            term for term in query_terms if term not in dimension_terms and term not in signal_rules.query_intent_noise_terms
        )

    return QuerySignalProfile(
        normalized_query=normalized_query,
        query_terms=query_terms,
        metric_terms=metric_terms,
        dimension_terms=frozenset(dimension_terms),
        wants_temporal=bool(query_terms & signal_rules.query_temporal_terms or re.search(r"\bultim(?:o|a|os|as)\b", normalized_query)),
        wants_aggregation=bool(query_terms & signal_rules.query_aggregation_terms),
        wants_grouping=bool(re.search(r"\bpor\b", normalized_query) or "group by" in normalized_query),
    )


def get_term_overlap_count(text: str, query_terms: frozenset[str], signal_rules: QuerySignalRules) -> int:
    """Cuenta cuantos terminos de la query aparecen en un texto de schema."""

    if not query_terms:
        return 0
    return len(extract_meaningful_terms(text, signal_rules) & set(query_terms))
