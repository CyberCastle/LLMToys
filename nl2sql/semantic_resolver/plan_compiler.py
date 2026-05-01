#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import partial
import re
from typing import TYPE_CHECKING, Iterable, Mapping

from nl2sql.utils.decision_models import DecisionIssue, dedupe_decision_issues
from nl2sql.utils.normalization import normalize_text_for_matching
from nl2sql.utils.schema_roles import detect_lookup_tables
from nl2sql.utils.semantic_filters import (
    iter_semantic_filter_payloads_from_assets,
    resolve_semantic_filter_selections,
)
from nl2sql.utils.sql_identifiers import TABLE_REFERENCE_RE

from .assets import MatchedAsset, SemanticPlan
from .config import resolve_compiler_rules_path, resolve_rules_path
from .dialects.base import ResolverDialect
from .dialects.registry import get_resolver_dialect
from .plan_compiler_joins import (
    build_join_graph,
    path_transit_tables as _path_transit_tables,
    shortest_join_path,
    split_join_edge as _split_join_edge,
)
from .plan_intent import QueryFormMatch, detect_intent, match_query_forms
from .plan_model import (
    build_compiled_semantic_plan,
    CandidatePlan,
    CandidatePlanSet,
    CompiledSemanticPlan,
    MetricScoreTrace,
    PlanIntent,
    PlanMeasure,
    PlanPostAggregation,
    PlanRanking,
    PlanTimeFilter,
)
from .rules_loader import (
    CompilerRules,
    SemanticDerivedMetric,
    SemanticJoinPath,
    load_compiler_rules,
    load_semantic_derived_metrics,
    load_semantic_join_paths,
)

if TYPE_CHECKING:
    from .config import SemanticResolverConfig


_normalize_text = partial(normalize_text_for_matching, keep_underscore=False)


def _default_rules() -> CompilerRules:
    """Carga las reglas del compilador desde la ruta por defecto (env o assets/).

    Se usa como fallback cuando las funciones publicas se llaman sin `rules`
    explicitamente, por ejemplo desde tests o scripts externos.
    """
    return load_compiler_rules(str(resolve_compiler_rules_path()), resolve_rules_path())


def _tokenize(text: str) -> tuple[str, ...]:
    return tuple(token for token in re.findall(r"[a-z0-9_]+", _normalize_text(text)) if token)


def _dedupe_preserve(values: Iterable[str]) -> list[str]:
    seen_values: set[str] = set()
    deduped_values: list[str] = []
    for value in values:
        if value in seen_values:
            continue
        seen_values.add(value)
        deduped_values.append(value)
    return deduped_values


def _measure_has_state_filter(measure: PlanMeasure | None) -> bool:
    """Detecta si la medida ya aterrizo un subconjunto de estado en su formula."""

    if measure is None:
        return False

    formula_norm = _normalize_text(measure.formula)
    if "estado" not in formula_norm and "status" not in formula_norm:
        return False
    return "case when" in formula_norm or "filter" in formula_norm


def _detect_unmapped_status_qualifiers(
    query_norm: str,
    measure: PlanMeasure | None,
    rules: CompilerRules,
) -> list[str]:
    """Emite warnings cuando la query pide un estado y la medida no lo refleja."""

    query_tokens = set(_tokenize(query_norm))
    matched_tokens = sorted(query_tokens & rules.status_hint_tokens)
    if not matched_tokens:
        return []
    if _measure_has_state_filter(measure):
        return []
    return [f"unmapped_qualifier_in_question:{token}" for token in matched_tokens]


def _get_accepted_assets(plan: SemanticPlan) -> list[MatchedAsset]:
    if plan.all_assets:
        return [asset for asset in plan.all_assets if asset.rejected_reason is None]

    accepted_assets: list[MatchedAsset] = []
    for assets in plan.assets_by_kind.values():
        accepted_assets.extend(assets)
    return accepted_assets


def _get_assets_by_kind(plan: SemanticPlan, kind: str) -> list[MatchedAsset]:
    accepted_assets = [asset for asset in _get_accepted_assets(plan) if asset.asset.kind == kind]
    accepted_assets.sort(
        key=lambda asset: (
            asset.rerank_score,
            asset.compatibility_score,
            asset.embedding_score,
        ),
        reverse=True,
    )
    return accepted_assets


def _build_entity_to_table(entity_assets: list[MatchedAsset]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for matched_asset in entity_assets:
        source_table = matched_asset.asset.payload.get("source_table")
        if isinstance(source_table, str) and source_table:
            mapping[matched_asset.asset.name] = source_table
    return mapping


def _build_entity_keys(entity_assets: list[MatchedAsset]) -> dict[str, str]:
    entity_keys: dict[str, str] = {}
    for matched_asset in entity_assets:
        source_table = matched_asset.asset.payload.get("source_table")
        key = matched_asset.asset.payload.get("key")
        if isinstance(key, str) and key:
            entity_keys[matched_asset.asset.name] = key
            continue
        if isinstance(source_table, str) and source_table:
            entity_keys[matched_asset.asset.name] = f"{source_table}.id"
    return entity_keys


def _build_entity_aliases(
    entity_assets: list[MatchedAsset],
    synonym_assets: list[MatchedAsset],
) -> dict[str, tuple[str, ...]]:
    """Construye aliases comparables por entidad para frases de agrupacion."""

    aliases_by_entity: dict[str, list[str]] = {}
    for synonym_asset in synonym_assets:
        entity_name = synonym_asset.asset.payload.get("entity")
        if not isinstance(entity_name, str):
            continue
        aliases_by_entity.setdefault(entity_name, []).extend((entity_name, entity_name.replace("_", " ")))
        raw_synonyms = synonym_asset.asset.payload.get("synonyms")
        if isinstance(raw_synonyms, list):
            aliases_by_entity[entity_name].extend(str(raw_synonym) for raw_synonym in raw_synonyms if isinstance(raw_synonym, str))

    for matched_asset in entity_assets:
        aliases_by_entity.setdefault(
            matched_asset.asset.name,
            [matched_asset.asset.name, matched_asset.asset.name.replace("_", " ")],
        )

    return {entity_name: tuple(alias for alias in aliases if alias.strip()) for entity_name, aliases in aliases_by_entity.items()}


def _extract_group_phrases(
    query_norm: str,
    rules: CompilerRules,
    *,
    query_form_match: QueryFormMatch | None = None,
) -> list[str]:
    group_phrases: list[str] = []
    if query_form_match is not None:
        raw_dimension = query_form_match.groups.get("dimension")
        if isinstance(raw_dimension, str) and raw_dimension.strip():
            group_phrases.append(raw_dimension.strip())
    for match in rules.group_by_pattern.finditer(query_norm):
        for raw_phrase in re.split(r"\s+y\s+|\s+e\s+|,", match.group(1)):
            phrase = raw_phrase.strip()
            if phrase:
                group_phrases.append(phrase)
    return _dedupe_preserve(group_phrases)


def _extract_measure_terms(
    query_norm: str,
    group_phrases: list[str],
    rules: CompilerRules,
    *,
    query_form_match: QueryFormMatch | None = None,
) -> tuple[str, ...]:
    if query_form_match is not None:
        raw_measure = query_form_match.groups.get("measure")
        if isinstance(raw_measure, str) and raw_measure.strip():
            measure_terms = [token for token in _tokenize(raw_measure) if token not in rules.stop_tokens]
            if measure_terms:
                return tuple(measure_terms)

    for pattern in rules.measure_hint_patterns:
        match = pattern.search(query_norm)
        if match is not None:
            measure_terms = [token for token in _tokenize(match.group(1)) if token not in rules.stop_tokens]
            if measure_terms:
                return tuple(measure_terms)

    group_tokens = {token for phrase in group_phrases for token in _tokenize(phrase)}
    return tuple(token for token in _tokenize(query_norm) if token not in rules.stop_tokens and token not in group_tokens)


def _infer_source_table(
    payload: Mapping[str, object],
    formula: str,
    entity_to_table: dict[str, str],
) -> str | None:
    source_table = payload.get("source_table")
    if isinstance(source_table, str) and source_table:
        return source_table

    entity_name = payload.get("entity")
    if isinstance(entity_name, str) and entity_name in entity_to_table:
        return entity_to_table[entity_name]

    formula_matches = [match.group(1) for match in TABLE_REFERENCE_RE.finditer(formula)]
    if formula_matches:
        return formula_matches[0]
    return None


def _query_requests_ratio(query_tokens: set[str], rules: CompilerRules) -> bool:
    """Indica si la query pide explicitamente una tasa/porcentaje."""

    return bool(query_tokens & rules.ratio_query_tokens)


def _score_metric_phrase_match(matched_asset: MatchedAsset, query_norm: str, rules: CompilerRules) -> float:
    """Premia alias y ejemplos declarados en YAML cuando coinciden con la query."""

    payload = matched_asset.asset.payload
    best_score = 0.0
    phrase_rules = rules.metric_phrase_scoring

    candidate_phrases = [matched_asset.asset.name.replace("_", " ")]
    raw_synonyms = payload.get("synonyms")
    if isinstance(raw_synonyms, list):
        candidate_phrases.extend(str(item) for item in raw_synonyms if isinstance(item, str) and item.strip())

    for candidate_phrase in candidate_phrases:
        normalized_phrase = _normalize_text(candidate_phrase)
        if not normalized_phrase:
            continue
        if normalized_phrase == query_norm:
            best_score = max(best_score, phrase_rules.exact_query_score)
        elif normalized_phrase in query_norm:
            best_score = max(best_score, phrase_rules.containment_score)

    raw_examples = payload.get("examples")
    if not isinstance(raw_examples, list):
        return best_score

    for raw_example in raw_examples:
        if not isinstance(raw_example, Mapping):
            continue
        question = raw_example.get("question")
        expected_metric = raw_example.get("expected_metric")
        if not isinstance(question, str) or expected_metric != matched_asset.asset.name:
            continue
        if _normalize_text(question) == query_norm:
            return max(best_score, phrase_rules.exact_example_score)
    return best_score


def _score_metric_candidate(
    matched_asset: MatchedAsset,
    query_norm: str,
    measure_terms: tuple[str, ...],
    query_tokens: set[str],
    intent: PlanIntent,
    entity_to_table: dict[str, str],
    hinted_entities: set[str],
    rules: CompilerRules,
) -> tuple[float, dict[str, float]]:
    payload = matched_asset.asset.payload
    formula = str(payload.get("formula") or payload.get("expr") or payload.get("source") or "").strip()
    if not formula:
        return float("-inf"), {}

    source_table = _infer_source_table(payload, formula, entity_to_table)
    if source_table is None:
        return float("-inf"), {}
    searchable_text = " ".join(
        filter(
            None,
            (
                matched_asset.asset.name,
                str(payload.get("description") or ""),
                formula,
                str(payload.get("entity") or ""),
                source_table or "",
            ),
        )
    )
    searchable_tokens = set(_tokenize(searchable_text))
    measure_overlap = len(searchable_tokens & set(measure_terms))
    ratio_requested = _query_requests_ratio(query_tokens, rules)
    weights = rules.metric_scoring

    score_components: dict[str, float] = {
        "compatibility": matched_asset.compatibility_score * weights.compatibility_weight,
        "embedding": matched_asset.embedding_score * weights.embedding_weight,
        "measure_overlap": measure_overlap * weights.measure_overlap_weight,
        "phrase_match": _score_metric_phrase_match(matched_asset, query_norm, rules),
    }

    entity_name = payload.get("entity")
    if isinstance(entity_name, str) and entity_name in hinted_entities:
        score_components["hinted_entity_boost"] = weights.hinted_entity_boost
    if source_table is not None and any(token in _tokenize(source_table) for token in measure_terms):
        score_components["source_table_hint_boost"] = weights.source_table_hint_boost
    if re.fullmatch(r"count_distinct\([a-z_][a-z0-9_]*\.[a-z_][a-z0-9_]*\)", formula):
        score_components["count_distinct_bonus"] = weights.count_distinct_bonus
    if "count" in formula and intent == "post_aggregated_metric":
        score_components["post_aggregated_count_bonus"] = weights.post_aggregated_count_bonus
    if "total" in _normalize_text(matched_asset.asset.name):
        score_components["total_metric_name_bonus"] = weights.total_metric_name_bonus
    if "/" in formula and not ratio_requested:
        score_components["ratio_not_requested_penalty"] = weights.ratio_not_requested_penalty

    status_tokens = query_tokens.intersection(rules.status_hint_tokens)
    if status_tokens and not ratio_requested:
        metric_view = PlanMeasure(name=matched_asset.asset.name, formula=formula, source_table=source_table)
        if _measure_has_state_filter(metric_view):
            score_components["state_filter_token_boost"] = len(status_tokens & searchable_tokens) * weights.state_filter_token_boost
        else:
            score_components["missing_state_filter_penalty"] = weights.missing_state_filter_penalty
    if "case when" in formula.lower() and not query_tokens.intersection(rules.status_hint_tokens):
        score_components["unexpected_case_when_penalty"] = weights.unexpected_case_when_penalty
    return sum(score_components.values()), score_components


@dataclass(frozen=True)
class MeasureCandidate:
    """Metrica candidata ya puntuada y materializada como medida del plan."""

    asset_id: str
    measure: PlanMeasure
    score: float
    trace: MetricScoreTrace


def _apply_metric_trace_selection(
    metric_score_trace: list[MetricScoreTrace],
    selected_measure_name: str | None,
) -> list[MetricScoreTrace]:
    """Marca la metrica seleccionada dentro del trace puntuado."""

    return [
        replace(
            trace,
            selected=selected_measure_name is not None and trace.metric_name == selected_measure_name,
            rejected_reason=(None if selected_measure_name is not None and trace.metric_name == selected_measure_name else "not_selected"),
        )
        for trace in metric_score_trace
    ]


def rank_measure_candidates(
    metric_assets: list[MatchedAsset],
    *,
    query_norm: str,
    entity_to_table: dict[str, str],
    intent: PlanIntent,
    hinted_entities: set[str],
    rules: CompilerRules | None = None,
    query_form_match: QueryFormMatch | None = None,
) -> tuple[list[MeasureCandidate], list[MetricScoreTrace]]:
    """Ordena metricas candidatas para permitir seleccion y reparacion posteriores."""

    if rules is None:
        rules = _default_rules()
    group_phrases = _extract_group_phrases(query_norm, rules, query_form_match=query_form_match)
    measure_terms = _extract_measure_terms(
        query_norm,
        group_phrases,
        rules,
        query_form_match=query_form_match,
    )
    query_tokens = set(_tokenize(query_norm))

    candidates: list[MeasureCandidate] = []
    metric_score_trace: list[MetricScoreTrace] = []
    for matched_asset in metric_assets:
        payload = matched_asset.asset.payload
        formula = str(payload.get("formula") or payload.get("expr") or payload.get("source") or "").strip()
        if not formula:
            continue
        source_table = _infer_source_table(payload, formula, entity_to_table)
        if source_table is None:
            continue

        candidate_score, score_components = _score_metric_candidate(
            matched_asset,
            query_norm,
            measure_terms,
            query_tokens,
            intent,
            entity_to_table,
            hinted_entities,
            rules,
        )
        trace = MetricScoreTrace(
            metric_name=matched_asset.asset.name,
            components=score_components,
            total_score=candidate_score,
        )
        metric_score_trace.append(trace)
        candidates.append(
            MeasureCandidate(
                asset_id=matched_asset.asset.asset_id,
                measure=PlanMeasure(
                    name=matched_asset.asset.name,
                    formula=formula,
                    source_table=source_table,
                ),
                score=candidate_score,
                trace=trace,
            )
        )

    candidates.sort(key=lambda candidate: candidate.score, reverse=True)
    metric_score_trace.sort(key=lambda trace: trace.total_score, reverse=True)
    return candidates, metric_score_trace


def extract_measure(
    metric_assets: list[MatchedAsset],
    *,
    query_norm: str,
    entity_to_table: dict[str, str],
    intent: PlanIntent,
    hinted_entities: set[str],
    rules: CompilerRules | None = None,
) -> tuple[PlanMeasure | None, list[MetricScoreTrace]]:
    """Selecciona la medida mas compatible con la porcion agregada de la query."""

    candidates, metric_score_trace = rank_measure_candidates(
        metric_assets,
        query_norm=query_norm,
        entity_to_table=entity_to_table,
        intent=intent,
        hinted_entities=hinted_entities,
        rules=rules,
    )
    if not candidates:
        return None, metric_score_trace
    selected_measure = candidates[0].measure
    return selected_measure, _apply_metric_trace_selection(metric_score_trace, selected_measure.name)


def select_base_entity(measure: PlanMeasure, entity_assets: list[MatchedAsset]) -> tuple[str, str]:
    """Resuelve la entidad base desde la fuente real de la medida.

    Regla canonica: la entidad base corresponde a la tabla donde vive el grano de
    la medida principal. Las dimensiones usadas para agrupar pueden venir de otra
    entidad enlazada por joins, pero no deben desplazar a la base_entity.
    """

    for matched_asset in entity_assets:
        source_table = matched_asset.asset.payload.get("source_table")
        if source_table == measure.source_table:
            return matched_asset.asset.name, f"{measure.source_table}.id"
    return measure.source_table, f"{measure.source_table}.id"


def _match_entity_group_field(
    phrase_norm: str,
    entity_assets: list[MatchedAsset],
    synonym_assets: list[MatchedAsset],
    entity_keys: dict[str, str],
    rules: CompilerRules,
) -> str | None:
    """Resuelve la clave tecnica de la entidad mencionada en la frase."""

    matched_entity = _match_entity_name(phrase_norm, entity_assets, synonym_assets, rules)
    if matched_entity is None:
        return None
    return entity_keys.get(matched_entity[0])


def _match_entity_name(
    phrase_norm: str,
    entity_assets: list[MatchedAsset],
    synonym_assets: list[MatchedAsset],
    rules: CompilerRules,
) -> tuple[str, float] | None:
    """Detecta la entidad mas probable mencionada en una frase libre."""

    best_match: tuple[float, str] | None = None
    phrase_tokens = set(_tokenize(phrase_norm))
    aliases_by_entity = _build_entity_aliases(entity_assets, synonym_assets)
    matching_rules = rules.entity_matching

    for matched_asset in entity_assets:
        entity_name = matched_asset.asset.name
        aliases = aliases_by_entity.get(entity_name, (entity_name, entity_name.replace("_", " ")))
        best_alias_score = 0.0
        for alias in aliases:
            alias_norm = _normalize_text(alias)
            alias_tokens = set(_tokenize(alias_norm))
            if alias_norm == phrase_norm:
                best_alias_score = max(best_alias_score, matching_rules.alias_exact_score)
                continue
            if alias_norm and (alias_norm in phrase_norm or phrase_norm in alias_norm):
                best_alias_score = max(best_alias_score, matching_rules.alias_containment_score)
            if alias_tokens:
                overlap = len(alias_tokens & phrase_tokens) / max(len(alias_tokens), 1)
                best_alias_score = max(best_alias_score, overlap)

        if best_alias_score <= 0.0:
            continue
        if best_match is None or best_alias_score > best_match[0]:
            best_match = (best_alias_score, entity_name)

    if best_match is None:
        return None
    return best_match[1], best_match[0]


def _score_dimension_display_preference(
    matched_asset: MatchedAsset,
    *,
    entity_name: str,
    entity_table: str,
    rules: CompilerRules,
) -> float:
    """Puntua si una dimension sirve como etiqueta legible para un ranking."""

    payload = matched_asset.asset.payload
    if payload.get("entity") != entity_name:
        return -1.0

    source_value = payload.get("source") or payload.get("field")
    if not isinstance(source_value, str) or not source_value.strip():
        return -1.0

    type_value = _normalize_text(str(payload.get("type") or ""))
    search_text = (
        _normalize_text(
            " ".join(
                (
                    matched_asset.asset.name,
                    str(source_value),
                    str(payload.get("field") or ""),
                    type_value,
                )
            )
        )
        .replace("_", " ")
        .replace(".", " ")
    )
    search_tokens = {token for token in search_text.split() if token}

    preference_rules = rules.ranking_dimension_preference
    positive_signal = sum(weight for token, weight in preference_rules.positive_token_weights if token in search_tokens)
    negative_signal = sum(preference_rules.negative_token_penalty for token in preference_rules.negative_tokens if token in search_tokens)
    if positive_signal <= negative_signal:
        return -1.0

    score = positive_signal - negative_signal
    if type_value in preference_rules.string_type_hints:
        score += preference_rules.string_type_bonus
    if matched_asset.asset.name.startswith(f"{entity_name}_"):
        score += preference_rules.entity_prefix_bonus
    if entity_table and f"{entity_table}." in source_value:
        score += preference_rules.same_table_bonus
    return score


def _match_preferred_dimension_field_for_entity(
    phrase_norm: str,
    *,
    entity_assets: list[MatchedAsset],
    dimension_assets: list[MatchedAsset],
    synonym_assets: list[MatchedAsset],
    rules: CompilerRules,
) -> str | None:
    """Selecciona una dimension descriptiva cuando la frase solo nombra la entidad."""

    matched_entity = _match_entity_name(phrase_norm, entity_assets, synonym_assets, rules)
    if matched_entity is None:
        return None

    entity_name, entity_match_score = matched_entity
    entity_to_table = _build_entity_to_table(entity_assets)
    entity_table = entity_to_table.get(entity_name, "")

    best_match: tuple[float, str] | None = None
    for matched_asset in dimension_assets:
        score = _score_dimension_display_preference(
            matched_asset,
            entity_name=entity_name,
            entity_table=entity_table,
            rules=rules,
        )
        if score <= 0.0:
            continue
        source_value = matched_asset.asset.payload.get("source") or matched_asset.asset.payload.get("field")
        if not isinstance(source_value, str) or not source_value.strip():
            continue
        candidate_score = score + entity_match_score
        if best_match is None or candidate_score > best_match[0]:
            best_match = (candidate_score, source_value)

    return best_match[1] if best_match is not None else None


def _match_field_from_assets(
    phrase_norm: str,
    assets: list[MatchedAsset],
    candidate_keys: tuple[str, ...],
    rules: CompilerRules,
) -> str | None:
    best_match: tuple[float, str] | None = None
    phrase_tokens = set(_tokenize(phrase_norm))
    field_rules = rules.field_matching
    for matched_asset in assets:
        payload = matched_asset.asset.payload
        field_value = None
        for candidate_key in candidate_keys:
            value = payload.get(candidate_key)
            if isinstance(value, str) and value:
                field_value = value
                break
        if field_value is None:
            continue

        search_text = " ".join(
            filter(
                None,
                (
                    matched_asset.asset.name,
                    str(payload.get("entity") or ""),
                    field_value,
                ),
            )
        )
        search_text_norm = _normalize_text(search_text)
        search_tokens = set(_tokenize(search_text))
        overlap = len(search_tokens & phrase_tokens)
        if overlap <= 0 and phrase_norm not in search_text_norm:
            continue

        candidate_score = float(overlap)
        if phrase_norm in search_text_norm:
            candidate_score += field_rules.phrase_contains_bonus
        if best_match is None or candidate_score > best_match[0]:
            best_match = (candidate_score, field_value)

    return best_match[1] if best_match is not None else None


def extract_group_by(
    query_norm: str,
    *,
    entity_assets: list[MatchedAsset],
    dimension_assets: list[MatchedAsset],
    filter_assets: list[MatchedAsset],
    synonym_assets: list[MatchedAsset],
    base_entity: str,
    base_table: str,
    rules: CompilerRules | None = None,
    query_form_match: QueryFormMatch | None = None,
) -> list[str]:
    """Resuelve dimensiones de agrupacion sin confundirlas con la entidad base."""

    if rules is None:
        rules = _default_rules()
    entity_keys = _build_entity_keys(entity_assets)
    base_terms = {_normalize_text(base_entity), _normalize_text(base_table)}
    resolved_group_by: list[str] = []
    ranking_dimension_phrase = query_form_match.groups.get("dimension") if query_form_match is not None else None
    for group_phrase in _extract_group_phrases(query_norm, rules, query_form_match=query_form_match):
        phrase_norm = _normalize_text(group_phrase)
        prefer_dimension_field = bool(
            query_form_match is not None
            and query_form_match.intent == "ranking"
            and isinstance(ranking_dimension_phrase, str)
            and phrase_norm == _normalize_text(ranking_dimension_phrase)
        )
        group_field = None
        if prefer_dimension_field:
            group_field = _match_field_from_assets(phrase_norm, dimension_assets, ("source", "field"), rules)
            if group_field is None:
                group_field = _match_preferred_dimension_field_for_entity(
                    phrase_norm,
                    entity_assets=entity_assets,
                    dimension_assets=dimension_assets,
                    synonym_assets=synonym_assets,
                    rules=rules,
                )
            if group_field is None:
                group_field = _match_field_from_assets(phrase_norm, filter_assets, ("field", "source"), rules)
            if group_field is None:
                group_field = _match_entity_group_field(phrase_norm, entity_assets, synonym_assets, entity_keys, rules)
        else:
            group_field = _match_entity_group_field(phrase_norm, entity_assets, synonym_assets, entity_keys, rules)
            if group_field is None:
                group_field = _match_field_from_assets(phrase_norm, filter_assets, ("field", "source"), rules)
            if group_field is None:
                group_field = _match_field_from_assets(phrase_norm, dimension_assets, ("field", "source"), rules)
        if group_field is None or "." not in group_field:
            continue

        group_table = group_field.split(".", 1)[0]
        if group_table == base_table and phrase_norm not in base_terms:
            continue
        resolved_group_by.append(group_field)

    return _dedupe_preserve(resolved_group_by)


def _build_resolved_expressions(
    rule_dialect_values: dict[str, str],
    canonical_value: str,
    dialect: ResolverDialect | None,
) -> dict[str, str]:
    """Materializa el dict ``resolved_expressions`` de un ``PlanTimeFilter``.

    Copia las expresiones precomputadas del YAML para todos los dialectos y,
    cuando hay un dialecto activo que no aparece en el YAML, intenta obtener
    una expresion de respaldo via ``dialect.render_time_expression``. Si el
    fallback tambien falla, simplemente no se agrega la clave; el solver SQL
    seguira teniendo ``value`` canonico para reinterpretar.
    """

    resolved: dict[str, str] = dict(rule_dialect_values)
    if dialect is not None and dialect.name and dialect.name not in resolved:
        fallback = dialect.render_time_expression(canonical_value, compiler_rules=rules)
        if isinstance(fallback, str) and fallback.strip():
            resolved[dialect.name] = fallback.strip()
    return resolved


def extract_time_filter(
    query_norm: str,
    *,
    entity_assets: list[MatchedAsset],
    dimension_assets: list[MatchedAsset],
    base_entity: str,
    base_table: str,
    rules: CompilerRules | None = None,
    dialect: ResolverDialect | None = None,
) -> tuple[PlanTimeFilter | None, list[str]]:
    warnings: list[str] = []
    if rules is None:
        rules = _default_rules()
    matched_time_rule: tuple[str, str] | None = None
    matched_dialect_values: dict[str, str] = {}
    # Los patrones de tiempo vienen del YAML; se comparan en el orden declarado
    # y se usa el primero que coincide con la query normalizada. El mapping
    # ``dialect_values`` es agnostico al motor: el dialecto activo decide cual
    # entrada concreta materializar en ``resolved_expressions``.
    for time_pattern_rule in rules.time_patterns:
        if time_pattern_rule.pattern.search(query_norm):
            matched_time_rule = (time_pattern_rule.operator, time_pattern_rule.value)
            matched_dialect_values = time_pattern_rule.dialect_values
            break

    if matched_time_rule is None:
        return None, warnings

    resolved_expressions = _build_resolved_expressions(matched_dialect_values, matched_time_rule[1], dialect)
    for matched_asset in entity_assets:
        if matched_asset.asset.name != base_entity:
            continue
        for key_name in ("time_field", "date_field"):
            field = matched_asset.asset.payload.get(key_name)
            if isinstance(field, str) and field:
                return (
                    PlanTimeFilter(
                        field=field,
                        operator=matched_time_rule[0],
                        value=matched_time_rule[1],
                        resolved_expressions=resolved_expressions,
                    ),
                    warnings,
                )

    best_candidate: tuple[float, str] | None = None
    time_field_rules = rules.time_field_scoring
    for matched_asset in dimension_assets:
        payload = matched_asset.asset.payload
        source = payload.get("field") or payload.get("source")
        if not isinstance(source, str) or "." not in source:
            continue
        table_name = source.split(".", 1)[0]
        if table_name != base_table:
            continue
        candidate_score = time_field_rules.base_score
        entity_name = payload.get("entity")
        if entity_name == base_entity:
            candidate_score += time_field_rules.entity_match_bonus
        column_type = str(payload.get("type") or "")
        if any(type_hint in _normalize_text(column_type) for type_hint in rules.temporal_type_hints):
            candidate_score += time_field_rules.temporal_type_bonus
        if any(term in _normalize_text(f"{matched_asset.asset.name} {source}") for term in time_field_rules.temporal_name_terms):
            candidate_score += time_field_rules.temporal_name_bonus
        if best_candidate is None or candidate_score > best_candidate[0]:
            best_candidate = (candidate_score, source)

    if best_candidate is None:
        warnings.append(f"time_expression_found_but_no_time_field_on_{base_entity}")
        return None, warnings

    return (
        PlanTimeFilter(
            field=best_candidate[1],
            operator=matched_time_rule[0],
            value=matched_time_rule[1],
            resolved_expressions=resolved_expressions,
        ),
        warnings,
    )


def build_post_aggregation(
    intent: PlanIntent,
    query_norm: str,
    *,
    has_group_by: bool,
    default_function: str,
    rules: CompilerRules,
) -> tuple[PlanPostAggregation | None, str | None]:
    if intent != "post_aggregated_metric":
        return None, None
    post_aggregation_rules = rules.post_aggregation
    if not has_group_by:
        return None, post_aggregation_rules.missing_group_by_warning
    normalized_default_function = default_function.strip().lower()
    selected_function = post_aggregation_rules.default_functions.get(normalized_default_function, normalized_default_function)
    if _query_requests_ratio(set(_tokenize(query_norm)), rules):
        selected_function = post_aggregation_rules.ratio_function
    return PlanPostAggregation(function=selected_function, over=post_aggregation_rules.over), None  # type: ignore[arg-type]


def _detect_lookup_tables(pruned_schema: Mapping[str, object] | None, rules: CompilerRules) -> frozenset[str]:
    """Identifica tablas catalogo/lookup a partir de un pruned_schema.

    La heuristica replica la que usa el semantic_prune: pocas columnas,
    sin FKs propias, sin columnas temporales o numericas dominantes.
    Esto permite que el plan_compiler refuerce la misma politica aun cuando
    el pruning ya haya dejado una ruta via catalogo.
    Los umbrales y listas de tipos vienen del YAML de reglas del compilador.
    """
    return detect_lookup_tables(
        pruned_schema,
        lookup_column_count_max=rules.lookup_column_count_max,
        lookup_max_numeric_columns=rules.lookup_max_numeric_columns,
        temporal_type_hints=rules.lookup_temporal_type_hints,
        numeric_type_hints=rules.lookup_numeric_type_hints,
        identifier_column_names=rules.lookup_identifier_column_names,
        identifier_suffixes=rules.lookup_identifier_suffixes,
    )


def _merge_join_path(existing_join_path: list[str], candidate_join_path: list[str]) -> list[str]:
    seen_edges = set(existing_join_path)
    merged_join_path = list(existing_join_path)
    for join_edge in candidate_join_path:
        if join_edge in seen_edges:
            continue
        seen_edges.add(join_edge)
        merged_join_path.append(join_edge)
    return merged_join_path


def _extract_formula_tables(formula: str) -> list[str]:
    """Extrae tablas fisicas referenciadas en una formula de medida."""

    seen_tables: set[str] = set()
    ordered_tables: list[str] = []
    for match in TABLE_REFERENCE_RE.finditer(formula):
        table_name = match.group(1)
        if table_name in seen_tables:
            continue
        seen_tables.add(table_name)
        ordered_tables.append(table_name)
    return ordered_tables


def _join_path_contains_table(join_path: list[str], table_name: str) -> bool:
    """Indica si una tabla ya forma parte de la ruta de joins compilada."""

    for join_edge in join_path:
        join_sides = _split_join_edge(join_edge)
        if join_sides is None:
            continue
        left_side, right_side = join_sides
        for side in (left_side, right_side):
            if "." not in side:
                continue
            if side.split(".", 1)[0].strip() == table_name:
                return True
    return False


def _augment_join_path_with_required_tables(
    *,
    join_path: list[str],
    target_tables: Iterable[str],
    base_table: str,
    join_graph: dict[str, list[tuple[str, str]]],
    lookup_tables: frozenset[str],
    reason: str,
) -> tuple[list[str], list[str]]:
    """Agrega joins para tablas adicionales requeridas por el plan compilado."""

    warnings: list[str] = []
    augmented_join_path = list(join_path)
    for target_table in _dedupe_preserve(table.strip() for table in target_tables if isinstance(table, str)):
        if not target_table or target_table == base_table or _join_path_contains_table(augmented_join_path, target_table):
            continue

        candidate_join_path = shortest_join_path(
            join_graph,
            base_table,
            target_table,
            forbidden_transit=lookup_tables,
        )
        if not candidate_join_path:
            candidate_join_path = shortest_join_path(join_graph, base_table, target_table)
        if not candidate_join_path:
            warnings.append(f"no_join_path_from_{base_table}_to_{target_table}_for_{reason}")
            continue

        augmented_join_path = _merge_join_path(augmented_join_path, candidate_join_path)

    return augmented_join_path, warnings


def _augment_join_path_with_measure_support_tables(
    *,
    join_path: list[str],
    measure: PlanMeasure | None,
    base_table: str,
    join_graph: dict[str, list[tuple[str, str]]],
    lookup_tables: frozenset[str],
) -> tuple[list[str], list[str]]:
    """Agrega joins necesarios para tablas fisicas usadas dentro de la formula."""

    if measure is None:
        return join_path, []

    return _augment_join_path_with_required_tables(
        join_path=join_path,
        target_tables=_extract_formula_tables(measure.formula),
        base_table=base_table,
        join_graph=join_graph,
        lookup_tables=lookup_tables,
        reason="measure",
    )


def _select_semantic_model(
    model_assets: list[MatchedAsset],
    base_table: str,
    group_by: list[str],
    rules: CompilerRules,
) -> str | None:
    if not model_assets:
        return None

    group_tables = {group_field.split(".", 1)[0] for group_field in group_by if "." in group_field}
    best_model: tuple[float, str] | None = None
    model_rules = rules.semantic_model_scoring
    for matched_asset in model_assets:
        payload = matched_asset.asset.payload
        core_tables = payload.get("core_tables")
        model_tables = {str(table_name) for table_name in core_tables} if isinstance(core_tables, list) else set()
        grain_tables_raw = payload.get("grain")
        grain_tables = {str(table_name) for table_name in grain_tables_raw} if isinstance(grain_tables_raw, list) else set()

        candidate_score = (
            matched_asset.compatibility_score * model_rules.compatibility_weight
            + matched_asset.embedding_score * model_rules.embedding_weight
        )

        # Senal fuerte: la tabla base de la medida figura en el grano del modelo.
        # El grano es mas especifico que core_tables y refleja mejor el sujeto
        # real del modelo frente a tablas que solo aparecen como contexto
        # estructural dentro del mismo modelo.
        if base_table in grain_tables:
            candidate_score += model_rules.grain_table_bonus
        elif base_table in model_tables:
            candidate_score += model_rules.core_table_bonus

        candidate_score += len(model_tables & group_tables) * model_rules.group_table_overlap_weight

        # Penalizacion fuerte cuando la propia descripcion del modelo advierte
        # que no debe usarse para JOINs automaticos (joins referenciales, sin
        # FKs formales, no deterministicos). Estos modelos suelen ser puente
        # hacia datos externos y no sirven para compilar un plan ejecutable
        # desde solo las core_tables.
        description_norm = _normalize_text(str(payload.get("description") or ""))
        if any(warning_token in description_norm for warning_token in rules.model_warning_tokens):
            candidate_score -= model_rules.warning_penalty

        if best_model is None or candidate_score > best_model[0]:
            best_model = (candidate_score, matched_asset.asset.name)
    return best_model[1] if best_model is not None else None


def _extract_required_tables(base_table: str, join_path: list[str], group_by: list[str]) -> list[str]:
    """Devuelve la lista ordenada de tablas que el JOIN obliga a incluir.

    Se construye desde base_table + todas las tablas mencionadas en join_path
    y en group_by, preservando el orden canonico base -> bridges -> dimensiones.
    """
    seen_tables: set[str] = set()
    ordered_tables: list[str] = []

    def _add(table_name: str) -> None:
        if table_name in seen_tables:
            return
        seen_tables.add(table_name)
        ordered_tables.append(table_name)

    _add(base_table)
    for edge_label in join_path:
        join_sides = _split_join_edge(edge_label)
        if join_sides is None:
            continue
        left_side, right_side = join_sides
        for side in (left_side, right_side):
            if "." in side:
                _add(side.split(".", 1)[0])
    for group_field in group_by:
        if "." in group_field:
            _add(group_field.split(".", 1)[0])
    return ordered_tables


def _build_deterministic_issues(*, base_entity: str, warnings: list[str]) -> list[DecisionIssue]:
    """Promueve warnings estructuralmente invalidantes a issues bloqueantes."""

    issues: list[DecisionIssue] = []
    for warning in warnings:
        if warning.startswith("time_expression_found_but_no_time_field_on_"):
            unresolved_entity = warning.removeprefix("time_expression_found_but_no_time_field_on_") or base_entity
            issues.append(
                DecisionIssue(
                    stage="semantic_compilation",
                    code="time_filter_unresolved",
                    severity="error",
                    message=("La consulta pide un filtro temporal pero la entidad base no tiene un " "campo temporal operativo resoluble."),
                    context={
                        "base_entity": unresolved_entity,
                        "warning": warning,
                    },
                )
            )
        elif warning.startswith("no_join_path_from_") and warning.endswith("_for_time_filter"):
            issues.append(
                DecisionIssue(
                    stage="semantic_compilation",
                    code="time_filter_join_path_unresolved",
                    severity="error",
                    message=(
                        "La consulta pide un filtro temporal pero el resolver no pudo construir "
                        "el join hacia la tabla temporal requerida."
                    ),
                    context={
                        "base_entity": base_entity,
                        "warning": warning,
                    },
                )
            )
        elif warning.startswith("no_join_path_from_") and warning.endswith("_for_selected_filter"):
            issues.append(
                DecisionIssue(
                    stage="semantic_compilation",
                    code="selected_filter_join_path_unresolved",
                    severity="error",
                    message=(
                        "La consulta pide un filtro semantico pero el resolver no pudo construir "
                        "el join hacia la tabla filtrada requerida."
                    ),
                    context={
                        "base_entity": base_entity,
                        "warning": warning,
                    },
                )
            )
    return dedupe_decision_issues(issues)


def _estimate_confidence(plan: CompiledSemanticPlan, rules: CompilerRules) -> float:
    confidence_rules = rules.confidence_scoring
    confidence = confidence_rules.base_score
    if plan.measure is not None:
        confidence += confidence_rules.measure_bonus
    if plan.group_by:
        confidence += confidence_rules.group_by_bonus
    if plan.ranking is not None:
        confidence += confidence_rules.ranking_bonus
    if plan.time_filter is not None:
        confidence += confidence_rules.time_filter_bonus
    if plan.post_aggregation is not None:
        confidence += confidence_rules.post_aggregation_bonus
    if plan.semantic_model is not None:
        confidence += confidence_rules.semantic_model_bonus
    if plan.join_path or not plan.group_by:
        confidence += confidence_rules.join_path_or_scalar_bonus
    confidence -= min(
        confidence_rules.warning_penalty_cap,
        len(plan.warnings) * confidence_rules.warning_penalty_per_item,
    )
    if any(warning.startswith("unmapped_qualifier_in_question:") for warning in plan.warnings):
        confidence -= confidence_rules.unmapped_qualifier_penalty
    return max(0.0, min(1.0, confidence))


def _coerce_positive_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if not isinstance(value, (int, float, str)):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _resolve_ranking(query_form_match: QueryFormMatch | None, rules: CompilerRules) -> PlanRanking | None:
    if query_form_match is None or query_form_match.intent != "ranking":
        return None
    direction = str(query_form_match.output.get("sort_direction") or "desc").strip().lower()
    normalized_direction = direction if direction in {"asc", "desc"} else "desc"
    limit = _coerce_positive_int(query_form_match.groups.get("limit"))
    if limit is None:
        limit = _coerce_positive_int(query_form_match.output.get("default_limit"))
    if limit is None:
        limit = rules.selection_tuning.ranking_default_limit
    return PlanRanking(limit=limit, direction=normalized_direction)  # type: ignore[arg-type]


def _build_selection_rationale(
    *,
    selected_plan: CompiledSemanticPlan,
    selected_score: float,
    query_form_match: QueryFormMatch | None,
) -> str:
    measure_name = selected_plan.measure.name if selected_plan.measure is not None else "none"
    if query_form_match is not None:
        return f"selected_highest_scoring_candidate_for_query_form:{query_form_match.name}:{measure_name}:{selected_score:.2f}"
    return f"selected_highest_metric_score:{measure_name}:{selected_score:.2f}"


def _candidate_issues(candidate_plan: CompiledSemanticPlan, *, selected_measure_name: str | None) -> list[str]:
    issues: list[str] = []
    if candidate_plan.measure is None:
        issues.append("no_measure_resolved")
    elif selected_measure_name is not None and candidate_plan.measure.name != selected_measure_name:
        issues.append("not_selected")
    if candidate_plan.ranking is not None and not candidate_plan.group_by:
        issues.append("ranking_dimension_unresolved")
    return issues


def _candidate_from_plan(
    candidate_plan: CompiledSemanticPlan,
    *,
    score: float,
    selected_measure_name: str | None,
) -> CandidatePlan:
    return CandidatePlan(
        intent=candidate_plan.intent,
        base_entity=candidate_plan.base_entity,
        grain=candidate_plan.grain,
        semantic_model=candidate_plan.semantic_model,
        measure=candidate_plan.measure,
        group_by=list(candidate_plan.group_by),
        final_group_by=list(candidate_plan.final_group_by),
        selected_filters=list(candidate_plan.selected_filters),
        time_filter=candidate_plan.time_filter,
        ranking=candidate_plan.ranking,
        post_aggregation=candidate_plan.post_aggregation,
        join_path=list(candidate_plan.join_path),
        required_tables=list(candidate_plan.required_tables),
        warnings=list(candidate_plan.warnings),
        confidence=candidate_plan.confidence,
        score=score,
        issues=_candidate_issues(candidate_plan, selected_measure_name=selected_measure_name),
        join_path_hint=candidate_plan.join_path_hint,
        derived_metric_ref=candidate_plan.derived_metric_ref,
        population_scope=candidate_plan.population_scope,
        base_group_by=list(candidate_plan.base_group_by),
        intermediate_alias=candidate_plan.intermediate_alias,
    )


def _build_compiled_candidate(
    *,
    query: str,
    query_norm: str,
    intent: PlanIntent,
    measure: PlanMeasure | None,
    query_form_match: QueryFormMatch | None,
    accepted_entities: list[MatchedAsset],
    accepted_dimensions: list[MatchedAsset],
    accepted_filters: list[MatchedAsset],
    accepted_synonyms: list[MatchedAsset],
    accepted_models: list[MatchedAsset],
    entity_to_table: dict[str, str],
    pruned_schema: Mapping[str, object] | None,
    rules: CompilerRules,
    dialect: ResolverDialect | None,
    default_post_aggregation: str,
    join_graph: dict[str, list[tuple[str, str]]],
    lookup_tables: frozenset[str],
    semantic_join_paths: tuple[SemanticJoinPath, ...],
    derived_metrics: tuple[SemanticDerivedMetric, ...],
) -> CompiledSemanticPlan:
    warnings: list[str] = []
    if measure is None:
        warnings.append("no_measure_resolved")
        if accepted_entities:
            base_entity = accepted_entities[0].asset.name
            base_table = entity_to_table.get(base_entity, accepted_entities[0].asset.name)
        else:
            base_entity = rules.plan_fallbacks.base_entity
            base_table = rules.plan_fallbacks.base_table
        grain = f"{base_table}.id"
    else:
        base_entity, grain = select_base_entity(measure, accepted_entities)
        base_table = measure.source_table

    warnings.extend(_detect_unmapped_status_qualifiers(query_norm, measure, rules))
    group_by = extract_group_by(
        query_norm,
        entity_assets=accepted_entities,
        dimension_assets=accepted_dimensions,
        filter_assets=accepted_filters,
        synonym_assets=accepted_synonyms,
        base_entity=base_entity,
        base_table=base_table,
        rules=rules,
        query_form_match=query_form_match,
    )
    ranking = _resolve_ranking(query_form_match, rules)
    if ranking is not None and not group_by:
        warnings.append("ranking_dimension_unresolved")

    time_filter, time_warnings = extract_time_filter(
        query_norm,
        entity_assets=accepted_entities,
        dimension_assets=accepted_dimensions,
        base_entity=base_entity,
        base_table=base_table,
        rules=rules,
        dialect=dialect,
    )
    warnings.extend(time_warnings)

    post_aggregation, post_aggregation_warning = build_post_aggregation(
        intent,
        query_norm,
        has_group_by=bool(group_by),
        default_function=default_post_aggregation,
        rules=rules,
    )
    if post_aggregation_warning is not None:
        warnings.append(post_aggregation_warning)

    join_path: list[str] = []
    join_path_hints: list[str] = []
    for group_field in group_by:
        if "." not in group_field:
            continue
        target_table = group_field.split(".", 1)[0]
        if target_table == base_table:
            continue

        matching_join_path = _find_matching_join_path(
            semantic_join_paths,
            base_table,
            target_table,
            entity_to_table,
        )
        if matching_join_path is not None:
            oriented_edges = _orient_semantic_join_path(matching_join_path, base_table)
            if oriented_edges:
                join_path = _merge_join_path(join_path, oriented_edges)
                if matching_join_path.name not in join_path_hints:
                    join_path_hints.append(matching_join_path.name)
                continue

        candidate_join_path = shortest_join_path(
            join_graph,
            base_table,
            target_table,
            forbidden_transit=lookup_tables,
        )
        if not candidate_join_path:
            fallback_path = shortest_join_path(join_graph, base_table, target_table)
            if fallback_path:
                transit_tables = _path_transit_tables(fallback_path, base_table, target_table)
                lookup_transit = transit_tables & lookup_tables
                if lookup_transit:
                    for lookup_name in sorted(lookup_transit):
                        warnings.append(f"join_path_rejected_via_lookup:{lookup_name}:{base_table}_to_{target_table}")
                    warnings.append(f"no_valid_join_path_from_{base_table}_to_{target_table}")
                    continue
                candidate_join_path = fallback_path
            else:
                warnings.append(f"no_join_path_from_{base_table}_to_{target_table}")
                continue
        join_path = _merge_join_path(join_path, candidate_join_path)

    matched_derived_metric = _find_matching_derived_metric(
        derived_metrics,
        intent=intent,
        measure=measure,
        group_by=group_by,
        post_aggregation=post_aggregation,
    )

    if matched_derived_metric is not None:
        if post_aggregation is None and matched_derived_metric.post_aggregation:
            post_aggregation = PlanPostAggregation(
                function=matched_derived_metric.post_aggregation,  # type: ignore[arg-type]
                over=rules.post_aggregation.over,
            )
        if measure is not None:
            normalized_base_measure = _normalize_formula(matched_derived_metric.base_measure)
            if normalized_base_measure not in {
                _normalize_formula(measure.name),
                _normalize_formula(measure.formula),
            }:
                measure = replace(measure, formula=matched_derived_metric.base_measure)

    join_path, measure_join_warnings = _augment_join_path_with_measure_support_tables(
        join_path=join_path,
        measure=measure,
        base_table=base_table,
        join_graph=join_graph,
        lookup_tables=lookup_tables,
    )
    warnings.extend(measure_join_warnings)

    time_filter_tables: list[str] = []
    if time_filter is not None and "." in time_filter.field:
        time_filter_tables.append(time_filter.field.split(".", 1)[0])
    join_path, time_filter_join_warnings = _augment_join_path_with_required_tables(
        join_path=join_path,
        target_tables=time_filter_tables,
        base_table=base_table,
        join_graph=join_graph,
        lookup_tables=lookup_tables,
        reason="time_filter",
    )
    warnings.extend(time_filter_join_warnings)

    derived_metric_ref = matched_derived_metric.name if matched_derived_metric is not None else None
    if matched_derived_metric is not None and matched_derived_metric.join_path_hint is not None:
        if matched_derived_metric.join_path_hint not in join_path_hints:
            join_path_hints.insert(0, matched_derived_metric.join_path_hint)

    base_group_by: list[str] = []
    intermediate_alias: str | None = None
    if matched_derived_metric is not None and matched_derived_metric.base_group_by:
        base_group_by = list(matched_derived_metric.base_group_by)
    elif intent == "post_aggregated_metric" and group_by:
        base_group_by = list(group_by)

    if matched_derived_metric is not None:
        intermediate_alias = matched_derived_metric.name
    elif intent == "post_aggregated_metric" and measure is not None:
        intermediate_alias = f"{measure.name}_por_nivel_base"

    final_group_by = [] if intent == "post_aggregated_metric" else list(group_by)
    selected_filters = resolve_semantic_filter_selections(
        query,
        iter_semantic_filter_payloads_from_assets(accepted_filters),
        grouped_fields=[*group_by, *base_group_by, *final_group_by],
    )

    population_scope = None
    if intent == "post_aggregated_metric" and group_by:
        population_scope = rules.plan_fallbacks.population_scope_default
        warnings.append(rules.plan_fallbacks.population_scope_warning)

    join_path_hint_value = join_path_hints[0] if join_path_hints else None

    selected_filter_tables = [filter_obj.field.split(".", 1)[0] for filter_obj in selected_filters if "." in filter_obj.field]
    join_path, selected_filter_join_warnings = _augment_join_path_with_required_tables(
        join_path=join_path,
        target_tables=selected_filter_tables,
        base_table=base_table,
        join_graph=join_graph,
        lookup_tables=lookup_tables,
        reason="selected_filter",
    )
    warnings.extend(selected_filter_join_warnings)

    deduped_warnings = _dedupe_preserve(warnings)
    issues = _build_deterministic_issues(base_entity=base_entity, warnings=deduped_warnings)

    compiled_candidate = build_compiled_semantic_plan(
        query=query,
        semantic_model=_select_semantic_model(accepted_models, base_table, group_by, rules),
        intent=intent,
        base_entity=base_entity,
        grain=grain,
        measure=measure,
        group_by=group_by,
        final_group_by=final_group_by,
        selected_filters=selected_filters,
        time_filter=time_filter,
        ranking=ranking,
        post_aggregation=post_aggregation,
        join_path=join_path,
        required_tables=_extract_required_tables(base_table, join_path, group_by),
        warnings=deduped_warnings,
        confidence=0.0,
        join_path_hint=join_path_hint_value,
        derived_metric_ref=derived_metric_ref,
        population_scope=population_scope,
        base_group_by=base_group_by,
        intermediate_alias=intermediate_alias,
        query_form_name=(query_form_match.name if query_form_match is not None else None),
        issues=issues,
    )
    return replace(compiled_candidate, confidence=_estimate_confidence(compiled_candidate, rules))


def _normalize_formula(formula: str) -> str:
    """Normaliza formulas para comparacion tolerante a espacios y mayusculas."""
    return re.sub(r"\s+", "", formula.strip().lower())


def _find_matching_join_path(
    join_paths: tuple[SemanticJoinPath, ...],
    base_table: str,
    target_table: str,
    entity_to_table: Mapping[str, str],
) -> SemanticJoinPath | None:
    """Selecciona un semantic_join_path canonico que conecte dos tablas.

    La comparacion se hace a nivel de entidad resuelta a su source_table, y
    tolera ambos sentidos del path: si la ruta declarada va en el sentido
    inverso al del plan, igual se considera aplicable.
    Devuelve el primer match para conservar el orden del YAML como fuente de
    verdad.
    """
    if not join_paths:
        return None

    def _resolve_entity_table(entity_name: str) -> str:
        table_name = entity_to_table.get(entity_name)
        return table_name if isinstance(table_name, str) and table_name else entity_name

    endpoints = {base_table, target_table}
    for join_path in join_paths:
        from_table = _resolve_entity_table(join_path.from_entity)
        to_table = _resolve_entity_table(join_path.to_entity)
        if from_table == to_table:
            continue
        if {from_table, to_table} == endpoints:
            return join_path
    return None


def _reverse_join_edge(edge: str) -> str:
    if "=" not in edge:
        return edge
    left, right = (side.strip() for side in edge.split("=", 1))
    return f"{right} = {left}"


def _orient_semantic_join_path(join_path: SemanticJoinPath, base_table: str) -> list[str]:
    """Orienta los edges de una ruta canonica para que arranquen en base_table.

    El YAML declara rutas siempre desde from_entity hacia to_entity; el plan
    puede ir en sentido contrario, por lo que se invierte tanto el orden de
    los edges como los dos lados de cada igualdad.
    """
    edges = list(join_path.path)
    if not edges:
        return edges
    first_edge = edges[0]
    if "=" in first_edge:
        left_side = first_edge.split("=", 1)[0].strip()
        if "." in left_side and left_side.split(".", 1)[0] == base_table:
            return edges
    return [_reverse_join_edge(edge) for edge in reversed(edges)]


def _find_matching_derived_metric(
    derived_metrics: tuple[SemanticDerivedMetric, ...],
    *,
    intent: PlanIntent,
    measure: PlanMeasure | None,
    group_by: list[str],
    post_aggregation: PlanPostAggregation | None,
) -> SemanticDerivedMetric | None:
    """Empareja el plan actual con una metrica derivada registrada en el YAML.

    Un match por ``base_measure`` requiere que la expresion coincida al
    normalizar espacios y mayusculas, que ``base_group_by`` coincida
    exactamente con ``group_by`` y que la funcion de ``post_aggregation`` sea
    consistente. Cuando la metrica seleccionada y la derivada comparten nombre,
    el YAML actua como contrato explicito de forma SQL y no depende de que la
    pregunta exprese una post-agregacion textual.
    """
    if measure is None or not derived_metrics:
        return None

    normalized_formula = _normalize_formula(measure.formula)
    normalized_measure_name = _normalize_formula(measure.name)
    group_by_tuple = tuple(group_by)
    post_agg_function = post_aggregation.function if post_aggregation is not None else None

    for derived_metric in derived_metrics:
        normalized_derived_name = _normalize_formula(derived_metric.name)
        if normalized_derived_name == normalized_measure_name:
            if post_agg_function is not None and derived_metric.post_aggregation:
                if derived_metric.post_aggregation.lower() != post_agg_function.lower():
                    continue
            return derived_metric

        if intent != "post_aggregated_metric":
            continue

        normalized_base_measure = _normalize_formula(derived_metric.base_measure)
        if normalized_base_measure not in {normalized_formula, normalized_measure_name}:
            continue
        if tuple(derived_metric.base_group_by) != group_by_tuple:
            continue
        if post_agg_function is not None and derived_metric.post_aggregation:
            if derived_metric.post_aggregation.lower() != post_agg_function.lower():
                continue
        return derived_metric
    return None


def compile_semantic_plan(
    plan: SemanticPlan,
    query: str,
    *,
    config: SemanticResolverConfig | None = None,
    pruned_schema: Mapping[str, object] | None = None,
    dialect: ResolverDialect | None = None,
) -> CompiledSemanticPlan:
    """Compila el ranking de activos en un plan final con estructura obligatoria."""

    compiler_rules_path = str(config.compiler_rules_path) if config is not None else str(resolve_compiler_rules_path())
    semantic_rules_path = str(config.rules_path) if config is not None else resolve_rules_path()
    rules = load_compiler_rules(compiler_rules_path, semantic_rules_path)

    # Si el caller no inyecta un dialecto explicito, se materializa el declarado
    # en config para mantener los planes especializados sin acoplar el codigo a
    # un motor concreto. Si tampoco hay config, el plan queda agnostico
    # (resolved_expressions vacio, value canonico intacto) y el solver decide.
    if dialect is None and config is not None and getattr(config, "dialect", None):
        dialect = get_resolver_dialect(config.dialect)

    query_norm = _normalize_text(query)
    query_form_match = match_query_forms(query_norm, rules)
    intent = detect_intent(query_norm, rules)
    accepted_entities = _get_assets_by_kind(plan, "semantic_entities")
    accepted_metrics = _get_assets_by_kind(plan, "semantic_metrics")
    accepted_dimensions = _get_assets_by_kind(plan, "semantic_dimensions")
    accepted_filters = _get_assets_by_kind(plan, "semantic_filters")
    accepted_relationships = _get_assets_by_kind(plan, "semantic_relationships")
    accepted_synonyms = _get_assets_by_kind(plan, "semantic_synonyms")
    accepted_models = _get_assets_by_kind(plan, "semantic_models")

    entity_to_table = _build_entity_to_table(accepted_entities)
    hinted_entities = {
        str(entity_name) for entity_name in plan.diagnostics.get("synonym_entities_detected", []) if isinstance(entity_name, str)
    }
    default_post_aggregation = "avg"
    if config is not None:
        default_post_aggregation = config.default_post_aggregation_function

    join_graph = build_join_graph(accepted_relationships, pruned_schema)
    lookup_tables = _detect_lookup_tables(pruned_schema, rules)
    semantic_join_paths = load_semantic_join_paths(semantic_rules_path)
    derived_metrics = load_semantic_derived_metrics(semantic_rules_path)

    measure_candidates, metric_score_trace = rank_measure_candidates(
        accepted_metrics,
        query_norm=query_norm,
        entity_to_table=entity_to_table,
        intent=intent,
        hinted_entities=hinted_entities,
        rules=rules,
        query_form_match=query_form_match,
    )

    candidate_limit = rules.selection_tuning.measure_candidate_limit
    compiled_candidates: list[tuple[CompiledSemanticPlan, float]] = []
    for measure_candidate in measure_candidates[:candidate_limit]:
        compiled_candidates.append(
            (
                _build_compiled_candidate(
                    query=query,
                    query_norm=query_norm,
                    intent=intent,
                    measure=measure_candidate.measure,
                    query_form_match=query_form_match,
                    accepted_entities=accepted_entities,
                    accepted_dimensions=accepted_dimensions,
                    accepted_filters=accepted_filters,
                    accepted_synonyms=accepted_synonyms,
                    accepted_models=accepted_models,
                    entity_to_table=entity_to_table,
                    pruned_schema=pruned_schema,
                    rules=rules,
                    dialect=dialect,
                    default_post_aggregation=default_post_aggregation,
                    join_graph=join_graph,
                    lookup_tables=lookup_tables,
                    semantic_join_paths=semantic_join_paths,
                    derived_metrics=derived_metrics,
                ),
                measure_candidate.score,
            )
        )

    if not compiled_candidates:
        compiled_candidates.append(
            (
                _build_compiled_candidate(
                    query=query,
                    query_norm=query_norm,
                    intent=intent,
                    measure=None,
                    query_form_match=query_form_match,
                    accepted_entities=accepted_entities,
                    accepted_dimensions=accepted_dimensions,
                    accepted_filters=accepted_filters,
                    accepted_synonyms=accepted_synonyms,
                    accepted_models=accepted_models,
                    entity_to_table=entity_to_table,
                    pruned_schema=pruned_schema,
                    rules=rules,
                    dialect=dialect,
                    default_post_aggregation=default_post_aggregation,
                    join_graph=join_graph,
                    lookup_tables=lookup_tables,
                    semantic_join_paths=semantic_join_paths,
                    derived_metrics=derived_metrics,
                ),
                float("-inf"),
            )
        )

    selected_plan, selected_score = compiled_candidates[0]
    selected_measure_name = selected_plan.measure.name if selected_plan.measure is not None else None
    candidate_plan_set = CandidatePlanSet(
        selected_index=0,
        selection_rationale=_build_selection_rationale(
            selected_plan=selected_plan,
            selected_score=selected_score,
            query_form_match=query_form_match,
        ),
        candidates=[
            _candidate_from_plan(candidate_plan, score=score, selected_measure_name=selected_measure_name)
            for candidate_plan, score in compiled_candidates
        ],
    )
    return replace(
        selected_plan,
        metric_score_trace=_apply_metric_trace_selection(metric_score_trace, selected_measure_name),
        candidate_plan_set=candidate_plan_set,
    )
