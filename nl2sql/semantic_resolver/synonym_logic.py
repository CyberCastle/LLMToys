#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass

from nl2sql.utils.normalization import normalize_text_for_matching
from nl2sql.utils.spanish_morphology import pluralize_token

from .assets import SemanticAsset
from .compatibility import build_direct_table_references
from .config import resolve_compiler_rules_path
from .rules_loader import SynonymScoringRules, load_compiler_rules


@dataclass(frozen=True)
class SynonymResolution:
    """Resultado de resolver sinonimos contra la query original."""

    retrieval_query: str
    matched_entities: tuple[str, ...]
    matched_tables: tuple[str, ...]
    matched_models: tuple[str, ...]
    matched_aliases_by_entity: dict[str, tuple[str, ...]]
    entity_confidence: dict[str, float]


def _build_aliases(asset: SemanticAsset, plural_suffix_rules: tuple[tuple[str, str, int], ...]) -> set[str]:
    entity = str(asset.payload.get("entity") or asset.name).strip()
    aliases: set[str] = set()

    if entity:
        aliases.add(entity)
        aliases.add(entity.replace("_", " "))
        entity_tokens = [token for token in entity.split("_") if token]
        primary_tokens = entity_tokens[:1]
        for token in primary_tokens:
            aliases.add(token)
            aliases.add(pluralize_token(token, plural_suffix_rules))

    raw_synonyms = asset.payload.get("synonyms")
    if isinstance(raw_synonyms, list):
        for raw_synonym in raw_synonyms:
            synonym = str(raw_synonym).strip()
            if not synonym:
                continue
            aliases.add(synonym)
            aliases.add(synonym.replace("_", " "))
            normalized_synonym = normalize_text_for_matching(synonym, keep_underscore=False)
            if normalized_synonym and " " not in normalized_synonym:
                aliases.add(pluralize_token(normalized_synonym, plural_suffix_rules))

    return {alias for alias in aliases if alias.strip()}


def _resolve_synonym_scoring_rules(scoring_rules: SynonymScoringRules | None) -> SynonymScoringRules:
    if scoring_rules is not None:
        return scoring_rules
    return load_compiler_rules(str(resolve_compiler_rules_path())).synonym_scoring


def _match_alias_strength(
    query_text: str,
    query_tokens: tuple[str, ...],
    alias: str,
    scoring_rules: SynonymScoringRules,
) -> float:
    normalized_alias = normalize_text_for_matching(alias, keep_underscore=False)
    if not normalized_alias:
        return 0.0

    query_padded = f" {query_text} "
    alias_padded = f" {normalized_alias} "
    if alias_padded in query_padded:
        return scoring_rules.exact_phrase_strength

    alias_tokens = tuple(token for token in normalized_alias.split() if token)
    if not alias_tokens:
        return 0.0

    if len(alias_tokens) == 1:
        alias_token = alias_tokens[0]
        if alias_token in query_tokens:
            return scoring_rules.single_token_exact_strength
        if len(alias_token) >= scoring_rules.prefix_min_token_length and any(
            len(token) >= scoring_rules.prefix_min_token_length and (token.startswith(alias_token) or alias_token.startswith(token))
            for token in query_tokens
        ):
            return scoring_rules.single_token_prefix_strength
        return 0.0

    matched_tokens = 0
    for alias_token in alias_tokens:
        if any(
            token == alias_token
            or (
                len(alias_token) >= scoring_rules.prefix_min_token_length
                and len(token) >= scoring_rules.prefix_min_token_length
                and (token.startswith(alias_token) or alias_token.startswith(token))
            )
            for token in query_tokens
        ):
            matched_tokens += 1

    if matched_tokens == len(alias_tokens):
        return scoring_rules.multi_token_full_strength
    if len(alias_tokens) >= scoring_rules.multi_token_partial_min_tokens and matched_tokens >= len(alias_tokens) - 1:
        return scoring_rules.multi_token_partial_strength
    return 0.0


def resolve_query_synonyms(
    query: str,
    assets: list[SemanticAsset],
    *,
    entity_to_table: dict[str, str],
    model_to_tables: dict[str, set[str]],
    max_entities: int,
    enable_query_expansion: bool,
    scoring_rules: SynonymScoringRules | None = None,
) -> SynonymResolution:
    """Detecta entidades sugeridas por sinonimos y construye una query expandida."""

    normalized_query = normalize_text_for_matching(query, keep_underscore=False)
    query_tokens = tuple(token for token in normalized_query.split() if token)
    active_scoring_rules = _resolve_synonym_scoring_rules(scoring_rules)
    entity_confidence: dict[str, float] = {}
    entity_aliases: dict[str, set[str]] = {}

    for asset in assets:
        if asset.kind != "semantic_synonyms":
            continue

        entity = str(asset.payload.get("entity") or asset.name).strip()
        if not entity:
            continue

        for alias in _build_aliases(asset, active_scoring_rules.plural_suffix_rules):
            match_strength = _match_alias_strength(normalized_query, query_tokens, alias, active_scoring_rules)
            if match_strength <= 0.0:
                continue

            entity_confidence[entity] = max(entity_confidence.get(entity, 0.0), match_strength)
            entity_aliases.setdefault(entity, set()).add(alias)

    ranked_entities = sorted(
        entity_confidence,
        key=lambda entity: (entity_confidence[entity], entity),
        reverse=True,
    )
    ranked_entities = ranked_entities[: max(0, max_entities)]

    matched_tables = tuple(sorted({entity_to_table[entity] for entity in ranked_entities if entity in entity_to_table}))
    matched_models = tuple(sorted(model_name for model_name, tables in model_to_tables.items() if set(matched_tables) & tables))

    retrieval_query = query
    if enable_query_expansion and ranked_entities:
        retrieval_query = (
            f"{query}\n\n"
            f"Semantic entities hinted by query: {', '.join(ranked_entities)}\n"
            f"Semantic models hinted by query: {', '.join(matched_models) or 'none'}"
        )

    return SynonymResolution(
        retrieval_query=retrieval_query,
        matched_entities=tuple(ranked_entities),
        matched_tables=matched_tables,
        matched_models=matched_models,
        matched_aliases_by_entity={entity: tuple(sorted(entity_aliases.get(entity, set()))) for entity in ranked_entities},
        entity_confidence={entity: entity_confidence[entity] for entity in ranked_entities},
    )


def compute_synonym_boost(
    asset: SemanticAsset,
    resolution: SynonymResolution,
    *,
    entity_to_table: dict[str, str],
    model_to_tables: dict[str, set[str]],
    direct_boost: float,
    related_boost: float,
    scoring_rules: SynonymScoringRules | None = None,
) -> float:
    """Calcula un refuerzo semantico para activos alineados con sinonimos detectados."""

    if not resolution.matched_entities:
        return 0.0

    active_scoring_rules = _resolve_synonym_scoring_rules(scoring_rules)
    matched_entity_set = set(resolution.matched_entities)
    matched_table_set = set(resolution.matched_tables)
    matched_model_set = set(resolution.matched_models)
    boost = 0.0

    payload_entity = asset.payload.get("entity")
    asset_entity = str(payload_entity).strip() if isinstance(payload_entity, str) else ""

    if asset.kind == "semantic_synonyms" and asset_entity in matched_entity_set:
        boost += direct_boost * resolution.entity_confidence.get(asset_entity, 1.0)

    if asset.kind.startswith("semantic_entities") and asset.name in matched_entity_set:
        boost += direct_boost * active_scoring_rules.entity_direct_confidence_multiplier * resolution.entity_confidence.get(asset.name, 1.0)

    if asset_entity in matched_entity_set:
        boost += related_boost * resolution.entity_confidence.get(asset_entity, 1.0)

    model_name = asset.payload.get("model")
    if isinstance(model_name, str) and model_name in matched_model_set:
        boost += related_boost * active_scoring_rules.model_related_multiplier

    referenced_tables = build_direct_table_references(
        asset,
        entity_to_table=entity_to_table,
        model_to_tables=model_to_tables,
    )
    if referenced_tables and matched_table_set:
        overlap_ratio = len(referenced_tables & matched_table_set) / len(referenced_tables)
        boost += related_boost * overlap_ratio

    return min(boost, direct_boost + (related_boost * active_scoring_rules.max_related_boost_multiplier))
