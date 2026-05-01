#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Annotated, Any

from pydantic import (
    BeforeValidator,
    Field,
    ValidationInfo,
    field_validator,
    model_validator,
)

from nl2sql.utils.decision_models import StrictModel
from nl2sql.utils.normalization import normalize_text_for_matching
from nl2sql.utils.semantic_contract import SemanticContract

PatternType = re.Pattern[str]


def _require_text(value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Debe ser un string no vacio.")
    return value.strip()


def _normalize_scalar_text(value: object) -> str:
    normalized = str(value).strip()
    if not normalized:
        raise ValueError("Debe ser un string no vacio.")
    return normalized


def _require_int(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("Debe ser un entero.")
    return value


def _require_non_negative_int(value: object) -> int:
    normalized = _require_int(value)
    if normalized < 0:
        raise ValueError("Debe ser un entero no negativo.")
    return normalized


def _require_float(value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("Debe ser un valor numerico.")
    return float(value)


def _require_bool(value: object) -> bool:
    if not isinstance(value, bool):
        raise ValueError("Debe ser un booleano.")
    return value


NonEmptyStr = Annotated[str, BeforeValidator(_require_text)]
StrictIntValue = Annotated[int, BeforeValidator(_require_int)]
NonNegativeInt = Annotated[int, BeforeValidator(_require_non_negative_int)]
NumericValue = Annotated[float, BeforeValidator(_require_float)]
StrictBoolValue = Annotated[bool, BeforeValidator(_require_bool)]


class SemanticPruneEmbeddingPrompt(StrictModel):
    """Prompts del embedding del semantic prune."""

    task_instruction: NonEmptyStr
    query_template: NonEmptyStr


class SemanticPruneListwisePrompt(StrictModel):
    """Prompts del rerank listwise del semantic prune."""

    task_instruction: NonEmptyStr
    prompt_template: NonEmptyStr


class SemanticPrunePromptRules(StrictModel):
    """Bloque de prompts del semantic prune."""

    embedding: SemanticPruneEmbeddingPrompt
    listwise_rerank: SemanticPruneListwisePrompt


class TableSignalHeuristics(StrictModel):
    """Heuristicas de scoring de tablas en semantic prune."""

    lexical_bonus_cap: NumericValue
    lexical_overlap_weight: NumericValue
    richness_bonus_cap: NumericValue
    richness_extra_column_weight: NumericValue
    temporal_bonus_base: NumericValue
    temporal_bonus_per_column: NumericValue
    temporal_bonus_cap: NumericValue
    grouping_bonus_base: NumericValue
    grouping_bonus_per_fk: NumericValue
    grouping_bonus_cap: NumericValue
    aggregation_structural_bonus: NumericValue
    aggregation_numeric_fk_bonus: NumericValue
    lookup_penalty: NumericValue
    output_min: NumericValue
    output_max: NumericValue


class ColumnSignalHeuristics(StrictModel):
    """Heuristicas de scoring de columnas en semantic prune."""

    lexical_bonus_cap: NumericValue
    lexical_overlap_weight: NumericValue
    temporal_type_bonus: NumericValue
    temporal_name_bonus: NumericValue
    output_min: NumericValue
    output_max: NumericValue


class SeedSelectionHeuristics(StrictModel):
    """Umbrales de seleccion de tablas ancla."""

    anchor_min_floor: NumericValue
    anchor_min_margin: NumericValue


class RelationshipExpansionHeuristics(StrictModel):
    """Reglas de expansion relacional alrededor de las anclas."""

    outbound_hops: StrictIntValue
    inbound_hops: StrictIntValue
    max_neighbors_per_table: StrictIntValue
    outbound_min_score: NumericValue
    inbound_min_score: NumericValue
    bridge_max_hops: StrictIntValue
    bridge_table_min_score: NumericValue
    unused_foreign_key_score_margin: NumericValue


class FkPathHeuristics(StrictModel):
    """Reglas del bridge duro por camino FK."""

    enabled: StrictBoolValue
    max_hops: StrictIntValue
    anchor_min_overlap: StrictIntValue
    max_anchors_per_role: StrictIntValue


class StructureProfileHeuristics(StrictModel):
    """Heuristicas estructurales de lookup y densidad semantica."""

    lookup_column_count_max: StrictIntValue
    lookup_max_numeric_columns: StrictIntValue
    lookup_min_descriptor_columns: StrictIntValue
    lookup_descriptor_margin: StrictIntValue
    richness_extra_columns_start: StrictIntValue
    temporal_bonus_column_cap: StrictIntValue
    grouping_bonus_fk_cap: StrictIntValue
    aggregation_min_column_count: StrictIntValue
    aggregation_min_numeric_columns: StrictIntValue
    derived_table_suffixes: tuple[str, ...]

    @field_validator("derived_table_suffixes", mode="before")
    @classmethod
    def _normalize_derived_suffixes(cls, value: object) -> tuple[str, ...]:
        return _normalize_text_tuple(value, allow_empty=False)


class RelationshipScoringHeuristics(StrictModel):
    """Pesos del score semantico de joins."""

    primary_evidence_weight: NumericValue
    supporting_evidence_weight: NumericValue
    lookup_edge_penalty: NumericValue


class HeuristicRules(StrictModel):
    """Conjunto de heuristicas declarativas del semantic prune."""

    table_selection_reason_priority: dict[str, StrictIntValue]
    table_signal: TableSignalHeuristics
    column_signal: ColumnSignalHeuristics
    seed_selection: SeedSelectionHeuristics
    relationship_expansion: RelationshipExpansionHeuristics
    fk_path: FkPathHeuristics
    structure_profile: StructureProfileHeuristics
    relationship_scoring: RelationshipScoringHeuristics

    @field_validator("table_selection_reason_priority")
    @classmethod
    def _validate_priority_map(cls, value: dict[str, int]) -> dict[str, int]:
        if not value:
            raise ValueError("table_selection_reason_priority no puede quedar vacio.")
        normalized = {str(key).strip(): raw_value for key, raw_value in value.items() if str(key).strip()}
        if not normalized:
            raise ValueError("table_selection_reason_priority no puede quedar vacio.")
        return normalized


class QuerySignalRules(StrictModel):
    """Reglas lexicas precompiladas para interpretar la pregunta."""

    groupby_dimension_patterns: tuple[PatternType, ...]
    query_term_stopwords: frozenset[str]
    query_intent_noise_terms: frozenset[str]
    query_temporal_terms: frozenset[str]
    query_aggregation_terms: frozenset[str]
    temporal_column_name_terms: frozenset[str]
    lookup_descriptor_terms: frozenset[str]
    documental_terms: frozenset[str] = frozenset()
    query_enrichment_temporal_hints: tuple[tuple[PatternType, str], ...] = ()
    query_enrichment_aggregation_hints: tuple[tuple[str, str], ...] = ()
    singular_suffix_rules: tuple[tuple[str, str, int], ...] = ()

    @field_validator("groupby_dimension_patterns", mode="before")
    @classmethod
    def _compile_groupby_patterns(cls, value: object) -> tuple[PatternType, ...]:
        return _compile_pattern_tuple(value, allow_empty=False)

    @field_validator(
        "query_term_stopwords",
        "query_intent_noise_terms",
        "query_temporal_terms",
        "query_aggregation_terms",
        "temporal_column_name_terms",
        "lookup_descriptor_terms",
        mode="before",
    )
    @classmethod
    def _normalize_required_token_sets(cls, value: object) -> frozenset[str]:
        return _normalize_token_set(value, allow_empty=False)

    @field_validator("documental_terms", mode="before")
    @classmethod
    def _normalize_optional_token_sets(cls, value: object) -> frozenset[str]:
        return _normalize_token_set(value, allow_empty=True)

    @field_validator("query_enrichment_temporal_hints", mode="before")
    @classmethod
    def _normalize_temporal_hints(cls, value: object) -> tuple[tuple[PatternType, str], ...]:
        if value in (None, {}):
            return ()
        if not isinstance(value, dict):
            raise ValueError("query_enrichment_temporal_hints debe ser un mapping.")
        normalized: list[tuple[PatternType, str]] = []
        for raw_pattern, raw_hint in value.items():
            pattern = _require_text(raw_pattern)
            hint = _require_text(raw_hint)
            normalized.append((re.compile(pattern), hint))
        if not normalized:
            raise ValueError("query_enrichment_temporal_hints no puede quedar vacio cuando se declara.")
        return tuple(normalized)

    @field_validator("query_enrichment_aggregation_hints", mode="before")
    @classmethod
    def _normalize_aggregation_hints(cls, value: object) -> tuple[tuple[str, str], ...]:
        if value in (None, {}):
            return ()
        if not isinstance(value, dict):
            raise ValueError("query_enrichment_aggregation_hints debe ser un mapping.")
        normalized: list[tuple[str, str]] = []
        for raw_key, raw_hint in value.items():
            key = _require_text(raw_key).lower()
            hint = _require_text(raw_hint)
            normalized.append((key, hint))
        if not normalized:
            raise ValueError("query_enrichment_aggregation_hints no puede quedar vacio cuando se declara.")
        return tuple(normalized)

    @field_validator("singular_suffix_rules", mode="before")
    @classmethod
    def _normalize_singular_suffix_rules(cls, value: object) -> tuple[tuple[str, str, int], ...]:
        return _normalize_suffix_rewrite_rules(value, allow_empty=True)


class SemanticPruneTextFormattingRules(StrictModel):
    """Limites declarativos de serializacion compacta para prune."""

    table_column_summary_limit: NonNegativeInt
    table_foreign_key_limit: NonNegativeInt
    column_relation_summary_limit: NonNegativeInt
    eos_token_budget_cushion: NonNegativeInt


class SemanticPruneRuntimeTuning(StrictModel):
    """Knobs operativos del semantic prune."""

    top_k_matches: StrictIntValue
    top_k_tables: StrictIntValue
    top_k_columns_per_table: StrictIntValue
    min_score: NumericValue
    table_score_doc_weight: NumericValue
    table_score_column_topn: StrictIntValue
    mmr_enabled: StrictBoolValue
    mmr_lambda: NumericValue
    mmr_candidate_pool_size: StrictIntValue
    adaptive_threshold_k_sigma: NumericValue
    adaptive_threshold_k_sigma_columns: NumericValue
    table_listwise_input_docs: StrictIntValue
    column_listwise_input_docs: StrictIntValue
    max_tokens_per_doc: StrictIntValue
    listwise_min_tokens_per_doc: StrictIntValue
    listwise_token_step: StrictIntValue
    listwise_score_alpha: NumericValue
    preview_chars: StrictIntValue


class SemanticPruneSettings(StrictModel):
    """Configuracion completa y validada del semantic prune."""

    prompts: SemanticPrunePromptRules
    heuristic_rules: HeuristicRules
    query_signal_rules: QuerySignalRules
    text_formatting: SemanticPruneTextFormattingRules
    runtime_tuning: SemanticPruneRuntimeTuning


class SemanticResolverEmbeddingPrompt(StrictModel):
    """Prompts del embedding del resolver semantico."""

    query_instruction: NonEmptyStr
    query_template: NonEmptyStr


class SemanticResolverRerankPrompt(StrictModel):
    """Prompts del reranker binario del resolver."""

    instruction: NonEmptyStr
    system_prompt: NonEmptyStr
    user_prompt_template: NonEmptyStr


class SemanticResolverVerifierPrompt(StrictModel):
    """Prompts del verificador semantico local."""

    system_prompt: NonEmptyStr
    user_prompt_template: NonEmptyStr


class SemanticResolverPromptRules(StrictModel):
    """Bloque de prompts del semantic resolver."""

    embedding: SemanticResolverEmbeddingPrompt
    rerank: SemanticResolverRerankPrompt
    verifier: SemanticResolverVerifierPrompt


class SemanticResolverRuntimeTuning(StrictModel):
    """Tuning operativo del semantic resolver."""

    top_k_retrieval: StrictIntValue
    top_k_rerank: StrictIntValue
    min_embedding_score: NumericValue
    min_rerank_score: NumericValue
    compatibility_min_score: NumericValue
    synonym_query_expansion_max_entities: StrictIntValue
    synonym_direct_boost: NumericValue
    synonym_related_boost: NumericValue
    rerank_batch_size: StrictIntValue
    rerank_max_document_chars: StrictIntValue
    rerank_logprobs: StrictIntValue
    rerank_prompt_token_margin: StrictIntValue
    rerank_positive_token_candidates: tuple[str, ...]
    rerank_negative_token_candidates: tuple[str, ...]
    rerank_suffix: NonEmptyStr
    verifier_few_shot_limit: StrictIntValue
    max_semantic_repair_attempts: StrictIntValue
    default_post_aggregation_function: str
    per_kind_caps: dict[str, StrictIntValue]

    @field_validator(
        "rerank_positive_token_candidates",
        "rerank_negative_token_candidates",
        mode="before",
    )
    @classmethod
    def _normalize_text_tuples(cls, value: object, info: ValidationInfo) -> tuple[str, ...]:
        return _normalize_text_tuple(
            value,
            allow_empty=False,
            true_token="yes",
            false_token="no",
            ignore_non_strings=True,
        )

    @field_validator("default_post_aggregation_function")
    @classmethod
    def _validate_post_agg_function(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in {"avg", "ratio", "sum"}:
            raise ValueError("default_post_aggregation_function debe ser avg, ratio o sum.")
        return normalized

    @field_validator("per_kind_caps")
    @classmethod
    def _validate_per_kind_caps(cls, value: dict[str, int]) -> dict[str, int]:
        if not value:
            raise ValueError("per_kind_caps no puede quedar vacio.")
        normalized = {str(key).strip(): raw_value for key, raw_value in value.items() if str(key).strip()}
        if not normalized:
            raise ValueError("per_kind_caps no puede quedar vacio.")
        return normalized


class TimePatternRule(StrictModel):
    """Regla declarativa de tiempo relativo para el compilador."""

    pattern: PatternType
    operator: NonEmptyStr
    value: NonEmptyStr
    dialect_values: dict[str, str] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _extract_dialect_values(cls, value: object) -> object:
        if not isinstance(value, dict):
            return value
        payload = dict(value)
        dialect_values: dict[str, str] = {}
        for raw_key in list(payload.keys()):
            if not isinstance(raw_key, str) or raw_key == "value" or not raw_key.startswith("value_"):
                continue
            raw_dialect_value = payload.pop(raw_key)
            if isinstance(raw_dialect_value, str) and raw_dialect_value.strip():
                dialect_values[raw_key[len("value_") :]] = raw_dialect_value.strip()
        payload["dialect_values"] = dialect_values
        return payload

    @field_validator("pattern", mode="before")
    @classmethod
    def _compile_pattern(cls, value: object) -> PatternType:
        return _compile_pattern(value)

    @field_validator("dialect_values")
    @classmethod
    def _validate_dialect_values(cls, value: dict[str, str]) -> dict[str, str]:
        return {str(key).strip(): _require_text(raw_value) for key, raw_value in value.items() if str(key).strip()}


class QueryFormRule(StrictModel):
    """Forma declarativa de consulta recuperada desde semantic rules."""

    name: NonEmptyStr
    intent: NonEmptyStr
    patterns: tuple[PatternType, ...]
    output: dict[str, object] = Field(default_factory=dict)
    description: str | None = None

    @field_validator("patterns", mode="before")
    @classmethod
    def _compile_patterns(cls, value: object) -> tuple[PatternType, ...]:
        return _compile_pattern_tuple(value, allow_empty=False)

    @field_validator("description")
    @classmethod
    def _normalize_description(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None


class MetricScoringWeights(StrictModel):
    """Pesos usados al puntuar metricas candidatas."""

    compatibility_weight: NumericValue
    embedding_weight: NumericValue
    measure_overlap_weight: NumericValue
    hinted_entity_boost: NumericValue
    source_table_hint_boost: NumericValue
    count_distinct_bonus: NumericValue
    post_aggregated_count_bonus: NumericValue
    total_metric_name_bonus: NumericValue
    ratio_not_requested_penalty: NumericValue
    missing_state_filter_penalty: NumericValue
    state_filter_token_boost: NumericValue
    unexpected_case_when_penalty: NumericValue


class ExampleExpansionRules(StrictModel):
    """Pesos heredados cuando un ejemplo curado expande activos."""

    inherited_embedding_weight: NumericValue
    inherited_rerank_weight: NumericValue


class RankingDimensionPreferenceRules(StrictModel):
    """Heuristicas para elegir dimensiones legibles en rankings."""

    positive_token_weights: tuple[tuple[str, float], ...]
    negative_tokens: frozenset[str]
    negative_token_penalty: NumericValue
    string_type_hints: frozenset[str]
    string_type_bonus: NumericValue
    entity_prefix_bonus: NumericValue
    same_table_bonus: NumericValue

    @field_validator("positive_token_weights", mode="before")
    @classmethod
    def _normalize_weighted_tokens(cls, value: object) -> tuple[tuple[str, float], ...]:
        if not isinstance(value, dict) or not value:
            raise ValueError("positive_token_weights debe ser un mapping no vacio.")
        normalized: list[tuple[str, float]] = []
        for raw_token, raw_weight in value.items():
            token = _require_text(raw_token)
            normalized.append((token, _require_float(raw_weight)))
        return tuple(normalized)

    @field_validator("negative_tokens", "string_type_hints", mode="before")
    @classmethod
    def _normalize_sets(cls, value: object) -> frozenset[str]:
        return _normalize_token_set(value, allow_empty=False)


class SynonymScoringRules(StrictModel):
    """Thresholds declarativos para matching de sinonimos."""

    exact_phrase_strength: NumericValue
    single_token_exact_strength: NumericValue
    single_token_prefix_strength: NumericValue
    multi_token_full_strength: NumericValue
    multi_token_partial_strength: NumericValue
    prefix_min_token_length: StrictIntValue
    multi_token_partial_min_tokens: StrictIntValue
    entity_direct_confidence_multiplier: NumericValue
    model_related_multiplier: NumericValue
    max_related_boost_multiplier: NumericValue
    plural_suffix_rules: tuple[tuple[str, str, int], ...] = ()

    @field_validator("plural_suffix_rules", mode="before")
    @classmethod
    def _normalize_plural_suffix_rules(cls, value: object) -> tuple[tuple[str, str, int], ...]:
        return _normalize_suffix_rewrite_rules(value, allow_empty=True)


class MetricPhraseScoringRules(StrictModel):
    """Bonos por coincidencia textual de metricas."""

    exact_query_score: NumericValue
    containment_score: NumericValue
    exact_example_score: NumericValue


class EntityMatchingRules(StrictModel):
    """Pesos de matching de entidades por alias."""

    alias_exact_score: NumericValue
    alias_containment_score: NumericValue


class FieldMatchingRules(StrictModel):
    """Pesos para resolver campos desde frase libre."""

    phrase_contains_bonus: NumericValue


class TimeFieldScoringRules(StrictModel):
    """Pesos para inferir el campo temporal operativo."""

    base_score: NumericValue
    entity_match_bonus: NumericValue
    temporal_type_bonus: NumericValue
    temporal_name_bonus: NumericValue
    temporal_name_terms: frozenset[str]

    @field_validator("temporal_name_terms", mode="before")
    @classmethod
    def _normalize_temporal_terms(cls, value: object) -> frozenset[str]:
        return _normalize_token_set(value, allow_empty=False)


class SemanticModelScoringRules(StrictModel):
    """Pesos para elegir el semantic model final."""

    compatibility_weight: NumericValue
    embedding_weight: NumericValue
    grain_table_bonus: NumericValue
    core_table_bonus: NumericValue
    group_table_overlap_weight: NumericValue
    warning_penalty: NumericValue


class ConfidenceScoringRules(StrictModel):
    """Pesos de confianza final del plan compilado."""

    base_score: NumericValue
    measure_bonus: NumericValue
    group_by_bonus: NumericValue
    ranking_bonus: NumericValue
    time_filter_bonus: NumericValue
    post_aggregation_bonus: NumericValue
    semantic_model_bonus: NumericValue
    join_path_or_scalar_bonus: NumericValue
    warning_penalty_per_item: NumericValue
    warning_penalty_cap: NumericValue
    unmapped_qualifier_penalty: NumericValue


class SelectionTuningRules(StrictModel):
    """Caps declarativos del ranking y de la exploracion de candidatos."""

    ranking_default_limit: StrictIntValue
    measure_candidate_limit: StrictIntValue


class PostAggregationRules(StrictModel):
    """Reglas declarativas para materializar post-aggregations."""

    default_functions: dict[str, str]
    ratio_function: str
    over: str
    missing_group_by_warning: NonEmptyStr

    @field_validator("default_functions")
    @classmethod
    def _validate_default_functions(cls, value: dict[str, str]) -> dict[str, str]:
        if not value:
            raise ValueError("default_functions no puede quedar vacio.")
        allowed = {"avg", "count", "max", "min", "ratio", "sum"}
        normalized: dict[str, str] = {}
        for raw_key, raw_value in value.items():
            key = _require_text(raw_key)
            normalized_value = _require_text(raw_value).lower()
            if normalized_value not in allowed:
                raise ValueError("post_aggregation.default_functions contiene un valor invalido.")
            normalized[key] = normalized_value
        return normalized

    @field_validator("ratio_function")
    @classmethod
    def _validate_ratio_function(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in {"avg", "count", "max", "min", "ratio", "sum"}:
            raise ValueError("ratio_function invalido.")
        return normalized

    @field_validator("over")
    @classmethod
    def _validate_over(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in {"grouped_measure", "rows"}:
            raise ValueError("over debe ser grouped_measure o rows.")
        return normalized


class PlanFallbackRules(StrictModel):
    """Fallbacks tecnicos del compilador cuando faltan activos resolubles."""

    base_entity: NonEmptyStr
    base_table: NonEmptyStr
    population_scope_default: str
    population_scope_warning: NonEmptyStr

    @field_validator("population_scope_default")
    @classmethod
    def _validate_population_scope(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in {"active_entities_only", "all_entities_including_zero"}:
            raise ValueError("population_scope_default invalido.")
        return normalized


class AssetTextFormattingRules(StrictModel):
    """Claves declarativas que se serializan para embed y rerank."""

    body_keys: tuple[str, ...]
    scalar_keys: tuple[str, ...]
    list_keys: tuple[str, ...]

    @field_validator("body_keys", "scalar_keys", "list_keys", mode="before")
    @classmethod
    def _normalize_keys(cls, value: object) -> tuple[str, ...]:
        return _normalize_text_tuple(value, allow_empty=False)


class SemanticExampleSelectionRules(StrictModel):
    """Bonos declarativos para seleccionar ejemplos curados."""

    exact_match_bonus: NumericValue
    containment_bonus: NumericValue


class CompilerRules(StrictModel):
    """Reglas completas del compilador semantico, ya listas para usar."""

    intent_patterns_post_agg: tuple[PatternType, ...]
    intent_patterns_ranking: tuple[PatternType, ...]
    intent_patterns_filter: tuple[PatternType, ...]
    intent_patterns_metric: tuple[PatternType, ...]
    group_by_pattern: PatternType
    measure_hint_patterns: tuple[PatternType, ...]
    time_patterns: tuple[TimePatternRule, ...]
    ratio_query_tokens: frozenset[str]
    stop_tokens: frozenset[str]
    status_hint_tokens: frozenset[str]
    temporal_type_hints: frozenset[str]
    model_warning_tokens: tuple[str, ...]
    lookup_numeric_type_hints: frozenset[str]
    lookup_temporal_type_hints: frozenset[str]
    metric_scoring: MetricScoringWeights
    example_expansion: ExampleExpansionRules
    ranking_dimension_preference: RankingDimensionPreferenceRules
    synonym_scoring: SynonymScoringRules
    metric_phrase_scoring: MetricPhraseScoringRules
    entity_matching: EntityMatchingRules
    field_matching: FieldMatchingRules
    time_field_scoring: TimeFieldScoringRules
    semantic_model_scoring: SemanticModelScoringRules
    confidence_scoring: ConfidenceScoringRules
    selection_tuning: SelectionTuningRules
    post_aggregation: PostAggregationRules
    plan_fallbacks: PlanFallbackRules
    asset_text_formatting: AssetTextFormattingRules
    semantic_example_selection: SemanticExampleSelectionRules
    query_forms: tuple[QueryFormRule, ...] = ()
    lookup_column_count_max: StrictIntValue = 4
    lookup_max_numeric_columns: StrictIntValue = 1
    lookup_identifier_column_names: frozenset[str] = frozenset()
    lookup_identifier_suffixes: tuple[str, ...] = ()

    @field_validator(
        "intent_patterns_post_agg",
        "intent_patterns_ranking",
        "intent_patterns_filter",
        "intent_patterns_metric",
        "measure_hint_patterns",
        mode="before",
    )
    @classmethod
    def _compile_pattern_lists(cls, value: object) -> tuple[PatternType, ...]:
        return _compile_pattern_tuple(value, allow_empty=False)

    @field_validator("group_by_pattern", mode="before")
    @classmethod
    def _compile_single_pattern(cls, value: object) -> PatternType:
        return _compile_pattern(value)

    @field_validator(
        "ratio_query_tokens",
        "stop_tokens",
        "status_hint_tokens",
        "temporal_type_hints",
        "lookup_numeric_type_hints",
        "lookup_temporal_type_hints",
        "lookup_identifier_column_names",
        mode="before",
    )
    @classmethod
    def _normalize_token_sets(cls, value: object) -> frozenset[str]:
        return _normalize_token_set(value, allow_empty=False)

    @field_validator("model_warning_tokens", "lookup_identifier_suffixes", mode="before")
    @classmethod
    def _normalize_text_lists(cls, value: object) -> tuple[str, ...]:
        return _normalize_text_tuple(value, allow_empty=False)

    @field_validator("time_patterns", mode="before")
    @classmethod
    def _validate_time_patterns(cls, value: object) -> tuple[TimePatternRule, ...]:
        if not isinstance(value, list) or not value:
            raise ValueError("time_patterns debe ser una lista no vacia.")
        return tuple(TimePatternRule.model_validate(item) for item in value)

    @field_validator("query_forms", mode="before")
    @classmethod
    def _validate_query_forms(cls, value: object) -> tuple[QueryFormRule, ...]:
        if value in (None, []):
            return ()
        if isinstance(value, tuple):
            return tuple(QueryFormRule.model_validate(item) for item in value)
        if not isinstance(value, list):
            raise ValueError("query_forms debe ser una lista cuando se declara.")
        return tuple(QueryFormRule.model_validate(item) for item in value)


class SemanticResolverVerificationRules(StrictModel):
    """Limites declarativos del verificador semantico local."""

    prompt_token_safety_margin: NonNegativeInt
    max_tables_rich: NonNegativeInt
    max_tables_tight: NonNegativeInt
    max_tables_minimal: NonNegativeInt
    max_columns_per_table_rich: NonNegativeInt
    max_columns_per_table_tight: NonNegativeInt
    max_columns_per_table_minimal: NonNegativeInt
    measure_formula_chars_rich: NonNegativeInt
    measure_formula_chars_minimal: NonNegativeInt
    schema_foreign_key_limit_rich: NonNegativeInt
    max_warnings: NonNegativeInt
    max_join_edges: NonNegativeInt
    example_question_chars: NonNegativeInt
    example_metric_limit: NonNegativeInt
    example_dimension_limit: NonNegativeInt
    repair_measure_match_bonus: NumericValue
    repair_group_by_overlap_weight: NumericValue
    repair_join_table_overlap_weight: NumericValue
    repair_same_intent_bonus: NumericValue
    repair_same_measure_bonus: NumericValue
    recoverable_repairabilities: frozenset[str]
    non_recoverable_failure_classes: frozenset[str]

    @field_validator("recoverable_repairabilities", "non_recoverable_failure_classes", mode="before")
    @classmethod
    def _normalize_repairability_sets(cls, value: object) -> frozenset[str]:
        return _normalize_token_set(value, allow_empty=False)


class SemanticResolverSettings(StrictModel):
    """Configuracion completa y validada del semantic resolver."""

    prompts: SemanticResolverPromptRules
    runtime_tuning: SemanticResolverRuntimeTuning
    compiler_rules: CompilerRules
    verification: SemanticResolverVerificationRules


class SqlSolverSpecGenerationPrompt(StrictModel):
    """Prompt principal de generacion del solver SQL."""

    system_prompt: NonEmptyStr
    user_prompt_template: NonEmptyStr


class SqlSolverPromptRules(StrictModel):
    """Bloque de prompts del solver SQL."""

    spec_generation: SqlSolverSpecGenerationPrompt


class SolverFilterValueRules(StrictModel):
    """Reglas lexicas para extraer valores literales de filtros."""

    stop_tokens: frozenset[str] = frozenset()
    leading_connectors: tuple[str, ...] = ()
    separator_patterns: tuple[str, ...] = ()
    bare_value_pattern: NonEmptyStr = "[A-Za-z0-9][A-Za-z0-9_.:/-]*"

    @field_validator("stop_tokens", mode="before")
    @classmethod
    def _normalize_stop_tokens(cls, value: object) -> frozenset[str]:
        if value in (None, []):
            return frozenset()
        if not isinstance(value, list):
            raise ValueError("stop_tokens debe ser una lista.")
        normalized = {
            normalize_text_for_matching(raw_token)
            for raw_token in value
            if isinstance(raw_token, str) and normalize_text_for_matching(raw_token)
        }
        return frozenset(normalized)

    @field_validator("leading_connectors", "separator_patterns", mode="before")
    @classmethod
    def _normalize_optional_text_tuples(cls, value: object) -> tuple[str, ...]:
        if value in (None, []):
            return ()
        return _normalize_text_tuple(value, allow_empty=False)


class SQLGenerationTuningRules(StrictModel):
    """Tuning declarativo del prompt budgeting y repair del solver."""

    end_of_output_marker: NonEmptyStr
    prompt_token_safety_margin: NonNegativeInt
    detail_listing_default_limit: NonNegativeInt
    max_columns_per_table_rich: NonNegativeInt
    max_columns_per_table_minimal: NonNegativeInt
    default_few_shot_limit: NonNegativeInt
    lean_few_shot_limit: NonNegativeInt
    min_cpu_offload_gb: NumericValue
    example_metric_limit: NonNegativeInt
    example_dimension_limit: NonNegativeInt
    example_filter_limit: NonNegativeInt
    ranking_default_limit: NonNegativeInt
    rich_candidate_plan_limit: NonNegativeInt
    minimal_candidate_plan_limit: NonNegativeInt
    semantic_context_confidence_threshold: NumericValue
    retry_rules: tuple[tuple[frozenset[str], str], ...] = ()

    @field_validator("retry_rules", mode="before")
    @classmethod
    def _normalize_retry_rules(cls, value: object) -> tuple[tuple[frozenset[str], str], ...]:
        if value in (None, []):
            return ()
        if not isinstance(value, list):
            raise ValueError("retry_rules debe ser una lista.")
        normalized: list[tuple[frozenset[str], str]] = []
        for index, raw_rule in enumerate(value):
            if not isinstance(raw_rule, dict):
                raise ValueError(f"retry_rules[{index}] debe ser un mapping.")
            issue_codes = _normalize_token_set(raw_rule.get("issue_codes"), allow_empty=False)
            guidance = _require_text(raw_rule.get("guidance"))
            normalized.append((issue_codes, guidance))
        return tuple(normalized)


class SqlSolverRuntimeTuning(StrictModel):
    """Knobs operativos del solver SQL local."""

    max_retries: StrictIntValue
    llm_dtype: NonEmptyStr
    max_model_len: StrictIntValue
    max_tokens: StrictIntValue
    temperature: NumericValue
    gpu_memory_utilization: NumericValue
    enforce_eager: StrictBoolValue
    cpu_offload_gb: NumericValue
    swap_space_gb: NumericValue
    fail_on_validation_error: StrictBoolValue

    @field_validator("llm_dtype")
    @classmethod
    def _validate_llm_dtype(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in {"auto", "bfloat16", "float16", "float32"}:
            raise ValueError("llm_dtype debe ser auto, bfloat16, float16 o float32.")
        return normalized


class SqlSolverSettings(StrictModel):
    """Configuracion completa y validada del solver SQL."""

    prompts: SqlSolverPromptRules
    filter_value_rules: SolverFilterValueRules
    generation_tuning: SQLGenerationTuningRules
    runtime_tuning: SqlSolverRuntimeTuning


class NarrativePromptRules(StrictModel):
    """Prompt narrativo final del orquestador."""

    system: NonEmptyStr
    user_template: NonEmptyStr


class OrchestratorSettings(StrictModel):
    """Seccion del YAML interno usada por el orquestador."""

    narrative_prompt: NarrativePromptRules


class SemanticResolverCompilerRulesConfig(StrictModel):
    """Shape crudo del bloque compiler_rules en settings.yaml."""

    intent_patterns: dict[str, list[str]]
    group_by_pattern: NonEmptyStr
    measure_hint_patterns: list[str]
    ratio_query_tokens: list[str]
    time_patterns: list[dict[str, object]]
    stop_tokens: list[object]
    status_hint_tokens: list[str]
    temporal_type_hints: list[str]
    model_warning_tokens: list[str]
    lookup_numeric_type_hints: list[str]
    lookup_temporal_type_hints: list[str]
    lookup_column_count_max: StrictIntValue
    lookup_max_numeric_columns: StrictIntValue
    lookup_identifier_column_names: list[str]
    lookup_identifier_suffixes: list[str]
    metric_scoring: MetricScoringWeights
    example_expansion: ExampleExpansionRules
    ranking_dimension_preference: RankingDimensionPreferenceRules
    synonym_scoring: SynonymScoringRules
    metric_phrase_scoring: MetricPhraseScoringRules
    entity_matching: EntityMatchingRules
    field_matching: FieldMatchingRules
    time_field_scoring: TimeFieldScoringRules
    semantic_model_scoring: SemanticModelScoringRules
    confidence_scoring: ConfidenceScoringRules
    selection_tuning: SelectionTuningRules
    post_aggregation: PostAggregationRules
    plan_fallbacks: PlanFallbackRules
    asset_text_formatting: AssetTextFormattingRules
    semantic_example_selection: SemanticExampleSelectionRules

    def to_runtime(self, *, query_forms: tuple[QueryFormRule, ...] = ()) -> CompilerRules:
        """Compila el shape crudo del YAML a reglas listas para el runtime."""

        return CompilerRules(
            intent_patterns_post_agg=self.intent_patterns.get("post_aggregated_metric", []),
            intent_patterns_ranking=self.intent_patterns.get("ranking", []),
            intent_patterns_filter=self.intent_patterns.get("filter_metric", []),
            intent_patterns_metric=self.intent_patterns.get("simple_metric", []),
            group_by_pattern=self.group_by_pattern,
            measure_hint_patterns=self.measure_hint_patterns,
            time_patterns=self.time_patterns,
            ratio_query_tokens=self.ratio_query_tokens,
            stop_tokens=self.stop_tokens,
            status_hint_tokens=self.status_hint_tokens,
            temporal_type_hints=self.temporal_type_hints,
            model_warning_tokens=self.model_warning_tokens,
            lookup_numeric_type_hints=self.lookup_numeric_type_hints,
            lookup_temporal_type_hints=self.lookup_temporal_type_hints,
            metric_scoring=self.metric_scoring,
            example_expansion=self.example_expansion,
            ranking_dimension_preference=self.ranking_dimension_preference,
            synonym_scoring=self.synonym_scoring,
            metric_phrase_scoring=self.metric_phrase_scoring,
            entity_matching=self.entity_matching,
            field_matching=self.field_matching,
            time_field_scoring=self.time_field_scoring,
            semantic_model_scoring=self.semantic_model_scoring,
            confidence_scoring=self.confidence_scoring,
            selection_tuning=self.selection_tuning,
            post_aggregation=self.post_aggregation,
            plan_fallbacks=self.plan_fallbacks,
            asset_text_formatting=self.asset_text_formatting,
            semantic_example_selection=self.semantic_example_selection,
            query_forms=query_forms,
            lookup_column_count_max=self.lookup_column_count_max,
            lookup_max_numeric_columns=self.lookup_max_numeric_columns,
            lookup_identifier_column_names=self.lookup_identifier_column_names,
            lookup_identifier_suffixes=self.lookup_identifier_suffixes,
        )


class SemanticResolverSettingsConfig(StrictModel):
    """Shape crudo del resolver antes de compilar compiler_rules."""

    prompts: SemanticResolverPromptRules
    runtime_tuning: SemanticResolverRuntimeTuning
    compiler_rules: SemanticResolverCompilerRulesConfig
    verification: SemanticResolverVerificationRules

    def to_runtime(self, *, query_forms: tuple[QueryFormRule, ...] = ()) -> SemanticResolverSettings:
        """Compila la configuracion del resolver a su shape final de runtime."""

        return SemanticResolverSettings(
            prompts=self.prompts,
            runtime_tuning=self.runtime_tuning,
            compiler_rules=self.compiler_rules.to_runtime(query_forms=query_forms),
            verification=self.verification,
        )


class NL2SQLSettingsConfig(StrictModel):
    """Documento raiz validado de settings.yaml antes de derivados runtime."""

    semantic_prune: SemanticPruneSettings
    semantic_resolver: SemanticResolverSettingsConfig
    sql_solver: SqlSolverSettings
    orchestrator: OrchestratorSettings

    def to_runtime(self, *, query_forms: tuple[QueryFormRule, ...] = ()) -> "NL2SQLSettings":
        """Materializa el documento final con reglas listas para consumo."""

        return NL2SQLSettings(
            semantic_prune=self.semantic_prune,
            semantic_resolver=self.semantic_resolver.to_runtime(query_forms=query_forms),
            sql_solver=self.sql_solver,
            orchestrator=self.orchestrator,
        )


class NL2SQLSettings(StrictModel):
    """Documento raiz final de settings.yaml listo para consumo."""

    semantic_prune: SemanticPruneSettings
    semantic_resolver: SemanticResolverSettings
    sql_solver: SqlSolverSettings
    orchestrator: OrchestratorSettings


@dataclass(frozen=True)
class NL2SQLRuntimeBundle:
    """Bundle central con settings validados y semantic contract listo."""

    settings_path: Path
    semantic_rules_path: Path
    settings: NL2SQLSettings
    semantic_contract: SemanticContract


def _compile_pattern(value: object) -> PatternType:
    return re.compile(_require_text(value))


def _compile_pattern_tuple(value: object, *, allow_empty: bool) -> tuple[PatternType, ...]:
    if value in (None, []):
        if allow_empty:
            return ()
        raise ValueError("Debe ser una lista no vacia de patrones.")
    if not isinstance(value, list):
        raise ValueError("Debe ser una lista de patrones.")
    compiled = tuple(_compile_pattern(raw_pattern) for raw_pattern in value)
    if not compiled and not allow_empty:
        raise ValueError("La lista de patrones no puede quedar vacia.")
    return compiled


def _normalize_text_tuple(
    value: object,
    *,
    allow_empty: bool,
    true_token: str | None = None,
    false_token: str | None = None,
    ignore_non_strings: bool = False,
) -> tuple[str, ...]:
    if value in (None, []):
        if allow_empty:
            return ()
        raise ValueError("Debe ser una lista no vacia de strings.")
    if not isinstance(value, list):
        raise ValueError("Debe ser una lista de strings.")
    normalized_values: list[str] = []
    for raw_value in value:
        if isinstance(raw_value, str):
            candidate = raw_value.strip()
        elif isinstance(raw_value, bool) and true_token is not None and false_token is not None:
            candidate = true_token if raw_value else false_token
        elif ignore_non_strings:
            continue
        else:
            candidate = _normalize_scalar_text(raw_value)
        if candidate:
            normalized_values.append(candidate)
    normalized = tuple(normalized_values)
    if not normalized and not allow_empty:
        raise ValueError("La lista no puede quedar vacia.")
    return normalized


def _normalize_token_set(
    value: object,
    *,
    allow_empty: bool,
    true_token: str = "on",
    false_token: str = "off",
) -> frozenset[str]:
    if value in (None, []):
        if allow_empty:
            return frozenset()
        raise ValueError("Debe ser una lista no vacia de tokens.")
    if not isinstance(value, list):
        raise ValueError("Debe ser una lista de tokens.")
    normalized_values: set[str] = set()
    for raw_value in value:
        if isinstance(raw_value, str):
            candidate = raw_value.strip().lower()
        elif isinstance(raw_value, bool):
            candidate = true_token if raw_value else false_token
        else:
            candidate = str(raw_value).strip().lower()
        if candidate:
            normalized_values.add(candidate)
    normalized = frozenset(normalized_values)
    if not normalized and not allow_empty:
        raise ValueError("La lista de tokens no puede quedar vacia.")
    return normalized


def _normalize_suffix_rewrite_rules(value: object, *, allow_empty: bool) -> tuple[tuple[str, str, int], ...]:
    """Normaliza reglas declarativas de reescritura de sufijos."""

    if value in (None, []):
        if allow_empty:
            return ()
        raise ValueError("Debe ser una lista no vacia de reglas de sufijos.")
    if not isinstance(value, list):
        raise ValueError("Debe ser una lista de reglas de sufijos.")

    normalized_rules: list[tuple[str, str, int]] = []
    for index, raw_rule in enumerate(value):
        if not isinstance(raw_rule, dict):
            raise ValueError(f"suffix_rules[{index}] debe ser un mapping.")
        raw_suffix = raw_rule.get("suffix")
        if not isinstance(raw_suffix, str):
            raise ValueError(f"suffix_rules[{index}].suffix debe ser un string.")
        suffix = raw_suffix.strip().lower()
        replacement = str(raw_rule.get("replacement", "")).strip().lower()
        min_length = _require_non_negative_int(raw_rule.get("min_length", 0))
        if not suffix and not replacement:
            raise ValueError(f"suffix_rules[{index}] debe declarar suffix o replacement.")
        normalized_rules.append((suffix, replacement, min_length))
    if not normalized_rules and not allow_empty:
        raise ValueError("La lista de reglas de sufijos no puede quedar vacia.")
    return tuple(normalized_rules)
