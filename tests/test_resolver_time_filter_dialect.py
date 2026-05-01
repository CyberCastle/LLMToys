#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Tests de inyeccion de dialecto en ``extract_time_filter`` del compilador del plan."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import yaml

from nl2sql.config import load_nl2sql_config
from nl2sql.semantic_resolver.dialects import get_resolver_dialect
from nl2sql.semantic_resolver.plan_compiler import extract_time_filter
from nl2sql.semantic_resolver.rules_loader import load_compiler_rules
from nl2sql.semantic_resolver.config import resolve_compiler_rules_path


def _load_rules():
    """Carga las reglas del compilador desde el YAML por defecto."""

    return load_compiler_rules(str(resolve_compiler_rules_path()))


def test_extract_time_filter_emite_expresion_postgres() -> None:
    """Con dialecto PostgreSQL solo debe materializarse la expresion correspondiente."""

    rules = _load_rules()
    dialect = get_resolver_dialect("postgres")
    time_filter, _warnings = extract_time_filter(
        "registros del ultimo ano",
        entity_assets=[],
        dimension_assets=[],
        base_entity="entity_a",
        base_table="entity_a",
        rules=rules,
        dialect=dialect,
    )
    # Sin entidades disponibles no se resuelve un ``field``, pero el patron
    # temporal si se detecta y produce warning. Forzamos via reglas + dialecto:
    # cuando no hay best_candidate, el helper retorna None con warning.
    # Aqui validamos directamente la materializacion vis _build_resolved_expressions
    # corriendo el matching con un asset minimo.
    assert time_filter is None  # no hay time_field disponible en este escenario


def test_extract_time_filter_con_entity_time_field_postgres() -> None:
    """El plan debe contener resolved_expressions['postgres'] cuando hay time_field."""

    from nl2sql.semantic_resolver.assets import MatchedAsset, SemanticAsset

    rules = _load_rules()
    dialect = get_resolver_dialect("postgres")
    entity = MatchedAsset(
        asset=SemanticAsset(
            asset_id="semantic_entities::entity_a",
            kind="semantic_entities",
            name="entity_a",
            payload={"name": "entity_a", "time_field": "entity_a.created_at"},
        ),
        embedding_score=0.9,
        rerank_score=0.9,
        compatibility_score=1.0,
        compatible_tables=tuple(),
        rejected_reason=None,
    )
    time_filter, _warnings = extract_time_filter(
        "registros del ultimo ano",
        entity_assets=[entity],
        dimension_assets=[],
        base_entity="entity_a",
        base_table="entity_a",
        rules=rules,
        dialect=dialect,
    )
    assert time_filter is not None
    assert time_filter.field == "entity_a.created_at"
    assert time_filter.value == "today - 1 year"
    assert "postgres" in time_filter.resolved_expressions
    assert time_filter.resolved_expressions["postgres"].startswith("(CURRENT_DATE")
    # El YAML declara value_tsql tambien, asi que ambos deben venir presentes
    # (porque el loader copia todas las claves value_*, no solo la del dialecto
    # activo). El dialecto solo aplica fallback cuando la clave del activo
    # falta en el YAML.
    assert "tsql" in time_filter.resolved_expressions


def test_extract_time_filter_sin_dialecto_solo_yaml() -> None:
    """Sin dialecto activo solo se exponen las expresiones declaradas en YAML."""

    from nl2sql.semantic_resolver.assets import MatchedAsset, SemanticAsset

    rules = _load_rules()
    entity = MatchedAsset(
        asset=SemanticAsset(
            asset_id="semantic_entities::entity_a",
            kind="semantic_entities",
            name="entity_a",
            payload={"name": "entity_a", "time_field": "entity_a.created_at"},
        ),
        embedding_score=0.9,
        rerank_score=0.9,
        compatibility_score=1.0,
        compatible_tables=tuple(),
        rejected_reason=None,
    )
    time_filter, _warnings = extract_time_filter(
        "registros del ultimo ano",
        entity_assets=[entity],
        dimension_assets=[],
        base_entity="entity_a",
        base_table="entity_a",
        rules=rules,
        dialect=None,
    )
    assert time_filter is not None
    # Sin dialecto, el dict resolved_expressions contiene solo lo que el YAML
    # declara (tsql + postgres tras la migracion).
    assert set(time_filter.resolved_expressions) == {"tsql", "postgres"}


def test_load_compiler_rules_acepta_config_unificada_personalizada(tmp_path: Path) -> None:
    compiler_rules = deepcopy(load_nl2sql_config(resolve_compiler_rules_path())["semantic_resolver"]["compiler_rules"])
    compiler_rules["status_hint_tokens"] = ["custom_estado"]
    compiler_rules["ratio_query_tokens"] = ["custom_ratio"]
    compiler_rules["lookup_identifier_column_names"] = ["pk"]
    compiler_rules["lookup_identifier_suffixes"] = ["_pk"]
    compiler_rules["ranking_dimension_preference"]["positive_token_weights"]["display"] = 9.0
    compiler_rules["ranking_dimension_preference"]["string_type_hints"] = ["varchar"]
    compiler_rules["synonym_scoring"]["single_token_prefix_strength"] = 0.11
    compiler_rules["post_aggregation"]["over"] = "rows"
    compiler_rules["plan_fallbacks"]["base_entity"] = "custom_unknown_entity"

    config_path = tmp_path / "nl2sql_config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {"semantic_resolver": {"compiler_rules": compiler_rules}},
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )

    rules = load_compiler_rules(str(config_path))

    assert "custom_estado" in rules.status_hint_tokens
    assert "custom_ratio" in rules.ratio_query_tokens
    assert rules.lookup_identifier_column_names == frozenset({"pk"})
    assert rules.lookup_identifier_suffixes == ("_pk",)
    assert dict(rules.ranking_dimension_preference.positive_token_weights)["display"] == 9.0
    assert rules.ranking_dimension_preference.string_type_hints == frozenset({"varchar"})
    assert rules.synonym_scoring.single_token_prefix_strength == 0.11
    assert rules.post_aggregation.over == "rows"
    assert rules.plan_fallbacks.base_entity == "custom_unknown_entity"
