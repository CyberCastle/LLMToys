#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from nl2sql.semantic_resolver.assets import SemanticAsset
from nl2sql.semantic_resolver.rules_loader import build_reference_maps
from nl2sql.semantic_resolver.synonym_logic import resolve_query_synonyms


def test_resolve_query_synonyms_detecta_plural_de_sigla_corta() -> None:
    """Valida que consultas con una sigla plural corta activen la entidad base."""

    assets = [
        SemanticAsset(
            asset_id="semantic_entities::entity_a",
            kind="semantic_entities",
            name="entity_a",
            payload={"name": "entity_a", "source_table": "entity_a"},
        ),
        SemanticAsset(
            asset_id="semantic_synonyms::entity_a",
            kind="semantic_synonyms",
            name="entity_a",
            payload={"entity": "entity_a", "synonyms": ["EA"]},
        ),
        SemanticAsset(
            asset_id="semantic_models::model_alpha",
            kind="semantic_models",
            name="model_alpha",
            payload={"name": "model_alpha", "core_tables": ["entity_a"]},
        ),
    ]
    entity_to_table, model_to_tables = build_reference_maps(assets)

    resolution = resolve_query_synonyms(
        "cual es el promedio de EAs activas por entidad_c",
        assets,
        entity_to_table=entity_to_table,
        model_to_tables=model_to_tables,
        max_entities=5,
        enable_query_expansion=True,
    )

    assert "entity_a" in resolution.matched_entities
    assert "entity_a" in resolution.matched_tables
    assert "model_alpha" in resolution.matched_models
    assert "Semantic entities hinted by query: entity_a" in resolution.retrieval_query
