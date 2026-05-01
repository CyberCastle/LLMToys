#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import pytest

from nl2sql.semantic_resolver.assets import SemanticAsset
from nl2sql.semantic_resolver.resolver import expand_reranked_candidates_from_examples


def _asset(kind: str, name: str, payload: dict[str, object] | None = None) -> SemanticAsset:
    """Crea activos semanticos minimos para pruebas del resolver."""

    return SemanticAsset(
        asset_id=f"{kind}::{name}",
        kind=kind,
        name=name,
        payload=payload or {"name": name},
    )


def test_expand_reranked_candidates_from_examples_inyecta_metricas_y_entidades() -> None:
    """Un ejemplo recuperado debe arrastrar sus metricas curadas al plan."""

    example = _asset(
        "semantic_examples",
        "entity_a_active",
        {
            "question": "¿Cuántas entidades_a están activas?",
            "model": "model_alpha",
            "metrics": ["metric_count_a_active"],
            "dimensions": ["status_a_name"],
        },
    )
    assets = [
        example,
        _asset(
            "semantic_metrics",
            "metric_count_a_active",
            {"name": "metric_count_a_active", "entity": "entity_a"},
        ),
        _asset(
            "semantic_dimensions",
            "status_a_name",
            {
                "name": "status_a_name",
                "entity": "entity_a",
                "source": "status_a.name",
            },
        ),
        _asset(
            "semantic_entities",
            "entity_a",
            {"name": "entity_a", "source_table": "entity_a"},
        ),
        _asset("semantic_models", "model_alpha", {"name": "model_alpha"}),
    ]

    expanded, trace = expand_reranked_candidates_from_examples(
        [(example, 0.70, 0.20)],
        assets,
        inherited_embedding_weight=0.80,
        inherited_rerank_weight=0.70,
    )
    expanded_ids = {asset.asset_id for asset, _embedding_score, _rerank_score in expanded}
    inherited_scores = {
        asset.asset_id: (_embedding_score, _rerank_score)
        for asset, _embedding_score, _rerank_score in expanded
        if asset.asset_id != example.asset_id
    }

    assert "semantic_metrics::metric_count_a_active" in expanded_ids
    assert "semantic_dimensions::status_a_name" in expanded_ids
    assert "semantic_entities::entity_a" in expanded_ids
    assert "semantic_models::model_alpha" in expanded_ids
    assert inherited_scores["semantic_metrics::metric_count_a_active"] == pytest.approx((0.56, 0.14))
    assert len(trace) == 3
