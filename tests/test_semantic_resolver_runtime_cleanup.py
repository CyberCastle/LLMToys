#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from nl2sql.semantic_resolver.assets import SemanticAsset
from nl2sql.semantic_resolver.config import SemanticResolverConfig
from nl2sql.semantic_resolver.resolver import run_semantic_resolver


def test_run_semantic_resolver_releases_cuda_between_sequential_engines() -> None:
    asset = SemanticAsset(
        asset_id="semantic_metrics::metric_count_a",
        kind="semantic_metrics",
        name="metric_count_a",
        payload={"name": "metric_count_a", "source_table": "entity_a"},
    )
    config = SemanticResolverConfig(enable_plan_compiler=False, sequential_engines=True)

    with (
        patch("nl2sql.semantic_resolver.resolver.load_semantic_rules", return_value=[asset]),
        patch("nl2sql.semantic_resolver.resolver.build_reference_maps", return_value=({}, {})),
        patch(
            "nl2sql.semantic_resolver.resolver.resolve_query_synonyms",
            return_value=SimpleNamespace(
                retrieval_query="promedio de registros por entidad_c",
                matched_entities=(),
                matched_tables=(),
                matched_models=(),
            ),
        ),
        patch(
            "nl2sql.semantic_resolver.resolver.get_embedding_runtime",
            return_value=SimpleNamespace(llm=object(), effective_max_model_len=8192),
        ),
        patch("nl2sql.semantic_resolver.resolver.embed_query", return_value=np.array([1.0], dtype=np.float32)),
        patch(
            "nl2sql.semantic_resolver.resolver.embed_assets_cached",
            return_value=(
                np.array([[1.0]], dtype=np.float32),
                SimpleNamespace(enabled=True, hits=1, misses=0, cache_path="/tmp/cache"),
            ),
        ),
        patch(
            "nl2sql.semantic_resolver.resolver._retrieve_top_k_with_boosts",
            return_value=[(asset, 0.9, 0.9)],
        ),
        patch("nl2sql.semantic_resolver.resolver.clear_embedding_runtime") as clear_embedding_mock,
        patch(
            "nl2sql.semantic_resolver.resolver.get_reranker_runtime",
            return_value=SimpleNamespace(effective_max_model_len=4096),
        ),
        patch(
            "nl2sql.semantic_resolver.resolver.rerank_candidates",
            return_value=[(asset, 0.9, 0.8)],
        ),
        patch("nl2sql.semantic_resolver.resolver.clear_reranker_runtime") as clear_reranker_mock,
        patch(
            "nl2sql.semantic_resolver.resolver.score_compatibility",
            return_value=(1.0, ("entity_a",), None),
        ),
        patch("nl2sql.semantic_resolver.resolver.release_cuda_memory") as release_cuda_memory_mock,
    ):
        semantic_plan = run_semantic_resolver(
            query="cual es el promedio de registros por entidad_c en el ultimo ano?",
            pruned_schema={"entity_a": {"columns": []}},
            config=config,
        )

    assert semantic_plan.diagnostics["num_after_retrieval"] == 1
    assert semantic_plan.diagnostics["num_after_rerank"] == 1
    clear_embedding_mock.assert_called_once()
    clear_reranker_mock.assert_called_once()
    assert release_cuda_memory_mock.call_count == 2
