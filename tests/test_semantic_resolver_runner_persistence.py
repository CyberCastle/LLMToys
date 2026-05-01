#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path

import yaml

import run_semantic_resolver as resolver_runner
from nl2sql.semantic_resolver.assets import MatchedAsset, SemanticAsset, SemanticPlan
from nl2sql.semantic_resolver.plan_model import CompiledSemanticPlan, PlanMeasure


def _matched_asset(kind: str, name: str) -> MatchedAsset:
    return MatchedAsset(
        asset=SemanticAsset(
            asset_id=f"{kind}::{name}",
            kind=kind,
            name=name,
            payload={"name": name, "source_table": name},
        ),
        embedding_score=0.9,
        rerank_score=0.8,
        compatibility_score=1.0,
        compatible_tables=(name,),
        rejected_reason=None,
    )


def test_load_pruned_schema_requires_existing_yaml(tmp_path: Path, monkeypatch) -> None:
    missing_path = tmp_path / "missing_semantic_pruned_schema.yaml"
    monkeypatch.setattr(resolver_runner, "PRUNED_SCHEMA_PATH", str(missing_path))

    try:
        resolver_runner._load_pruned_schema()
    except FileNotFoundError as exc:
        assert str(missing_path) in str(exc)
    else:
        raise AssertionError("_load_pruned_schema debio fallar cuando falta el YAML")


def test_persist_semantic_plan_splits_retrieved_and_compiled(tmp_path: Path, monkeypatch) -> None:
    output_path = tmp_path / "semantic_plan.yaml"
    source_pruned_schema_path = tmp_path / "semantic_pruned_schema.yaml"
    monkeypatch.setattr(resolver_runner, "SEMANTIC_PLAN_OUTPUT_PATH", str(output_path))
    monkeypatch.setattr(resolver_runner, "PRUNED_SCHEMA_PATH", str(source_pruned_schema_path))
    monkeypatch.setattr(resolver_runner, "RULES_PATH", "schema-docs/semantic_rules.yaml")

    accepted_metric = _matched_asset("semantic_metrics", "metric_count_a")
    semantic_plan = SemanticPlan(
        query="promedio de registros por entidad_c",
        assets_by_kind={"semantic_metrics": [accepted_metric]},
        all_assets=[accepted_metric],
        pruned_tables=("entity_a", "entity_c"),
        diagnostics={"num_accepted": 1},
        compiled_plan=CompiledSemanticPlan(
            query="promedio de registros por entidad_c",
            semantic_model="model_alpha",
            intent="post_aggregated_metric",
            base_entity="entity_a",
            grain="entity_a.id",
            measure=PlanMeasure(
                name="metric_count_a",
                formula="count_distinct(entity_a.id)",
                source_table="entity_a",
            ),
            group_by=["entity_c.id"],
            join_path=["entity_a.entity_c_id = entity_c.id"],
            required_tables=["entity_a", "entity_c"],
        ),
    )

    persisted_path = resolver_runner._persist_semantic_plan(semantic_plan)

    assert persisted_path == output_path
    payload = yaml.safe_load(output_path.read_text(encoding="utf-8"))
    assert payload["source_pruned_schema_path"] == str(source_pruned_schema_path)
    assert payload["source_rules_path"] == "schema-docs/semantic_rules.yaml"
    assert payload["semantic_plan"]["retrieved_candidates"]["query"] == "promedio de registros por entidad_c"
    assert "compiled_plan" not in payload["semantic_plan"]["retrieved_candidates"]
    assert payload["semantic_plan"]["compiled_plan"]["base_entity"] == "entity_a"
    assert payload["semantic_plan"]["compiled_plan"]["group_by"] == ["entity_c.id"]
