#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Callable

import yaml

from nl2sql.config import load_nl2sql_config
from nl2sql.semantic_resolver.assets import MatchedAsset, SemanticAsset, SemanticPlan
from nl2sql.semantic_resolver.config import SemanticResolverConfig, resolve_compiler_rules_path
from nl2sql.semantic_resolver.plan_compiler import (
    _detect_lookup_tables,
    build_join_graph,
    compile_semantic_plan,
    extract_group_by,
    select_base_entity,
    shortest_join_path,
)
from nl2sql.semantic_resolver.plan_intent import detect_intent
from nl2sql.semantic_resolver.plan_model import (
    PlanMeasure,
    PlanPostAggregation,
    PlanRanking,
    PlanTimeFilter,
)
from nl2sql.semantic_resolver.rules_loader import load_compiler_rules
from tests.generic_domain import (
    generic_schema_tables,
    generic_semantic_contract_payload,
)

QUERY = "average of metric_count_a by entity_c last year"
UNMAPPED_STATUS_QUERY = "average of rejected metric_count_a by entity_c last year"
ACTIVE_QUERY = "average of metric_count_a_active by entity_c last year"
LOSS_QUERY = "average of metric_count_b_lost by entity_c last year"
RANKING_QUERY = "top 5 de entity_c con mas entity_a en el ultimo ano"
RANKING_SYNONYM_QUERY = "top 5 de groups con mas entity_a en el ultimo ano"
SCALAR_TIME_QUERY = "metric_count_a last year"
SCALAR_FILTER_QUERY = "metric_count_a with status b = 7"


def _write_rules(tmp_path: Path) -> Path:
    rules_path = tmp_path / "semantic_rules.yaml"
    rules_path.write_text(
        yaml.safe_dump(generic_semantic_contract_payload(), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return rules_path


def _config(tmp_path: Path) -> SemanticResolverConfig:
    return SemanticResolverConfig(rules_path=str(_write_rules(tmp_path)))


def _write_compiler_rules_config(
    tmp_path: Path,
    *,
    mutate: Callable[[dict[str, Any]], None] | None = None,
) -> Path:
    compiler_rules = deepcopy(load_nl2sql_config(resolve_compiler_rules_path())["semantic_resolver"]["compiler_rules"])
    if mutate is not None:
        mutate(compiler_rules)

    config_path = tmp_path / "nl2sql_config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {"semantic_resolver": {"compiler_rules": compiler_rules}},
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    return config_path


def _matched_asset(
    kind: str,
    name: str,
    payload: dict[str, object],
    *,
    embedding_score: float = 0.80,
    rerank_score: float = 0.80,
    compatibility_score: float = 1.0,
) -> MatchedAsset:
    return MatchedAsset(
        asset=SemanticAsset(
            asset_id=f"{kind}::{name}",
            kind=kind,
            name=name,
            payload=payload,
        ),
        embedding_score=embedding_score,
        rerank_score=rerank_score,
        compatibility_score=compatibility_score,
        compatible_tables=tuple(),
        rejected_reason=None,
    )


def _build_plan_assets() -> list[MatchedAsset]:
    return [
        _matched_asset(
            "semantic_entities",
            "entity_a",
            {
                "name": "entity_a",
                "business_definition": "Primary operational record in the generic domain.",
                "source_table": "entity_a",
                "key": "entity_a.id",
            },
        ),
        _matched_asset(
            "semantic_entities",
            "entity_b",
            {
                "name": "entity_b",
                "business_definition": "Commercial record associated to the grouping entity.",
                "source_table": "entity_b",
                "key": "entity_b.id",
            },
        ),
        _matched_asset(
            "semantic_entities",
            "entity_c",
            {
                "name": "entity_c",
                "business_definition": "Grouping entity in the generic domain.",
                "source_table": "entity_c",
                "key": "entity_c.id",
            },
        ),
        _matched_asset(
            "semantic_entities",
            "status_a",
            {
                "name": "status_a",
                "business_definition": "Operational status catalog.",
                "source_table": "status_a",
                "key": "status_a.id",
            },
        ),
        _matched_asset(
            "semantic_entities",
            "status_b",
            {
                "name": "status_b",
                "business_definition": "Commercial status catalog.",
                "source_table": "status_b",
                "key": "status_b.id",
            },
        ),
        _matched_asset(
            "semantic_metrics",
            "metric_count_a_active",
            {
                "name": "metric_count_a_active",
                "entity": "entity_a",
                "formula": "count_distinct(case when status_a.name = 'Active' then entity_a.id end)",
            },
            embedding_score=0.91,
        ),
        _matched_asset(
            "semantic_metrics",
            "metric_count_a",
            {
                "name": "metric_count_a",
                "entity": "entity_a",
                "formula": "count_distinct(entity_a.id)",
            },
            embedding_score=0.88,
        ),
        _matched_asset(
            "semantic_metrics",
            "metric_count_c",
            {
                "name": "metric_count_c",
                "entity": "entity_c",
                "formula": "count_distinct(entity_c.id)",
            },
            embedding_score=0.82,
        ),
        _matched_asset(
            "semantic_metrics",
            "metric_count_b_lost",
            {
                "name": "metric_count_b_lost",
                "entity": "entity_b",
                "formula": "count_distinct(case when status_b.name = 'Archived' then entity_b.id end)",
                "synonyms": ["archived entity_b records"],
                "examples": [{"question": LOSS_QUERY, "expected_metric": "metric_count_b_lost"}],
            },
            embedding_score=0.68,
            compatibility_score=0.5,
        ),
        _matched_asset(
            "semantic_metrics",
            "metric_ratio_b_lost",
            {
                "name": "metric_ratio_b_lost",
                "entity": "entity_b",
                "formula": "metric_count_b_lost / nullif(metric_count_b, 0)",
                "synonyms": ["ratio of archived entity_b records"],
            },
            embedding_score=0.77,
            compatibility_score=1.0,
        ),
        _matched_asset(
            "semantic_dimensions",
            "entity_a_created_at",
            {
                "name": "entity_a_created_at",
                "entity": "entity_a",
                "source": "entity_a.created_at",
                "type": "datetime",
            },
        ),
        _matched_asset(
            "semantic_dimensions",
            "entity_b_requested_at",
            {
                "name": "entity_b_requested_at",
                "entity": "entity_b",
                "source": "entity_b.requested_at",
                "type": "date",
            },
        ),
        _matched_asset(
            "semantic_dimensions",
            "entity_c_label",
            {
                "name": "entity_c_label",
                "entity": "entity_c",
                "source": "entity_c.display_name",
                "type": "string",
            },
        ),
        _matched_asset(
            "semantic_filters",
            "by_entity_c",
            {
                "name": "by_entity_c",
                "field": "entity_c.id",
            },
        ),
        _matched_asset(
            "semantic_synonyms",
            "entity_c",
            {
                "entity": "entity_c",
                "synonyms": ["group", "aggregator"],
            },
        ),
        _matched_asset(
            "semantic_synonyms",
            "entity_a",
            {
                "entity": "entity_a",
                "synonyms": ["record", "ea"],
            },
        ),
        _matched_asset(
            "semantic_models",
            "model_alpha",
            {
                "name": "model_alpha",
                "core_tables": [
                    "entity_a",
                    "entity_b_version",
                    "entity_b",
                    "bridge_contact",
                    "entity_c_site",
                    "entity_c",
                    "status_a",
                ],
            },
        ),
        _matched_asset(
            "semantic_models",
            "model_beta",
            {
                "name": "model_beta",
                "core_tables": [
                    "entity_b",
                    "bridge_contact",
                    "entity_c_site",
                    "entity_c",
                    "status_b",
                ],
            },
            embedding_score=0.70,
        ),
        _matched_asset(
            "semantic_relationships",
            "entity_c_site.entity_c_id",
            {"from": "entity_c_site.entity_c_id", "to": "entity_c.id"},
        ),
        _matched_asset(
            "semantic_relationships",
            "bridge_contact.entity_c_site_id",
            {"from": "bridge_contact.entity_c_site_id", "to": "entity_c_site.id"},
        ),
        _matched_asset(
            "semantic_relationships",
            "entity_b.bridge_contact_id",
            {"from": "entity_b.bridge_contact_id", "to": "bridge_contact.id"},
        ),
        _matched_asset(
            "semantic_relationships",
            "entity_b.status_b_id",
            {"from": "entity_b.status_b_id", "to": "status_b.id"},
        ),
        _matched_asset(
            "semantic_relationships",
            "entity_b_version.entity_b_id",
            {"from": "entity_b_version.entity_b_id", "to": "entity_b.id"},
        ),
        _matched_asset(
            "semantic_relationships",
            "entity_a.entity_b_version_id",
            {"from": "entity_a.entity_b_version_id", "to": "entity_b_version.id"},
        ),
        _matched_asset(
            "semantic_relationships",
            "entity_a.status_a_id",
            {"from": "entity_a.status_a_id", "to": "status_a.id"},
        ),
    ]


def _build_pruned_schema() -> dict[str, object]:
    return generic_schema_tables()


def _build_plan(query: str = QUERY) -> SemanticPlan:
    assets = _build_plan_assets()
    assets_by_kind: dict[str, list[MatchedAsset]] = {}
    for asset in assets:
        assets_by_kind.setdefault(asset.asset.kind, []).append(asset)
    synonym_entities = ["entity_b", "entity_c"] if "metric_count_b_lost" in query else ["entity_a", "entity_c"]
    return SemanticPlan(
        query=query,
        assets_by_kind=assets_by_kind,
        all_assets=assets,
        pruned_tables=tuple(sorted(_build_pruned_schema())),
        diagnostics={"synonym_entities_detected": synonym_entities},
    )


def test_detect_intent_post_aggregated(tmp_path: Path) -> None:
    config = _config(tmp_path)
    rules = load_compiler_rules(str(config.compiler_rules_path))

    assert detect_intent(QUERY, rules) == "post_aggregated_metric"


def test_base_entity_follows_measure_table() -> None:
    measure = PlanMeasure(
        name="metric_count_a",
        formula="count_distinct(entity_a.id)",
        source_table="entity_a",
    )
    entity_assets = [asset for asset in _build_plan_assets() if asset.asset.kind == "semantic_entities"]
    base_entity, grain = select_base_entity(measure, entity_assets)

    assert base_entity == "entity_a"
    assert grain == "entity_a.id"


def test_shortest_join_path_resolves_multi_hop_entity_c_to_entity_a() -> None:
    relationship_assets = [asset for asset in _build_plan_assets() if asset.asset.kind == "semantic_relationships"]
    graph = build_join_graph(relationship_assets, _build_pruned_schema())
    join_path = shortest_join_path(graph, "entity_a", "entity_c")

    assert join_path == [
        "entity_a.entity_b_version_id = entity_b_version.id",
        "entity_b_version.entity_b_id = entity_b.id",
        "entity_b.bridge_contact_id = bridge_contact.id",
        "bridge_contact.entity_c_site_id = entity_c_site.id",
        "entity_c_site.entity_c_id = entity_c.id",
    ]


def test_build_join_graph_accepts_normalized_foreign_key_column_key() -> None:
    graph = build_join_graph(
        [],
        {
            "entity_b": {
                "columns": {"id": {}, "status_b_id": {}},
                "foreign_keys": [{"column": "status_b_id", "ref_table": "status_b", "ref_col": "id"}],
            },
            "status_b": {
                "columns": {"id": {}, "name": {}},
                "foreign_keys": [],
            },
        },
    )

    assert shortest_join_path(graph, "entity_b", "status_b") == ["entity_b.status_b_id = status_b.id"]


def test_lookup_detection_accepts_dict_columns_and_column_types(tmp_path: Path) -> None:
    config = _config(tmp_path)
    rules = load_compiler_rules(str(config.compiler_rules_path))
    lookup_tables = _detect_lookup_tables(
        {
            "status_b": {
                "columns": {"id": {}, "name": {}},
                "column_types": {"id": "BIGINT", "name": "VARCHAR"},
                "foreign_keys": [],
            },
            "event_log": {
                "columns": {"id": {}, "created_at": {}},
                "column_types": {"id": "BIGINT", "created_at": "DATETIME"},
                "foreign_keys": [],
            },
        },
        rules,
    )

    assert "status_b" in lookup_tables
    assert "event_log" not in lookup_tables


def test_compile_semantic_plan_builds_two_level_aggregation(tmp_path: Path) -> None:
    config = _config(tmp_path)
    compiled_plan = compile_semantic_plan(
        _build_plan(),
        QUERY,
        config=config,
        pruned_schema=_build_pruned_schema(),
    )

    assert compiled_plan.intent == "post_aggregated_metric"
    assert compiled_plan.semantic_model == "model_alpha"
    assert compiled_plan.base_entity == "entity_a"
    assert compiled_plan.grain == "entity_a.id"
    assert compiled_plan.measure is not None
    assert compiled_plan.measure.name == "metric_count_a"
    assert compiled_plan.metric_score_trace
    selected_trace = next(trace for trace in compiled_plan.metric_score_trace if trace.selected)
    assert selected_trace.metric_name == "metric_count_a"
    assert selected_trace.components["compatibility"] > 0.0
    assert compiled_plan.group_by == ["entity_c.id"]
    assert compiled_plan.time_filter == PlanTimeFilter(
        field="entity_a.created_at",
        operator=">=",
        value="today - 1 year",
        resolved_expressions={
            "tsql": "DATEADD(year, -1, CAST(GETDATE() AS date))",
            "postgres": "(CURRENT_DATE - INTERVAL '1 year')",
        },
    )
    assert compiled_plan.post_aggregation == PlanPostAggregation(function="avg", over="grouped_measure")
    assert compiled_plan.join_path == [
        "entity_a.entity_b_version_id = entity_b_version.id",
        "entity_b_version.entity_b_id = entity_b.id",
        "entity_b.bridge_contact_id = bridge_contact.id",
        "bridge_contact.entity_c_site_id = entity_c_site.id",
        "entity_c_site.entity_c_id = entity_c.id",
    ]
    assert compiled_plan.join_path_hint == "entity_c_to_a_via_b"
    assert compiled_plan.derived_metric_ref == "metric_avg_a_per_c"
    assert compiled_plan.population_scope == "active_entities_only"
    assert "population_scope_defaulted_to_active_entities_only" in compiled_plan.warnings
    assert set(compiled_plan.required_tables) == {
        "entity_a",
        "entity_b_version",
        "entity_b",
        "bridge_contact",
        "entity_c_site",
        "entity_c",
    }
    assert compiled_plan.base_group_by == ["entity_c.id"]
    assert compiled_plan.intermediate_alias == "metric_avg_a_per_c"
    assert compiled_plan.confidence > 0.70
    assert compiled_plan.candidate_plan_set is not None
    assert compiled_plan.candidate_plan_set.selected_index == 0


def test_compile_semantic_plan_uses_query_form_for_ranking(tmp_path: Path) -> None:
    config = _config(tmp_path)
    compiled_plan = compile_semantic_plan(
        _build_plan(query=RANKING_QUERY),
        RANKING_QUERY,
        config=config,
        pruned_schema=_build_pruned_schema(),
    )

    assert compiled_plan.intent == "ranking"
    assert compiled_plan.measure is not None
    assert compiled_plan.measure.name == "metric_count_a"
    assert compiled_plan.group_by == ["entity_c.display_name"]
    assert compiled_plan.final_group_by == ["entity_c.display_name"]
    assert compiled_plan.ranking == PlanRanking(limit=5, direction="desc")
    assert compiled_plan.time_filter == PlanTimeFilter(
        field="entity_a.created_at",
        operator=">=",
        value="today - 1 year",
        resolved_expressions={
            "tsql": "DATEADD(year, -1, CAST(GETDATE() AS date))",
            "postgres": "(CURRENT_DATE - INTERVAL '1 year')",
        },
    )
    assert compiled_plan.post_aggregation is None
    assert compiled_plan.derived_metric_ref is None
    assert compiled_plan.candidate_plan_set is not None
    assert compiled_plan.candidate_plan_set.candidates[0].group_by == ["entity_c.display_name"]


def test_compile_semantic_plan_adds_join_path_for_parent_time_field(tmp_path: Path) -> None:
    config = _config(tmp_path)
    assets: list[MatchedAsset] = []
    for asset in _build_plan_assets():
        if asset.asset.kind == "semantic_entities" and asset.asset.name == "entity_a":
            assets.append(
                _matched_asset(
                    "semantic_entities",
                    "entity_a",
                    {
                        "name": "entity_a",
                        "business_definition": "Primary operational record in the generic domain.",
                        "source_table": "entity_a",
                        "key": "entity_a.id",
                        "time_field": "entity_b.requested_at",
                    },
                )
            )
            continue
        assets.append(asset)

    assets_by_kind: dict[str, list[MatchedAsset]] = {}
    for asset in assets:
        assets_by_kind.setdefault(asset.asset.kind, []).append(asset)

    compiled_plan = compile_semantic_plan(
        SemanticPlan(
            query=SCALAR_TIME_QUERY,
            assets_by_kind=assets_by_kind,
            all_assets=assets,
            pruned_tables=tuple(sorted(_build_pruned_schema())),
            diagnostics={"synonym_entities_detected": ["entity_a"]},
        ),
        SCALAR_TIME_QUERY,
        config=config,
        pruned_schema=_build_pruned_schema(),
    )

    assert compiled_plan.time_filter == PlanTimeFilter(
        field="entity_b.requested_at",
        operator=">=",
        value="today - 1 year",
        resolved_expressions={
            "tsql": "DATEADD(year, -1, CAST(GETDATE() AS date))",
            "postgres": "(CURRENT_DATE - INTERVAL '1 year')",
        },
    )
    assert compiled_plan.join_path == [
        "entity_a.entity_b_version_id = entity_b_version.id",
        "entity_b_version.entity_b_id = entity_b.id",
    ]
    assert compiled_plan.required_tables == ["entity_a", "entity_b_version", "entity_b"]


def test_compile_semantic_plan_promotes_unresolved_time_filter_to_blocking_issue(tmp_path: Path) -> None:
    config = _config(tmp_path)
    assets = [
        asset
        for asset in _build_plan_assets()
        if not (asset.asset.kind == "semantic_dimensions" and asset.asset.name in {"entity_a_created_at", "entity_b_requested_at"})
    ]
    assets_by_kind: dict[str, list[MatchedAsset]] = {}
    for asset in assets:
        assets_by_kind.setdefault(asset.asset.kind, []).append(asset)

    compiled_plan = compile_semantic_plan(
        SemanticPlan(
            query=SCALAR_TIME_QUERY,
            assets_by_kind=assets_by_kind,
            all_assets=assets,
            pruned_tables=tuple(sorted(_build_pruned_schema())),
            diagnostics={"synonym_entities_detected": ["entity_a"]},
        ),
        SCALAR_TIME_QUERY,
        config=config,
        pruned_schema=_build_pruned_schema(),
    )

    assert compiled_plan.time_filter is None
    assert "time_expression_found_but_no_time_field_on_entity_a" in compiled_plan.warnings
    assert any(issue.code == "time_filter_unresolved" and issue.severity == "error" for issue in compiled_plan.issues)


def test_compile_semantic_plan_adds_join_path_for_selected_filter_field(tmp_path: Path) -> None:
    config = _config(tmp_path)
    assets = list(_build_plan_assets())
    assets.append(
        _matched_asset(
            "semantic_filters",
            "by_status_b",
            {
                "name": "by_status_b",
                "field": "status_b.id",
                "operator": "equals",
                "synonyms": ["status b"],
            },
        )
    )

    assets_by_kind: dict[str, list[MatchedAsset]] = {}
    for asset in assets:
        assets_by_kind.setdefault(asset.asset.kind, []).append(asset)

    compiled_plan = compile_semantic_plan(
        SemanticPlan(
            query=SCALAR_FILTER_QUERY,
            assets_by_kind=assets_by_kind,
            all_assets=assets,
            pruned_tables=tuple(sorted(_build_pruned_schema())),
            diagnostics={"synonym_entities_detected": ["entity_a"]},
        ),
        SCALAR_FILTER_QUERY,
        config=config,
        pruned_schema=_build_pruned_schema(),
    )

    assert [
        (selected_filter.field, selected_filter.operator, selected_filter.value) for selected_filter in compiled_plan.selected_filters
    ] == [("status_b.id", "=", "7")]
    assert compiled_plan.join_path == [
        "entity_a.entity_b_version_id = entity_b_version.id",
        "entity_b_version.entity_b_id = entity_b.id",
        "entity_b.status_b_id = status_b.id",
    ]
    assert compiled_plan.required_tables == ["entity_a", "entity_b_version", "entity_b", "status_b"]


def test_compile_semantic_plan_ranking_does_not_inherit_derived_metric_shape_by_base_group_by_match(tmp_path: Path) -> None:
    config = _config(tmp_path)
    assets = [
        asset for asset in _build_plan_assets() if not (asset.asset.kind == "semantic_dimensions" and asset.asset.name == "entity_c_label")
    ]
    assets.append(
        _matched_asset(
            "semantic_dimensions",
            "entity_c_id_dimension",
            {
                "name": "entity_c_id_dimension",
                "entity": "entity_c",
                "source": "entity_c.id",
                "type": "id",
            },
        )
    )
    assets_by_kind: dict[str, list[MatchedAsset]] = {}
    for asset in assets:
        assets_by_kind.setdefault(asset.asset.kind, []).append(asset)

    compiled_plan = compile_semantic_plan(
        SemanticPlan(
            query=RANKING_QUERY,
            assets_by_kind=assets_by_kind,
            all_assets=assets,
            pruned_tables=tuple(sorted(_build_pruned_schema())),
            diagnostics={"synonym_entities_detected": ["entity_a", "entity_c"]},
        ),
        RANKING_QUERY,
        config=config,
        pruned_schema=_build_pruned_schema(),
    )

    assert compiled_plan.intent == "ranking"
    assert compiled_plan.group_by == ["entity_c.id"]
    assert compiled_plan.post_aggregation is None
    assert compiled_plan.derived_metric_ref is None
    assert compiled_plan.base_group_by == []
    assert compiled_plan.intermediate_alias is None


def test_compile_semantic_plan_ranking_prefers_descriptive_dimension_for_entity_phrase(tmp_path: Path) -> None:
    config = _config(tmp_path)
    compiled_plan = compile_semantic_plan(
        _build_plan(query=RANKING_SYNONYM_QUERY),
        RANKING_SYNONYM_QUERY,
        config=config,
        pruned_schema=_build_pruned_schema(),
    )

    assert compiled_plan.intent == "ranking"
    assert compiled_plan.measure is not None
    assert compiled_plan.measure.name == "metric_count_a"
    assert compiled_plan.group_by == ["entity_c.display_name"]
    assert compiled_plan.final_group_by == ["entity_c.display_name"]
    assert compiled_plan.ranking == PlanRanking(limit=5, direction="desc")
    assert compiled_plan.post_aggregation is None
    assert compiled_plan.derived_metric_ref is None


def test_compile_semantic_plan_ranking_usa_heuristica_desde_config_unificada(tmp_path: Path) -> None:
    assets = _build_plan_assets()
    assets.append(
        _matched_asset(
            "semantic_dimensions",
            "entity_c_code",
            {
                "name": "entity_c_code",
                "entity": "entity_c",
                "source": "entity_c.code",
                "type": "string",
            },
        )
    )
    assets_by_kind: dict[str, list[MatchedAsset]] = {}
    for asset in assets:
        assets_by_kind.setdefault(asset.asset.kind, []).append(asset)

    compiler_rules_path = _write_compiler_rules_config(
        tmp_path,
        mutate=lambda compiler_rules: compiler_rules["ranking_dimension_preference"].update(
            {
                "positive_token_weights": {"code": 10.0},
                "negative_tokens": ["display", "label", "name"],
                "negative_token_penalty": 1.5,
                "string_type_bonus": 0.0,
                "entity_prefix_bonus": 0.0,
                "same_table_bonus": 0.0,
            }
        ),
    )
    config = SemanticResolverConfig(
        rules_path=str(_write_rules(tmp_path)),
        compiler_rules_path=compiler_rules_path,
    )

    compiled_plan = compile_semantic_plan(
        SemanticPlan(
            query=RANKING_SYNONYM_QUERY,
            assets_by_kind=assets_by_kind,
            all_assets=assets,
            pruned_tables=tuple(sorted(_build_pruned_schema())),
            diagnostics={"synonym_entities_detected": ["entity_a", "entity_c"]},
        ),
        RANKING_SYNONYM_QUERY,
        config=config,
        pruned_schema=_build_pruned_schema(),
    )

    assert compiled_plan.group_by == ["entity_c.code"]
    assert compiled_plan.final_group_by == ["entity_c.code"]


def test_compile_semantic_plan_usa_fallbacks_desde_config_unificada(tmp_path: Path) -> None:
    compiler_rules_path = _write_compiler_rules_config(
        tmp_path,
        mutate=lambda compiler_rules: compiler_rules["plan_fallbacks"].update(
            {
                "base_entity": "fallback_entity",
                "base_table": "fallback_table",
            }
        ),
    )
    config = SemanticResolverConfig(
        rules_path=str(_write_rules(tmp_path)),
        compiler_rules_path=compiler_rules_path,
    )

    compiled_plan = compile_semantic_plan(
        SemanticPlan(
            query="que datos hay?",
            assets_by_kind={},
            all_assets=[],
            pruned_tables=(),
            diagnostics={},
        ),
        "que datos hay?",
        config=config,
        pruned_schema={},
    )

    assert compiled_plan.base_entity == "fallback_entity"
    assert compiled_plan.grain == "fallback_table.id"


def test_compile_semantic_plan_applies_named_derived_metric_shape(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    assets = [
        *_build_plan_assets(),
        _matched_asset(
            "semantic_metrics",
            "metric_balance_a",
            {
                "name": "metric_balance_a",
                "entity": "entity_a",
                "formula": "coalesce(sum(entity_a.amount_total), 0) - coalesce(sum(entity_b.id), 0)",
            },
            embedding_score=0.95,
        ),
    ]
    assets_by_kind: dict[str, list[MatchedAsset]] = {}
    for asset in assets:
        assets_by_kind.setdefault(asset.asset.kind, []).append(asset)

    compiled_plan = compile_semantic_plan(
        SemanticPlan(
            query="what is metric_balance_a",
            assets_by_kind=assets_by_kind,
            all_assets=assets,
            pruned_tables=tuple(sorted(_build_pruned_schema())),
            diagnostics={"synonym_entities_detected": ["entity_a"]},
        ),
        "what is metric_balance_a",
        config=config,
        pruned_schema=_build_pruned_schema(),
    )

    assert compiled_plan.intent == "lookup"
    assert compiled_plan.measure is not None
    assert compiled_plan.measure.name == "metric_balance_a"
    assert compiled_plan.measure.formula == "entity_a.amount_total - coalesce(sum(entity_b.id), 0)"
    assert compiled_plan.derived_metric_ref == "metric_balance_a"
    assert compiled_plan.post_aggregation == PlanPostAggregation(function="sum", over="grouped_measure")
    assert compiled_plan.base_group_by == ["entity_a.id", "entity_a.amount_total"]
    assert compiled_plan.intermediate_alias == "metric_balance_a"


def test_compile_semantic_plan_warns_on_unmapped_status_qualifier(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    baseline_plan = compile_semantic_plan(
        _build_plan(query=QUERY),
        QUERY,
        config=config,
        pruned_schema=_build_pruned_schema(),
    )

    compiled_plan = compile_semantic_plan(
        _build_plan(query=UNMAPPED_STATUS_QUERY),
        UNMAPPED_STATUS_QUERY,
        config=config,
        pruned_schema=_build_pruned_schema(),
    )

    assert compiled_plan.measure is not None
    assert compiled_plan.measure.name == "metric_count_a"
    assert "unmapped_qualifier_in_question:rejected" in compiled_plan.warnings
    assert compiled_plan.confidence < baseline_plan.confidence


def test_compile_semantic_plan_uses_status_a_for_active_entity_a(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    compiled_plan = compile_semantic_plan(
        _build_plan(query=ACTIVE_QUERY),
        ACTIVE_QUERY,
        config=config,
        pruned_schema=_build_pruned_schema(),
    )

    assert compiled_plan.measure is not None
    assert compiled_plan.measure.name == "metric_count_a_active"
    assert "status_a.name = 'Active'" in compiled_plan.measure.formula
    assert "entity_a.status_a_id = status_a.id" in compiled_plan.join_path
    assert compiled_plan.time_filter == PlanTimeFilter(
        field="entity_a.created_at",
        operator=">=",
        value="today - 1 year",
        resolved_expressions={
            "tsql": "DATEADD(year, -1, CAST(GETDATE() AS date))",
            "postgres": "(CURRENT_DATE - INTERVAL '1 year')",
        },
    )
    assert "unmapped_qualifier_in_question:active" not in compiled_plan.warnings


def test_compile_semantic_plan_prefers_loss_base_metric_over_ratio_metric(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    compiled_plan = compile_semantic_plan(
        _build_plan(query=LOSS_QUERY),
        LOSS_QUERY,
        config=config,
        pruned_schema=_build_pruned_schema(),
    )

    assert compiled_plan.measure is not None
    assert compiled_plan.measure.name == "metric_count_b_lost"
    assert "status_b.name = 'Archived'" in compiled_plan.measure.formula
    assert "entity_b.status_b_id = status_b.id" in compiled_plan.join_path
    assert compiled_plan.derived_metric_ref == "metric_avg_b_lost_per_c"
    assert compiled_plan.join_path_hint == "entity_c_to_b_via_site"
    assert "status_b" in compiled_plan.required_tables
