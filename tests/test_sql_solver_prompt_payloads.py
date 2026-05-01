#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import yaml

from nl2sql.sql_solver_generator.sql_generator import (
    build_query_spec_from_plan,
    prepare_prompt_payloads,
    resolve_required_tables,
)


def test_prepare_prompt_payloads_reduce_context_without_losing_relevant_signal() -> None:
    semantic_plan = {
        "compiled_plan": {
            "query": "promedio de registros por entidad_c",
            "semantic_model": "model_alpha",
            "intent": "post_aggregated_metric",
            "base_entity": "entity_a",
            "grain": "entity_a.id",
            "measure": {
                "name": "metric_count_a",
                "formula": "count_distinct(entity_a.id)",
                "source_table": "entity_a",
            },
            "group_by": ["entity_c.id"],
            "time_filter": {
                "field": "entity_a.created_at",
                "operator": ">=",
                "value": "today - 1 year",
            },
            "join_path": [
                "entity_a.entity_b_version_id = entity_b_version.id",
                "entity_b_version.entity_b_id = entity_b.id",
                "entity_b.bridge_contact_id = bridge_contact.id",
                "bridge_contact.entity_c_site_id = entity_c_site.id",
                "entity_c_site.entity_c_id = entity_c.id",
            ],
            "required_tables": [
                "entity_a",
                "entity_b_version",
                "entity_b",
                "bridge_contact",
                "entity_c_site",
                "entity_c",
            ],
            "join_path_hint": "entity_c_to_a_via_b",
            "derived_metric_ref": "metric_avg_a_per_c",
        },
        "retrieved_candidates": {
            "query": "promedio de registros por entidad_c",
            "pruned_tables": [
                "entity_a",
                "entity_b_version",
                "entity_b",
                "bridge_contact",
                "entity_c_site",
                "entity_c",
                "noise_table",
            ],
            "assets_by_kind": {"semantic_metrics": [{"name": "noise_metric"}]},
        },
    }
    pruned_schema = {
        "entity_a": {
            "columns": {"id": {}, "created_at": {}, "entity_b_version_id": {}},
            "column_types": {
                "id": "BIGINT",
                "created_at": "DATETIME",
                "entity_b_version_id": "BIGINT",
            },
            "column_descriptions": {"id": "identifier"},
            "primary_keys": ["id"],
            "foreign_keys": [
                {
                    "column": "entity_b_version_id",
                    "ref_table": "entity_b_version",
                    "ref_col": "id",
                }
            ],
            "description": "fact table",
            "role": "fact",
        },
        "entity_b_version": {
            "columns": {"id": {}, "entity_b_id": {}},
            "column_types": {"id": "BIGINT", "entity_b_id": "BIGINT"},
            "primary_keys": ["id"],
            "foreign_keys": [{"column": "entity_b_id", "ref_table": "entity_b", "ref_col": "id"}],
            "role": "bridge",
        },
        "entity_b": {
            "columns": {"id": {}, "bridge_contact_id": {}},
            "column_types": {"id": "BIGINT", "bridge_contact_id": "BIGINT"},
            "primary_keys": ["id"],
            "foreign_keys": [
                {
                    "column": "bridge_contact_id",
                    "ref_table": "bridge_contact",
                    "ref_col": "id",
                }
            ],
            "role": "bridge",
        },
        "bridge_contact": {
            "columns": {"id": {}, "entity_c_site_id": {}},
            "column_types": {"id": "BIGINT", "entity_c_site_id": "BIGINT"},
            "primary_keys": ["id"],
            "foreign_keys": [
                {
                    "column": "entity_c_site_id",
                    "ref_table": "entity_c_site",
                    "ref_col": "id",
                }
            ],
            "role": "bridge",
        },
        "entity_c_site": {
            "columns": {"id": {}, "entity_c_id": {}},
            "column_types": {"id": "BIGINT", "entity_c_id": "BIGINT"},
            "primary_keys": ["id"],
            "foreign_keys": [{"column": "entity_c_id", "ref_table": "entity_c", "ref_col": "id"}],
            "role": "bridge",
        },
        "entity_c": {
            "columns": {"id": {}},
            "column_types": {"id": "BIGINT"},
            "column_descriptions": {"id": "identifier"},
            "primary_keys": ["id"],
            "foreign_keys": [],
            "description": "grouping entity",
            "role": "dimension",
        },
        "noise_table": {
            "columns": {"id": {}},
            "column_types": {"id": "BIGINT"},
            "column_descriptions": {"id": "identifier"},
            "primary_keys": ["id"],
            "foreign_keys": [],
            "description": "noise",
            "role": "noise",
        },
    }

    compact_plan, compact_schema = prepare_prompt_payloads(
        semantic_plan=semantic_plan,
        pruned_schema=pruned_schema,
    )

    assert "assets_by_kind" not in yaml.safe_dump(compact_plan, sort_keys=False, allow_unicode=True)
    assert sorted(compact_schema.keys()) == [
        "bridge_contact",
        "entity_a",
        "entity_b",
        "entity_b_version",
        "entity_c",
        "entity_c_site",
    ]
    assert compact_schema["entity_a"]["columns"]["created_at"] == "DATETIME"
    assert "column_descriptions" not in yaml.safe_dump(compact_schema, sort_keys=False, allow_unicode=True)


def test_build_spec_payload_sets_default_limit_for_lookup_detail_listing() -> None:
    spec = build_query_spec_from_plan(
        {
            "compiled_plan": {
                "intent": "lookup",
                "base_entity": "entity_c",
            }
        },
        "tsql",
    )

    payload = spec.model_dump(mode="python")
    assert spec.query_type == "detail_listing"
    assert spec.limit == 100
    assert spec.selected_metrics == []
    assert "assumptions" not in payload
    assert "ordering" not in payload


def test_build_spec_payload_uses_ranking_limit_from_compiled_plan() -> None:
    spec = build_query_spec_from_plan(
        {
            "compiled_plan": {
                "intent": "ranking",
                "base_entity": "entity_a",
                "measure": {
                    "name": "metric_count_a",
                    "formula": "count_distinct(entity_a.id)",
                    "source_table": "entity_a",
                },
                "group_by": ["entity_c.display_name"],
                "ranking": {"limit": 5, "direction": "desc"},
            }
        },
        "tsql",
    )

    assert spec.query_type == "ranking"
    assert spec.limit == 5


def test_build_spec_payload_keeps_metric_for_lookup_with_measure() -> None:
    spec = build_query_spec_from_plan(
        {
            "compiled_plan": {
                "intent": "lookup",
                "base_entity": "entity_c",
                "measure": {
                    "name": "metric_amount_c",
                    "formula": "sum(entity_c.amount_total)",
                    "source_table": "entity_c",
                },
            }
        },
        "tsql",
    )

    assert spec.query_type == "scalar_metric"
    assert spec.limit is None
    assert [metric.name for metric in spec.selected_metrics] == ["metric_amount_c"]


def test_build_spec_payload_resolves_semantic_filter_value_from_query() -> None:
    spec = build_query_spec_from_plan(
        {
            "compiled_plan": {
                "query": "cual es metric_amount_c con business code ABC-123",
                "intent": "simple_metric",
                "base_entity": "entity_c",
                "measure": {
                    "name": "metric_amount_c",
                    "formula": "sum(entity_c.amount_total)",
                    "source_table": "entity_c",
                },
            },
            "retrieved_candidates": {
                "assets_by_kind": {
                    "semantic_filters": [
                        {
                            "asset": {
                                "name": "by_business_code",
                                "payload": {
                                    "name": "by_business_code",
                                    "field": "entity_c.business_code",
                                    "operator": "equals",
                                    "value_type": "string",
                                    "synonyms": ["business code"],
                                },
                            }
                        }
                    ]
                }
            },
        },
        "tsql",
    )

    assert [(filter_obj.field, filter_obj.operator, filter_obj.value) for filter_obj in spec.selected_filters] == [
        ("entity_c.business_code", "=", "ABC-123")
    ]


def test_build_spec_payload_skips_group_by_alias_without_explicit_filter_value() -> None:
    spec = build_query_spec_from_plan(
        {
            "compiled_plan": {
                "query": "cual es el promedio de cotizaciones perdidas por cliente en el ultimo ano?",
                "intent": "post_aggregated_metric",
                "base_entity": "cotizacion",
                "measure": {
                    "name": "cotizaciones_perdidas",
                    "formula": "count_distinct(case when estado_cotizacion.nombre = 'Lost' then cotizacion.id end)",
                    "source_table": "cotizacion",
                },
                "group_by": ["cliente.id"],
                "base_group_by": ["cliente.id"],
            },
            "retrieved_candidates": {
                "assets_by_kind": {
                    "semantic_filters": [
                        {
                            "asset": {
                                "name": "por_cliente",
                                "payload": {
                                    "name": "por_cliente",
                                    "field": "cliente.id",
                                    "operator": "equals",
                                },
                            }
                        }
                    ]
                }
            },
        },
        "tsql",
    )

    assert spec.selected_filters == []


def test_build_spec_payload_keeps_explicit_filter_value_on_grouped_field() -> None:
    spec = build_query_spec_from_plan(
        {
            "compiled_plan": {
                "query": "cual es el promedio de cotizaciones perdidas por cliente = ACME en el ultimo ano?",
                "intent": "post_aggregated_metric",
                "base_entity": "cotizacion",
                "measure": {
                    "name": "cotizaciones_perdidas",
                    "formula": "count_distinct(case when estado_cotizacion.nombre = 'Lost' then cotizacion.id end)",
                    "source_table": "cotizacion",
                },
                "group_by": ["cliente.id"],
                "base_group_by": ["cliente.id"],
            },
            "retrieved_candidates": {
                "assets_by_kind": {
                    "semantic_filters": [
                        {
                            "asset": {
                                "name": "por_cliente",
                                "payload": {
                                    "name": "por_cliente",
                                    "field": "cliente.id",
                                    "operator": "equals",
                                },
                            }
                        }
                    ]
                }
            },
        },
        "tsql",
    )

    assert [(filter_obj.field, filter_obj.operator, filter_obj.value) for filter_obj in spec.selected_filters] == [
        ("cliente.id", "=", "ACME")
    ]


def test_build_spec_payload_uses_compiled_selected_filters_when_present() -> None:
    spec = build_query_spec_from_plan(
        {
            "compiled_plan": {
                "query": "metric_count_a with status b = 7",
                "intent": "simple_metric",
                "base_entity": "entity_a",
                "measure": {
                    "name": "metric_count_a",
                    "formula": "count_distinct(entity_a.id)",
                    "source_table": "entity_a",
                },
                "selected_filters": [
                    {
                        "name": "by_status_b",
                        "field": "status_b.id",
                        "operator": "=",
                        "value": "7",
                        "source": "semantic_filter",
                    }
                ],
            }
        },
        "tsql",
    )

    assert [(filter_obj.field, filter_obj.operator, filter_obj.value) for filter_obj in spec.selected_filters] == [
        ("status_b.id", "=", "7")
    ]


def test_prepare_prompt_payloads_keep_support_table_needed_by_measure_formula() -> None:
    semantic_plan = {
        "compiled_plan": {
            "query": "promedio de registros_b archivados por entidad_c",
            "semantic_model": "model_beta",
            "intent": "post_aggregated_metric",
            "base_entity": "entity_b",
            "grain": "entity_b.id",
            "measure": {
                "name": "metric_count_b_lost",
                "formula": "count_distinct(case when status_b.name = 'Archived' then entity_b.id end)",
                "source_table": "entity_b",
            },
            "group_by": ["entity_c.id"],
            "time_filter": {
                "field": "entity_b.requested_at",
                "operator": ">=",
                "value": "today - 1 year",
            },
            "join_path": [
                "entity_b.bridge_contact_id = bridge_contact.id",
                "bridge_contact.entity_c_site_id = entity_c_site.id",
                "entity_c_site.entity_c_id = entity_c.id",
                "entity_b.status_b_id = status_b.id",
            ],
            "required_tables": [
                "entity_b",
                "bridge_contact",
                "entity_c_site",
                "entity_c",
                "status_b",
            ],
            "join_path_hint": "entity_c_to_b_via_site",
            "derived_metric_ref": "metric_avg_b_lost_per_c",
        }
    }
    pruned_schema = {
        "entity_b": {
            "columns": {
                "id": {},
                "bridge_contact_id": {},
                "status_b_id": {},
                "requested_at": {},
            },
            "column_types": {
                "id": "BIGINT",
                "bridge_contact_id": "BIGINT",
                "status_b_id": "BIGINT",
                "requested_at": "DATE",
            },
            "primary_keys": ["id"],
            "foreign_keys": [
                {
                    "column": "bridge_contact_id",
                    "ref_table": "bridge_contact",
                    "ref_col": "id",
                },
                {"column": "status_b_id", "ref_table": "status_b", "ref_col": "id"},
            ],
            "role": "fact",
        },
        "bridge_contact": {
            "columns": {"id": {}, "entity_c_site_id": {}},
            "column_types": {"id": "BIGINT", "entity_c_site_id": "BIGINT"},
            "primary_keys": ["id"],
            "foreign_keys": [
                {
                    "column": "entity_c_site_id",
                    "ref_table": "entity_c_site",
                    "ref_col": "id",
                }
            ],
            "role": "bridge",
        },
        "entity_c_site": {
            "columns": {"id": {}, "entity_c_id": {}},
            "column_types": {"id": "BIGINT", "entity_c_id": "BIGINT"},
            "primary_keys": ["id"],
            "foreign_keys": [{"column": "entity_c_id", "ref_table": "entity_c", "ref_col": "id"}],
            "role": "bridge",
        },
        "entity_c": {
            "columns": {"id": {}},
            "column_types": {"id": "BIGINT"},
            "primary_keys": ["id"],
            "foreign_keys": [],
            "role": "dimension",
        },
        "status_b": {
            "columns": {"id": {}, "name": {}},
            "column_types": {"id": "BIGINT", "name": "VARCHAR"},
            "primary_keys": ["id"],
            "foreign_keys": [],
            "role": "dimension",
        },
    }

    _compact_plan, compact_schema = prepare_prompt_payloads(
        semantic_plan=semantic_plan,
        pruned_schema=pruned_schema,
    )

    assert sorted(compact_schema.keys()) == [
        "bridge_contact",
        "entity_b",
        "entity_c",
        "entity_c_site",
        "status_b",
    ]
    assert any(fk["ref_table"] == "status_b" for fk in compact_schema["entity_b"]["foreign_keys"])


def test_prepare_prompt_payloads_infer_required_tables_when_field_is_absent() -> None:
    semantic_plan = {
        "compiled_plan": {
            "query": "promedio de registros_b archivados por entidad_c",
            "intent": "post_aggregated_metric",
            "base_entity": "entity_b",
            "grain": "entity_b.id",
            "measure": {
                "name": "metric_count_b_lost",
                "formula": "count_distinct(case when status_b.name = 'Archived' then entity_b.id end)",
                "source_table": "entity_b",
            },
            "group_by": ["entity_c.id"],
            "time_filter": {
                "field": "entity_b.requested_at",
                "operator": ">=",
                "value": "today - 1 year",
            },
            "join_path": [
                "entity_b.bridge_contact_id=bridge_contact.id",
                "bridge_contact.entity_c_site_id=entity_c_site.id",
                "entity_c_site.entity_c_id=entity_c.id",
                "entity_b.status_b_id=status_b.id",
            ],
        }
    }
    pruned_schema = {
        table_name: {
            "columns": {"id": {}},
            "column_types": {"id": "BIGINT"},
            "primary_keys": ["id"],
            "foreign_keys": [],
        }
        for table_name in (
            "entity_b",
            "status_b",
            "bridge_contact",
            "entity_c_site",
            "entity_c",
            "noise_table",
        )
    }

    _compact_plan, compact_schema = prepare_prompt_payloads(
        semantic_plan=semantic_plan,
        pruned_schema=pruned_schema,
    )

    assert sorted(compact_schema) == [
        "bridge_contact",
        "entity_b",
        "entity_c",
        "entity_c_site",
        "status_b",
    ]
    assert "noise_table" not in compact_schema


def test_resolve_required_tables_preserves_explicit_empty_list() -> None:
    required_tables = resolve_required_tables(
        {"compiled_plan": {"required_tables": []}},
        {"entity_c": {"columns": {"id": {}}}, "noise_table": {"columns": {"id": {}}}},
    )

    assert required_tables == []


def test_prepare_prompt_payloads_include_solver_semantic_context_when_plan_is_uncertain() -> None:
    compact_plan, _compact_schema = prepare_prompt_payloads(
        semantic_plan={
            "compiled_plan": {
                "query": "top 5 de entity_c con mas entity_a en el ultimo ano",
                "intent": "ranking",
                "base_entity": "entity_a",
                "grain": "entity_a.id",
                "measure": {
                    "name": "metric_count_c",
                    "formula": "count_distinct(entity_c.id)",
                    "source_table": "entity_c",
                },
                "group_by": ["entity_c.display_name"],
                "ranking": {"limit": 5, "direction": "desc"},
                "confidence": 0.48,
                "verification": {
                    "is_semantically_aligned": False,
                    "failure_class": "recoverable_semantic_mismatch",
                    "repairability": "high",
                    "wrong_metric": "metric_count_c",
                    "suggested_measure": "metric_count_a",
                    "suggested_join_tables": ["entity_b", "bridge_contact"],
                    "confidence": 0.81,
                },
                "candidate_plan_set": {
                    "selected_index": 0,
                    "selection_rationale": "selected_highest_metric_score",
                    "candidates": [
                        {
                            "intent": "ranking",
                            "measure": {"name": "metric_count_c", "formula": "count_distinct(entity_c.id)", "source_table": "entity_c"},
                            "group_by": ["entity_c.display_name"],
                            "ranking": {"limit": 5, "direction": "desc"},
                            "confidence": 0.48,
                            "issues": ["not_selected"],
                        },
                        {
                            "intent": "ranking",
                            "measure": {"name": "metric_count_a", "formula": "count_distinct(entity_a.id)", "source_table": "entity_a"},
                            "group_by": ["entity_c.display_name"],
                            "ranking": {"limit": 5, "direction": "desc"},
                            "confidence": 0.74,
                            "issues": [],
                            "required_tables": ["entity_a", "entity_b", "bridge_contact", "entity_c_site", "entity_c"],
                        },
                    ],
                },
            }
        },
        pruned_schema={
            "entity_a": {"columns": {"id": {}}, "foreign_keys": []},
            "entity_c": {"columns": {"id": {}, "display_name": {}}, "foreign_keys": []},
        },
    )

    semantic_context = compact_plan["compiled_plan"]["solver_semantic_context"]
    assert semantic_context["selected_plan_confidence"] == 0.48
    assert semantic_context["verification"]["suggested_measure"] == "metric_count_a"
    assert semantic_context["candidate_plan_set"]["candidates"][1]["measure"]["name"] == "metric_count_a"


def test_prepare_prompt_payloads_omit_solver_semantic_context_when_plan_is_confident() -> None:
    compact_plan, _compact_schema = prepare_prompt_payloads(
        semantic_plan={
            "compiled_plan": {
                "query": "cual es el promedio de ordenes por cliente en el ultimo ano",
                "intent": "post_aggregated_metric",
                "base_entity": "orden_trabajo",
                "grain": "orden_trabajo.id",
                "measure": {
                    "name": "ordenes_trabajo_totales",
                    "formula": "count_distinct(orden_trabajo.id)",
                    "source_table": "orden_trabajo",
                },
                "group_by": ["cliente.id"],
                "time_filter": {
                    "field": "orden_trabajo.fecha_creacion",
                    "operator": ">=",
                    "value": "today - 1 year",
                },
                "post_aggregation": {"function": "avg", "over": "grouped_measure"},
                "join_path": [
                    "orden_trabajo.cotizacion_version_id = cotizacion_version.id",
                    "cotizacion_version.cotizacion_id = cotizacion.id",
                    "cotizacion.contacto_id = contacto_sucursal.id",
                    "contacto_sucursal.sucursal_id = sucursal.id",
                    "sucursal.cliente_id = cliente.id",
                ],
                "required_tables": [
                    "orden_trabajo",
                    "cotizacion_version",
                    "cotizacion",
                    "contacto_sucursal",
                    "sucursal",
                    "cliente",
                ],
                "confidence": 0.94,
                "candidate_plan_set": {
                    "selected_index": 0,
                    "selection_rationale": "selected_highest_metric_score",
                    "candidates": [
                        {
                            "intent": "post_aggregated_metric",
                            "measure": {
                                "name": "ordenes_trabajo_totales",
                                "formula": "count_distinct(orden_trabajo.id)",
                                "source_table": "orden_trabajo",
                            },
                            "group_by": ["cliente.id"],
                            "confidence": 0.94,
                            "issues": [],
                        },
                        {
                            "intent": "post_aggregated_metric",
                            "measure": {
                                "name": "clientes_totales",
                                "formula": "count_distinct(cliente.id)",
                                "source_table": "cliente",
                            },
                            "group_by": ["cliente.id"],
                            "confidence": 0.62,
                            "issues": ["not_selected"],
                        },
                    ],
                },
            }
        },
        pruned_schema={
            "orden_trabajo": {"columns": {"id": {}, "fecha_creacion": {}, "cotizacion_version_id": {}}, "foreign_keys": []},
            "cotizacion_version": {"columns": {"id": {}, "cotizacion_id": {}}, "foreign_keys": []},
            "cotizacion": {"columns": {"id": {}, "contacto_id": {}}, "foreign_keys": []},
            "contacto_sucursal": {"columns": {"id": {}, "sucursal_id": {}}, "foreign_keys": []},
            "sucursal": {"columns": {"id": {}, "cliente_id": {}}, "foreign_keys": []},
            "cliente": {"columns": {"id": {}}, "foreign_keys": []},
        },
    )

    assert "solver_semantic_context" not in compact_plan["compiled_plan"]
