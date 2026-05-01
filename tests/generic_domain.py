#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Helpers compartidos para tests con dominio generico."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from nl2sql.semantic_resolver.plan_model import CompiledSemanticPlan, PlanMeasure, PlanPostAggregation, PlanTimeFilter
from nl2sql.utils.semantic_contract import SemanticContract, load_semantic_contract

_FIXTURE_PATH = Path(__file__).resolve().parent / "fixtures" / "generic_domain.yaml"


def load_generic_domain_fixture() -> dict[str, Any]:
    """Carga el fixture YAML del dominio generico usado por la suite."""

    payload = yaml.safe_load(_FIXTURE_PATH.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError("generic_domain.yaml invalido: la raiz debe ser un mapping")
    return payload


def generic_semantic_contract_payload() -> dict[str, Any]:
    """Devuelve el payload minimo compatible con ``load_semantic_contract``."""

    fixture = load_generic_domain_fixture()
    return {"semantic_contract": deepcopy(fixture["semantic_contract"])}


def generic_semantic_contract() -> SemanticContract:
    """Valida y retorna el contrato semantico generico de tests."""

    return load_semantic_contract(generic_semantic_contract_payload())


def generic_schema_tables() -> dict[str, Any]:
    """Entrega una copia aislada del mini esquema generico."""

    fixture = load_generic_domain_fixture()
    schema = fixture.get("schema", {})
    tables = schema.get("tables", {}) if isinstance(schema, dict) else {}
    if not isinstance(tables, dict):
        raise ValueError("generic_domain.yaml invalido: schema.tables debe ser un mapping")
    return deepcopy(tables)


def build_compiled_plan_active_a_per_c() -> CompiledSemanticPlan:
    """Construye un plan compilado de dos niveles sobre el dominio generico."""

    return CompiledSemanticPlan(
        query="cual es el promedio de entidades_a con estado activo por entidad_c en el ultimo ano?",
        semantic_model="model_alpha",
        intent="post_aggregated_metric",
        base_entity="entity_a",
        grain="entity_a.id",
        measure=PlanMeasure(
            name="metric_count_a_active",
            formula="count_distinct(case when status_a.name = 'Active' then entity_a.id end)",
            source_table="entity_a",
        ),
        group_by=["entity_c.id"],
        time_filter=PlanTimeFilter(
            field="entity_a.created_at",
            operator=">=",
            value="today - 1 year",
            resolved_expressions={"tsql": "DATEADD(YEAR, -1, CAST(GETDATE() AS DATE))"},
        ),
        post_aggregation=PlanPostAggregation(function="avg", over="grouped_measure"),
        join_path=[
            "entity_a.entity_b_version_id = entity_b_version.id",
            "entity_b_version.entity_b_id = entity_b.id",
            "entity_b.bridge_contact_id = bridge_contact.id",
            "bridge_contact.entity_c_site_id = entity_c_site.id",
            "entity_c_site.entity_c_id = entity_c.id",
            "entity_a.status_a_id = status_a.id",
        ],
        required_tables=[
            "entity_a",
            "entity_b_version",
            "entity_b",
            "bridge_contact",
            "entity_c_site",
            "entity_c",
            "status_a",
        ],
        warnings=["population_scope_defaulted_to_active_entities_only"],
        confidence=0.94,
        join_path_hint="entity_c_to_a_via_b",
        derived_metric_ref="metric_avg_a_active_per_c",
        population_scope="active_entities_only",
        base_group_by=["entity_c.id"],
        intermediate_alias="metric_avg_a_active_per_c",
    )


def build_semantic_plan_mapping_active_a_per_c() -> dict[str, Any]:
    """Serializa el plan generico al shape usado por prompt/verifier tests."""

    compiled = build_compiled_plan_active_a_per_c()
    return {
        "compiled_plan": {
            "query": compiled.query,
            "semantic_model": compiled.semantic_model,
            "intent": compiled.intent,
            "base_entity": compiled.base_entity,
            "grain": compiled.grain,
            "measure": {
                "name": compiled.measure.name if compiled.measure else "",
                "formula": compiled.measure.formula if compiled.measure else "",
                "source_table": compiled.measure.source_table if compiled.measure else "",
            },
            "group_by": list(compiled.group_by),
            "time_filter": {
                "field": compiled.time_filter.field,
                "operator": compiled.time_filter.operator,
                "value": compiled.time_filter.value,
            },
            "post_aggregation": {
                "function": compiled.post_aggregation.function if compiled.post_aggregation else "avg",
                "over": compiled.post_aggregation.over if compiled.post_aggregation else "grouped_measure",
            },
            "join_path": list(compiled.join_path),
            "required_tables": list(compiled.required_tables),
            "warnings": list(compiled.warnings),
            "join_path_hint": compiled.join_path_hint,
            "derived_metric_ref": compiled.derived_metric_ref,
            "population_scope": compiled.population_scope,
            "base_group_by": list(compiled.base_group_by),
            "intermediate_alias": compiled.intermediate_alias,
        }
    }
