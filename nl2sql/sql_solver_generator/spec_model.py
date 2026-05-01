#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, field_validator

from nl2sql.utils.decision_models import StrictModel

QueryType = Literal[
    "scalar_metric",
    "grouped_metric",
    "derived_metric",
    "ranking",
    "detail_listing",
]
Dialect = Literal["tsql", "postgres"]
JoinType = Literal["inner", "left"]
JoinSource = Literal["semantic_join_paths", "semantic_relationships", "physical_fk"]
AggregationLevel = Literal["row", "entity", "grouped", "derived_two_level"]
PostAggregation = Literal["avg", "sum", "count", "max", "min", "rank_top_n", "percentiles", "none"]


class SQLMetric(StrictModel):
    """Metrica ya compilada para consumo del solver SQL."""

    name: str
    formula: str
    entity: str
    aggregation_level: AggregationLevel


class SQLDimension(StrictModel):
    """Dimension declarada en el contrato SQL tipado."""

    name: str
    source: str
    entity: str
    type: str


class SQLFilter(StrictModel):
    """Filtro de negocio o de seguridad ya resuelto para el solver."""

    name: str
    field: str
    operator: str
    value: Any
    source: str = "semantic_filter"


class SQLTimeFilter(StrictModel):
    """Filtro temporal resuelto para el dialecto activo."""

    field: str
    operator: str
    value: str
    resolved_expression: str


class SQLJoinEdge(StrictModel):
    """Arista individual del plan de joins que el LLM debe respetar."""

    left_table: str
    left_column: str
    right_table: str
    right_column: str
    join_type: JoinType = "inner"


class SQLJoinPlan(StrictModel):
    """Ruta de joins permitida para la consulta final."""

    path_name: str
    joins: list[SQLJoinEdge] = Field(default_factory=list)
    source: JoinSource = "semantic_join_paths"


class SQLQuerySpec(StrictModel):
    """Contrato SQL compilado a partir del plan semantico."""

    query_type: QueryType
    dialect: Dialect
    base_entity: str
    base_table: str
    selected_metrics: list[SQLMetric] = Field(default_factory=list)
    selected_dimensions: list[SQLDimension] = Field(default_factory=list)
    selected_filters: list[SQLFilter] = Field(default_factory=list)
    time_filter: SQLTimeFilter | None = None
    join_plan: list[SQLJoinPlan] = Field(default_factory=list)
    base_group_by: list[str] = Field(default_factory=list)
    final_group_by: list[str] = Field(default_factory=list)
    post_aggregation: PostAggregation = "none"
    limit: int | None = None
    warnings: list[str] = Field(default_factory=list)


class SQLGenerationPayload(StrictModel):
    """Salida JSON exacta que debe producir el LLM generador local."""

    final_sql: str = Field(min_length=1)

    @field_validator("final_sql")
    @classmethod
    def _strip_final_sql(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("final_sql no puede estar vacio")
        return cleaned
