#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass, field as dataclass_field
from typing import TYPE_CHECKING, Literal, Sequence, cast, get_args

from nl2sql.utils.decision_models import DecisionIssue
from nl2sql.utils.semantic_filters import SemanticFilterSelection

if TYPE_CHECKING:
    from .verification import SemanticVerificationResult

PlanIntent = Literal[
    "simple_metric",
    "post_aggregated_metric",
    "filtered_metric",
    "ranking",
    "lookup",
]
PLAN_INTENT_VALUES = cast(tuple[PlanIntent, ...], get_args(PlanIntent))
PLAN_INTENT_SET = frozenset(PLAN_INTENT_VALUES)

PostAggregationFunction = Literal["avg", "sum", "min", "max", "count", "ratio"]
RankingDirection = Literal["asc", "desc"]

# Alcance poblacional del plan. Distingue si una metrica promedio cuenta solo
# entidades con actividad (joins INNER) vs. incluir entidades con cero
# ocurrencias (LEFT JOIN desde la entidad dimension). Este campo es informativo
# para el solver SQL: el compilador lo fija por defecto con un warning si la
# query no es explicita.
PopulationScope = Literal["active_entities_only", "all_entities_including_zero"]


@dataclass(frozen=True)
class PlanMeasure:
    """Medida principal del plan ya resuelta a nivel semantico."""

    name: str
    formula: str
    source_table: str


@dataclass(frozen=True)
class PlanPostAggregation:
    """Operacion que ocurre despues de agrupar la medida base."""

    function: PostAggregationFunction
    over: Literal["grouped_measure", "rows"] = "grouped_measure"


@dataclass(frozen=True)
class PlanRanking:
    """Especificacion de ranking derivada de una forma declarativa o de la query."""

    limit: int
    direction: RankingDirection


@dataclass(frozen=True)
class PlanTimeFilter:
    """Filtro temporal ya aterrizado a un campo concreto.

    ``value`` mantiene la expresion canonica (p.ej. ``"today - 1 year"``).
    ``resolved_expressions`` mapea ``dialect.name`` -> expresion SQL lista
    para inyectar en el WHERE. El solver consume la entrada correspondiente
    al dialecto activo o, en su defecto, reinterpreta ``value`` mediante su
    propio ``SqlDialect``. Mantener un dict en vez de un campo por dialecto
    deja al modelo agnostico al motor de base de datos.
    """

    field: str
    operator: str
    value: str
    resolved_expressions: dict[str, str] = dataclass_field(default_factory=dict)


@dataclass(frozen=True)
class MetricScoreTrace:
    """Explica como se puntuo una metrica candidata del compilador."""

    metric_name: str
    components: dict[str, float] = dataclass_field(default_factory=dict)
    total_score: float = 0.0
    selected: bool = False
    rejected_reason: str | None = None


@dataclass(frozen=True)
class CandidatePlan:
    """Resumen serializable de un plan candidato compilado por el resolver."""

    intent: PlanIntent
    base_entity: str
    grain: str
    semantic_model: str | None = None
    measure: PlanMeasure | None = None
    group_by: list[str] = dataclass_field(default_factory=list)
    final_group_by: list[str] = dataclass_field(default_factory=list)
    selected_filters: list[SemanticFilterSelection] = dataclass_field(default_factory=list)
    time_filter: PlanTimeFilter | None = None
    ranking: PlanRanking | None = None
    post_aggregation: PlanPostAggregation | None = None
    join_path: list[str] = dataclass_field(default_factory=list)
    required_tables: list[str] = dataclass_field(default_factory=list)
    warnings: list[str] = dataclass_field(default_factory=list)
    confidence: float = 0.0
    score: float = 0.0
    issues: list[str] = dataclass_field(default_factory=list)
    join_path_hint: str | None = None
    derived_metric_ref: str | None = None
    population_scope: PopulationScope | None = None
    base_group_by: list[str] = dataclass_field(default_factory=list)
    intermediate_alias: str | None = None


@dataclass(frozen=True)
class CandidatePlanSet:
    """Coleccion ordenada de planes candidatos generados por el compilador."""

    selected_index: int = 0
    selection_rationale: str = ""
    candidates: list[CandidatePlan] = dataclass_field(default_factory=list)


@dataclass(frozen=True)
class CompiledSemanticPlan:
    """Plan compilado listo para consumir por un solver o SQL generator."""

    query: str
    semantic_model: str | None
    intent: PlanIntent
    base_entity: str
    grain: str
    measure: PlanMeasure | None
    group_by: list[str] = dataclass_field(default_factory=list)
    final_group_by: list[str] = dataclass_field(default_factory=list)
    selected_filters: list[SemanticFilterSelection] = dataclass_field(default_factory=list)
    time_filter: PlanTimeFilter | None = None
    ranking: PlanRanking | None = None
    post_aggregation: PlanPostAggregation | None = None
    join_path: list[str] = dataclass_field(default_factory=list)
    required_tables: list[str] = dataclass_field(default_factory=list)
    warnings: list[str] = dataclass_field(default_factory=list)
    confidence: float = 0.0
    # Hint de ruta canonica proveniente de semantic_join_paths. Cuando el
    # compilador detecta que la ruta final coincide con una ruta declarada en
    # semantic_rules.yaml, esta referencia se propaga para que el solver SQL
    # pueda reusar el orden de joins recomendado.
    join_path_hint: str | None = None
    # Referencia a una metrica derivada de dos niveles definida en
    # semantic_derived_metrics. Cuando no es None, el solver debe construir
    # CTE/subquery con base_measure agrupada por base_group_by y aplicar
    # post_aggregation al resultado intermedio.
    derived_metric_ref: str | None = None
    # Ver PopulationScope. Solo aplica a intents agregados y se usa para
    # decidir INNER vs LEFT JOIN en la salida SQL.
    population_scope: PopulationScope | None = None
    # Agrupacion del primer nivel de una metrica derivada (ej. COUNT(...) por
    # una entidad agrupadora). Cuando se trata de `post_aggregated_metric`, el solver debe
    # construir una CTE/subquery que agrupe por `base_group_by` antes de
    # aplicar `post_aggregation`. Para planes `simple_metric` puede quedar
    # vacio y el group_by equivalente sigue siendo el unico nivel.
    base_group_by: list[str] = dataclass_field(default_factory=list)
    # Alias del resultado intermedio producido por la CTE/subquery del primer
    # nivel. El solver usa este alias en el select exterior (por ejemplo
    # `AVG(metric_alias)`) en vez de un identificador generico
    # tipo `grouped_measure`.
    intermediate_alias: str | None = None
    metric_score_trace: list[MetricScoreTrace] = dataclass_field(default_factory=list)
    candidate_plan_set: CandidatePlanSet | None = None
    query_form_name: str | None = None
    verification: SemanticVerificationResult | None = None
    issues: list[DecisionIssue] = dataclass_field(default_factory=list)


def build_compiled_semantic_plan(
    *,
    query: str,
    semantic_model: str | None,
    intent: PlanIntent,
    base_entity: str,
    grain: str,
    measure: PlanMeasure | None,
    group_by: Sequence[str] = (),
    final_group_by: Sequence[str] = (),
    selected_filters: Sequence[SemanticFilterSelection] = (),
    time_filter: PlanTimeFilter | None = None,
    ranking: PlanRanking | None = None,
    post_aggregation: PlanPostAggregation | None = None,
    join_path: Sequence[str] = (),
    required_tables: Sequence[str] = (),
    warnings: Sequence[str] = (),
    confidence: float = 0.0,
    join_path_hint: str | None = None,
    derived_metric_ref: str | None = None,
    population_scope: PopulationScope | None = None,
    base_group_by: Sequence[str] = (),
    intermediate_alias: str | None = None,
    metric_score_trace: Sequence[MetricScoreTrace] = (),
    candidate_plan_set: CandidatePlanSet | None = None,
    query_form_name: str | None = None,
    verification: SemanticVerificationResult | None = None,
    issues: Sequence[DecisionIssue] = (),
) -> CompiledSemanticPlan:
    """Construye un plan compilado copiando contenedores mutables para evitar aliasing."""

    return CompiledSemanticPlan(
        query=query,
        semantic_model=semantic_model,
        intent=intent,
        base_entity=base_entity,
        grain=grain,
        measure=measure,
        group_by=list(group_by),
        final_group_by=list(final_group_by),
        selected_filters=list(selected_filters),
        time_filter=time_filter,
        ranking=ranking,
        post_aggregation=post_aggregation,
        join_path=list(join_path),
        required_tables=list(required_tables),
        warnings=list(warnings),
        confidence=confidence,
        join_path_hint=join_path_hint,
        derived_metric_ref=derived_metric_ref,
        population_scope=population_scope,
        base_group_by=list(base_group_by),
        intermediate_alias=intermediate_alias,
        metric_score_trace=list(metric_score_trace),
        candidate_plan_set=candidate_plan_set,
        query_form_name=query_form_name,
        verification=verification,
        issues=list(issues),
    )
