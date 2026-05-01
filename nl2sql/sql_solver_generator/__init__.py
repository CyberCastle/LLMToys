"""API publica del sql_solver_generator.

- Recibe SemanticPlan + pruned_schema + semantic_rules como datos inyectables
  (dicts o rutas YAML). Nunca importa codigo de otros modulos del repo.
- Compila SQLQuerySpec de forma deterministica desde SemanticPlan y delega al
  generador SQL la generacion de final_sql.
- Dialecto configurable (tsql | postgres) via interfaz SqlDialect.
"""

from __future__ import annotations

from .config import SolverConfig
from .contracts import SolverInput, SolverMetadata, SolverOutput
from .dialects.registry import get_dialect
from .reporting import render_solver_result
from .solver import run_sql_solver
from .spec_model import SQLDimension, SQLFilter, SQLGenerationPayload, SQLJoinPlan, SQLMetric, SQLQuerySpec

__all__ = [
    "SolverConfig",
    "SolverInput",
    "SolverMetadata",
    "SolverOutput",
    "SQLDimension",
    "SQLFilter",
    "SQLGenerationPayload",
    "SQLJoinPlan",
    "SQLMetric",
    "SQLQuerySpec",
    "get_dialect",
    "render_solver_result",
    "run_sql_solver",
]
