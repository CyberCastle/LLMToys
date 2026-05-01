"""API publica del Semantic Resolver."""

from __future__ import annotations

from .assets import MatchedAsset, SemanticAsset, SemanticPlan
from .config import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_RERANKER_MODEL,
    SemanticResolverConfig,
)
from .plan_compiler import compile_semantic_plan
from .plan_model import (
    CandidatePlan,
    CandidatePlanSet,
    CompiledSemanticPlan,
    PlanMeasure,
    PlanPostAggregation,
    PlanRanking,
    PlanTimeFilter,
)
from .plan_repair import repair_compiled_plan
from .reporting import (
    build_semantic_plan_report,
    persist_semantic_plan,
    render_semantic_plan,
)
from .resolver import release_semantic_resolver_runtimes, run_semantic_resolver
from .verification import SemanticVerificationResult, verify_compiled_plan

__all__ = [
    "DEFAULT_EMBEDDING_MODEL",
    "DEFAULT_RERANKER_MODEL",
    "CandidatePlan",
    "CandidatePlanSet",
    "CompiledSemanticPlan",
    "MatchedAsset",
    "PlanMeasure",
    "PlanPostAggregation",
    "PlanRanking",
    "PlanTimeFilter",
    "SemanticAsset",
    "SemanticPlan",
    "SemanticResolverConfig",
    "SemanticVerificationResult",
    "build_semantic_plan_report",
    "compile_semantic_plan",
    "persist_semantic_plan",
    "repair_compiled_plan",
    "release_semantic_resolver_runtimes",
    "render_semantic_plan",
    "run_semantic_resolver",
    "verify_compiled_plan",
]
