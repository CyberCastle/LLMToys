#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Literal

from nl2sql.config import (
    CompilerRules,
    SemanticResolverRuntimeTuning,
    SemanticResolverSettings,
    SemanticResolverVerificationRules,
    env_bool,
    env_float,
    env_int,
    env_path,
    env_str,
    load_semantic_resolver_settings,
    resolve_nl2sql_config_path,
    resolve_semantic_rules_path,
)
from nl2sql.utils.semantic_contract import SemanticContract

EmbeddingDType = Literal["auto", "bfloat16", "float16", "float32"]
DefaultPostAggregationFunction = Literal["avg", "ratio", "sum"]
ResolverDialectName = Literal["tsql", "postgres"]

DEFAULT_EMBEDDING_MODEL = "Qwen/Qwen3-Embedding-0.6B"
DEFAULT_RERANKER_MODEL = "Qwen/Qwen3-Reranker-0.6B"
DEFAULT_VERIFIER_MODEL = "Qwen/Qwen2.5-7B-Instruct-GPTQ-Int4"
DEFAULT_SEMANTIC_SECTIONS = (
    "semantic_models",
    "semantic_entities",
    "semantic_dimensions",
    "semantic_metrics",
    "semantic_filters",
    "semantic_business_rules",
    "semantic_relationships",
    "semantic_synonyms",
    "semantic_examples",
    "semantic_constraints",
)

# La ruta del YAML semantico sale del entorno para permitir moverlo sin tocar
# codigo. Se resuelve con helper explicito para mantener el mismo patron que
# usa semantic_prune con sus reglas internas configurables.
RULES_PATH_ENV_VAR = "SEMANTIC_RULES_PATH"
DEFAULT_RULES_PATH = "schema-docs/semantic_rules.yaml"
EMBEDDING_CACHE_DIR_ENV_VAR = "SEMANTIC_RESOLVER_EMBEDDING_CACHE_DIR"
DEFAULT_EMBEDDING_CACHE_DIR = Path(".cache/semantic_resolver_embeddings")

# Variable de entorno que selecciona el dialecto SQL del compilador del plan.
# Mantener la seleccion en config + .env permite que el modulo siga siendo
# agnostico: la implementacion concreta vive en ``semantic_resolver.dialects``.
DIALECT_ENV_VAR = "SQL_DIALECT"
DEFAULT_DIALECT: ResolverDialectName = "tsql"


def resolve_rules_path() -> str:
    """Resuelve la ruta del YAML de reglas semanticas desde entorno."""

    return str(resolve_semantic_rules_path())


def resolve_prompts_path() -> Path:
    """Resuelve la ruta del YAML unificado de configuracion del resolver."""

    return resolve_nl2sql_config_path()


def resolve_embedding_cache_dir() -> Path:
    """Resuelve la carpeta del cache persistente de embeddings del resolver."""

    return env_path(EMBEDDING_CACHE_DIR_ENV_VAR, DEFAULT_EMBEDDING_CACHE_DIR)


def resolve_compiler_rules_path() -> Path:
    """Resuelve la ruta del YAML unificado para el compilador del resolver."""

    return resolve_nl2sql_config_path()


def resolve_runtime_tuning_path() -> Path:
    """Resuelve la ruta del YAML unificado para el tuning operativo del resolver."""

    return resolve_nl2sql_config_path()


def resolve_dialect_name() -> str:
    """Resuelve el nombre del dialecto SQL activo desde entorno.

    Cae al :data:`DEFAULT_DIALECT` cuando ``SQL_DIALECT`` no esta definido.
    No instancia el dialecto en si para evitar acoplar la config al submodulo
    ``semantic_resolver.dialects`` durante la importacion.
    """

    return env_str(DIALECT_ENV_VAR, DEFAULT_DIALECT).lower()


@lru_cache(maxsize=8)
def load_runtime_tuning(
    path: str | Path | None = None,
) -> SemanticResolverRuntimeTuning:
    """Devuelve el tuning tipado ya validado del resolver."""

    resolved_path = Path(path).expanduser().resolve() if path is not None else resolve_runtime_tuning_path()
    return load_semantic_resolver_settings(resolved_path).runtime_tuning


@dataclass(frozen=True)
class SemanticResolverConfig:
    """Configuracion del pipeline de retrieval, rerank y compatibilidad."""

    settings: SemanticResolverSettings | None = None
    semantic_contract: SemanticContract | None = None
    compiler_rules: CompilerRules | None = None
    verification_rules: SemanticResolverVerificationRules | None = None
    rules_path: str = field(default_factory=resolve_rules_path)
    prompts_path: Path = field(default_factory=resolve_prompts_path)
    sections: tuple[str, ...] = DEFAULT_SEMANTIC_SECTIONS
    embedding_model: str = field(default_factory=lambda: env_str("SEMANTIC_RESOLVER_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL))
    reranker_model: str = field(default_factory=lambda: env_str("SEMANTIC_RESOLVER_RERANKER_MODEL", DEFAULT_RERANKER_MODEL))
    dtype: EmbeddingDType = field(default_factory=lambda: env_str("SEMANTIC_RESOLVER_DTYPE", "auto"))
    max_model_len: int = field(default_factory=lambda: env_int("SEMANTIC_RESOLVER_MAX_MODEL_LEN", 8192))
    gpu_memory_utilization_embed: float = field(default_factory=lambda: env_float("SEMANTIC_RESOLVER_GPU_MEMORY_UTILIZATION_EMBED", 0.30))
    gpu_memory_utilization_rerank: float = field(default_factory=lambda: env_float("SEMANTIC_RESOLVER_GPU_MEMORY_UTILIZATION_RERANK", 0.25))
    tensor_parallel_size: int = field(default_factory=lambda: env_int("SEMANTIC_RESOLVER_TENSOR_PARALLEL_SIZE", 1))
    trust_remote_code: bool = field(default_factory=lambda: env_bool("SEMANTIC_RESOLVER_TRUST_REMOTE_CODE", True))
    query_instruction: str = ""
    embedding_query_template: str = ""
    reranker_instruction: str = ""
    rerank_system_prompt: str = ""
    rerank_user_prompt_template: str = ""
    verifier_model: str = field(default_factory=lambda: env_str("SEMANTIC_RESOLVER_VERIFIER_MODEL", DEFAULT_VERIFIER_MODEL))
    verifier_dtype: EmbeddingDType = field(
        default_factory=lambda: env_str("SEMANTIC_RESOLVER_VERIFIER_DTYPE", env_str("SEMANTIC_RESOLVER_DTYPE", "auto"))
    )
    verifier_system_prompt: str = ""
    verifier_user_prompt_template: str = ""
    verifier_temperature: float = field(default_factory=lambda: env_float("SEMANTIC_RESOLVER_VERIFIER_TEMPERATURE", 0.0))
    verifier_max_model_len: int = field(default_factory=lambda: env_int("SEMANTIC_RESOLVER_VERIFIER_MAX_MODEL_LEN", 2048))
    verifier_max_tokens: int = field(default_factory=lambda: env_int("SEMANTIC_RESOLVER_VERIFIER_MAX_TOKENS", 256))
    verifier_gpu_memory_utilization: float = field(
        default_factory=lambda: env_float("SEMANTIC_RESOLVER_VERIFIER_GPU_MEMORY_UTILIZATION", 0.82)
    )
    verifier_cpu_offload_gb: float = field(default_factory=lambda: env_float("SEMANTIC_RESOLVER_VERIFIER_CPU_OFFLOAD_GB", 0.0))
    verifier_enforce_eager: bool = field(default_factory=lambda: env_bool("SEMANTIC_RESOLVER_VERIFIER_ENFORCE_EAGER", True))
    verifier_few_shot_limit: int | None = None
    max_semantic_repair_attempts: int | None = None
    enable_semantic_verifier: bool = field(default_factory=lambda: env_bool("SEMANTIC_RESOLVER_ENABLE_SEMANTIC_VERIFIER", True))
    enable_embedding_cache: bool = field(default_factory=lambda: env_bool("SEMANTIC_RESOLVER_ENABLE_EMBEDDING_CACHE", True))
    embedding_cache_dir: Path = field(default_factory=resolve_embedding_cache_dir)

    # Retrieval y rerank tienen top-k distintos porque el primero prioriza recall
    # barato con coseno y el segundo aplica un filtro mas caro pero mas preciso.
    top_k_retrieval: int | None = None
    top_k_rerank: int | None = None
    min_embedding_score: float | None = None
    # En Qwen3-Reranker el score absoluto yes/no no siempre queda bien calibrado
    # en dominios propios; por defecto se usa para ordenar candidatos, no para
    # filtrar en duro, salvo que el usuario fije un umbral mayor a 0.
    min_rerank_score: float | None = None
    compatibility_min_score: float | None = None
    enable_synonym_query_expansion: bool = field(default_factory=lambda: env_bool("SEMANTIC_RESOLVER_ENABLE_SYNONYM_QUERY_EXPANSION", True))
    enable_synonym_score_boost: bool = field(default_factory=lambda: env_bool("SEMANTIC_RESOLVER_ENABLE_SYNONYM_SCORE_BOOST", True))
    synonym_query_expansion_max_entities: int | None = None
    synonym_direct_boost: float | None = None
    synonym_related_boost: float | None = None

    rerank_batch_size: int | None = None
    rerank_max_document_chars: int | None = None
    rerank_logprobs: int | None = None
    rerank_prompt_token_margin: int | None = None
    sequential_engines: bool = field(default_factory=lambda: env_bool("SEMANTIC_RESOLVER_SEQUENTIAL_ENGINES", True))
    show_rejected_assets: bool = field(default_factory=lambda: env_bool("SEMANTIC_RESOLVER_SHOW_REJECTED_ASSETS", True))
    enable_plan_compiler: bool = field(default_factory=lambda: env_bool("SEMANTIC_RESOLVER_ENABLE_PLAN_COMPILER", True))
    # Ruta al YAML unificado de configuracion del resolver.
    compiler_rules_path: Path = field(default_factory=resolve_compiler_rules_path)
    default_post_aggregation_function: DefaultPostAggregationFunction | None = None
    # Nombre del dialecto SQL para el cual el plan_compiler debe materializar
    # las expresiones temporales (``value_<dialect>`` en
    # `semantic_resolver.compiler_rules`).
    # No se almacena la instancia para que la config siga siendo serializable
    # y el submodulo ``semantic_resolver.dialects`` solo se importe cuando se
    # necesita instanciar el dialecto.
    dialect: str = field(default_factory=resolve_dialect_name)

    # Los caps por tipo evitan que metricas o sinonimos dominen el plan final
    # y obligan a que el resultado conserve diversidad semantica util.
    per_kind_caps: dict[str, int] | None = None

    def __post_init__(self) -> None:
        """Completa prompts y plantillas desde el YAML configurado.

        Si el caller no fuerza un override puntual, el resolver toma la fuente
        canonica de prompts desde `prompts_path` para que el ajuste de negocio
        viva en YAML y no en literales Python.
        """

        active_settings = self.settings or load_semantic_resolver_settings(self.prompts_path, self.rules_path)
        prompt_rules = active_settings.prompts
        runtime_tuning = active_settings.runtime_tuning
        for field_name in (
            "verifier_few_shot_limit",
            "max_semantic_repair_attempts",
            "top_k_retrieval",
            "top_k_rerank",
            "min_embedding_score",
            "min_rerank_score",
            "compatibility_min_score",
            "synonym_query_expansion_max_entities",
            "synonym_direct_boost",
            "synonym_related_boost",
            "rerank_batch_size",
            "rerank_max_document_chars",
            "rerank_logprobs",
            "rerank_prompt_token_margin",
            "default_post_aggregation_function",
        ):
            if getattr(self, field_name) is None:
                object.__setattr__(self, field_name, getattr(runtime_tuning, field_name))
        if self.per_kind_caps is None:
            object.__setattr__(self, "per_kind_caps", dict(runtime_tuning.per_kind_caps))
        if self.compiler_rules is None:
            object.__setattr__(self, "compiler_rules", active_settings.compiler_rules)
        if self.verification_rules is None:
            object.__setattr__(self, "verification_rules", active_settings.verification)
        if not self.query_instruction:
            object.__setattr__(self, "query_instruction", prompt_rules.embedding.query_instruction)
        if not self.embedding_query_template:
            object.__setattr__(self, "embedding_query_template", prompt_rules.embedding.query_template)
        if not self.reranker_instruction:
            object.__setattr__(self, "reranker_instruction", prompt_rules.rerank.instruction)
        if not self.rerank_system_prompt:
            object.__setattr__(self, "rerank_system_prompt", prompt_rules.rerank.system_prompt)
        if not self.rerank_user_prompt_template:
            object.__setattr__(
                self,
                "rerank_user_prompt_template",
                prompt_rules.rerank.user_prompt_template,
            )
        if not self.verifier_system_prompt:
            object.__setattr__(self, "verifier_system_prompt", prompt_rules.verifier.system_prompt)
        if not self.verifier_user_prompt_template:
            object.__setattr__(
                self,
                "verifier_user_prompt_template",
                prompt_rules.verifier.user_prompt_template,
            )
