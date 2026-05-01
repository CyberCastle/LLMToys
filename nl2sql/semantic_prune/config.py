#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Literal

from nl2sql.config import (
    HeuristicRules,
    QuerySignalRules,
    SemanticPruneRuntimeTuning,
    SemanticPruneSettings,
    SemanticPruneTextFormattingRules,
    env_bool,
    env_float,
    env_int,
    env_path,
    env_str,
    load_semantic_prune_settings,
    resolve_nl2sql_config_path,
)

EmbeddingDType = Literal["auto", "half", "float16", "bfloat16", "float", "float32"]

DEFAULT_MODEL = "Alibaba-NLP/E2Rank-0.6B"
DEFAULT_POOLING_TYPE = "LAST"
DEFAULT_NORMALIZE = True
DEFAULT_MAX_MODEL_LEN = 30464
DEFAULT_GPU_MEMORY_UTILIZATION = 0.30
DEFAULT_TENSOR_PARALLEL_SIZE = 1
DEFAULT_CACHE_DIR = Path(".cache/schema_embeddings")
EOS_TOKEN = "<|endoftext|>"


def resolve_prompts_path() -> Path:
    """Resuelve la ruta del YAML unificado de configuracion de prune."""

    return resolve_nl2sql_config_path()


def resolve_runtime_tuning_path() -> Path:
    """Resuelve la ruta del YAML unificado para el tuning de semantic prune."""

    return resolve_nl2sql_config_path()


def resolve_cache_dir() -> Path:
    """Resuelve la carpeta del cache persistente del pruning."""

    return env_path("SEMANTIC_PRUNE_CACHE_DIR", DEFAULT_CACHE_DIR)


@lru_cache(maxsize=8)
def load_runtime_tuning(path: str | Path | None = None) -> SemanticPruneRuntimeTuning:
    """Devuelve el tuning tipado ya validado del semantic prune."""

    resolved_path = Path(path).expanduser().resolve() if path is not None else resolve_runtime_tuning_path()
    return load_semantic_prune_settings(resolved_path).runtime_tuning


@dataclass(frozen=True)
class SemanticSchemaPruningConfig:
    """Configuracion operativa del pipeline E2Rank.

    La mayoria de los knobs conserva los nombres del flujo previo para poder
    reutilizar heuristicas ya validadas de scoring, thresholds y expansion FK.
    Los controles `fk_path_*` gobiernan la expansion estructural dura que fuerza
    el camino minimo entre una tabla de metrica y una de dimension cuando la
    query mezcla ambas y el score semantico por si solo deja fuera tablas puente.

    Los defaults canonicos de expansion relacional y FK path viven ahora en el
    YAML apuntado por `heuristic_rules_path`. Los campos `relationship_*` y
    `fk_path_*` quedan como overrides puntuales para una corrida concreta sin
    necesidad de editar el YAML compartido.
    """

    query: str
    settings: SemanticPruneSettings | None = None
    query_signal_rules: QuerySignalRules | None = None
    heuristic_rules: HeuristicRules | None = None
    text_formatting_rules: SemanticPruneTextFormattingRules | None = None
    model: str = field(default_factory=lambda: env_str("SEMANTIC_PRUNE_MODEL", DEFAULT_MODEL))
    dtype: EmbeddingDType = field(default_factory=lambda: env_str("SEMANTIC_PRUNE_DTYPE", "auto"))
    signal_rules_path: Path = field(default_factory=resolve_prompts_path)
    heuristic_rules_path: Path = field(default_factory=resolve_prompts_path)
    prompts_path: Path = field(default_factory=resolve_prompts_path)
    max_model_len: int = field(default_factory=lambda: env_int("SEMANTIC_PRUNE_MAX_MODEL_LEN", DEFAULT_MAX_MODEL_LEN))
    gpu_memory_utilization: float = field(
        default_factory=lambda: env_float("SEMANTIC_PRUNE_GPU_MEMORY_UTILIZATION", DEFAULT_GPU_MEMORY_UTILIZATION)
    )
    tensor_parallel_size: int = field(default_factory=lambda: env_int("SEMANTIC_PRUNE_TENSOR_PARALLEL_SIZE", DEFAULT_TENSOR_PARALLEL_SIZE))
    embedding_task_instruction: str = ""
    embedding_query_template: str = ""
    rerank_task_instruction: str = ""
    listwise_prompt_template: str = ""
    eos_token: str = EOS_TOKEN
    enable_query_enrichment: bool = field(default_factory=lambda: env_bool("SEMANTIC_PRUNE_ENABLE_QUERY_ENRICHMENT", True))
    listwise_uses_enriched_query: bool = field(default_factory=lambda: env_bool("SEMANTIC_PRUNE_LISTWISE_USES_ENRICHED_QUERY", False))
    enable_embedding_cache: bool = field(default_factory=lambda: env_bool("SEMANTIC_PRUNE_ENABLE_EMBEDDING_CACHE", True))
    cache_dir: Path = field(default_factory=resolve_cache_dir)
    top_k_matches: int | None = None
    top_k_tables: int | None = None
    top_k_columns_per_table: int | None = None
    min_score: float | None = None
    table_score_doc_weight: float | None = None
    table_score_column_topn: int | None = None
    relationship_expansion_outbound_hops: int | None = None
    relationship_expansion_inbound_hops: int | None = None
    relationship_expansion_max_neighbors_per_table: int | None = None
    relationship_expansion_min_score: float | None = None
    relationship_expansion_inbound_min_score: float | None = None
    relationship_bridge_max_hops: int | None = None
    relationship_bridge_table_min_score: float | None = None
    enable_fk_path_expansion: bool | None = None
    fk_path_max_hops: int | None = None
    fk_path_anchor_min_overlap: int | None = None
    fk_path_max_anchors_per_role: int | None = None
    mmr_enabled: bool | None = None
    mmr_lambda: float | None = None
    mmr_candidate_pool_size: int | None = None
    adaptive_threshold_k_sigma: float | None = None
    adaptive_threshold_k_sigma_columns: float | None = None
    table_listwise_input_docs: int | None = None
    column_listwise_input_docs: int | None = None
    max_tokens_per_doc: int | None = None
    listwise_min_tokens_per_doc: int | None = None
    listwise_token_step: int | None = None
    listwise_score_alpha: float | None = None
    preview_chars: int | None = None
    show_match_text: bool = field(default_factory=lambda: env_bool("SEMANTIC_PRUNE_SHOW_MATCH_TEXT", False))
    show_pruned_schema: bool = field(default_factory=lambda: env_bool("SEMANTIC_PRUNE_SHOW_PRUNED_SCHEMA", True))
    # Ruta al YAML de reglas semanticas (el mismo del semantic_resolver). Cuando
    # se provee, el pruning honra `semantic_join_paths` como bridges autoritativos
    # y conserva dependencias declaradas por metricas (`source_catalog`,
    # `required_relationships`, referencias tabla.col en `formula`).
    semantic_rules_path: str | None = None

    def __post_init__(self) -> None:
        """Completa prompts y plantillas desde el YAML configurado.

        Los campos siguen siendo overrideables por constructor, pero cuando se
        dejan vacios el runtime toma el valor canonico del archivo apuntado por
        `prompts_path`.
        """

        active_settings = self.settings or load_semantic_prune_settings(self.prompts_path)
        prompt_rules = active_settings.prompts
        runtime_tuning = active_settings.runtime_tuning
        for field_name in (
            "top_k_matches",
            "top_k_tables",
            "top_k_columns_per_table",
            "min_score",
            "table_score_doc_weight",
            "table_score_column_topn",
            "mmr_enabled",
            "mmr_lambda",
            "mmr_candidate_pool_size",
            "adaptive_threshold_k_sigma",
            "adaptive_threshold_k_sigma_columns",
            "table_listwise_input_docs",
            "column_listwise_input_docs",
            "max_tokens_per_doc",
            "listwise_min_tokens_per_doc",
            "listwise_token_step",
            "listwise_score_alpha",
            "preview_chars",
        ):
            if getattr(self, field_name) is None:
                object.__setattr__(self, field_name, getattr(runtime_tuning, field_name))
        if self.query_signal_rules is None:
            object.__setattr__(self, "query_signal_rules", active_settings.query_signal_rules)
        if self.heuristic_rules is None:
            object.__setattr__(self, "heuristic_rules", active_settings.heuristic_rules)
        if self.text_formatting_rules is None:
            object.__setattr__(self, "text_formatting_rules", active_settings.text_formatting)
        if not self.embedding_task_instruction:
            object.__setattr__(
                self,
                "embedding_task_instruction",
                prompt_rules.embedding.task_instruction,
            )
        if not self.embedding_query_template:
            object.__setattr__(self, "embedding_query_template", prompt_rules.embedding.query_template)
        if not self.rerank_task_instruction:
            object.__setattr__(
                self,
                "rerank_task_instruction",
                prompt_rules.listwise_rerank.task_instruction,
            )
        if not self.listwise_prompt_template:
            object.__setattr__(
                self,
                "listwise_prompt_template",
                prompt_rules.listwise_rerank.prompt_template,
            )
