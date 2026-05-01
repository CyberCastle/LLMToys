#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import SemanticSchemaPruningConfig
from .e2rank_engine import E2RankRuntime, get_e2rank_runtime
from .embedding_stage import EmbeddingCacheStats, embed_documents_cached, embed_query
from .query_enrichment import enrich_query_for_retrieval
from .rerank_stage import (
    ListwiseRerankDiagnostics,
    execute_prepared_listwise_reranks,
    prepare_listwise_rerank,
    rerank_listwise,
)
from .reporting import render_semantic_schema_pruning_report
from .schema_logic import build_column_documents, build_table_document, get_schema_columns, serialize_schema_graph_path
from .scoring import (
    apply_listwise_score_adjustment,
    apply_mmr_diversification,
    build_ranked_documents,
    build_semantic_outputs,
    finalize_ranks,
)
from .text_formatting import build_column_listwise_text, build_table_listwise_text


@dataclass(frozen=True)
class SemanticSchemaPruningResult:
    schema_documents: list[dict[str, str]]
    retrieval_query: str
    ranked_documents: list[dict[str, object]]
    semantic_seed_tables: list[str]
    metric_anchor_tables: list[str]
    dimension_anchor_tables: list[str]
    fk_paths: list[list[dict[str, str]]]
    table_min_score: float
    column_min_score: float
    pruned_schema: dict[str, object]
    cache_stats: EmbeddingCacheStats
    table_listwise_diagnostics: ListwiseRerankDiagnostics
    column_listwise_diagnostics: ListwiseRerankDiagnostics
    requested_max_model_len: int
    effective_max_model_len: int


def _build_schema_documents(
    schema: dict[str, object],
    *,
    text_formatting_rules=None,
) -> list[dict[str, str]]:
    documents: list[dict[str, str]] = []
    for table_name, raw_table_info in schema.items():
        if not isinstance(table_name, str) or not isinstance(raw_table_info, dict):
            continue

        table_document = build_table_document(table_name, raw_table_info, schema)
        table_document["listwise_text"] = build_table_listwise_text(
            table_name,
            raw_table_info,
            schema,
            text_formatting_rules,
        )
        documents.append(table_document)

        column_type_lookup = dict(get_schema_columns(raw_table_info))
        for column_document in build_column_documents(table_name, raw_table_info, schema):
            column_name = column_document.get("column", "")
            column_document["listwise_text"] = build_column_listwise_text(
                table_name,
                raw_table_info,
                schema,
                str(column_name),
                column_type_lookup.get(str(column_name), ""),
                text_formatting_rules,
            )
            documents.append(column_document)

    return documents


def _coerce_int(value: object, *, default: int = -1) -> int:
    """Convierte valores heterogeneos a int sin romper tipado estatico."""

    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return default
    return default


def _coerce_float(value: object, *, default: float = 0.0) -> float:
    """Convierte valores heterogeneos a float con fallback controlado."""

    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return default
    return default


def _slice_embeddings(document_embeddings, ranked_documents: list[dict[str, object]]) -> Any:
    indices = [_coerce_int(document.get("document_index", -1)) for document in ranked_documents]
    return document_embeddings[indices]


def _rank_documents_with_e2rank(
    runtime: E2RankRuntime,
    documents: list[dict[str, str]],
    retrieval_query: str,
    config: SemanticSchemaPruningConfig,
) -> tuple[list[dict[str, object]], EmbeddingCacheStats, ListwiseRerankDiagnostics, ListwiseRerankDiagnostics]:
    query_embedding = embed_query(
        runtime.llm,
        config.embedding_task_instruction,
        retrieval_query,
        config.embedding_query_template,
    )
    document_embeddings, cache_stats = embed_documents_cached(
        runtime.llm,
        documents,
        model=config.model,
        cache_dir=config.cache_dir,
        enable_cache=config.enable_embedding_cache,
        task_instruction=config.embedding_task_instruction,
        eos_token=config.eos_token,
    )

    ranked_documents = build_ranked_documents(
        query_embedding,
        document_embeddings,
        documents,
        query_input=retrieval_query,
    )

    table_ranked_documents = [document for document in ranked_documents if document.get("kind") == "table"]
    column_ranked_documents = [document for document in ranked_documents if document.get("kind") == "column"]
    table_embeddings = _slice_embeddings(document_embeddings, table_ranked_documents)
    column_embeddings = _slice_embeddings(document_embeddings, column_ranked_documents)
    listwise_query = retrieval_query if config.listwise_uses_enriched_query else config.query

    table_prepared_rerank, table_fallback_diagnostics = prepare_listwise_rerank(
        runtime.tokenizer,
        listwise_query,
        table_ranked_documents,
        table_embeddings,
        stage_name="tables",
        task=config.rerank_task_instruction,
        prompt_template=config.listwise_prompt_template,
        num_input_docs=config.table_listwise_input_docs,
        max_tokens_per_doc=config.max_tokens_per_doc,
        min_tokens_per_doc=config.listwise_min_tokens_per_doc,
        token_step=config.listwise_token_step,
        max_model_len=runtime.effective_max_model_len,
        eos_token=config.eos_token,
    )
    column_prepared_rerank, column_fallback_diagnostics = prepare_listwise_rerank(
        runtime.tokenizer,
        listwise_query,
        column_ranked_documents,
        column_embeddings,
        stage_name="columns",
        task=config.rerank_task_instruction,
        prompt_template=config.listwise_prompt_template,
        num_input_docs=config.column_listwise_input_docs,
        max_tokens_per_doc=config.max_tokens_per_doc,
        min_tokens_per_doc=config.listwise_min_tokens_per_doc,
        token_step=config.listwise_token_step,
        max_model_len=runtime.effective_max_model_len,
        eos_token=config.eos_token,
    )

    prepared_reranks = [
        prepared_rerank for prepared_rerank in (table_prepared_rerank, column_prepared_rerank) if prepared_rerank is not None
    ]
    table_listwise_scores: dict[str, float] = {}
    column_listwise_scores: dict[str, float] = {}
    table_listwise_diagnostics = table_fallback_diagnostics
    column_listwise_diagnostics = column_fallback_diagnostics

    if prepared_reranks:
        try:
            batched_results = execute_prepared_listwise_reranks(runtime.llm, prepared_reranks)
        except Exception:
            if table_prepared_rerank is not None:
                table_listwise_scores, table_listwise_diagnostics = rerank_listwise(
                    runtime.llm,
                    runtime.tokenizer,
                    listwise_query,
                    table_ranked_documents,
                    table_embeddings,
                    stage_name="tables",
                    task=config.rerank_task_instruction,
                    prompt_template=config.listwise_prompt_template,
                    num_input_docs=config.table_listwise_input_docs,
                    max_tokens_per_doc=config.max_tokens_per_doc,
                    min_tokens_per_doc=config.listwise_min_tokens_per_doc,
                    token_step=config.listwise_token_step,
                    max_model_len=runtime.effective_max_model_len,
                    eos_token=config.eos_token,
                )
            if column_prepared_rerank is not None:
                column_listwise_scores, column_listwise_diagnostics = rerank_listwise(
                    runtime.llm,
                    runtime.tokenizer,
                    listwise_query,
                    column_ranked_documents,
                    column_embeddings,
                    stage_name="columns",
                    task=config.rerank_task_instruction,
                    prompt_template=config.listwise_prompt_template,
                    num_input_docs=config.column_listwise_input_docs,
                    max_tokens_per_doc=config.max_tokens_per_doc,
                    min_tokens_per_doc=config.listwise_min_tokens_per_doc,
                    token_step=config.listwise_token_step,
                    max_model_len=runtime.effective_max_model_len,
                    eos_token=config.eos_token,
                )
        else:
            table_listwise_scores, table_listwise_diagnostics = batched_results.get(
                "tables",
                ({}, table_fallback_diagnostics),
            )
            column_listwise_scores, column_listwise_diagnostics = batched_results.get(
                "columns",
                ({}, column_fallback_diagnostics),
            )

    apply_listwise_score_adjustment(ranked_documents, table_listwise_scores, alpha=config.listwise_score_alpha)
    apply_listwise_score_adjustment(ranked_documents, column_listwise_scores, alpha=config.listwise_score_alpha)

    ranked_documents.sort(
        key=lambda item: _coerce_float(item.get("effective_score", item.get("score", 0.0))),
        reverse=True,
    )
    ranked_documents = apply_mmr_diversification(ranked_documents, document_embeddings, config)
    finalize_ranks(ranked_documents)

    return ranked_documents, cache_stats, table_listwise_diagnostics, column_listwise_diagnostics


def build_semantic_schema_pruning_result(
    config: SemanticSchemaPruningConfig,
    *,
    schema: dict[str, object] | None = None,
) -> SemanticSchemaPruningResult:
    """Ejecuta el pipeline E2Rank completo y devuelve una estructura reutilizable."""

    if schema is None:
        raise ValueError("build_semantic_schema_pruning_result requiere un schema explicito; el core no aplica fallbacks.")

    active_schema = schema
    if not active_schema:
        raise RuntimeError("No se pudo obtener el esquema de la base de datos.")

    schema_documents = _build_schema_documents(active_schema, text_formatting_rules=config.text_formatting_rules)
    if not schema_documents:
        raise RuntimeError("No se pudieron construir documentos semanticos a partir del esquema.")

    retrieval_query = (
        enrich_query_for_retrieval(config.query, signal_rules=config.query_signal_rules) if config.enable_query_enrichment else config.query
    )
    runtime = get_e2rank_runtime(
        config.model,
        dtype=config.dtype,
        max_model_len=config.max_model_len,
        gpu_memory_utilization=config.gpu_memory_utilization,
        tensor_parallel_size=config.tensor_parallel_size,
    )
    ranked_documents, cache_stats, table_listwise_diagnostics, column_listwise_diagnostics = _rank_documents_with_e2rank(
        runtime,
        schema_documents,
        retrieval_query,
        config,
    )

    score_context, pruned_schema, selection = build_semantic_outputs(ranked_documents, active_schema, config)
    return SemanticSchemaPruningResult(
        schema_documents=schema_documents,
        retrieval_query=retrieval_query,
        ranked_documents=ranked_documents,
        semantic_seed_tables=score_context.semantic_seed_tables,
        metric_anchor_tables=list(selection.metric_anchor_tables),
        dimension_anchor_tables=list(selection.dimension_anchor_tables),
        fk_paths=[serialize_schema_graph_path(path_edges) for path_edges in selection.fk_path_edges],
        table_min_score=score_context.table_min_score,
        column_min_score=score_context.column_min_score,
        pruned_schema=pruned_schema,
        cache_stats=cache_stats,
        table_listwise_diagnostics=table_listwise_diagnostics,
        column_listwise_diagnostics=column_listwise_diagnostics,
        requested_max_model_len=config.max_model_len,
        effective_max_model_len=runtime.effective_max_model_len,
    )


def run_semantic_schema_pruning(
    config: SemanticSchemaPruningConfig,
    *,
    schema: dict[str, object] | None = None,
) -> SemanticSchemaPruningResult:
    result = build_semantic_schema_pruning_result(config, schema=schema)
    print(render_semantic_schema_pruning_report(result, config))
    return result
