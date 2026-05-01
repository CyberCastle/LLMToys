#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from nl2sql.utils.text_utils import truncate_text
from nl2sql.utils.yaml_utils import normalize_for_yaml

from .config import SemanticSchemaPruningConfig
from .rerank_stage import ListwiseRerankDiagnostics
from .schema_logic import (
    as_float,
    load_heuristic_rules,
    resolve_fk_path_heuristics,
    resolve_relationship_expansion_heuristics,
)

if TYPE_CHECKING:
    from .schema_pruning import SemanticSchemaPruningResult


def persist_pruned_schema(result: object, *, query: str, out_path: str | Path) -> Path:
    """Persiste el artefacto `semantic_pruned_schema.yaml` para callers externos."""

    normalized_result = normalize_for_yaml(result)
    if not isinstance(normalized_result, dict):
        raise ValueError("Semantic prune result invalido: no se pudo serializar a mapping YAML-safe")

    payload = {
        "query": query,
        **normalized_result,
    }

    output_path = Path(out_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return output_path


def _format_document_identifier(document: dict[str, object]) -> str:
    table_name = document.get("table", "?")
    column_name = document.get("column")
    if isinstance(column_name, str) and column_name:
        return f"{table_name}.{column_name}"
    return str(table_name)


def _format_document_score(document: dict[str, object]) -> str:
    parts = [f"emb={as_float(document.get('score', 0.0)):.4f}"]
    listwise_score = document.get("listwise_score")
    if isinstance(listwise_score, (int, float)):
        parts.append(f"listwise={float(listwise_score):.4f}")
    parts.append(f"eff={as_float(document.get('effective_score', document.get('score', 0.0))):.4f}")
    return " | ".join(parts)


def _append_ranked_match(
    lines: list[str],
    document: dict[str, object],
    *,
    prefix: str,
    config: SemanticSchemaPruningConfig,
) -> None:
    kind = document.get("kind", "?")
    lines.append(f"{prefix}[{kind}] {_format_document_identifier(document)} -> {_format_document_score(document)}")
    if config.show_match_text:
        text = document.get("text")
        if isinstance(text, str):
            lines.append(f"{prefix}{truncate_text(text, config.preview_chars)}")


def render_ranked_matches(ranked_documents: list[dict[str, object]], config: SemanticSchemaPruningConfig) -> str:
    lines = ["TOP SEMANTIC MATCHES", "-" * 80, "Global Top-K:"]
    for index, document in enumerate(ranked_documents[: config.top_k_matches], start=1):
        _append_ranked_match(lines, document, prefix=f"{index:>2}. ", config=config)

    lines.extend(["-" * 80, "Top Matches Por Tabla:"])
    per_table_limit = max(1, min(3, config.top_k_columns_per_table))
    tables_in_order: list[str] = []
    matches_by_table: dict[str, list[dict[str, object]]] = {}

    for document in ranked_documents:
        table_name = document.get("table")
        if not isinstance(table_name, str):
            continue
        if table_name not in matches_by_table:
            if len(tables_in_order) >= config.top_k_tables:
                continue
            tables_in_order.append(table_name)
            matches_by_table[table_name] = []
        if len(matches_by_table[table_name]) >= per_table_limit:
            continue
        matches_by_table[table_name].append(document)

    for table_name in tables_in_order:
        lines.append(table_name)
        for document in matches_by_table.get(table_name, []):
            _append_ranked_match(lines, document, prefix="  - ", config=config)

    return "\n".join(lines)


def _render_listwise_status(diagnostics: ListwiseRerankDiagnostics) -> str:
    status = "applied" if diagnostics.applied else "embedding-only fallback"
    details = (
        f"cand={diagnostics.candidate_count}, "
        f"docs={diagnostics.used_input_docs}/{diagnostics.requested_input_docs}, "
        f"doc_max_tokens={diagnostics.used_max_tokens_per_doc}/{diagnostics.requested_max_tokens_per_doc}, "
        f"prompt_tokens={diagnostics.prompt_token_count}, "
        f"truncados={diagnostics.truncated_documents}"
    )
    if diagnostics.reason:
        details = f"{details}, reason={diagnostics.reason}"
    return f"{status} ({details})"


def _render_fk_paths(fk_paths: list[list[dict[str, str]]]) -> list[str]:
    if not fk_paths:
        return ["none"]

    rendered_paths: list[str] = []
    for path_index, path_steps in enumerate(fk_paths, start=1):
        if not path_steps:
            rendered_paths.append(f"{path_index}. empty")
            continue
        rendered_steps = [
            (
                f"{step.get('from_table', '?')}.{step.get('from_column', '?')}"
                f" -> {step.get('to_table', '?')}.{step.get('to_column', '?')}"
            )
            for step in path_steps
        ]
        rendered_paths.append(f"{path_index}. {' | '.join(rendered_steps)}")
    return rendered_paths


def render_semantic_schema_pruning_report(
    result: SemanticSchemaPruningResult,
    config: SemanticSchemaPruningConfig,
) -> str:
    heuristic_rules = config.heuristic_rules or load_heuristic_rules(str(config.heuristic_rules_path))
    relationship_expansion = resolve_relationship_expansion_heuristics(config, heuristic_rules)
    fk_path = resolve_fk_path_heuristics(config, heuristic_rules)
    lines = [
        "=" * 80,
        "SEMANTIC SCHEMA PRUNING REPORT (E2RANK)",
        "=" * 80,
        f"Query                : {config.query}",
    ]

    if result.retrieval_query != config.query:
        lines.append(f"Retrieval Query      : {result.retrieval_query}")

    lines.extend(
        [
            f"Model                : {config.model}",
            f"Max Model Len        : requested={result.requested_max_model_len} effective={result.effective_max_model_len}",
            f"Schema Documents     : {len(result.schema_documents)}",
            f"Seed Tables          : {', '.join(result.semantic_seed_tables) or 'none'}",
            f"Metric Anchors       : {', '.join(result.metric_anchor_tables) or 'none'}",
            f"Dimension Anchors    : {', '.join(result.dimension_anchor_tables) or 'none'}",
            f"Table Min Score      : {result.table_min_score:.2f}",
            f"Column Min Score     : {result.column_min_score:.2f}",
            f"Top Tables           : {config.top_k_tables}",
            f"Top Columns/Table    : {config.top_k_columns_per_table}",
            f"Outbound Hops        : {relationship_expansion.outbound_hops}",
            f"Inbound Hops         : {relationship_expansion.inbound_hops}",
            f"Neighbors/Table      : {relationship_expansion.max_neighbors_per_table}",
            f"Outbound Min Score   : {relationship_expansion.outbound_min_score:.2f}",
            f"Inbound Min Score    : {relationship_expansion.inbound_min_score:.2f}",
            f"Bridge Max Hops      : {relationship_expansion.bridge_max_hops}",
            f"Bridge Min Score     : {relationship_expansion.bridge_table_min_score:.2f}",
            f"FK Path Expansion    : {'on' if fk_path.enabled else 'off'} (hops={fk_path.max_hops})",
            f"FK Anchor Overlap    : {fk_path.anchor_min_overlap}",
            f"FK Anchors/Role      : {fk_path.max_anchors_per_role}",
            f"MMR                  : {'on' if config.mmr_enabled else 'off'}",
            f"Query Enrichment     : {'on' if config.enable_query_enrichment else 'off'}",
            f"Embedding Cache      : {'on' if result.cache_stats.enabled else 'off'} (hits={result.cache_stats.hits}, misses={result.cache_stats.misses})",
            f"Table Listwise       : {_render_listwise_status(result.table_listwise_diagnostics)}",
            f"Column Listwise      : {_render_listwise_status(result.column_listwise_diagnostics)}",
            "Schema Source        : DATABASE_URL reflection",
            "Reranker             : unified E2Rank listwise PRF (single pooling engine)",
            "-" * 80,
            render_ranked_matches(result.ranked_documents, config),
        ]
    )

    lines.extend(["-" * 80, "FK PATHS", "-" * 80, *_render_fk_paths(result.fk_paths)])

    if config.show_pruned_schema:
        lines.extend(
            [
                "-" * 80,
                "PRUNED SCHEMA",
                "-" * 80,
                yaml.safe_dump(
                    result.pruned_schema,
                    sort_keys=False,
                    allow_unicode=False,
                ).rstrip(),
            ]
        )

    return "\n".join(lines)
