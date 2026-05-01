#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Any

import numpy as np

from .config import SemanticSchemaPruningConfig
from .schema_logic import as_float, build_pruned_schema, build_semantic_score_context


def normalize_relevance_scores(scores: np.ndarray) -> np.ndarray:
    if scores.size == 0:
        return scores.astype(np.float32, copy=False)
    minimum_score = float(np.min(scores))
    maximum_score = float(np.max(scores))
    if np.isclose(minimum_score, maximum_score):
        return np.ones_like(scores, dtype=np.float32)
    return ((scores - minimum_score) / (maximum_score - minimum_score)).astype(np.float32, copy=False)


def build_ranked_documents(
    query_embedding: np.ndarray,
    document_embeddings: np.ndarray,
    documents: list[dict[str, str]],
    *,
    query_input: str,
) -> list[dict[str, object]]:
    similarity_scores = document_embeddings @ query_embedding
    ranked_documents: list[dict[str, object]] = []
    for document_index, document in enumerate(documents):
        ranked_documents.append(
            {
                **document,
                "document_index": document_index,
                "score": float(similarity_scores[document_index]),
                "effective_score": float(similarity_scores[document_index]),
                "query_input": query_input,
            }
        )

    ranked_documents.sort(key=lambda item: as_float(item.get("score", 0.0)), reverse=True)
    return ranked_documents


def apply_listwise_score_adjustment(
    ranked_documents: list[dict[str, object]],
    listwise_scores_by_id: dict[str, float],
    *,
    alpha: float,
) -> None:
    if not listwise_scores_by_id:
        return

    scoped_documents = [
        document for document in ranked_documents if isinstance(document.get("id"), str) and document["id"] in listwise_scores_by_id
    ]
    if not scoped_documents:
        return

    raw_scores = np.asarray([listwise_scores_by_id[str(document["id"])] for document in scoped_documents], dtype=np.float32)
    normalized_scores = normalize_relevance_scores(raw_scores)
    normalized_scores_by_id = {str(document["id"]): float(normalized_scores[index]) for index, document in enumerate(scoped_documents)}

    for document in scoped_documents:
        document_id = str(document["id"])
        base_score = as_float(document.get("score", 0.0))
        normalized_score = normalized_scores_by_id[document_id]
        document["listwise_score"] = float(listwise_scores_by_id[document_id])
        document["listwise_score_normalized"] = normalized_score
        document["effective_score"] = base_score + alpha * (normalized_score - 0.5)


def _mmr_rerank(
    document_embeddings: np.ndarray,
    relevance_scores: np.ndarray,
    k: int,
    *,
    lambda_: float,
) -> list[int]:
    if document_embeddings.shape[0] == 0 or k <= 0:
        return []

    clamped_lambda = min(max(lambda_, 0.0), 1.0)
    selected: list[int] = []
    candidates = set(range(document_embeddings.shape[0]))
    max_similarity_to_selected = np.full(document_embeddings.shape[0], float("-inf"), dtype=np.float32)

    while candidates and len(selected) < k:
        best_candidate_index = -1
        best_candidate_score = float("-inf")

        for candidate_index in candidates:
            if not selected:
                candidate_score = float(relevance_scores[candidate_index])
            else:
                similarity_to_selected = float(max_similarity_to_selected[candidate_index])
                candidate_score = (
                    clamped_lambda * float(relevance_scores[candidate_index]) - (1.0 - clamped_lambda) * similarity_to_selected
                )

            if candidate_score > best_candidate_score:
                best_candidate_score = candidate_score
                best_candidate_index = candidate_index

        selected.append(best_candidate_index)
        candidates.remove(best_candidate_index)
        if candidates:
            candidate_indices = np.fromiter(candidates, dtype=np.int64)
            similarities = document_embeddings[candidate_indices] @ document_embeddings[best_candidate_index]
            max_similarity_to_selected[candidate_indices] = np.maximum(max_similarity_to_selected[candidate_indices], similarities)

    return selected


def apply_mmr_diversification(
    ranked_documents: list[dict[str, object]],
    document_embeddings: np.ndarray,
    config: SemanticSchemaPruningConfig,
) -> list[dict[str, object]]:
    if not config.mmr_enabled or len(ranked_documents) <= 1:
        return ranked_documents

    candidate_pool_size = min(
        len(ranked_documents),
        max(
            config.top_k_matches,
            config.top_k_tables * max(2, config.table_score_column_topn),
            config.mmr_candidate_pool_size,
        ),
    )
    if candidate_pool_size <= 1:
        return ranked_documents

    candidate_documents = ranked_documents[:candidate_pool_size]
    candidate_embedding_indices = [int(as_float(document.get("document_index", -1), -1.0)) for document in candidate_documents]
    candidate_embeddings = document_embeddings[candidate_embedding_indices]
    candidate_relevance = np.asarray(
        [as_float(document.get("effective_score", document.get("score", 0.0))) for document in candidate_documents],
        dtype=np.float32,
    )
    diversified_order = _mmr_rerank(
        candidate_embeddings,
        normalize_relevance_scores(candidate_relevance),
        candidate_pool_size,
        lambda_=config.mmr_lambda,
    )
    diversified_documents = [candidate_documents[index] for index in diversified_order]
    return diversified_documents + ranked_documents[candidate_pool_size:]


def finalize_ranks(ranked_documents: list[dict[str, object]]) -> None:
    for rank, document in enumerate(ranked_documents, start=1):
        document["rank"] = rank


def build_semantic_outputs(
    ranked_documents: list[dict[str, object]],
    schema: dict[str, object],
    config: SemanticSchemaPruningConfig,
) -> tuple[Any, dict[str, object], Any]:
    score_context = build_semantic_score_context(ranked_documents, schema, config)
    pruned_schema, selection = build_pruned_schema(score_context, schema, config)
    return score_context, pruned_schema, selection
