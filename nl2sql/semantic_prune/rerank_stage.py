#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from llm_core.tokenizer_utils import count_text_tokens
from nl2sql.utils.vector_math import normalize_vector

from .config import EOS_TOKEN
from .text_formatting import build_listwise_prompt, prepare_listwise_document_text


@dataclass(frozen=True)
class ListwiseRerankDiagnostics:
    stage_name: str
    applied: bool
    candidate_count: int
    requested_input_docs: int
    used_input_docs: int
    requested_max_tokens_per_doc: int
    used_max_tokens_per_doc: int
    prompt_token_count: int
    truncated_documents: int
    degraded_to_embedding_only: bool
    reason: str | None = None


@dataclass(frozen=True)
class PreparedListwiseRerank:
    stage_name: str
    pseudo_query: str
    candidate_ids: tuple[str | None, ...]
    candidate_embeddings: np.ndarray
    requested_input_docs: int
    used_input_docs: int
    requested_max_tokens_per_doc: int
    used_max_tokens_per_doc: int
    prompt_token_count: int
    truncated_documents: int


def _build_token_limit_candidates(
    requested_max_tokens_per_doc: int,
    min_tokens_per_doc: int,
    step: int,
) -> list[int]:
    candidates: list[int] = []
    current_value = max(requested_max_tokens_per_doc, min_tokens_per_doc)
    while current_value > min_tokens_per_doc:
        candidates.append(current_value)
        current_value -= max(1, step)
    candidates.append(min_tokens_per_doc)
    return list(dict.fromkeys(candidate for candidate in candidates if candidate > 0))


def _build_applied_diagnostics(prepared: PreparedListwiseRerank) -> ListwiseRerankDiagnostics:
    return ListwiseRerankDiagnostics(
        stage_name=prepared.stage_name,
        applied=True,
        candidate_count=len(prepared.candidate_ids),
        requested_input_docs=prepared.requested_input_docs,
        used_input_docs=prepared.used_input_docs,
        requested_max_tokens_per_doc=prepared.requested_max_tokens_per_doc,
        used_max_tokens_per_doc=prepared.used_max_tokens_per_doc,
        prompt_token_count=prepared.prompt_token_count,
        truncated_documents=prepared.truncated_documents,
        degraded_to_embedding_only=False,
        reason=None,
    )


def prepare_listwise_rerank(
    tokenizer: Any,
    query: str,
    candidate_documents: list[dict[str, object]],
    candidate_embeddings: np.ndarray,
    *,
    stage_name: str,
    task: str,
    prompt_template: str,
    num_input_docs: int,
    max_tokens_per_doc: int,
    min_tokens_per_doc: int,
    token_step: int,
    max_model_len: int,
    eos_token: str = EOS_TOKEN,
) -> tuple[PreparedListwiseRerank | None, ListwiseRerankDiagnostics]:
    """Prepara la pseudo-query listwise para poder batchear varias etapas en una sola llamada."""
    requested_input_docs = min(max(0, num_input_docs), len(candidate_documents))
    if requested_input_docs == 0 or candidate_embeddings.size == 0:
        return None, ListwiseRerankDiagnostics(
            stage_name=stage_name,
            applied=False,
            candidate_count=len(candidate_documents),
            requested_input_docs=requested_input_docs,
            used_input_docs=0,
            requested_max_tokens_per_doc=max_tokens_per_doc,
            used_max_tokens_per_doc=0,
            prompt_token_count=0,
            truncated_documents=0,
            degraded_to_embedding_only=True,
            reason="no_candidates",
        )

    token_limit_candidates = _build_token_limit_candidates(max_tokens_per_doc, min_tokens_per_doc, token_step)

    for current_max_tokens_per_doc in token_limit_candidates:
        prepared_documents: list[str] = []
        truncated_documents = 0
        for document in candidate_documents[:requested_input_docs]:
            source_text = str(document.get("listwise_text") or document.get("text") or "")
            prepared_text, was_truncated, _ = prepare_listwise_document_text(
                source_text,
                tokenizer,
                max_tokens_per_doc=current_max_tokens_per_doc,
                eos_token=eos_token,
            )
            prepared_documents.append(prepared_text)
            truncated_documents += int(was_truncated)

        for current_input_docs in range(requested_input_docs, 0, -1):
            pseudo_query = build_listwise_prompt(
                task,
                query,
                prepared_documents[:current_input_docs],
                tokenizer,
                prompt_template=prompt_template,
                num_input_docs=current_input_docs,
            )
            prompt_token_count = count_text_tokens(tokenizer, pseudo_query)
            if prompt_token_count > max_model_len:
                continue

            prepared_rerank = PreparedListwiseRerank(
                stage_name=stage_name,
                pseudo_query=pseudo_query,
                candidate_ids=tuple(
                    str(document.get("id")) if isinstance(document.get("id"), str) else None for document in candidate_documents
                ),
                candidate_embeddings=candidate_embeddings,
                requested_input_docs=requested_input_docs,
                used_input_docs=current_input_docs,
                requested_max_tokens_per_doc=max_tokens_per_doc,
                used_max_tokens_per_doc=current_max_tokens_per_doc,
                prompt_token_count=prompt_token_count,
                truncated_documents=truncated_documents,
            )
            return prepared_rerank, _build_applied_diagnostics(prepared_rerank)

    return None, ListwiseRerankDiagnostics(
        stage_name=stage_name,
        applied=False,
        candidate_count=len(candidate_documents),
        requested_input_docs=requested_input_docs,
        used_input_docs=0,
        requested_max_tokens_per_doc=max_tokens_per_doc,
        used_max_tokens_per_doc=min_tokens_per_doc,
        prompt_token_count=0,
        truncated_documents=0,
        degraded_to_embedding_only=True,
        reason="prompt_exceeded_context",
    )


def finalize_prepared_listwise_rerank(
    prepared: PreparedListwiseRerank,
    pseudo_query_embedding: np.ndarray,
) -> tuple[dict[str, float], ListwiseRerankDiagnostics]:
    normalized_embedding = normalize_vector(np.asarray(pseudo_query_embedding, dtype=np.float32))
    similarity_scores = prepared.candidate_embeddings @ normalized_embedding
    return (
        {
            document_id: float(similarity_scores[document_index])
            for document_index, document_id in enumerate(prepared.candidate_ids)
            if document_id is not None
        },
        _build_applied_diagnostics(prepared),
    )


def execute_prepared_listwise_reranks(
    llm: Any,
    prepared_reranks: list[PreparedListwiseRerank],
) -> dict[str, tuple[dict[str, float], ListwiseRerankDiagnostics]]:
    if not prepared_reranks:
        return {}

    outputs = llm.embed([prepared.pseudo_query for prepared in prepared_reranks])
    results: dict[str, tuple[dict[str, float], ListwiseRerankDiagnostics]] = {}
    for prepared, output in zip(prepared_reranks, outputs):
        results[prepared.stage_name] = finalize_prepared_listwise_rerank(
            prepared,
            np.asarray(output.outputs.embedding, dtype=np.float32),
        )
    return results


def rerank_listwise(
    llm: Any,
    tokenizer: Any,
    query: str,
    candidate_documents: list[dict[str, object]],
    candidate_embeddings: np.ndarray,
    *,
    stage_name: str,
    task: str,
    prompt_template: str,
    num_input_docs: int,
    max_tokens_per_doc: int,
    min_tokens_per_doc: int,
    token_step: int,
    max_model_len: int,
    eos_token: str = EOS_TOKEN,
) -> tuple[dict[str, float], ListwiseRerankDiagnostics]:
    """Ejecuta PRF listwise y degrada a solo-embedding si el prompt no cabe.

    E2Rank no reembebe los documentos candidatos: solo genera el embedding de la
    pseudo-query y lo compara por coseno con todos los candidatos ya vectorizados.
    """
    prepared_rerank, fallback_diagnostics = prepare_listwise_rerank(
        tokenizer,
        query,
        candidate_documents,
        candidate_embeddings,
        stage_name=stage_name,
        task=task,
        prompt_template=prompt_template,
        num_input_docs=num_input_docs,
        max_tokens_per_doc=max_tokens_per_doc,
        min_tokens_per_doc=min_tokens_per_doc,
        token_step=token_step,
        max_model_len=max_model_len,
        eos_token=eos_token,
    )
    if prepared_rerank is None:
        return {}, fallback_diagnostics

    try:
        output = llm.embed([prepared_rerank.pseudo_query])
    except Exception as error:
        return {}, ListwiseRerankDiagnostics(
            stage_name=stage_name,
            applied=False,
            candidate_count=len(candidate_documents),
            requested_input_docs=prepared_rerank.requested_input_docs,
            used_input_docs=0,
            requested_max_tokens_per_doc=max_tokens_per_doc,
            used_max_tokens_per_doc=min_tokens_per_doc,
            prompt_token_count=0,
            truncated_documents=prepared_rerank.truncated_documents,
            degraded_to_embedding_only=True,
            reason=str(error).strip() or "listwise_embed_failed",
        )

    return finalize_prepared_listwise_rerank(prepared_rerank, np.asarray(output[0].outputs.embedding, dtype=np.float32))
