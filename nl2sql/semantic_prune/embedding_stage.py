#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np

from nl2sql.utils.embedding_cache import (
    EmbeddingCacheStats,
    embed_with_cache,
)
from nl2sql.utils.vector_math import normalize_matrix, normalize_vector

from .config import EOS_TOKEN
from .text_formatting import append_eos, get_detailed_instruct


def _doc_hash(document: dict[str, str], *, eos_token: str, task_instruction: str) -> str:
    """Versiona la cache cuando cambian detalles que afectan el embedding final."""
    payload = f"{document['id']}::{document['text']}::{eos_token}::{task_instruction}".encode("utf-8")
    return hashlib.sha1(payload).hexdigest()


def embed_query(llm: Any, task: str, query: str, query_template: str) -> np.ndarray:
    """Embebe la query enriquecida usando la instruccion tipo E2Rank."""
    rendered_query = get_detailed_instruct(task, query, query_template)
    output = llm.embed([rendered_query])
    return normalize_vector(np.asarray(output[0].outputs.embedding, dtype=np.float32))


def embed_documents(llm: Any, documents: list[str], *, eos_token: str = EOS_TOKEN) -> np.ndarray:
    if not documents:
        return np.empty((0, 0), dtype=np.float32)

    outputs = llm.embed([append_eos(document, eos_token) for document in documents])
    return normalize_matrix(np.vstack([np.asarray(item.outputs.embedding, dtype=np.float32) for item in outputs]))


def embed_documents_cached(
    llm: Any,
    documents: list[dict[str, str]],
    *,
    model: str,
    cache_dir: Path,
    enable_cache: bool,
    task_instruction: str,
    eos_token: str = EOS_TOKEN,
) -> tuple[np.ndarray, EmbeddingCacheStats]:
    texts = [document["text"] for document in documents]
    return embed_with_cache(
        model=model,
        cache_dir=cache_dir,
        enable_cache=enable_cache,
        texts=texts,
        item_hashes=[_doc_hash(document, eos_token=eos_token, task_instruction=task_instruction) for document in documents],
        embed_fn=lambda missing_texts: embed_documents(llm, missing_texts, eos_token=eos_token),
    )
