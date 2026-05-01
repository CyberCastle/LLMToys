#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Helpers reutilizables para caches persistentes de embeddings."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class EmbeddingCacheStats:
    """Resumen del comportamiento del cache persistente de embeddings."""

    cache_path: Path
    enabled: bool
    hits: int
    misses: int


def resolve_embedding_cache_path(model: str, cache_dir: Path) -> Path:
    """Devuelve el archivo NPZ asociado a un modelo dentro del cache."""

    resolved_cache_dir = Path(cache_dir)
    resolved_cache_dir.mkdir(parents=True, exist_ok=True)
    return resolved_cache_dir / f"{model.replace('/', '_')}.npz"


def load_embedding_cache(cache_path: Path) -> dict[str, np.ndarray]:
    """Carga un cache NPZ e ignora archivos corruptos o incompletos."""

    if not cache_path.exists():
        return {}

    try:
        with np.load(cache_path, allow_pickle=False) as cache_file:
            return {cache_key: cache_file[cache_key].astype(np.float32, copy=False) for cache_key in cache_file.files}
    except Exception:
        return {}


def save_embedding_cache(cache_path: Path, cached_embeddings: dict[str, np.ndarray]) -> None:
    """Persiste embeddings usando escritura atómica para evitar archivos truncados."""

    temporary_path = cache_path.with_suffix(".tmp.npz")
    np.savez(str(temporary_path), **cached_embeddings)
    temporary_path.replace(cache_path)


def embed_with_cache(
    *,
    model: str,
    cache_dir: Path,
    enable_cache: bool,
    texts: list[str],
    item_hashes: list[str],
    embed_fn: Callable[[list[str]], np.ndarray],
) -> tuple[np.ndarray, EmbeddingCacheStats]:
    """Embebe textos reusando un cache NPZ indexado por hashes estables."""

    cache_path = resolve_embedding_cache_path(model, cache_dir)
    if len(texts) != len(item_hashes):
        raise ValueError("embed_with_cache requiere la misma cantidad de textos y hashes")
    if not texts:
        return (
            np.empty((0, 0), dtype=np.float32),
            EmbeddingCacheStats(cache_path=cache_path, enabled=enable_cache, hits=0, misses=0),
        )
    if not enable_cache:
        return (
            embed_fn(texts),
            EmbeddingCacheStats(cache_path=cache_path, enabled=False, hits=0, misses=len(texts)),
        )

    cached_embeddings = load_embedding_cache(cache_path)
    missing_items: list[tuple[int, str]] = []
    cache_hits = 0
    for item_index, item_hash in enumerate(item_hashes):
        cached_vector = cached_embeddings.get(item_hash)
        if isinstance(cached_vector, np.ndarray) and cached_vector.ndim == 1:
            cache_hits += 1
            continue
        missing_items.append((item_index, item_hash))

    if missing_items:
        missing_texts = [texts[item_index] for item_index, _item_hash in missing_items]
        new_embeddings = embed_fn(missing_texts)
        for embedding_row_index, (_item_index, item_hash) in enumerate(missing_items):
            cached_embeddings[item_hash] = new_embeddings[embedding_row_index]
        save_embedding_cache(cache_path, cached_embeddings)

    ordered_embeddings = np.vstack([cached_embeddings[item_hash] for item_hash in item_hashes]).astype(np.float32, copy=False)
    return (
        ordered_embeddings,
        EmbeddingCacheStats(cache_path=cache_path, enabled=True, hits=cache_hits, misses=len(missing_items)),
    )
