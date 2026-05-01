#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from nl2sql.semantic_resolver.assets import SemanticAsset
from nl2sql.semantic_resolver.embedding_stage import embed_assets_cached


@dataclass(frozen=True)
class _FakeEmbeddingPayload:
    embedding: list[float]


@dataclass(frozen=True)
class _FakeEmbeddingOutput:
    outputs: _FakeEmbeddingPayload


class _FakeLLM:
    """Stub minimo del engine de embeddings para pruebas de cache en disco."""

    def __init__(self) -> None:
        self.calls: list[list[str]] = []

    def embed(self, texts: list[str]) -> list[_FakeEmbeddingOutput]:
        self.calls.append(list(texts))
        outputs: list[_FakeEmbeddingOutput] = []
        for text in texts:
            ascii_sum = float(sum(ord(character) for character in text))
            outputs.append(
                _FakeEmbeddingOutput(
                    outputs=_FakeEmbeddingPayload(
                        embedding=[ascii_sum, float(len(text)), float((len(text) % 7) + 1)],
                    )
                )
            )
        return outputs


def _asset(asset_id: str, name: str) -> SemanticAsset:
    return SemanticAsset(
        asset_id=asset_id,
        kind="semantic_metric",
        name=name,
        payload={"name": name},
    )


def test_embed_assets_cached_reuses_disk_cache(tmp_path: Path) -> None:
    llm = _FakeLLM()
    assets = [_asset("metric::metric_alpha", "metric_alpha"), _asset("metric::metric_beta", "metric_beta")]
    texts = ["metrica alpha por entidad_c", "metrica beta por entidad_c"]

    embeddings_first, stats_first = embed_assets_cached(
        llm,
        assets,
        texts,
        model="Qwen/Qwen3-Embedding-0.6B",
        cache_dir=tmp_path,
        enable_cache=True,
    )
    embeddings_second, stats_second = embed_assets_cached(
        llm,
        assets,
        texts,
        model="Qwen/Qwen3-Embedding-0.6B",
        cache_dir=tmp_path,
        enable_cache=True,
    )

    assert stats_first.enabled is True
    assert stats_first.hits == 0
    assert stats_first.misses == 2
    assert stats_second.enabled is True
    assert stats_second.hits == 2
    assert stats_second.misses == 0
    assert len(llm.calls) == 1
    assert stats_second.cache_path.exists()
    assert np.allclose(embeddings_first, embeddings_second)


def test_embed_assets_cached_invalidates_only_changed_asset(tmp_path: Path) -> None:
    llm = _FakeLLM()
    assets = [_asset("metric::metric_alpha", "metric_alpha"), _asset("metric::metric_beta", "metric_beta")]
    original_texts = ["metrica alpha por entidad_c", "metrica beta por entidad_c"]
    updated_texts = ["metrica alpha por entidad_c", "metrica beta acumulada por entidad_c"]

    embed_assets_cached(
        llm,
        assets,
        original_texts,
        model="Qwen/Qwen3-Embedding-0.6B",
        cache_dir=tmp_path,
        enable_cache=True,
    )
    _embeddings_updated, stats_updated = embed_assets_cached(
        llm,
        assets,
        updated_texts,
        model="Qwen/Qwen3-Embedding-0.6B",
        cache_dir=tmp_path,
        enable_cache=True,
    )

    assert stats_updated.enabled is True
    assert stats_updated.hits == 1
    assert stats_updated.misses == 1
    assert len(llm.calls) == 2
    assert llm.calls[1] == ["metrica beta acumulada por entidad_c"]
