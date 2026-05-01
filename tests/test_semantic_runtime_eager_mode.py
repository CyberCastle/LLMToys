#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Pruebas de regresion para runtimes semanticos en modo eager."""

from __future__ import annotations

from types import SimpleNamespace

from nl2sql.semantic_prune import e2rank_engine
from nl2sql.semantic_resolver import embedding_stage, rerank_stage


class _FakeLLM:
    """Stub minimo de vLLM que expone solo lo necesario para estas pruebas."""

    def get_tokenizer(self) -> object:
        """Devuelve un tokenizer opaco para satisfacer el contrato del runtime."""

        return object()


def test_e2rank_runtime_forces_eager_mode(monkeypatch) -> None:
    """El runtime E2Rank debe desactivar compile/cudagraphs por defecto."""

    captured_kwargs: dict[str, object] = {}

    def _fake_llm(**kwargs):
        captured_kwargs.update(kwargs)
        return _FakeLLM()

    e2rank_engine.get_e2rank_runtime.cache_clear()
    e2rank_engine._REGISTERED_LLMS.clear()
    monkeypatch.setattr(e2rank_engine, "LLM", _fake_llm)

    runtime = e2rank_engine.get_e2rank_runtime(
        "Alibaba-NLP/E2Rank-0.6B",
        dtype="auto",
        max_model_len=30464,
        gpu_memory_utilization=0.30,
        tensor_parallel_size=1,
    )

    assert runtime.effective_max_model_len == 30464
    assert captured_kwargs["enforce_eager"] is True

    e2rank_engine.shutdown_cached_llm()


def test_embedding_runtime_forces_eager_mode(monkeypatch) -> None:
    """El runtime de embeddings del resolver debe cargar en eager."""

    captured_kwargs: dict[str, object] = {}

    def _fake_llm(**kwargs):
        captured_kwargs.update(kwargs)
        return _FakeLLM()

    embedding_stage.get_embedding_runtime.cache_clear()
    embedding_stage.REGISTERED_LLMS.clear()
    monkeypatch.setattr(embedding_stage, "LLM", _fake_llm)

    runtime = embedding_stage.get_embedding_runtime(
        "Qwen/Qwen3-Embedding-0.6B",
        dtype="auto",
        max_model_len=8192,
        gpu_memory_utilization=0.30,
        tensor_parallel_size=1,
        trust_remote_code=True,
    )

    assert runtime.effective_max_model_len == 8192
    assert captured_kwargs["enforce_eager"] is True

    embedding_stage.clear_embedding_runtime()


def test_reranker_runtime_forces_eager_mode(monkeypatch) -> None:
    """El runtime de rerank debe evitar compile/cudagraphs al inicializar."""

    captured_kwargs: dict[str, object] = {}

    class _FakeTokenizer:
        """Stub callable que expone el minimo contrato usado por el reranker."""

        def __init__(self) -> None:
            self.padding_side = "left"
            self.pad_token = None
            self.eos_token = "</s>"

        def __call__(self, _text: str, add_special_tokens: bool = False) -> SimpleNamespace:
            return SimpleNamespace(input_ids=[1])

        def encode(self, _text: str, add_special_tokens: bool = False) -> list[int]:
            return [1]

    fake_tokenizer = _FakeTokenizer()

    def _fake_tokenizer_from_pretrained(*_args, **_kwargs):
        return fake_tokenizer

    def _fake_llm(**kwargs):
        captured_kwargs.update(kwargs)
        return _FakeLLM()

    rerank_stage.get_reranker_runtime.cache_clear()
    rerank_stage.REGISTERED_LLMS.clear()
    monkeypatch.setattr(rerank_stage.AutoTokenizer, "from_pretrained", _fake_tokenizer_from_pretrained)
    monkeypatch.setattr(rerank_stage, "LLM", _fake_llm)

    runtime = rerank_stage.get_reranker_runtime(
        "Qwen/Qwen3-Reranker-0.6B",
        dtype="auto",
        max_model_len=8192,
        gpu_memory_utilization=0.25,
        tensor_parallel_size=1,
        trust_remote_code=True,
    )

    assert runtime.effective_max_model_len == 8192
    assert captured_kwargs["enforce_eager"] is True

    rerank_stage.clear_reranker_runtime()
