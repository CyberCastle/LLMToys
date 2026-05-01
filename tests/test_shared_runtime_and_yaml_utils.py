#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Pruebas para utilidades compartidas de runtime y YAML."""

from __future__ import annotations

from pathlib import Path

import yaml
import pytest

from llm_core.vllm_runtime_utils import (
    destroy_distributed_process_group,
    iter_exception_messages,
    release_cuda_memory,
    resolve_fallback_max_model_len,
    should_try_stepdown_fallback,
    shutdown_vllm_engine_once,
)
from nl2sql.utils.yaml_utils import load_yaml_mapping


def test_load_yaml_mapping_raises_for_missing_file(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.yaml"

    with pytest.raises(FileNotFoundError):
        load_yaml_mapping(missing_path, artifact_name="missing")


def test_load_yaml_mapping_raises_for_non_mapping_yaml(tmp_path: Path) -> None:
    payload_path = tmp_path / "list.yaml"
    payload_path.write_text(yaml.safe_dump(["a", "b"], sort_keys=False, allow_unicode=True), encoding="utf-8")

    with pytest.raises(ValueError):
        load_yaml_mapping(payload_path, artifact_name="list")


def test_load_yaml_mapping_returns_mapping_payload(tmp_path: Path) -> None:
    payload_path = tmp_path / "mapping.yaml"
    payload_path.write_text(yaml.safe_dump({"foo": 1, "bar": "baz"}, sort_keys=False, allow_unicode=True), encoding="utf-8")

    assert load_yaml_mapping(payload_path, artifact_name="mapping") == {"foo": 1, "bar": "baz"}


def test_release_cuda_memory_collects_gc_and_empties_cuda_cache_when_available(monkeypatch) -> None:
    calls: list[str] = []

    monkeypatch.setattr("llm_core.vllm_runtime_utils.gc.collect", lambda: calls.append("gc"))
    monkeypatch.setattr("llm_core.vllm_runtime_utils.torch.cuda.is_available", lambda: True)
    monkeypatch.setattr("llm_core.vllm_runtime_utils.torch.cuda.empty_cache", lambda: calls.append("cuda"))

    release_cuda_memory()

    assert calls == ["gc", "cuda"]


def test_release_cuda_memory_skips_empty_cache_when_cuda_is_unavailable(monkeypatch) -> None:
    calls: list[str] = []

    monkeypatch.setattr("llm_core.vllm_runtime_utils.gc.collect", lambda: calls.append("gc"))
    monkeypatch.setattr("llm_core.vllm_runtime_utils.torch.cuda.is_available", lambda: False)
    monkeypatch.setattr("llm_core.vllm_runtime_utils.torch.cuda.empty_cache", lambda: calls.append("cuda"))

    release_cuda_memory()

    assert calls == ["gc"]


def test_iter_exception_messages_walks_exception_chain() -> None:
    """Recolecta los mensajes utiles de `__cause__` y `__context__` sin duplicados."""

    root_error = ValueError("root")
    chained_error = RuntimeError("middle")
    chained_error.__cause__ = root_error
    top_error = RuntimeError("top")
    top_error.__cause__ = chained_error

    assert iter_exception_messages(top_error) == ["top", "middle", "root"]


def test_resolve_fallback_max_model_len_rounds_down_to_step() -> None:
    """El fallback de contexto debe redondear hacia abajo usando el step global."""

    error = RuntimeError("Engine core initialization failed: estimated maximum model length is 3187")

    assert resolve_fallback_max_model_len(4096, error) == 3072


def test_should_try_stepdown_fallback_detects_kv_cache_errors() -> None:
    """Los errores de engine/KV cache deben habilitar el reintento con menos contexto."""

    assert should_try_stepdown_fallback(RuntimeError("Engine core initialization failed")) is True
    assert should_try_stepdown_fallback(RuntimeError("KV cache no puede inicializarse")) is True
    assert should_try_stepdown_fallback(RuntimeError("otro error")) is False


def test_shutdown_vllm_engine_once_is_idempotent() -> None:
    """El shutdown del engine debe ejecutarse una sola vez por instancia de LLM."""

    calls: list[str] = []

    class _FakeEngineCore:
        def shutdown(self) -> None:
            calls.append("shutdown")

    class _FakeLLMEngine:
        def __init__(self) -> None:
            self.engine_core = _FakeEngineCore()

    class _FakeLLM:
        def __init__(self) -> None:
            self.llm_engine = _FakeLLMEngine()

    llm = _FakeLLM()
    shutdown_ids: set[int] = set()

    shutdown_vllm_engine_once(llm, shutdown_ids)
    shutdown_vllm_engine_once(llm, shutdown_ids)

    assert calls == ["shutdown"]


def test_destroy_distributed_process_group_is_safe_when_unavailable(monkeypatch) -> None:
    """La destruccion del process group debe no-op si PyTorch no inicializo distribuido."""

    calls: list[str] = []

    monkeypatch.setattr("llm_core.vllm_runtime_utils.torch.distributed.is_available", lambda: True)
    monkeypatch.setattr("llm_core.vllm_runtime_utils.torch.distributed.is_initialized", lambda: False)
    monkeypatch.setattr(
        "llm_core.vllm_runtime_utils.torch.distributed.destroy_process_group",
        lambda: calls.append("destroy"),
    )

    destroy_distributed_process_group()

    assert calls == []
