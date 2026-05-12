#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Pruebas del ciclo de vida comun de `VLLMModelRunner`."""

from __future__ import annotations

import pytest

from llm_core.vllm_engine import ModelRuntimeProfile, VLLMConfig
from llm_core.vllm_interface import VLLMModelRunner


class _ExplodingRunner(VLLMModelRunner):
    """Runner de prueba que falla en `generate()` para validar cleanup."""

    def __init__(self) -> None:
        super().__init__()
        self.cleaned = False

    def get_model_profile(self) -> ModelRuntimeProfile:
        return ModelRuntimeProfile(
            alias="exploding",
            canonical_model_name="exploding/model",
            model_name="exploding/model",
        )

    def configure(self, system_prompt: str, user_prompt: str) -> VLLMConfig:  # type: ignore[override]
        cfg = VLLMConfig(
            model="exploding/model",
            gpu_memory_utilization=0.5,
            max_model_len=64,
            max_tokens=8,
            temperature=0.0,
            top_p=1.0,
            top_k=1,
            repetition_penalty=1.0,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
        self.cfg = cfg
        return cfg

    def load_tokenizer(self):  # type: ignore[override]
        return object()

    def build_prompt(self, system_prompt: str, user_prompt: str) -> str:  # type: ignore[override]
        return "prompt"

    def load_model(self):  # type: ignore[override]
        return object()

    def generate(self, prompt: str) -> list[str]:  # type: ignore[override]
        raise RuntimeError("boom")

    def cleanup(self) -> None:  # type: ignore[override]
        self.cleaned = True


class _BaseRunner(VLLMModelRunner):
    """Runner minimo que reutiliza los hooks concretos de la clase base."""

    def get_model_profile(self) -> ModelRuntimeProfile:
        return ModelRuntimeProfile(
            alias="base",
            canonical_model_name="base/model",
            model_name="base/model",
        )


def test_run_always_calls_cleanup_when_generate_fails() -> None:
    """`run()` debe ejecutar cleanup incluso si la generacion lanza una excepcion."""

    runner = _ExplodingRunner()

    with pytest.raises(RuntimeError, match="boom"):
        runner.run("Sistema", "Usuario")

    assert runner.cleaned is True


def test_load_model_retries_with_lower_max_model_len(monkeypatch: pytest.MonkeyPatch) -> None:
    """La carga comun debe reintentar con menos contexto tras un error de KV cache."""

    runner = _BaseRunner()
    runner.cfg = VLLMConfig(
        model="base/model",
        gpu_memory_utilization=0.5,
        max_model_len=4096,
        max_tokens=1024,
        temperature=0.0,
        top_p=1.0,
        top_k=1,
        repetition_penalty=1.0,
    )
    calls: list[object] = []

    def _fake_build_llm_instance() -> object:
        calls.append(runner.cfg.max_model_len)
        if len([call for call in calls if isinstance(call, int)]) == 1:
            raise RuntimeError("Engine core initialization failed: estimated maximum model length is 3072")
        return object()

    monkeypatch.setattr(runner, "_build_llm_instance", _fake_build_llm_instance)
    monkeypatch.setattr("llm_core.vllm_interface.release_cuda_memory", lambda: calls.append("release"))

    loaded_model = runner.load_model()

    assert loaded_model is runner._llm
    assert runner.cfg.max_model_len == 3072
    assert runner.cfg.max_tokens == 1024
    assert calls == [4096, "release", 3072]


def test_load_tokenizer_respects_runtime_tokenizer_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """La carga comun debe respetar `tokenizer_mode` para alinear el prompt con vLLM."""

    runner = _BaseRunner()
    runner.cfg = VLLMConfig(
        model="mistralai/Ministral-3-8B-Reasoning-2512",
        tokenizer="mistralai/Ministral-3-8B-Reasoning-2512",
        gpu_memory_utilization=0.5,
        max_model_len=64,
        max_tokens=8,
        temperature=0.0,
        top_p=1.0,
        top_k=1,
        repetition_penalty=1.0,
        tokenizer_mode="mistral",
        trust_remote_code=True,
        hf_token="secret",
    )
    captured: dict[str, object] = {}

    def _fake_load_uncached_tokenizer(**kwargs: object) -> object:
        captured.update(kwargs)
        return object()

    monkeypatch.setattr("llm_core.vllm_interface.load_uncached_tokenizer", _fake_load_uncached_tokenizer)

    tokenizer = runner.load_tokenizer()

    assert tokenizer is runner._tokenizer
    assert captured["tokenizer_name"] == "mistralai/Ministral-3-8B-Reasoning-2512"
    assert captured["tokenizer_mode"] == "mistral"
    assert captured["hf_token"] == "secret"


def test_cleanup_base_clears_state_and_uses_runtime_utils(monkeypatch: pytest.MonkeyPatch) -> None:
    """El cleanup comun debe apagar el engine una vez y liberar estado interno."""

    runner = _BaseRunner()
    runner.cfg = VLLMConfig(
        model="base/model",
        gpu_memory_utilization=0.5,
        max_model_len=64,
        max_tokens=8,
        temperature=0.0,
        top_p=1.0,
        top_k=1,
        repetition_penalty=1.0,
    )
    runner._tokenizer = object()
    runner._llm = object()
    calls: list[str] = []

    monkeypatch.setattr(
        "llm_core.vllm_interface.shutdown_vllm_engine_once",
        lambda _llm, _shutdown_ids: calls.append("shutdown"),
    )
    monkeypatch.setattr("llm_core.vllm_interface.destroy_distributed_process_group", lambda: calls.append("destroy_pg"))
    monkeypatch.setattr("llm_core.vllm_interface.release_cuda_memory", lambda: calls.append("release_cuda"))

    runner.cleanup()

    assert runner.cfg is None
    assert runner._tokenizer is None
    assert runner._llm is None
    assert calls == ["shutdown", "destroy_pg", "release_cuda"]
