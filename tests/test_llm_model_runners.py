#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Pruebas para los runners concretos registrados en `llm_core`."""

from __future__ import annotations

import pytest

from llm_core.vllm_config_gemma4 import Gemma4Runner, MODEL as GEMMA4_MODEL
from llm_core.vllm_config_gemma4_e4b import Gemma4E4BRunner, MODEL as GEMMA4_E4B_MODEL
from llm_core.vllm_config_qwen36 import Qwen3Runner, MODEL as QWEN3_MODEL
from llm_core.vllm_engine import VLLMRuntimeDefaults


def test_gemma4_runner_configures_base_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    """Gemma 4 26B debe resolver su perfil sin tocar CUDA real ni rutas cuantizadas rotas."""

    monkeypatch.setattr("llm_core.vllm_engine.torch.cuda.is_available", lambda: False)

    runner = Gemma4Runner(runtime_defaults=VLLMRuntimeDefaults(async_scheduling=True, max_tokens=32))
    cfg = runner.configure("Sistema", "Usuario")

    assert cfg.model == GEMMA4_MODEL
    assert cfg.quantization is None
    assert cfg.async_scheduling is True
    assert cfg.temperature == pytest.approx(0.7)
    assert cfg.top_k == 50


def test_qwen3_runner_configures_profile_and_strips_think_blocks(monkeypatch: pytest.MonkeyPatch) -> None:
    """Qwen3 debe heredar el config builder comun y limpiar bloques `<think>`."""

    monkeypatch.setattr("llm_core.vllm_engine.torch.cuda.is_available", lambda: False)

    runner = Qwen3Runner(runtime_defaults=VLLMRuntimeDefaults(async_scheduling=True, max_tokens=64))
    cfg = runner.configure("Sistema", "Usuario")

    assert cfg.model == QWEN3_MODEL
    assert cfg.async_scheduling is True
    assert cfg.temperature == pytest.approx(0.6)
    assert cfg.top_k == 20
    assert runner._post_process_output("<think>razonamiento</think> respuesta") == "respuesta"
    assert runner._post_process_output("<think>incompleto") == ""


def test_gemma4_e4b_runner_uses_quantized_profile_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    """El perfil default de E4B debe seguir siendo el modo cuantizado seguro."""

    monkeypatch.delenv("GEMMA4_E4B_RUNTIME_MODE", raising=False)
    monkeypatch.setattr("llm_core.vllm_engine.torch.cuda.is_available", lambda: False)

    runner = Gemma4E4BRunner(runtime_defaults=VLLMRuntimeDefaults(max_tokens=4096))
    cfg = runner.configure("Sistema", "Usuario")

    assert cfg.model == "Chunity/gemma-4-E4B-it-AWQ-4bit"
    assert cfg.tokenizer == "Chunity/gemma-4-E4B-it-AWQ-4bit"
    assert cfg.quantization == "awq"
    assert cfg.dtype == "float16"
    assert cfg.gpu_memory_utilization == pytest.approx(0.82)
    assert cfg.cpu_offload_gb == 0.0
    assert cfg.enforce_eager is True
    assert cfg.max_model_len == 2048
    assert cfg.max_tokens == 2048
    assert cfg.max_num_batched_tokens == 2048
    assert cfg.max_num_seqs == 1
    assert cfg.block_size == 64
    assert cfg.async_scheduling is False


def test_gemma4_e4b_runner_quantized_mode_stays_gpu_only_on_target_hardware(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """La AWQ de E4B no debe activar offload en la 3080 Ti objetivo cuando hay VRAM suficiente."""

    monkeypatch.delenv("GEMMA4_E4B_RUNTIME_MODE", raising=False)
    monkeypatch.setattr("llm_core.vllm_engine.torch.cuda.is_available", lambda: True)
    monkeypatch.setattr(
        "llm_core.vllm_engine.torch.cuda.mem_get_info",
        lambda: (int(16.0 * (1024**3)), int(16.0 * (1024**3))),
    )
    monkeypatch.setattr("llm_core.vllm_engine._get_system_ram_gib", lambda: 64.0)

    runner = Gemma4E4BRunner(runtime_defaults=VLLMRuntimeDefaults(max_tokens=4096))
    cfg = runner.configure("Sistema", "Usuario")

    assert cfg.cpu_offload_gb == 0.0
    assert cfg.disable_hybrid_kv_cache_manager is False


def test_gemma4_e4b_runner_supports_bf16_offload_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    """El perfil alternativo de E4B debe separar claramente el modo bf16 + offload."""

    monkeypatch.setenv("GEMMA4_E4B_RUNTIME_MODE", "bf16_offload")
    monkeypatch.setattr("llm_core.vllm_engine.torch.cuda.is_available", lambda: False)

    runner = Gemma4E4BRunner(runtime_defaults=VLLMRuntimeDefaults(max_tokens=4096))
    cfg = runner.configure("Sistema", "Usuario")

    assert cfg.model == GEMMA4_E4B_MODEL
    assert cfg.quantization is None
    assert cfg.cpu_offload_gb == pytest.approx(8.0)
    assert cfg.max_model_len == 2048
    assert cfg.disable_hybrid_kv_cache_manager is True
