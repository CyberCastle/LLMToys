#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Pruebas unitarias para el core agnostico de `llm_core.vllm_engine`."""

from __future__ import annotations

import logging

import pytest

from llm_core.vllm_engine import (
    MemoryPlan,
    ModelRuntimeProfile,
    QuantizedVariant,
    VLLMConfig,
    VLLMRuntimeDefaults,
    build_vllm_config_from_profile,
    estimate_model_size_gib,
    plan_memory,
    print_effective_config,
    validate_config,
)


def _make_profile(
    *,
    alias: str = "fake",
    bf16_size: float = 20.0,
    awq_size: float = 5.0,
    is_moe: bool = False,
    quantized_variant: QuantizedVariant | None = None,
) -> ModelRuntimeProfile:
    """Construye un perfil minimo reutilizable para pruebas del planner."""

    size_estimates = {"bf16": bf16_size}
    if awq_size > 0.0:
        size_estimates["awq_4bit"] = awq_size
    return ModelRuntimeProfile(
        alias=alias,
        canonical_model_name=f"{alias}/base",
        model_name=f"{alias}/base",
        temperature=0.3,
        top_k=7,
        size_estimates_gib=size_estimates,
        is_moe=is_moe,
        quantized_variant=quantized_variant,
    )


def _make_config(**overrides: object) -> VLLMConfig:
    """Construye una configuracion valida y permite override por campo."""

    cfg = VLLMConfig(
        model="fake/model",
        gpu_memory_utilization=0.8,
        max_model_len=512,
        max_tokens=64,
        temperature=0.3,
        top_p=0.9,
        top_k=8,
        min_p=0.0,
        repetition_penalty=1.0,
    )
    for key, value in overrides.items():
        setattr(cfg, key, value)
    return cfg


def _patch_memory(
    monkeypatch: pytest.MonkeyPatch,
    *,
    free_gib: float,
    total_gib: float,
    ram_available_gib: float,
) -> None:
    """Mockea el estado de memoria visible para el planner."""

    monkeypatch.setattr("llm_core.vllm_engine.torch.cuda.is_available", lambda: True)
    monkeypatch.setattr(
        "llm_core.vllm_engine.torch.cuda.mem_get_info",
        lambda: (int(free_gib * (1024**3)), int(total_gib * (1024**3))),
    )
    monkeypatch.setattr("llm_core.vllm_engine._get_system_ram_gib", lambda: ram_available_gib)


def test_estimate_model_size_normalizes_quantization_aliases() -> None:
    """Los aliases reales de cuantizacion deben resolver al tamaño estimado correcto."""

    profile = _make_profile(awq_size=6.5)

    assert estimate_model_size_gib(profile, "bfloat16", "compressed-tensors") == pytest.approx(6.5)
    assert estimate_model_size_gib(profile, "bfloat16", "awq_marlin") == pytest.approx(6.5)


def test_build_llm_kwargs_includes_hf_token_and_omits_none() -> None:
    """La configuracion para `vllm.LLM` debe incluir auth y omitir campos `None`."""

    cfg = _make_config(hf_token="secret", block_size=None, revision=None)

    kwargs = cfg.build_llm_kwargs()

    assert kwargs["hf_token"] == "secret"
    assert kwargs["tokenizer"] == "fake/model"
    assert "revision" not in kwargs
    assert "block_size" not in kwargs


def test_build_sampling_params_uses_none_when_stop_is_empty() -> None:
    """La lista vacia de stop sequences no debe convertirse en lista vacia efectiva."""

    params = _make_config(stop=[]).build_sampling_params()

    assert params.stop is None


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"max_model_len": 0}, "max_model_len"),
        ({"max_tokens": 0}, "max_tokens"),
        ({"max_tokens": 1024, "max_model_len": 512}, "max_tokens no puede exceder"),
        ({"gpu_memory_utilization": 0.0}, "gpu_memory_utilization"),
        ({"top_p": 0.0}, "top_p"),
        ({"top_k": 0}, "top_k"),
        ({"min_p": 1.5}, "min_p"),
        ({"repetition_penalty": 0.0}, "repetition_penalty"),
        ({"cpu_offload_gb": -1.0}, "cpu_offload_gb"),
    ],
)
def test_validate_config_rejects_invalid_ranges(overrides: dict[str, object], message: str) -> None:
    """La validacion debe cubrir errores de rango y relaciones entre campos."""

    with pytest.raises(ValueError, match=message):
        validate_config(_make_config(**overrides))


def test_validate_config_logs_warning_for_expert_parallel(caplog: pytest.LogCaptureFixture) -> None:
    """Mantiene el warning cuando la topologia declarada es potencialmente riesgosa."""

    caplog.set_level(logging.WARNING)

    validate_config(_make_config(enable_expert_parallel=True, tensor_parallel_size=2))

    assert "enable_expert_parallel=True" in caplog.text


def test_plan_memory_uses_real_free_vram_budget(monkeypatch: pytest.MonkeyPatch) -> None:
    """La VRAM libre real debe poder reducir `gpu_memory_utilization` al planificar."""

    profile = _make_profile(bf16_size=6.0)
    _patch_memory(monkeypatch, free_gib=10.28, total_gib=15.61, ram_available_gib=64.0)

    plan = plan_memory(
        profile=profile,
        dtype="bfloat16",
        gpu_memory_utilization=0.90,
        quantization=None,
        tensor_parallel_size=1,
        auto_quantize=True,
        auto_cpu_offload=True,
        ram_reserve_gb=4.0,
        current_max_model_len=4096,
    )

    assert plan.gpu_memory_utilization == pytest.approx((10.28 - 0.25) / 15.61, rel=1e-4)


def test_plan_memory_returns_cpu_offload_when_ram_suffices(monkeypatch: pytest.MonkeyPatch) -> None:
    """Si no cabe integro en GPU pero si en VRAM+RAM, el planner debe activar offload."""

    profile = _make_profile(bf16_size=20.0)
    _patch_memory(monkeypatch, free_gib=16.0, total_gib=16.0, ram_available_gib=24.0)

    plan = plan_memory(
        profile=profile,
        dtype="bfloat16",
        gpu_memory_utilization=0.90,
        quantization=None,
        tensor_parallel_size=1,
        auto_quantize=True,
        auto_cpu_offload=True,
        ram_reserve_gb=4.0,
        current_max_model_len=4096,
    )

    assert plan.cpu_offload_gb > 0.0
    assert plan.enforce_eager is True
    assert plan.quantization is None


def test_plan_memory_selects_quantized_variant_when_bf16_does_not_fit(monkeypatch: pytest.MonkeyPatch) -> None:
    """Cuando bf16 no cabe, el planner debe poder mutar a una variante cuantizada conocida."""

    profile = _make_profile(
        bf16_size=60.0,
        awq_size=18.0,
        is_moe=True,
        quantized_variant=QuantizedVariant(
            model_name="vendor/fake-awq",
            tokenizer_name="vendor/fake-awq",
            quantization="awq_marlin",
        ),
    )
    _patch_memory(monkeypatch, free_gib=16.0, total_gib=16.0, ram_available_gib=14.0)

    plan = plan_memory(
        profile=profile,
        dtype="bfloat16",
        gpu_memory_utilization=0.90,
        quantization=None,
        tensor_parallel_size=1,
        auto_quantize=True,
        auto_cpu_offload=True,
        ram_reserve_gb=4.0,
        current_max_model_len=4096,
    )

    assert plan.model_override == "vendor/fake-awq"
    assert plan.tokenizer_override == "vendor/fake-awq"
    assert plan.quantization == "awq_marlin"
    assert plan.cpu_offload_gb > 0.0


def test_plan_memory_quantized_variant_fallback_respects_tensor_parallel_size(monkeypatch: pytest.MonkeyPatch) -> None:
    """El fallback heuristico de una variante cuantizada no debe dividir por TP dos veces."""

    profile = _make_profile(
        bf16_size=40.0,
        awq_size=0.0,
        quantized_variant=QuantizedVariant(
            model_name="vendor/fake-custom-quant",
            tokenizer_name="vendor/fake-custom-quant",
            size_key="custom_4bit",
            quantization="compressed-tensors",
        ),
    )
    _patch_memory(monkeypatch, free_gib=6.0, total_gib=6.0, ram_available_gib=5.0)

    plan = plan_memory(
        profile=profile,
        dtype="bfloat16",
        gpu_memory_utilization=0.90,
        quantization=None,
        tensor_parallel_size=2,
        auto_quantize=True,
        auto_cpu_offload=True,
        ram_reserve_gb=4.0,
        current_max_model_len=4096,
    )

    assert plan.model_override == "vendor/fake-custom-quant"
    assert plan.cpu_offload_gb > 0.0


def test_plan_memory_force_quantized_variant_prefers_known_variant(monkeypatch: pytest.MonkeyPatch) -> None:
    """El flag explicito debe permitir elegir la variante cuantizada aunque bf16 tambien quepa."""

    profile = _make_profile(
        bf16_size=8.0,
        awq_size=4.0,
        quantized_variant=QuantizedVariant(
            model_name="vendor/fake-awq",
            tokenizer_name="vendor/fake-awq",
            quantization="compressed-tensors",
            dtype="float16",
        ),
    )
    _patch_memory(monkeypatch, free_gib=16.0, total_gib=16.0, ram_available_gib=64.0)

    plan = plan_memory(
        profile=profile,
        dtype="bfloat16",
        gpu_memory_utilization=0.90,
        quantization=None,
        tensor_parallel_size=1,
        auto_quantize=True,
        force_quantized_variant=True,
        auto_cpu_offload=True,
        ram_reserve_gb=4.0,
        current_max_model_len=4096,
    )

    assert plan.model_override == "vendor/fake-awq"
    assert plan.tokenizer_override == "vendor/fake-awq"
    assert plan.quantization == "compressed-tensors"
    assert plan.dtype_override == "float16"
    assert plan.cpu_offload_gb == 0.0


def test_plan_memory_force_quantized_variant_requires_profile_support() -> None:
    """Forzar la variante cuantizada debe fallar si el perfil no define una ruta conocida."""

    profile = _make_profile(bf16_size=8.0, awq_size=0.0, quantized_variant=None)

    with pytest.raises(RuntimeError, match="no define `quantized_variant`"):
        plan_memory(
            profile=profile,
            dtype="bfloat16",
            gpu_memory_utilization=0.90,
            quantization=None,
            tensor_parallel_size=1,
            auto_quantize=True,
            force_quantized_variant=True,
            auto_cpu_offload=True,
            ram_reserve_gb=4.0,
            current_max_model_len=4096,
        )


def test_plan_memory_raises_when_auto_quantize_disabled_and_memory_is_insufficient(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Sin cuantizacion automatica, un presupuesto inviable debe fallar de forma explicita."""

    profile = _make_profile(bf16_size=60.0)
    _patch_memory(monkeypatch, free_gib=16.0, total_gib=16.0, ram_available_gib=12.0)

    with pytest.raises(RuntimeError, match="Memoria insuficiente"):
        plan_memory(
            profile=profile,
            dtype="bfloat16",
            gpu_memory_utilization=0.90,
            quantization=None,
            tensor_parallel_size=1,
            auto_quantize=False,
            auto_cpu_offload=True,
            ram_reserve_gb=4.0,
            current_max_model_len=4096,
        )


def test_plan_memory_validates_manual_quantization_instead_of_skipping(monkeypatch: pytest.MonkeyPatch) -> None:
    """Una cuantizacion explicita debe seguir validando capacidad real de memoria."""

    profile = _make_profile(bf16_size=40.0, awq_size=14.0)
    _patch_memory(monkeypatch, free_gib=16.0, total_gib=16.0, ram_available_gib=6.0)

    with pytest.raises(RuntimeError, match="quantization='awq'"):
        plan_memory(
            profile=profile,
            dtype="bfloat16",
            gpu_memory_utilization=0.90,
            quantization="awq",
            tensor_parallel_size=1,
            auto_quantize=True,
            auto_cpu_offload=True,
            ram_reserve_gb=4.0,
            current_max_model_len=4096,
        )


def test_build_vllm_config_from_profile_merges_profile_defaults_and_plan(monkeypatch: pytest.MonkeyPatch) -> None:
    """El builder comun debe propagar scheduling, caps y overrides del planner."""

    profile = ModelRuntimeProfile(
        alias="profile",
        canonical_model_name="profile/base",
        model_name="profile/base",
        temperature=0.2,
        top_k=11,
        size_estimates_gib={"bf16": 8.0},
        gpu_memory_utilization_cap=0.82,
        max_model_len_cap=2048,
        max_num_batched_tokens_default=2048,
        max_num_seqs_default=1,
        block_size_default=64,
        enforce_eager_default=True,
        async_scheduling_default=False,
    )
    monkeypatch.setattr(
        "llm_core.vllm_engine.plan_memory",
        lambda **_kwargs: MemoryPlan(gpu_memory_utilization=0.81),
    )

    cfg = build_vllm_config_from_profile(
        profile,
        runtime_defaults=VLLMRuntimeDefaults(async_scheduling=True, max_tokens=4096),
        system_prompt="Sistema",
        user_prompt="Usuario",
        hf_token="secret",
    )

    assert cfg.gpu_memory_utilization == pytest.approx(0.81)
    assert cfg.async_scheduling is True
    assert cfg.enforce_eager is True
    assert cfg.max_model_len == 2048
    assert cfg.max_tokens == 2048
    assert cfg.max_num_batched_tokens == 2048
    assert cfg.max_num_seqs == 1
    assert cfg.block_size == 64


def test_build_vllm_config_disables_hybrid_kv_cache_manager_with_cpu_offload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """El builder debe apagar HMA cuando haya offload de CPU efectivo para evitar fallos de init en vLLM."""

    profile = ModelRuntimeProfile(
        alias="profile",
        canonical_model_name="profile/base",
        model_name="profile/base",
        temperature=0.2,
        top_k=11,
        size_estimates_gib={"bf16": 8.0},
    )
    monkeypatch.setattr(
        "llm_core.vllm_engine.plan_memory",
        lambda **_kwargs: MemoryPlan(cpu_offload_gb=1.5),
    )

    cfg = build_vllm_config_from_profile(
        profile,
        runtime_defaults=VLLMRuntimeDefaults(),
        system_prompt="Sistema",
        user_prompt="Usuario",
        hf_token=None,
    )

    assert cfg.cpu_offload_gb == pytest.approx(1.5)
    assert cfg.disable_hybrid_kv_cache_manager is True


def test_print_effective_config_redacts_hf_token(caplog: pytest.LogCaptureFixture) -> None:
    """El diagnostico no debe filtrar tokens sensibles en logs."""

    caplog.set_level(logging.INFO)

    print_effective_config(_make_config(hf_token="super-secret"))

    assert "super-secret" not in caplog.text
    assert '"hf_token": "***"' in caplog.text
