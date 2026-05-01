#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
vllm_engine.py

Infraestructura agnostica para ejecutar modelos con vLLM.

Este modulo no debe conocer particularidades de un LLM concreto. Su
responsabilidad se limita a:

  - describir configuracion y perfiles de runtime con dataclasses tipadas
  - construir `VLLMConfig` a partir de un perfil y defaults compartidos
  - planificar memoria CPU/GPU sin side effects de import por modelo
  - renderizar prompts de chat, validar configuraciones y emitir diagnostico

Toda decision especifica de modelo vive en `VLLMModelRunner` y en los perfiles
de cada runner concreto.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from typing import Any, Mapping

import torch
from vllm import SamplingParams

from llm_core.vllm_runtime_utils import resolve_gpu_utilization_from_available_memory

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VLLMRuntimeDefaults:
    """Configuracion compartida e inmutable para runners basados en vLLM."""

    tokenizer: str | None = None
    revision: str | None = None
    tokenizer_revision: str | None = None
    trust_remote_code: bool = True
    tokenizer_mode: str = "auto"
    skip_tokenizer_init: bool = False
    seed: int = 42

    dtype: str = "bfloat16"
    quantization: str | None = None
    kv_cache_dtype: str = "auto"
    calculate_kv_scales: bool = False

    max_model_len: int = 4096
    disable_sliding_window: bool = False
    disable_hybrid_kv_cache_manager: bool = False

    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    distributed_executor_backend: str | None = None
    max_parallel_loading_workers: int | None = None

    enable_expert_parallel: bool = False
    enable_ep_weight_filter: bool = False
    all2all_backend: str | None = None
    enable_eplb: bool = False
    expert_placement_strategy: str = "linear"
    eplb_config: dict[str, Any] | None = None

    gpu_memory_utilization: float = 0.92
    block_size: int | None = None
    cpu_offload_gb: float = 0.0
    kv_offloading_size: float | None = None
    kv_offloading_backend: str = "native"
    num_gpu_blocks_override: int | None = None
    auto_cpu_offload: bool = True
    auto_quantize: bool = True
    ram_reserve_gb: float = 4.0

    max_num_batched_tokens: int | None = None
    max_num_seqs: int | None = None
    max_num_partial_prefills: int = 1
    max_long_partial_prefills: int = 1
    long_prefill_token_threshold: int = 0
    async_scheduling: bool | None = None
    enable_prefix_caching: bool = True
    prefix_caching_hash_algo: str = "sha256"

    enforce_eager: bool = False
    disable_custom_all_reduce: bool = False

    top_p: float = 0.95
    min_p: float = 0.0
    repetition_penalty: float = 1.0
    max_tokens: int = 512
    stop: tuple[str, ...] = ()

    print_config: bool = False


@dataclass(frozen=True)
class QuantizedVariant:
    """Describe una variante pre-cuantizada conocida para un perfil de modelo."""

    model_name: str
    tokenizer_name: str | None = None
    size_key: str = "awq_4bit"
    quantization: str | None = None
    kernel_backend: str | None = None
    dtype: str | None = None


@dataclass(frozen=True)
class ModelRuntimeProfile:
    """Perfil inmutable con toda la informacion especifica de un modelo."""

    alias: str
    canonical_model_name: str
    model_name: str
    tokenizer_name: str | None = None
    temperature: float = 0.7
    top_k: int = 50
    size_estimates_gib: Mapping[str, float] = field(default_factory=dict)
    is_moe: bool = False
    quantized_variant: QuantizedVariant | None = None
    gpu_memory_utilization_cap: float | None = None
    min_cpu_offload_gb: float = 0.0
    max_model_len_cap: int | None = None
    max_num_batched_tokens_default: int | None = None
    max_num_seqs_default: int | None = None
    block_size_default: int | None = None
    enforce_eager_default: bool = False
    async_scheduling_default: bool | None = None
    dtype_default: str | None = None
    quantization_default: str | None = None

    def effective_tokenizer_name(self) -> str | None:
        """Devuelve el tokenizer propio del perfil cuando difiere del modelo."""

        return self.tokenizer_name


@dataclass
class VLLMConfig:
    """Configuracion efectiva enviada al runtime de vLLM y al tokenizer."""

    model: str = ""
    tokenizer: str | None = None
    revision: str | None = None
    tokenizer_revision: str | None = None
    trust_remote_code: bool = True
    tokenizer_mode: str = "auto"
    skip_tokenizer_init: bool = False
    hf_token: str | None = None
    seed: int = 42

    dtype: str = "bfloat16"
    quantization: str | None = None
    kv_cache_dtype: str = "auto"
    calculate_kv_scales: bool = False

    max_model_len: int = 8192
    disable_sliding_window: bool = False
    disable_hybrid_kv_cache_manager: bool = False

    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    distributed_executor_backend: str | None = None
    max_parallel_loading_workers: int | None = None

    enable_expert_parallel: bool = False
    enable_ep_weight_filter: bool = False
    all2all_backend: str | None = None
    enable_eplb: bool = False
    expert_placement_strategy: str = "linear"
    eplb_config: dict[str, Any] | None = None

    gpu_memory_utilization: float = 0.92
    block_size: int | None = None
    cpu_offload_gb: float = 0.0
    kv_offloading_size: float | None = None
    kv_offloading_backend: str = "native"
    num_gpu_blocks_override: int | None = None

    max_num_batched_tokens: int | None = None
    max_num_seqs: int | None = None
    max_num_partial_prefills: int = 1
    max_long_partial_prefills: int = 1
    long_prefill_token_threshold: int = 0
    async_scheduling: bool | None = None
    enable_prefix_caching: bool = True
    prefix_caching_hash_algo: str = "sha256"

    enforce_eager: bool = False
    disable_custom_all_reduce: bool = False

    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 50
    min_p: float = 0.0
    repetition_penalty: float = 1.0
    max_tokens: int = 512
    stop: list[str] = field(default_factory=list)

    system_prompt: str = ""
    user_prompt: str = ""

    def build_llm_kwargs(self) -> dict[str, Any]:
        """Construye kwargs limpios para `vllm.LLM`, omitiendo `None`."""

        kwargs: dict[str, Any] = {
            "model": self.model,
            "tokenizer": self.tokenizer or self.model,
            "revision": self.revision,
            "tokenizer_revision": self.tokenizer_revision,
            "trust_remote_code": self.trust_remote_code,
            "tokenizer_mode": self.tokenizer_mode,
            "skip_tokenizer_init": self.skip_tokenizer_init,
            "dtype": self.dtype,
            "quantization": self.quantization,
            "kv_cache_dtype": self.kv_cache_dtype,
            "calculate_kv_scales": self.calculate_kv_scales,
            "max_model_len": self.max_model_len,
            "disable_sliding_window": self.disable_sliding_window,
            "disable_hybrid_kv_cache_manager": self.disable_hybrid_kv_cache_manager,
            "tensor_parallel_size": self.tensor_parallel_size,
            "pipeline_parallel_size": self.pipeline_parallel_size,
            "distributed_executor_backend": self.distributed_executor_backend,
            "max_parallel_loading_workers": self.max_parallel_loading_workers,
            "enable_expert_parallel": self.enable_expert_parallel,
            "enable_ep_weight_filter": self.enable_ep_weight_filter,
            "all2all_backend": self.all2all_backend,
            "enable_eplb": self.enable_eplb,
            "expert_placement_strategy": self.expert_placement_strategy,
            "eplb_config": self.eplb_config,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "block_size": self.block_size,
            "cpu_offload_gb": self.cpu_offload_gb,
            "kv_offloading_size": self.kv_offloading_size,
            "kv_offloading_backend": self.kv_offloading_backend,
            "num_gpu_blocks_override": self.num_gpu_blocks_override,
            "max_num_batched_tokens": self.max_num_batched_tokens,
            "max_num_seqs": self.max_num_seqs,
            "max_num_partial_prefills": self.max_num_partial_prefills,
            "max_long_partial_prefills": self.max_long_partial_prefills,
            "long_prefill_token_threshold": self.long_prefill_token_threshold,
            "async_scheduling": self.async_scheduling,
            "enable_prefix_caching": self.enable_prefix_caching,
            "prefix_caching_hash_algo": self.prefix_caching_hash_algo,
            "enforce_eager": self.enforce_eager,
            "disable_custom_all_reduce": self.disable_custom_all_reduce,
            "seed": self.seed,
            "hf_token": self.hf_token,
        }
        return {key: value for key, value in kwargs.items() if value is not None}

    def build_sampling_params(self) -> SamplingParams:
        """Construye `SamplingParams` a partir de la configuracion efectiva."""

        sampling_params = SamplingParams(
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
            min_p=self.min_p,
            repetition_penalty=self.repetition_penalty,
            max_tokens=self.max_tokens,
            stop=self.stop or None,
        )
        if not self.stop:
            sampling_params.stop = None
        return sampling_params


@dataclass
class MemoryPlan:
    """Resultado del planificador de memoria CPU/GPU."""

    cpu_offload_gb: float = 0.0
    quantization: str | None = None
    enforce_eager: bool = False
    max_model_len: int | None = None
    model_override: str | None = None
    tokenizer_override: str | None = None
    kernel_backend: str | None = None
    dtype_override: str | None = None
    gpu_memory_utilization: float | None = None


def build_prompt(tokenizer: Any, system_prompt: str, user_prompt: str) -> str:
    """Renderiza un prompt de chat usando el tokenizer si soporta templates."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    chat_template_fn = getattr(tokenizer, "apply_chat_template", None)
    if callable(chat_template_fn):
        prompt = chat_template_fn(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        if not isinstance(prompt, str):
            raise TypeError("El chat template del tokenizer debe devolver un str")
        return prompt
    return f"<system>\n{system_prompt}\n</system>\n<user>\n{user_prompt}\n</user>\n<assistant>\n"


def _get_system_ram_gib() -> float:
    """Retorna la RAM disponible del sistema en GiB."""

    import psutil

    mem = psutil.virtual_memory()
    return mem.available / (1024**3)


def normalize_quantization_alias(quantization: str | None) -> str | None:
    """Normaliza aliases de cuantizacion a claves de estimacion de pesos."""

    if quantization is None:
        return None

    normalized = quantization.strip().lower()
    alias_map = {
        "awq": "awq_4bit",
        "awq_marlin": "awq_4bit",
        "compressed-tensors": "awq_4bit",
        "cpu_awq": "awq_4bit",
        "gptq": "awq_4bit",
        "gptq_marlin": "awq_4bit",
        "bitsandbytes": "awq_4bit",
        "modelopt_fp4": "awq_4bit",
        "mxfp4": "awq_4bit",
        "petit_nvfp4": "awq_4bit",
        "fp8": "fp8",
        "fbgemm_fp8": "fp8",
        "modelopt_mxfp8": "fp8",
        "modelopt_mixed": "fp8",
        "mxfp8": "fp8",
        "experts_int8": "fp8",
    }
    return alias_map.get(normalized, normalized)


def estimate_model_size_gib(
    profile: ModelRuntimeProfile,
    dtype: str,
    quantization: str | None = None,
) -> float | None:
    """Estima el tamaño de pesos del modelo en GiB para un perfil dado."""

    sizes = dict(profile.size_estimates_gib)
    if not sizes:
        return None

    normalized_quantization = normalize_quantization_alias(quantization)
    if normalized_quantization == "awq_4bit":
        return sizes.get("awq_4bit", sizes.get("bf16", 0.0) / 4)
    if normalized_quantization == "fp8":
        return sizes.get("fp8", sizes.get("bf16", 0.0) / 2)
    if dtype in {"float32", "fp32"}:
        return sizes.get("bf16", 0.0) * 2
    return sizes.get("bf16")


def _resolve_variant_fit(
    *,
    weights_per_gpu_gib: float,
    vram_for_weights_gib: float,
    existing_cpu_offload_gb: float,
    ram_for_offload_gib: float,
    auto_cpu_offload: bool,
) -> tuple[bool, float, float]:
    """Determina si una variante cabe y cuanto offload requeriria."""

    if weights_per_gpu_gib <= vram_for_weights_gib:
        return True, existing_cpu_offload_gb, 0.0

    offload_needed = max(weights_per_gpu_gib - vram_for_weights_gib, 0.0)
    if offload_needed <= existing_cpu_offload_gb:
        return True, existing_cpu_offload_gb, offload_needed
    if auto_cpu_offload and offload_needed <= ram_for_offload_gib:
        return True, offload_needed, offload_needed
    return False, existing_cpu_offload_gb, offload_needed


def _build_memory_error(
    *,
    profile: ModelRuntimeProfile,
    weights_per_gpu_gib: float,
    vram_usable_gib: float,
    vram_for_weights_gib: float,
    ram_available_gib: float,
    ram_for_offload_gib: float,
    quantization: str | None,
) -> RuntimeError:
    """Construye un error consistente cuando el plan de memoria no es viable."""

    quantization_label = quantization or "bf16"
    return RuntimeError(
        "Memoria insuficiente para cargar el modelo con la configuracion solicitada. "
        f"alias={profile.alias!r}, quantization={quantization_label!r}, "
        f"pesos_por_gpu≈{weights_per_gpu_gib:.2f} GiB, vram_usable≈{vram_usable_gib:.2f} GiB, "
        f"vram_para_pesos≈{vram_for_weights_gib:.2f} GiB, ram_disponible≈{ram_available_gib:.2f} GiB, "
        f"ram_para_offload≈{ram_for_offload_gib:.2f} GiB."
    )


def plan_memory(
    *,
    profile: ModelRuntimeProfile,
    dtype: str,
    gpu_memory_utilization: float,
    quantization: str | None,
    tensor_parallel_size: int,
    auto_quantize: bool,
    auto_cpu_offload: bool,
    ram_reserve_gb: float,
    current_max_model_len: int,
    existing_cpu_offload_gb: float = 0.0,
) -> MemoryPlan:
    """Planifica memoria CPU/GPU para un perfil sin depender del modelo concreto."""

    plan = MemoryPlan(quantization=quantization)
    if not torch.cuda.is_available():
        return plan

    gib = 1024**3
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    free_gib = free_bytes / gib
    total_gib = total_bytes / gib

    effective_gpu_memory_utilization = resolve_gpu_utilization_from_available_memory(
        gpu_memory_utilization,
        free_memory_gib=free_gib,
        total_memory_gib=total_gib,
    )
    if effective_gpu_memory_utilization < gpu_memory_utilization:
        plan.gpu_memory_utilization = effective_gpu_memory_utilization
        logger.info(
            "Ajustando gpu_memory_utilization %.4f -> %.4f por VRAM libre real.",
            gpu_memory_utilization,
            effective_gpu_memory_utilization,
        )

    vram_usable = total_gib * (plan.gpu_memory_utilization or gpu_memory_utilization)
    vram_for_weights = vram_usable * 0.75
    ram_available = _get_system_ram_gib()
    ram_for_offload = max(0.0, ram_available - ram_reserve_gb)

    requested_model_size_gib = estimate_model_size_gib(profile, dtype, quantization)
    if requested_model_size_gib is None:
        logger.info("No hay estimacion de tamaño para el perfil %s; se omite el planner.", profile.alias)
        return plan

    requested_weights_per_gpu = requested_model_size_gib / tensor_parallel_size
    fits_requested_variant, suggested_offload, _offload_needed = _resolve_variant_fit(
        weights_per_gpu_gib=requested_weights_per_gpu,
        vram_for_weights_gib=vram_for_weights,
        existing_cpu_offload_gb=existing_cpu_offload_gb,
        ram_for_offload_gib=ram_for_offload,
        auto_cpu_offload=auto_cpu_offload,
    )
    if fits_requested_variant:
        if suggested_offload > existing_cpu_offload_gb:
            plan.cpu_offload_gb = suggested_offload
            plan.enforce_eager = True
        logger.info(
            "El perfil %s cabe sin cambiar de variante. pesos_por_gpu≈%.2f GiB, vram_para_pesos≈%.2f GiB, offload≈%.2f GiB.",
            profile.alias,
            requested_weights_per_gpu,
            vram_for_weights,
            plan.cpu_offload_gb,
        )
        return plan

    if quantization is not None:
        raise _build_memory_error(
            profile=profile,
            weights_per_gpu_gib=requested_weights_per_gpu,
            vram_usable_gib=vram_usable,
            vram_for_weights_gib=vram_for_weights,
            ram_available_gib=ram_available,
            ram_for_offload_gib=ram_for_offload,
            quantization=quantization,
        )

    if not auto_quantize:
        raise _build_memory_error(
            profile=profile,
            weights_per_gpu_gib=requested_weights_per_gpu,
            vram_usable_gib=vram_usable,
            vram_for_weights_gib=vram_for_weights,
            ram_available_gib=ram_available,
            ram_for_offload_gib=ram_for_offload,
            quantization=None,
        )

    quantized_variant = profile.quantized_variant
    if quantized_variant is not None:
        quantized_size_gib = estimate_model_size_gib(profile, dtype, quantized_variant.size_key)
        if quantized_size_gib is None:
            quantized_size_gib = requested_weights_per_gpu / (3 if profile.is_moe else 4)
        quantized_weights_per_gpu = quantized_size_gib / tensor_parallel_size
        fits_quantized_variant, suggested_quantized_offload, quantized_offload_needed = _resolve_variant_fit(
            weights_per_gpu_gib=quantized_weights_per_gpu,
            vram_for_weights_gib=vram_for_weights,
            existing_cpu_offload_gb=existing_cpu_offload_gb,
            ram_for_offload_gib=ram_for_offload,
            auto_cpu_offload=auto_cpu_offload,
        )
        plan.model_override = quantized_variant.model_name
        plan.tokenizer_override = quantized_variant.tokenizer_name
        plan.quantization = quantized_variant.quantization
        plan.kernel_backend = quantized_variant.kernel_backend
        plan.dtype_override = quantized_variant.dtype
        if fits_quantized_variant:
            if suggested_quantized_offload > existing_cpu_offload_gb:
                plan.cpu_offload_gb = suggested_quantized_offload
                plan.enforce_eager = True
            logger.info(
                "Auto-quantize selecciona la variante %s para %s. pesos_por_gpu≈%.2f GiB, offload≈%.2f GiB.",
                quantized_variant.model_name,
                profile.alias,
                quantized_weights_per_gpu,
                plan.cpu_offload_gb,
            )
            return plan

        plan.cpu_offload_gb = max(existing_cpu_offload_gb, min(quantized_offload_needed, ram_for_offload))
        plan.enforce_eager = True
        plan.max_model_len = min(current_max_model_len, 2048)
        logger.warning(
            "Memoria muy ajustada para %s incluso con la variante cuantizada %s. offload≈%.2f GiB, max_model_len=%s.",
            profile.alias,
            quantized_variant.model_name,
            plan.cpu_offload_gb,
            plan.max_model_len,
        )
        return plan

    fallback_quantization = "fp8" if profile.is_moe else "bitsandbytes"
    quantized_size_gib = estimate_model_size_gib(profile, dtype, fallback_quantization)
    if quantized_size_gib is None:
        quantized_size_gib = requested_weights_per_gpu / (2 if profile.is_moe else 4)
    quantized_weights_per_gpu = quantized_size_gib / tensor_parallel_size
    fits_fallback_variant, suggested_fallback_offload, fallback_offload_needed = _resolve_variant_fit(
        weights_per_gpu_gib=quantized_weights_per_gpu,
        vram_for_weights_gib=vram_for_weights,
        existing_cpu_offload_gb=existing_cpu_offload_gb,
        ram_for_offload_gib=ram_for_offload,
        auto_cpu_offload=auto_cpu_offload,
    )
    plan.quantization = fallback_quantization
    if fits_fallback_variant:
        if suggested_fallback_offload > existing_cpu_offload_gb:
            plan.cpu_offload_gb = suggested_fallback_offload
            plan.enforce_eager = True
        logger.info(
            "Auto-quantize usara %s para %s. pesos_por_gpu≈%.2f GiB, offload≈%.2f GiB.",
            fallback_quantization,
            profile.alias,
            quantized_weights_per_gpu,
            plan.cpu_offload_gb,
        )
        return plan

    plan.cpu_offload_gb = max(existing_cpu_offload_gb, min(fallback_offload_needed, ram_for_offload))
    plan.enforce_eager = True
    plan.max_model_len = min(current_max_model_len, 2048)
    logger.warning(
        "Memoria muy ajustada para %s con cuantizacion %s. offload≈%.2f GiB, max_model_len=%s.",
        profile.alias,
        fallback_quantization,
        plan.cpu_offload_gb,
        plan.max_model_len,
    )
    return plan


def build_vllm_config_from_profile(
    profile: ModelRuntimeProfile,
    *,
    runtime_defaults: VLLMRuntimeDefaults,
    system_prompt: str,
    user_prompt: str,
    hf_token: str | None,
) -> VLLMConfig:
    """Construye una `VLLMConfig` completa a partir de perfil + defaults."""

    requested_gpu_memory_utilization = runtime_defaults.gpu_memory_utilization
    if profile.gpu_memory_utilization_cap is not None:
        requested_gpu_memory_utilization = min(
            requested_gpu_memory_utilization,
            profile.gpu_memory_utilization_cap,
        )

    requested_cpu_offload_gb = max(runtime_defaults.cpu_offload_gb, profile.min_cpu_offload_gb)
    requested_quantization = profile.quantization_default or runtime_defaults.quantization
    requested_dtype = profile.dtype_default or runtime_defaults.dtype
    requested_max_model_len = runtime_defaults.max_model_len
    if profile.max_model_len_cap is not None:
        requested_max_model_len = min(requested_max_model_len, profile.max_model_len_cap)

    memory_plan = plan_memory(
        profile=profile,
        dtype=requested_dtype,
        gpu_memory_utilization=requested_gpu_memory_utilization,
        quantization=requested_quantization,
        tensor_parallel_size=runtime_defaults.tensor_parallel_size,
        auto_quantize=runtime_defaults.auto_quantize,
        auto_cpu_offload=runtime_defaults.auto_cpu_offload,
        ram_reserve_gb=runtime_defaults.ram_reserve_gb,
        current_max_model_len=requested_max_model_len,
        existing_cpu_offload_gb=requested_cpu_offload_gb,
    )

    model_name = memory_plan.model_override or profile.model_name
    tokenizer_name = memory_plan.tokenizer_override or runtime_defaults.tokenizer or profile.effective_tokenizer_name()
    effective_dtype = memory_plan.dtype_override or requested_dtype
    effective_quantization = memory_plan.quantization if memory_plan.quantization is not None else requested_quantization
    effective_gpu_memory_utilization = memory_plan.gpu_memory_utilization or requested_gpu_memory_utilization
    effective_cpu_offload_gb = max(requested_cpu_offload_gb, memory_plan.cpu_offload_gb)
    effective_max_model_len = requested_max_model_len
    if memory_plan.max_model_len is not None:
        effective_max_model_len = min(effective_max_model_len, memory_plan.max_model_len)

    max_num_batched_tokens = runtime_defaults.max_num_batched_tokens
    if max_num_batched_tokens is None and profile.max_num_batched_tokens_default is not None:
        max_num_batched_tokens = min(profile.max_num_batched_tokens_default, effective_max_model_len)
    elif max_num_batched_tokens is not None:
        max_num_batched_tokens = min(max_num_batched_tokens, effective_max_model_len)

    max_num_seqs = runtime_defaults.max_num_seqs or profile.max_num_seqs_default
    async_scheduling = (
        runtime_defaults.async_scheduling if runtime_defaults.async_scheduling is not None else profile.async_scheduling_default
    )
    block_size = runtime_defaults.block_size or profile.block_size_default
    enforce_eager = runtime_defaults.enforce_eager or profile.enforce_eager_default or memory_plan.enforce_eager
    max_tokens = min(runtime_defaults.max_tokens, effective_max_model_len)
    disable_hybrid_kv_cache_manager = runtime_defaults.disable_hybrid_kv_cache_manager
    if effective_cpu_offload_gb > 0.0:
        disable_hybrid_kv_cache_manager = True

    cfg = VLLMConfig(
        model=model_name,
        tokenizer=tokenizer_name,
        revision=runtime_defaults.revision,
        tokenizer_revision=runtime_defaults.tokenizer_revision,
        trust_remote_code=runtime_defaults.trust_remote_code,
        tokenizer_mode=runtime_defaults.tokenizer_mode,
        skip_tokenizer_init=runtime_defaults.skip_tokenizer_init,
        hf_token=hf_token,
        seed=runtime_defaults.seed,
        dtype=effective_dtype,
        quantization=effective_quantization,
        kv_cache_dtype=runtime_defaults.kv_cache_dtype,
        calculate_kv_scales=runtime_defaults.calculate_kv_scales,
        max_model_len=effective_max_model_len,
        disable_sliding_window=runtime_defaults.disable_sliding_window,
        disable_hybrid_kv_cache_manager=disable_hybrid_kv_cache_manager,
        tensor_parallel_size=runtime_defaults.tensor_parallel_size,
        pipeline_parallel_size=runtime_defaults.pipeline_parallel_size,
        distributed_executor_backend=runtime_defaults.distributed_executor_backend,
        max_parallel_loading_workers=runtime_defaults.max_parallel_loading_workers,
        enable_expert_parallel=runtime_defaults.enable_expert_parallel,
        enable_ep_weight_filter=runtime_defaults.enable_ep_weight_filter,
        all2all_backend=runtime_defaults.all2all_backend,
        enable_eplb=runtime_defaults.enable_eplb,
        expert_placement_strategy=runtime_defaults.expert_placement_strategy,
        eplb_config=runtime_defaults.eplb_config,
        gpu_memory_utilization=effective_gpu_memory_utilization,
        block_size=block_size,
        cpu_offload_gb=effective_cpu_offload_gb,
        kv_offloading_size=runtime_defaults.kv_offloading_size,
        kv_offloading_backend=runtime_defaults.kv_offloading_backend,
        num_gpu_blocks_override=runtime_defaults.num_gpu_blocks_override,
        max_num_batched_tokens=max_num_batched_tokens,
        max_num_seqs=max_num_seqs,
        max_num_partial_prefills=runtime_defaults.max_num_partial_prefills,
        max_long_partial_prefills=runtime_defaults.max_long_partial_prefills,
        long_prefill_token_threshold=runtime_defaults.long_prefill_token_threshold,
        async_scheduling=async_scheduling,
        enable_prefix_caching=runtime_defaults.enable_prefix_caching,
        prefix_caching_hash_algo=runtime_defaults.prefix_caching_hash_algo,
        enforce_eager=enforce_eager,
        disable_custom_all_reduce=runtime_defaults.disable_custom_all_reduce,
        temperature=profile.temperature,
        top_p=runtime_defaults.top_p,
        top_k=profile.top_k,
        min_p=runtime_defaults.min_p,
        repetition_penalty=runtime_defaults.repetition_penalty,
        max_tokens=max_tokens,
        stop=list(runtime_defaults.stop),
        system_prompt=system_prompt,
        user_prompt=user_prompt,
    )

    if runtime_defaults.print_config:
        print_effective_config(cfg)

    return cfg


def validate_config(cfg: VLLMConfig) -> None:
    """Valida invariantes basicas antes de invocar el runtime de vLLM."""

    if cfg.max_model_len < 1:
        raise ValueError("max_model_len debe ser >= 1")
    if cfg.max_tokens < 1:
        raise ValueError("max_tokens debe ser >= 1")
    if cfg.max_tokens > cfg.max_model_len:
        raise ValueError("max_tokens no puede exceder max_model_len")
    if not (0.0 < cfg.gpu_memory_utilization <= 1.0):
        raise ValueError("gpu_memory_utilization debe estar en (0, 1]")
    if cfg.tensor_parallel_size < 1:
        raise ValueError("tensor_parallel_size debe ser >= 1")
    if cfg.pipeline_parallel_size < 1:
        raise ValueError("pipeline_parallel_size debe ser >= 1")
    if cfg.max_num_seqs is not None and cfg.max_num_seqs < 1:
        raise ValueError("max_num_seqs debe ser >= 1")
    if cfg.temperature < 0.0:
        raise ValueError("temperature debe ser >= 0")
    if not (0.0 < cfg.top_p <= 1.0):
        raise ValueError("top_p debe estar en (0, 1]")
    if cfg.top_k != -1 and cfg.top_k < 1:
        raise ValueError("top_k debe ser -1 o >= 1")
    if not (0.0 <= cfg.min_p <= 1.0):
        raise ValueError("min_p debe estar en [0, 1]")
    if cfg.repetition_penalty <= 0.0:
        raise ValueError("repetition_penalty debe ser > 0")
    if cfg.cpu_offload_gb < 0.0:
        raise ValueError("cpu_offload_gb no puede ser negativo")
    if cfg.enable_expert_parallel and cfg.tensor_parallel_size > 1:
        logger.warning("enable_expert_parallel=True con tensor_parallel_size>1; verifica compatibilidad de topologia y version de vLLM.")


def print_effective_config(cfg: VLLMConfig) -> None:
    """Registra la configuracion efectiva redactando secretos sensibles."""

    payload = asdict(cfg)
    if payload.get("hf_token"):
        payload["hf_token"] = "***"
    logger.info("Configuracion vLLM efectiva:\n%s", json.dumps(payload, indent=2, ensure_ascii=False))
