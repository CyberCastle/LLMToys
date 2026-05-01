#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Runner concreto para `google/gemma-4-E4B-it`."""

from __future__ import annotations

from llm_core.env import env_bool, env_float, env_int, env_str
from llm_core.vllm_engine import ModelRuntimeProfile, QuantizedVariant
from llm_core.vllm_interface import VLLMModelRunner

MODEL = "google/gemma-4-E4B-it"

DEFAULT_GEMMA4_E4B_QUANTIZED_MODEL = "Chunity/gemma-4-E4B-it-AWQ-4bit"
DEFAULT_GEMMA4_E4B_GPU_MEMORY_UTILIZATION = 0.82
DEFAULT_GEMMA4_E4B_CPU_OFFLOAD_GB = 8.0
DEFAULT_GEMMA4_E4B_MAX_MODEL_LEN = 2048
DEFAULT_GEMMA4_E4B_MAX_NUM_SEQS = 1
DEFAULT_GEMMA4_E4B_BLOCK_SIZE = 64
DEFAULT_GEMMA4_E4B_ENFORCE_EAGER = True
DEFAULT_GEMMA4_E4B_ASYNC_SCHEDULING = False
DEFAULT_GEMMA4_E4B_RUNTIME_MODE = "quantized"
DEFAULT_GEMMA4_E4B_QUANTIZATION = "awq"
DEFAULT_GEMMA4_E4B_QUANTIZED_DTYPE = "float16"


def load_gemma4_e4b_profile_from_env() -> ModelRuntimeProfile:
    """Resuelve el perfil efectivo de Gemma 4 E4B leyendo el entorno en runtime."""

    quantized_model = env_str("GEMMA4_E4B_QUANTIZED_MODEL", DEFAULT_GEMMA4_E4B_QUANTIZED_MODEL)
    quantized_tokenizer = env_str("GEMMA4_E4B_QUANTIZED_TOKENIZER", quantized_model)
    gpu_memory_utilization_cap = env_float(
        "GEMMA4_E4B_GPU_MEMORY_UTILIZATION",
        DEFAULT_GEMMA4_E4B_GPU_MEMORY_UTILIZATION,
    )
    cpu_offload_gb = env_float("GEMMA4_E4B_CPU_OFFLOAD_GB", DEFAULT_GEMMA4_E4B_CPU_OFFLOAD_GB)
    max_model_len_cap = env_int("GEMMA4_E4B_MAX_MODEL_LEN", DEFAULT_GEMMA4_E4B_MAX_MODEL_LEN)
    max_num_seqs_default = env_int("GEMMA4_E4B_MAX_NUM_SEQS", DEFAULT_GEMMA4_E4B_MAX_NUM_SEQS)
    block_size_default = env_int("GEMMA4_E4B_BLOCK_SIZE", DEFAULT_GEMMA4_E4B_BLOCK_SIZE)
    enforce_eager_default = env_bool("GEMMA4_E4B_ENFORCE_EAGER", DEFAULT_GEMMA4_E4B_ENFORCE_EAGER)
    async_scheduling_default = env_bool("GEMMA4_E4B_ASYNC_SCHEDULING", DEFAULT_GEMMA4_E4B_ASYNC_SCHEDULING)
    quantization_default = env_str("GEMMA4_E4B_QUANTIZATION", DEFAULT_GEMMA4_E4B_QUANTIZATION)
    quantized_dtype = env_str("GEMMA4_E4B_QUANTIZED_DTYPE", DEFAULT_GEMMA4_E4B_QUANTIZED_DTYPE)
    runtime_mode = env_str("GEMMA4_E4B_RUNTIME_MODE", DEFAULT_GEMMA4_E4B_RUNTIME_MODE).lower()

    shared_kwargs = {
        "alias": "gemma4_e4b",
        "canonical_model_name": MODEL,
        "temperature": 0.7,
        "top_k": 50,
        "size_estimates_gib": {
            "bf16": 18.0,
            "awq_4bit": 9.3,
        },
        "gpu_memory_utilization_cap": gpu_memory_utilization_cap,
        "max_model_len_cap": max_model_len_cap,
        "max_num_batched_tokens_default": max_model_len_cap,
        "max_num_seqs_default": max_num_seqs_default,
        "block_size_default": block_size_default,
        "enforce_eager_default": enforce_eager_default,
        "async_scheduling_default": async_scheduling_default,
    }
    quantized_variant = QuantizedVariant(
        model_name=quantized_model,
        tokenizer_name=quantized_tokenizer,
        size_key="awq_4bit",
        quantization=quantization_default,
        dtype=quantized_dtype,
        kernel_backend="awq",
    )

    if runtime_mode == "quantized":
        return ModelRuntimeProfile(
            model_name=quantized_model,
            tokenizer_name=quantized_tokenizer,
            quantization_default=quantization_default,
            dtype_default=quantized_dtype,
            **shared_kwargs,
        )
    if runtime_mode == "bf16_offload":
        return ModelRuntimeProfile(
            model_name=MODEL,
            tokenizer_name=None,
            min_cpu_offload_gb=cpu_offload_gb,
            quantized_variant=quantized_variant,
            **shared_kwargs,
        )

    raise ValueError("GEMMA4_E4B_RUNTIME_MODE debe ser 'quantized' o 'bf16_offload'; " f"valor recibido: {runtime_mode!r}.")


class Gemma4E4BRunner(VLLMModelRunner):
    """Runner vLLM para Gemma 4 E4B con perfil resuelto en runtime."""

    def get_model_profile(self) -> ModelRuntimeProfile:
        """Resuelve el perfil efectivo de Gemma 4 E4B leyendo el entorno."""

        return load_gemma4_e4b_profile_from_env()
