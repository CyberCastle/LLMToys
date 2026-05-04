#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Runner concreto para `google/gemma-4-26B-A4B-it`."""

from __future__ import annotations

from llm_core.vllm_engine import ModelRuntimeProfile, QuantizedVariant
from llm_core.vllm_interface import VLLMModelRunner

MODEL = "google/gemma-4-26B-A4B-it"
QUANTIZED_MODEL = "cyankiwi/gemma-4-26B-A4B-it-AWQ-4bit"

MODEL_PROFILE = ModelRuntimeProfile(
    alias="gemma4",
    canonical_model_name=MODEL,
    model_name=MODEL,
    temperature=0.7,
    top_k=50,
    size_estimates_gib={
        "bf16": 52.0,
        "awq_4bit": 16.0,
    },
    is_moe=True,
    quantized_variant=QuantizedVariant(
        model_name=QUANTIZED_MODEL,
        tokenizer_name=QUANTIZED_MODEL,
        size_key="awq_4bit",
        quantization="compressed-tensors",
        dtype="float16",
    ),
)


class Gemma4Runner(VLLMModelRunner):
    """Runner vLLM para Gemma 4 26B-A4B con offload CPU/GPU agnostico."""

    def get_model_profile(self) -> ModelRuntimeProfile:
        """Devuelve el perfil fijo del modelo Gemma 4 26B-A4B."""

        return MODEL_PROFILE
