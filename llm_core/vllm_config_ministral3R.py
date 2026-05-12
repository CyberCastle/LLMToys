#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Runner concreto para `mistralai/Ministral-3-8B-Reasoning-2512`."""

from __future__ import annotations

from typing import Any

from llm_core.vllm_engine import ModelRuntimeProfile, QuantizedVariant, VLLMConfig
from llm_core.vllm_interface import VLLMModelRunner

MODEL = "mistralai/Ministral-3-8B-Reasoning-2512"
QUANTIZED_MODEL = "cyankiwi/Ministral-3-8B-Reasoning-2512-AWQ-4bit"

MODEL_PROFILE = ModelRuntimeProfile(
    alias="ministral3",
    canonical_model_name=MODEL,
    model_name=MODEL,
    temperature=0.7,
    top_k=50,
    size_estimates_gib={
        "bf16": 24.0,
        "awq_4bit": 11.5,
    },
    quantized_variant=QuantizedVariant(
        model_name=QUANTIZED_MODEL,
        tokenizer_name=QUANTIZED_MODEL,
        size_key="awq_4bit",
        quantization="compressed-tensors",
        dtype="float16",
    ),
)


class Ministral3RRunner(VLLMModelRunner):
    """Runner vLLM para Ministral 3 con los overrides requeridos por Mistral."""

    def get_model_profile(self) -> ModelRuntimeProfile:
        """Devuelve el perfil fijo del modelo Ministral 3 8B Reasoning."""

        return MODEL_PROFILE

    def configure(self, system_prompt: str, user_prompt: str) -> VLLMConfig:
        """Fuerza `tokenizer_mode='mistral'` en la configuracion efectiva."""

        cfg = super().configure(system_prompt, user_prompt)
        cfg.tokenizer_mode = "mistral"
        return cfg

    def _build_llm_instance(self) -> Any:
        """Instancia vLLM con los formatos de config/carga recomendados por Mistral."""

        from vllm import LLM

        if self.cfg is None:
            raise RuntimeError("El runner debe configurarse antes de cargar el modelo")
        return LLM(
            **self.cfg.build_llm_kwargs(),
            config_format="mistral",
            load_format="mistral",
        )
