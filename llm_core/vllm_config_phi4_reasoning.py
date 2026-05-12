#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Runner concreto para `microsoft/Phi-4-reasoning`."""

from __future__ import annotations

import re

from llm_core.vllm_engine import ModelRuntimeProfile
from llm_core.vllm_interface import VLLMModelRunner

MODEL = "microsoft/Phi-4-reasoning"
QUANTIZED_MODEL = "ronantakizawa/phi-4-reasoning-awq"
THINK_BLOCK_RE = re.compile(r"<think>.*?(</think>|$)", re.DOTALL | re.IGNORECASE)

MODEL_PROFILE = ModelRuntimeProfile(
    alias="phi4_reasoning",
    canonical_model_name=MODEL,
    model_name=QUANTIZED_MODEL,
    tokenizer_name=QUANTIZED_MODEL,
    temperature=0.8,
    top_k=50,
    size_estimates_gib={
        "bf16": 28.0,
        "awq_4bit": 9.2,
    },
    max_model_len_cap=32768,
    quantization_default="awq",
    dtype_default="float16",
)


class Phi4ReasoningRunner(VLLMModelRunner):
    """Runner vLLM para Phi-4-reasoning usando AWQ como runtime por defecto."""

    def get_model_profile(self) -> ModelRuntimeProfile:
        """Devuelve el perfil fijo del modelo Phi-4-reasoning."""

        return MODEL_PROFILE

    def build_prompt(self, system_prompt: str, user_prompt: str) -> str:
        """Usa ChatML mínimo para evitar el template fijo de razonamiento de Phi."""

        return (
            f"<|im_start|>system<|im_sep|>{system_prompt.strip()}<|im_end|>\n"
            f"<|im_start|>user<|im_sep|>{user_prompt.strip()}<|im_end|>\n"
            "<|im_start|>assistant<|im_sep|>"
        )

    def _post_process_output(self, text: str) -> str:
        """Elimina el bloque `<think>...</think>` para extraer solo la respuesta final."""

        return THINK_BLOCK_RE.sub("", text).strip()
