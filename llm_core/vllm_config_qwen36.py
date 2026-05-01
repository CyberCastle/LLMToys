#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Runner concreto para `Qwen/Qwen3-30B-A3B`."""

from __future__ import annotations

import re

from llm_core.vllm_engine import ModelRuntimeProfile, QuantizedVariant
from llm_core.vllm_interface import VLLMModelRunner

MODEL = "Qwen/Qwen3-30B-A3B"
THINK_BLOCK_RE = re.compile(r"<think>.*?(</think>|$)", re.DOTALL)

MODEL_PROFILE = ModelRuntimeProfile(
    alias="qwen3",
    canonical_model_name=MODEL,
    model_name=MODEL,
    temperature=0.6,
    top_k=20,
    size_estimates_gib={
        "bf16": 60.0,
        "awq_4bit": 19.0,
    },
    is_moe=True,
    quantized_variant=QuantizedVariant(
        model_name="QuixiAI/Qwen3-30B-A3B-AWQ",
        tokenizer_name="QuixiAI/Qwen3-30B-A3B-AWQ",
        size_key="awq_4bit",
        quantization="awq_marlin",
        kernel_backend="marlin",
    ),
)


class Qwen3Runner(VLLMModelRunner):
    """Runner vLLM para Qwen3 con limpieza de bloques `<think>...</think>`."""

    def get_model_profile(self) -> ModelRuntimeProfile:
        """Devuelve el perfil fijo del modelo Qwen3-30B-A3B."""

        return MODEL_PROFILE

    def _post_process_output(self, text: str) -> str:
        """Elimina el razonamiento interno emitido dentro de `<think>...</think>`."""

        return THINK_BLOCK_RE.sub("", text).strip()
