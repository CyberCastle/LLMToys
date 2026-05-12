#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Script unificado para ejecutar modelos con runners vLLM registrados."""

from __future__ import annotations

from dotenv import load_dotenv

load_dotenv()

from llm_core.model_registry import build_runner
from llm_core.vllm_engine import VLLMRuntimeDefaults

# =====================================================================
# SELECCIÓN DE MODELO — Cambia esta variable para elegir el modelo
# =====================================================================
ACTIVE_MODEL = "ministral3r"  # "gemma4" | "gemma4_e4b" | "qwen3" | "ministral3i" | "ministral3r" | "phi4_reasoning"
# =====================================================================

# =====================================================================
# RUNTIME FLAGS — Overrides puntuales del planner/builder.
# =====================================================================
FORCE_QUANTIZED_VARIANT = True  # Forzar el uso de la variante cuantizada del modelo, si el perfil lo soporta.
# =====================================================================

# =====================================================================
# PROMPT — Define aquí la tarea a ejecutar
# =====================================================================
SYSTEM_PROMPT = "Eres un asistente técnico preciso."
USER_PROMPT = "Explica qué es un modelo Mixture-of-Experts."
# =====================================================================

# =====================================================================
# DEFAULTS — Ajusta aqui la configuracion compartida del runner.
# Ejemplo:
#   RUNTIME_DEFAULTS = VLLMRuntimeDefaults(max_model_len=8192, gpu_memory_utilization=0.90)
# =====================================================================
RUNTIME_DEFAULTS = VLLMRuntimeDefaults(
    max_tokens=4096,
    force_quantized_variant=FORCE_QUANTIZED_VARIANT,
)
# =====================================================================


def main() -> None:
    """Ejecuta el runner configurado por `ACTIVE_MODEL`."""

    runner = build_runner(ACTIVE_MODEL, runtime_defaults=RUNTIME_DEFAULTS)

    results = runner.run(SYSTEM_PROMPT, USER_PROMPT)

    print("\n" + "=" * 80)
    print("RESPUESTA")
    print("=" * 80)
    for text in results:
        print(text)
        print("-" * 80)


if __name__ == "__main__":
    main()
