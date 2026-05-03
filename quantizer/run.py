#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Runner del cuantizador generico."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

if __package__ in {None, ""}:
    # Permite ejecutar `uv run quantizer/run.py` sin romper los imports del paquete.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Reduce el riesgo de fragmentacion en flujos AWQ/GPTQ que usan offload y subgrafos.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from quantizer.config import QuantizerConfig
from quantizer.memory_preflight import (
    enforce_memory_preflight_policy,
    evaluate_quantization_memory_preflight,
    format_memory_preflight_report,
)
from quantizer.quantizer import quantize_model
from quantizer.vllm_smoke_test import smoke_test


def _should_skip_memory_preflight() -> bool:
    """Permite omitir el preflight en hijos ya validados por el proceso padre."""

    raw_value = os.getenv("QUANTIZER_SKIP_MEMORY_PREFLIGHT", "").strip().lower()
    return raw_value in {"1", "true", "yes", "on"}


def _run_memory_preflight(config: QuantizerConfig, scheme: str, *, launched_via_subprocess: bool) -> None:
    """Ejecuta el preflight, imprime el diagnostico y aplica el guardrail configurado."""

    if config.memory_preflight_mode == "off" or _should_skip_memory_preflight():
        return

    result = evaluate_quantization_memory_preflight(
        config,
        scheme,
        launched_via_subprocess=launched_via_subprocess,
    )
    print(format_memory_preflight_report(result))
    enforce_memory_preflight_policy(result, config.memory_preflight_mode)


def _run_scheme_subprocess(scheme: str) -> None:
    """Lanza un esquema en un proceso limpio para liberar VRAM entre corridas."""

    env = os.environ.copy()
    env["QUANTIZE_SCHEME"] = scheme
    env["QUANTIZER_SKIP_MEMORY_PREFLIGHT"] = "true"
    subprocess.run(
        [sys.executable, str(Path(__file__).resolve())],
        check=True,
        env=env,
    )


def main() -> None:
    """Ejecuta AWQ, GPTQ o ambos esquemas usando solo variables de entorno."""

    load_dotenv()
    config = QuantizerConfig.from_env()
    if config.quantize_scheme == "both":
        print("[quantizer] Ejecutando AWQ y GPTQ en procesos separados para liberar VRAM entre esquemas.")
        for scheme in ["awq", "gptq"]:
            _run_memory_preflight(config, scheme, launched_via_subprocess=True)
        for scheme in ["awq", "gptq"]:
            _run_scheme_subprocess(scheme)
        return

    scheme = config.quantize_scheme
    _run_memory_preflight(config, scheme, launched_via_subprocess=False)
    print(f"[quantizer] Iniciando cuantizacion {scheme.upper()} para {config.model_id}")
    output_path = quantize_model(config, scheme=scheme)
    print(f"[quantizer] Artefacto guardado en: {output_path}")
    if config.run_vllm_smoke_test:
        smoke_test(output_path)
        print(f"[quantizer] Prueba de humo vLLM completada para: {output_path}")
    else:
        print("[quantizer] Prueba de humo vLLM omitida. Activa QUANTIZER_RUN_VLLM_SMOKE_TEST=true si dispones de un entorno compatible.")


if __name__ == "__main__":
    main()
