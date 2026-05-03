#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Prueba de humo opcional para artefactos cuantizados en vLLM."""

from __future__ import annotations

import json
from pathlib import Path


def _ensure_quantization_config(model_dir: Path) -> None:
    """Verifica que el artefacto guardado exponga `quantization_config`."""

    config_path = model_dir / "config.json"
    if not config_path.exists():
        raise RuntimeError(f"No existe config.json en {model_dir}")

    payload = json.loads(config_path.read_text(encoding="utf-8"))
    if "quantization_config" not in payload:
        raise RuntimeError("El artefacto cuantizado no contiene quantization_config en config.json")


def smoke_test(model_dir: str | Path, max_model_len: int = 512) -> None:
    """Carga el artefacto cuantizado en vLLM y valida una generacion corta."""

    model_path = Path(model_dir).expanduser().resolve()
    _ensure_quantization_config(model_path)

    try:
        import torch
        from vllm import LLM, SamplingParams
    except ImportError as exc:
        raise RuntimeError(
            "La prueba de humo con vLLM es optativa y requiere un entorno separado con vLLM. "
            "El subproyecto tools/quantizer no fija vLLM porque el pin actual del root aun no "
            "coincide con la rama de compressed-tensors requerida por llmcompressor."
        ) from exc

    llm = None
    try:
        llm = LLM(
            model=str(model_path),
            dtype="auto",
            enforce_eager=True,
            max_model_len=max_model_len,
            gpu_memory_utilization=0.70,
        )
        outputs = llm.generate(
            ["Hello, my name is"],
            sampling_params=SamplingParams(max_tokens=32, temperature=0.0),
        )
        text = outputs[0].outputs[0].text if outputs and outputs[0].outputs else ""
        if not text.strip():
            raise RuntimeError("El artefacto cargo en vLLM, pero genero una salida vacia")
    except Exception as exc:
        raise RuntimeError(f"No fue posible validar el artefacto cuantizado con vLLM desde {model_path}") from exc
    finally:
        if llm is not None:
            del llm
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
