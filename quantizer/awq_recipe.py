#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Recetas AWQ configurables via YAML por arquitectura."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import yaml

from .config import DEFAULT_AWQ_MAPPINGS_PATH

try:
    from llmcompressor.modifiers.quantization import QuantizationModifier
    from llmcompressor.modifiers.transform.awq import AWQMapping, AWQModifier
except ImportError:

    @dataclass(slots=True)
    class AWQMapping:
        """Shim liviano para ejecutar pruebas unitarias sin llmcompressor."""

        smooth_layer: str
        balance_layers: list[str]
        activation_hook_target: str | None = None

    @dataclass(slots=True)
    class AWQModifier:
        """Shim liviano para ejecutar pruebas unitarias sin llmcompressor."""

        mappings: list[AWQMapping] | None = None
        duo_scaling: bool | str = "both"
        offload_device: Any | None = None

    @dataclass(slots=True)
    class QuantizationModifier:
        """Shim liviano para ejecutar pruebas unitarias sin llmcompressor."""

        targets: Any = "Linear"
        scheme: str = "W4A16_ASYM"
        ignore: list[str] | None = None


def resolve_awq_mappings(
    model_type: str,
    mappings_path: Path | None = None,
) -> list[AWQMapping]:
    """Carga `awq_mappings.yaml` y devuelve los mappings de la arquitectura pedida."""

    active_path = mappings_path or DEFAULT_AWQ_MAPPINGS_PATH
    payload = yaml.safe_load(active_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError("awq_mappings.yaml invalido: la raiz debe ser un mapping")

    normalized_model_type = model_type.strip().lower()
    if normalized_model_type not in payload:
        supported = ", ".join(sorted(payload))
        raise ValueError(f"No existe mapping AWQ para '{normalized_model_type}'. Arquitecturas soportadas: {supported}")

    raw_mappings = payload[normalized_model_type].get("mappings", [])
    mappings: list[AWQMapping] = []
    for raw_mapping in raw_mappings:
        target_layers = raw_mapping.get("balance_layers") or raw_mapping.get("target_layers")
        if not target_layers:
            raise ValueError("Cada mapping AWQ debe declarar 'target_layers' o 'balance_layers'")
        mappings.append(
            AWQMapping(
                smooth_layer=raw_mapping["smooth_layer"],
                balance_layers=list(target_layers),
                activation_hook_target=raw_mapping.get("activation_hook_target"),
            )
        )
    return mappings


def build_awq_recipe(
    model_type: str,
    mappings_path: Path | None = None,
    *,
    sequential_onloading: bool = True,
) -> list[Any]:
    """Construye la receta AWQ + cuantizacion int4 para la arquitectura indicada."""

    mappings = resolve_awq_mappings(model_type, mappings_path)
    awq_modifier = AWQModifier(
        mappings=mappings,
        duo_scaling="both",
        offload_device=torch.device("cpu") if sequential_onloading else None,
    )
    quantization_modifier = QuantizationModifier(
        targets="Linear",
        scheme="W4A16_ASYM",
        ignore=["lm_head"],
    )
    return [awq_modifier, quantization_modifier]
