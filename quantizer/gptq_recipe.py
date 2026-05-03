#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Receta GPTQ W4A16 para el cuantizador generico."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

try:
    from llmcompressor.modifiers.gptq import GPTQModifier
except ImportError:

    @dataclass(slots=True)
    class GPTQModifier:
        """Shim liviano para ejecutar pruebas unitarias sin llmcompressor."""

        targets: Any = "Linear"
        scheme: str = "W4A16"
        ignore: list[str] | None = None
        block_size: int = 128
        offload_hessians: bool = True


def build_gptq_recipe() -> list[Any]:
    """Construye la receta GPTQ W4A16 con el preset int4 por grupos de 128."""

    return [
        GPTQModifier(
            targets="Linear",
            scheme="W4A16",
            ignore=["lm_head"],
            block_size=128,
            offload_hessians=True,
        )
    ]
