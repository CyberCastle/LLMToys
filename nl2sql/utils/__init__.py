#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Utilidades compartidas del pipeline NL2SQL.

Este paquete contiene helpers transversales y libres de lógica de negocio que
pueden ser reutilizados por `orchestrator`, `semantic_prune`,
`semantic_resolver` y `sql_solver_generator` sin crear acoplamiento entre las
etapas del pipeline.
"""

from __future__ import annotations

from .collections import dedupe_preserve_order
from .normalization import normalize_text_for_matching
from .text_utils import collapse_whitespace, truncate_text
from .yaml_utils import load_yaml_cached, load_yaml_mapping, load_yaml_value, normalize_for_yaml

__all__ = [
    "collapse_whitespace",
    "dedupe_preserve_order",
    "load_yaml_cached",
    "load_yaml_mapping",
    "load_yaml_value",
    "normalize_for_yaml",
    "normalize_text_for_matching",
    "truncate_text",
]
