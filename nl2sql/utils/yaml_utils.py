#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Helpers de lectura y serialización YAML compartidos por NL2SQL."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, is_dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import BaseModel
import yaml


def normalize_for_yaml(value: object) -> object:
    """Convierte dataclasses, `Path` y mappings a estructuras YAML-safe."""

    if isinstance(value, BaseModel):
        return normalize_for_yaml(value.model_dump(mode="python"))
    if is_dataclass(value):
        return normalize_for_yaml(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): normalize_for_yaml(child) for key, child in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [normalize_for_yaml(child) for child in value]
    return value


def load_yaml_value(path: str | Path) -> Any:
    """Lee un archivo YAML completo retornando el valor crudo parseado."""

    resolved_path = Path(path).expanduser().resolve()
    return yaml.safe_load(resolved_path.read_text(encoding="utf-8"))


@lru_cache(maxsize=16)
def load_yaml_cached(path: str) -> Any:
    """Lee y cachea un YAML por ruta resuelta para assets de solo lectura."""

    return load_yaml_value(path)


def load_yaml_mapping(path: str | Path, *, artifact_name: str) -> dict[str, Any]:
    """Carga un YAML cuya raíz debe ser un mapping, con error homogéneo."""

    payload = load_yaml_value(path) or {}
    if not isinstance(payload, Mapping):
        raise ValueError(f"{artifact_name} invalido: la raiz debe ser un mapping YAML")
    return dict(payload)
