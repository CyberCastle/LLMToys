#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from nl2sql.config import AssetTextFormattingRules, load_semantic_resolver_settings, resolve_nl2sql_config_path
from nl2sql.utils.text_utils import truncate_text

from .assets import SemanticAsset


@lru_cache(maxsize=4)
def load_asset_text_formatting_rules(path: str | Path | None = None) -> AssetTextFormattingRules:
    """Devuelve las reglas tipadas de serializacion de activos semanticos."""

    resolved_path = Path(path).expanduser().resolve() if path is not None else resolve_nl2sql_config_path()
    return load_semantic_resolver_settings(resolved_path).compiler_rules.asset_text_formatting


def format_query_for_embedding(query: str, instruction: str, template: str) -> str:
    """Renderiza la query de embedding con una plantilla configurable."""

    return template.format(instruction=instruction, query=query)


def format_asset_text(asset: SemanticAsset, formatting_rules: AssetTextFormattingRules | None = None) -> str:
    """Serializa un activo en texto denso y controlado para embedding y rerank.

    Se antepone el tipo de activo para que el embedding capte si se trata de una
    entidad, métrica, regla o ejemplo. No se vuelcan todos los campos del payload
    porque muchas claves operativas solo añaden ruido y empeoran la recuperación.
    """

    payload = asset.payload
    active_rules = formatting_rules or load_asset_text_formatting_rules()
    parts: list[str] = [f"[{asset.kind}] {asset.name}"]

    for key in active_rules.body_keys:
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            parts.append(value.strip())

    for key in active_rules.scalar_keys:
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            parts.append(f"{key}: {value.strip()}")

    for key in active_rules.list_keys:
        value = payload.get(key)
        if isinstance(value, list) and value:
            rendered = ", ".join(str(item) for item in value)
            parts.append(f"{key}: {rendered}")

    handled_keys = set(active_rules.body_keys) | set(active_rules.scalar_keys) | set(active_rules.list_keys) | {"name", "id"}
    for key, value in payload.items():
        if key in handled_keys:
            continue
        if isinstance(value, str) and value.strip():
            parts.append(f"{key}: {value.strip()}")

    return " | ".join(parts)


def build_rerank_document_text(
    asset: SemanticAsset,
    *,
    max_chars: int,
    formatting_rules: AssetTextFormattingRules | None = None,
) -> str:
    """Recorta el documento preservando el encabezado y el texto mas informativo."""

    text = format_asset_text(asset, formatting_rules=formatting_rules)
    return truncate_text(text, max_chars)
