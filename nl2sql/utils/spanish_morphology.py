#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Morfología española mínima configurada desde `settings.yaml`."""

from __future__ import annotations

from collections.abc import Iterable


def singularize_token(token: str, rules: Iterable[tuple[str, str, int]] = ()) -> str:
    """Devuelve una variante singular aplicando reglas `(suffix, replacement, min_len)`."""

    normalized = token.strip().lower()
    if not normalized:
        return normalized
    for suffix, replacement, min_length in rules:
        if len(normalized) >= min_length and normalized.endswith(suffix):
            base = normalized[: -len(suffix)] if suffix else normalized
            return f"{base}{replacement}"
    return normalized


def pluralize_token(token: str, rules: Iterable[tuple[str, str, int]] = ()) -> str:
    """Devuelve una variante plural aplicando reglas `(suffix, replacement, min_len)`."""

    normalized = token.strip().lower()
    if not normalized:
        return normalized
    for suffix, replacement, min_length in rules:
        if len(normalized) >= min_length and normalized.endswith(suffix):
            base = normalized[: -len(suffix)] if suffix else normalized
            return f"{base}{replacement}"
    return normalized
