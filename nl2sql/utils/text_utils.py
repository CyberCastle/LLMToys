#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Utilidades de texto simples y reutilizables del pipeline NL2SQL."""

from __future__ import annotations


def collapse_whitespace(text: str) -> str:
    """Compacta espacios y saltos de línea a un único separador."""

    return " ".join(text.split())


def truncate_text(text: str, max_chars: int) -> str:
    """Recorta texto de forma estable sin romper casos pequeños."""

    if max_chars <= 0 or len(text) <= max_chars:
        return text
    if max_chars <= 3:
        return text[:max_chars]
    return text[: max_chars - 3].rstrip() + "..."
