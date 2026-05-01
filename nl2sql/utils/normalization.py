#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Normalización textual compartida entre etapas del pipeline NL2SQL."""

from __future__ import annotations

import re
import unicodedata

_WHITESPACE_RE = re.compile(r"\s+")
_NON_WORD_RE = re.compile(r"[^a-z0-9_ ]+")


def normalize_text_for_matching(
    text: str,
    *,
    lowercase: bool = True,
    keep_underscore: bool = True,
    ascii_fallback: bool = False,
    non_word_replacement: str = " ",
) -> str:
    """Normaliza texto para matching léxico robusto.

    Elimina diacríticos, compacta whitespace y opcionalmente fuerza minúsculas
    para que las heurísticas de matching no dependan de variaciones cosméticas.
    """

    decomposed = unicodedata.normalize("NFKD", text or "")
    stripped = "".join(character for character in decomposed if not unicodedata.combining(character))
    if ascii_fallback:
        stripped = stripped.encode("ascii", "ignore").decode("ascii")
    collapsed = _WHITESPACE_RE.sub(" ", stripped).strip()
    result = collapsed.lower() if lowercase else collapsed
    result = result.replace("-", " ").replace("/", " ")
    if not keep_underscore:
        result = result.replace("_", " ")
    return _WHITESPACE_RE.sub(" ", _NON_WORD_RE.sub(non_word_replacement, result)).strip()


def slugify_identifier(value: str, *, fallback: str = "item") -> str:
    """Normaliza texto libre a un identificador estable en snake_case."""

    normalized = normalize_text_for_matching(value, keep_underscore=True, ascii_fallback=True)
    slug = re.sub(r"[^a-z0-9_]+", "_", normalized).strip("_")
    return slug or fallback
