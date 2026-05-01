#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Helpers de colecciones compartidos por el pipeline NL2SQL."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TypeVar

T = TypeVar("T")


def dedupe_preserve_order(values: Iterable[T]) -> list[T]:
    """Elimina duplicados preservando el orden de aparición original."""

    seen: set[T] = set()
    output: list[T] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        output.append(value)
    return output
