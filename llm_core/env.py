#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Lectura consistente de variables de entorno para runtimes locales."""

from __future__ import annotations

import os


def env_str(env_key: str, default: str) -> str:
    """Lee un string del entorno aplicando trimming y fallback estable."""

    value = os.getenv(env_key)
    if value is None or not value.strip():
        return default
    return value.strip()


def env_int(env_key: str, default: int) -> int:
    """Lee un entero del entorno sin aceptar valores vacios."""

    return int(env_str(env_key, str(default)))


def env_float(env_key: str, default: float) -> float:
    """Lee un float del entorno sin aceptar valores vacios."""

    return float(env_str(env_key, str(default)))


def env_bool(env_key: str, default: bool) -> bool:
    """Lee un booleano del entorno usando convenciones comunes."""

    default_value = "1" if default else "0"
    return env_str(env_key, default_value).lower() in {"1", "true", "yes", "on"}
