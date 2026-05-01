#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Utilidades de acceso y normalizacion de resultados SQLAlchemy."""

from __future__ import annotations

from .engine_factory import build_engine, resolve_database_url
from .result_normalizer import rows_to_dicts

__all__ = [
    "build_engine",
    "resolve_database_url",
    "rows_to_dicts",
]
