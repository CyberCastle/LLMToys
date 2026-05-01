#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Factoria de SQLAlchemy Engine para ejecucion read-only del SQL generado."""

from __future__ import annotations

import os

from sqlalchemy import create_engine
from sqlalchemy.engine import Engine


def resolve_database_url(dialect: str) -> str:
    """Resuelve la URL SQLAlchemy desde entorno con fallback por dialecto."""

    generic_url = os.environ.get("NL2SQL_DATABASE_URL") or os.environ.get("DATABASE_URL")
    if generic_url:
        return generic_url
    if dialect == "tsql":
        configured_url = os.environ.get("NL2SQL_TSQL_URL")
    elif dialect == "postgres":
        configured_url = os.environ.get("NL2SQL_POSTGRES_URL")
    else:
        raise ValueError(f"Dialecto no soportado: {dialect}")
    if configured_url is None or not configured_url.strip():
        raise ValueError(
            "No existe una URL SQLAlchemy configurada para el dialecto solicitado. "
            "Usa NL2SQL_DATABASE_URL, DATABASE_URL o la variable especifica del dialecto."
        )
    return configured_url.strip()


def build_engine(dialect: str) -> Engine:
    """Construye un `Engine` reutilizable con `pool_pre_ping` habilitado."""

    return create_engine(
        resolve_database_url(dialect),
        future=True,
        pool_pre_ping=True,
    )
