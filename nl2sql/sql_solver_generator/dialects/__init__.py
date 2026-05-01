#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from .base import SqlDialect
from .postgres import PostgresDialect
from .registry import get_dialect
from .tsql import TsqlDialect

__all__ = ["PostgresDialect", "SqlDialect", "TsqlDialect", "get_dialect"]
