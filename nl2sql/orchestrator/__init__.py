#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""API publica del orquestador NL2SQL."""

from __future__ import annotations

from .config import NL2SQLConfig
from .contracts import NL2SQLRequest, NL2SQLResponse, StageArtifact
from .pipeline import build_nl2sql_pipeline, run_nl2sql, run_nl2sql_batch

__all__ = [
    "NL2SQLConfig",
    "NL2SQLRequest",
    "NL2SQLResponse",
    "StageArtifact",
    "build_nl2sql_pipeline",
    "run_nl2sql",
    "run_nl2sql_batch",
]
