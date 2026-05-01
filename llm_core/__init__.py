#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""API publica minima de `llm_core` para runners vLLM genericos."""

from __future__ import annotations

from llm_core.model_registry import build_runner, list_supported_models, resolve_model_name
from llm_core.vllm_engine import VLLMRuntimeDefaults
from llm_core.vllm_interface import VLLMModelRunner

__all__ = [
    "VLLMModelRunner",
    "VLLMRuntimeDefaults",
    "build_runner",
    "list_supported_models",
    "resolve_model_name",
]
