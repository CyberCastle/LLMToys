#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass

from ..llm_router import LlmAttempt, LlmRouter


@dataclass(frozen=True)
class GenerationResult:
    attempt: LlmAttempt


def run_generation(router: LlmRouter, **inputs) -> GenerationResult:
    return GenerationResult(attempt=router.run(**inputs))
