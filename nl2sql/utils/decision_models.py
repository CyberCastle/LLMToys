#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    """Modelo base inmutable y sin campos implicitos para contratos NL2SQL."""

    model_config = ConfigDict(extra="forbid", frozen=True, arbitrary_types_allowed=True)


class DecisionIssue(StrictModel):
    """Issue estructurado emitido por una etapa del pipeline NL2SQL."""

    stage: str
    code: str
    severity: Literal["warning", "error"]
    message: str
    context: dict[str, Any] = Field(default_factory=dict)

    def __str__(self) -> str:
        return f"{self.stage}:{self.code}"


def dedupe_decision_issues(values: list[DecisionIssue]) -> list[DecisionIssue]:
    """Elimina issues duplicados preservando el orden de aparicion."""

    seen: set[tuple[str, str, str, str, str]] = set()
    output: list[DecisionIssue] = []
    for value in values:
        identity = (
            value.stage,
            value.code,
            value.severity,
            value.message,
            json.dumps(value.context, sort_keys=True, ensure_ascii=False, default=str),
        )
        if identity in seen:
            continue
        seen.add(identity)
        output.append(value)
    return output