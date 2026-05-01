#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json

from pydantic import ValidationError

from .spec_model import SQLGenerationPayload

def render_generation_contract_json() -> str:
    """Expone un esquema JSON compacto para instruir al LLM del solver."""

    schema = SQLGenerationPayload.model_json_schema()
    compact_schema = {
        "type": schema.get("type", "object"),
        "required": schema.get("required", []),
        "properties": schema.get("properties", {}),
        "additionalProperties": False,
    }
    return json.dumps(compact_schema, ensure_ascii=False, indent=2)


def validate_generation_payload_json(raw_text: str) -> SQLGenerationPayload:
    """Valida el JSON emitido por el LLM contra el contrato local estricto."""

    candidate = raw_text.strip()
    if not candidate:
        raise ValueError("La salida del LLM esta vacia")
    try:
        return SQLGenerationPayload.model_validate_json(candidate)
    except ValidationError as exc:
        raise ValueError("La salida del LLM no cumple el contrato JSON esperado") from exc
