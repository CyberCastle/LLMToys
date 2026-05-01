#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Render y serializacion del resultado consolidado NL2SQL."""

from __future__ import annotations

from typing import Any

import yaml

from nl2sql.utils.yaml_utils import normalize_for_yaml

from .contracts import NL2SQLResponse


def build_nl2sql_report(response: NL2SQLResponse, *, rows_preview_limit: int = 10) -> str:
    """Construye un reporte legible por consola del flujo NL2SQL."""

    rows_preview = response.rows[:rows_preview_limit]
    lines = [
        "=" * 80,
        "NL2SQL REPORT",
        "=" * 80,
        f"Status               : {response.status}",
        f"Query                : {response.query}",
        f"Rows                 : {response.row_count}",
        f"Truncated            : {'yes' if response.truncated else 'no'}",
        "-" * 80,
        "FINAL SQL",
        "-" * 80,
        response.final_sql or "(sin SQL)",
        "-" * 80,
        "ROWS PREVIEW",
        "-" * 80,
        yaml.safe_dump(rows_preview or "(sin filas)", sort_keys=False, allow_unicode=True).rstrip(),
        "-" * 80,
        "NARRATIVE",
        "-" * 80,
        response.narrative or "(sin narrativa)",
        "-" * 80,
        "ARTIFACTS",
        "-" * 80,
    ]

    if not response.artifacts:
        lines.append("(sin artefactos)")
    else:
        for artifact in response.artifacts:
            lines.append(f"{artifact.name}: {artifact.path}")

    if response.warnings:
        lines.extend(["-" * 80, "WARNINGS", "-" * 80, *response.warnings])
    if response.issues:
        lines.extend(["-" * 80, "ISSUES", "-" * 80, *[f"{issue.code}: {issue.message}" for issue in response.issues]])

    return "\n".join(lines)


def render_nl2sql_response(response: NL2SQLResponse, *, rows_preview_limit: int = 10) -> None:
    """Imprime el reporte consolidado por consola."""

    print(build_nl2sql_report(response, rows_preview_limit=rows_preview_limit))


def serialize_nl2sql_response(response: NL2SQLResponse) -> dict[str, Any]:
    """Serializa la respuesta a un mapping apto para YAML/JSON."""

    normalized = normalize_for_yaml(response)
    if not isinstance(normalized, dict):
        raise ValueError("NL2SQLResponse invalida: no se pudo serializar a mapping")
    return normalized
