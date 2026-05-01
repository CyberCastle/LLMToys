#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Etapa LangChain que redacta la respuesta final con Gemma 4 E4B AWQ."""

from __future__ import annotations

from dataclasses import asdict
import re
import time
from pathlib import Path
from typing import Any, Mapping

from langchain_core.runnables import RunnableLambda
import yaml

from llm_core.prompt_optimizer import DEFAULT_SAFETY_MARGIN_TOKENS, count_prompt_tokens
from nl2sql.config import load_orchestrator_settings
from nl2sql.utils.yaml_utils import normalize_for_yaml

from ..config import NL2SQLConfig
from ..contracts import StageArtifact
from ..llm_manager import LLMManager

NARRATIVE_SQL_CHAR_LIMIT_COMPACT = 1200
NARRATIVE_SQL_CHAR_LIMIT_MINIMAL = 480


class NarrativePromptTooLongError(ValueError):
    """Error cuando el prompt narrativo no cabe ni tras degradar el payload."""


def _format_rows_preview(rows: list[dict[str, Any]], limit: int) -> str:
    """Serializa un subconjunto de filas para el prompt narrativo."""

    if not rows or limit <= 0:
        return "(sin filas)"
    return yaml.safe_dump(rows[:limit], sort_keys=False, allow_unicode=True).rstrip()


def _compact_sql_for_prompt(sql: str, *, max_chars: int) -> str:
    """Normaliza y trunca SQL largo para mantenerlo util dentro del budget."""

    normalized = re.sub(r"\s+", " ", sql or "").strip()
    if len(normalized) <= max_chars:
        return normalized
    return f"{normalized[: max_chars - 3].rstrip()}..."


def _render_narrative_prompt_variant(
    *,
    prompts: Mapping[str, str],
    query: str,
    sql: str,
    row_count: int,
    truncated: bool,
    rows: list[dict[str, Any]],
    rows_preview_limit: int,
    sql_char_limit: int,
) -> tuple[str, str]:
    """Construye una variante concreta del prompt narrativo."""

    system_prompt = prompts["system"]
    user_prompt = prompts["user_template"].format(
        query=query,
        sql=_compact_sql_for_prompt(sql, max_chars=sql_char_limit),
        row_count=row_count,
        truncated_marker=" - truncado" if truncated else "",
        rows_preview=_format_rows_preview(rows, rows_preview_limit),
    )
    return system_prompt, user_prompt


def select_narrative_prompt_variant(
    *,
    prompts: Mapping[str, str],
    query: str,
    sql: str,
    row_count: int,
    truncated: bool,
    rows: list[dict[str, Any]],
    config: NL2SQLConfig,
) -> tuple[str, str, dict[str, Any]]:
    """Elige la variante mas rica del prompt narrativo que cabe en contexto."""

    preview_limit = max(0, int(config.rows_preview_limit))
    row_limits: list[int] = []
    for candidate in (preview_limit, min(preview_limit, 10), min(preview_limit, 5), 1, 0):
        if candidate not in row_limits:
            row_limits.append(candidate)

    variants = (
        {"name": "full_sql_full_rows", "rows_limit": row_limits[0], "sql_char_limit": 20000},
        {"name": "full_sql_trimmed_rows", "rows_limit": row_limits[min(1, len(row_limits) - 1)], "sql_char_limit": 20000},
        {
            "name": "compact_sql_trimmed_rows",
            "rows_limit": row_limits[min(2, len(row_limits) - 1)],
            "sql_char_limit": NARRATIVE_SQL_CHAR_LIMIT_COMPACT,
        },
        {
            "name": "compact_sql_one_row",
            "rows_limit": 1 if 1 in row_limits else row_limits[-1],
            "sql_char_limit": NARRATIVE_SQL_CHAR_LIMIT_COMPACT,
        },
        {"name": "minimal_sql_no_rows", "rows_limit": 0, "sql_char_limit": NARRATIVE_SQL_CHAR_LIMIT_MINIMAL},
    )
    last_error: NarrativePromptTooLongError | None = None

    for variant in variants:
        system_prompt, user_prompt = _render_narrative_prompt_variant(
            prompts=prompts,
            query=query,
            sql=sql,
            row_count=row_count,
            truncated=truncated,
            rows=rows,
            rows_preview_limit=int(variant["rows_limit"]),
            sql_char_limit=int(variant["sql_char_limit"]),
        )
        stats = count_prompt_tokens(
            active_model=config.narrative.model_alias,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            safety_margin_tokens=DEFAULT_SAFETY_MARGIN_TOKENS,
        )
        if stats.final_prompt_tokens <= stats.available_input_tokens:
            return (
                system_prompt,
                user_prompt,
                {
                    "prompt_variant": variant["name"],
                    "rows_preview_limit": int(variant["rows_limit"]),
                    "sql_char_limit": int(variant["sql_char_limit"]),
                    "prompt_stats": asdict(stats),
                    "prompt_tokens": stats.final_prompt_tokens,
                    "max_model_len": stats.max_model_len,
                    "safety_margin_tokens": stats.safety_margin_tokens,
                },
            )
        last_error = NarrativePromptTooLongError(
            "El prompt narrativo excede el contexto seguro incluso tras degradar el preview de filas y el SQL. "
            f"final_prompt_tokens={stats.final_prompt_tokens}, available_input_tokens={stats.available_input_tokens}, "
            f"model={stats.model_name}."
        )

    if last_error is not None:
        raise last_error
    raise NarrativePromptTooLongError("No fue posible construir un prompt narrativo valido")


def _load_prompt_asset(prompt_asset_path: Path | None, *, narrative_prompt=None) -> Mapping[str, str]:
    """Carga y valida el asset de prompts de narrativa."""

    if narrative_prompt is not None:
        return {
            "system": narrative_prompt.system,
            "user_template": narrative_prompt.user_template,
        }
    if prompt_asset_path is None:
        raise ValueError("Falta la configuracion del prompt narrativo del orquestador.")
    prompt = load_orchestrator_settings(prompt_asset_path).narrative_prompt
    return {"system": prompt.system, "user_template": prompt.user_template}


def build_narrative_runnable(
    llm_manager: LLMManager,
    config: NL2SQLConfig,
    prompt_asset_path: Path | None = None,
) -> RunnableLambda:
    """Construye la etapa final de narrativa como `RunnableLambda`."""

    prompts = _load_prompt_asset(prompt_asset_path, narrative_prompt=config.narrative_prompt)

    def _run(state: dict[str, Any]) -> dict[str, Any]:
        if state.get("status", "ok") != "ok":
            return state
        request = state["request"]
        system_prompt, user_prompt, prompt_diagnostics = select_narrative_prompt_variant(
            prompts=prompts,
            query=request.query,
            sql=state.get("final_sql", ""),
            row_count=state.get("row_count", 0),
            truncated=bool(state.get("truncated")),
            rows=list(state.get("rows", [])),
            config=config,
        )
        runner = llm_manager.acquire_narrative_runner()

        started = time.perf_counter()
        try:
            outputs = runner.run(system_prompt, user_prompt)
        finally:
            llm_manager.release()

        narrative = outputs[0].strip() if outputs else ""
        payload = {
            "query": request.query,
            "sql": state.get("final_sql", ""),
            "row_count": state.get("row_count", 0),
            "truncated": state.get("truncated", False),
            "narrative": narrative,
            "prompt_diagnostics": prompt_diagnostics,
        }
        artifact_path = Path(request.out_dir) / "narrative_response.yaml"
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_path.write_text(
            yaml.safe_dump(normalize_for_yaml(payload), sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )

        state["narrative"] = narrative
        state["narrative_prompt_diagnostics"] = prompt_diagnostics
        state.setdefault("artifacts", []).append(
            StageArtifact(
                name="narrative",
                path=artifact_path,
                payload=payload,
                duration_seconds=time.perf_counter() - started,
            )
        )
        return state

    return RunnableLambda(_run, name="narrative_stage")
