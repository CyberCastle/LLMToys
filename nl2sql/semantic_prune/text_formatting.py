#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from nl2sql.config import SemanticPruneTextFormattingRules, load_semantic_prune_settings, resolve_nl2sql_config_path
from nl2sql.utils.text_utils import collapse_whitespace

from .schema_logic import (
    build_foreign_key_metadata_lookup,
    get_column_descriptions,
    get_primary_keys,
    get_schema_columns,
    get_table_description,
)

from .config import EOS_TOKEN


@lru_cache(maxsize=4)
def load_text_formatting_rules(path: str | Path | None = None) -> SemanticPruneTextFormattingRules:
    """Devuelve las reglas tipadas de compactacion textual del pruning."""

    resolved_path = Path(path).expanduser().resolve() if path is not None else resolve_nl2sql_config_path()
    return load_semantic_prune_settings(resolved_path).text_formatting


def _token_ids_for_text(tokenizer: Any, text: str) -> list[int]:
    tokenized = tokenizer(text, add_special_tokens=False)
    input_ids = getattr(tokenized, "input_ids", None)
    if input_ids is None and isinstance(tokenized, dict):
        input_ids = tokenized.get("input_ids")
    if isinstance(input_ids, list):
        if input_ids and isinstance(input_ids[0], list):
            return [int(token_id) for token_id in input_ids[0]]
        return [int(token_id) for token_id in input_ids]
    return [int(token_id) for token_id in tokenizer.encode(text, add_special_tokens=False)]


def get_detailed_instruct(task: str, query: str, template: str) -> str:
    """Renderiza la query de embedding usando una plantilla configurable."""

    return template.format(task=task, query=collapse_whitespace(query.strip()))


def strip_eos(text: str, eos_token: str = EOS_TOKEN) -> str:
    return text[: -len(eos_token)] if text.endswith(eos_token) else text


def append_eos(text: str, eos_token: str = EOS_TOKEN) -> str:
    stripped_text = strip_eos(text.strip(), eos_token)
    return f"{stripped_text}{eos_token}"


def build_table_listwise_text(
    table_name: str,
    table_info: dict[str, object],
    schema: dict[str, object],
    rules: SemanticPruneTextFormattingRules | None = None,
) -> str:
    """Construye un resumen compacto para el prompt listwise.

    El retrieval inicial usa la serializacion rica del flujo actual. Para la
    pseudo-query PRF interesa un resumen mas corto, con nombres clave, PK y joins.
    """
    active_rules = rules or load_text_formatting_rules()
    parts = [f"Tabla: {table_name}"]
    table_description = get_table_description(table_info)
    if table_description:
        parts.append(f"Descripcion: {table_description}")

    primary_keys = get_primary_keys(table_info)
    if primary_keys:
        parts.append(f"PK: {', '.join(primary_keys)}")

    column_descriptions = get_column_descriptions(table_info)
    rendered_columns: list[str] = []
    for column_name, column_type in get_schema_columns(table_info)[: active_rules.table_column_summary_limit]:
        rendered_column = f"{column_name} ({column_type})"
        column_description = column_descriptions.get(column_name)
        if column_description:
            rendered_column = f"{rendered_column}: {column_description}"
        rendered_columns.append(rendered_column)
    if rendered_columns:
        parts.append(f"Columnas clave: {'; '.join(rendered_columns)}")

    join_summaries: list[str] = []
    foreign_key_lookup = build_foreign_key_metadata_lookup(table_name, table_info, schema)
    for column_name, relations in list(foreign_key_lookup.items())[: active_rules.table_foreign_key_limit]:
        relation_targets = ", ".join(relation["relation_text"] for relation in relations[: active_rules.column_relation_summary_limit])
        join_summaries.append(f"{column_name} -> {relation_targets}")
    if join_summaries:
        parts.append(f"Joins: {'; '.join(join_summaries)}")

    return " | ".join(parts)


def build_column_listwise_text(
    table_name: str,
    table_info: dict[str, object],
    schema: dict[str, object],
    column_name: str,
    column_type: str,
    rules: SemanticPruneTextFormattingRules | None = None,
) -> str:
    active_rules = rules or load_text_formatting_rules()
    foreign_key_lookup = build_foreign_key_metadata_lookup(table_name, table_info, schema)
    column_descriptions = get_column_descriptions(table_info)

    parts = [
        f"Tabla: {table_name}",
        f"Columna: {column_name}",
        f"Tipo: {column_type}",
    ]

    table_description = get_table_description(table_info)
    if table_description:
        parts.append(f"Descripcion tabla: {table_description}")

    column_description = column_descriptions.get(column_name)
    if column_description:
        parts.append(f"Descripcion columna: {column_description}")

    relations = foreign_key_lookup.get(column_name, [])
    if relations:
        relation_targets = ", ".join(relation["relation_text"] for relation in relations[: active_rules.column_relation_summary_limit])
        business_relations = "; ".join(
            dict.fromkeys(relation["verbalized_join"] for relation in relations[: active_rules.column_relation_summary_limit])
        )
        parts.append(f"FK: {relation_targets}")
        if business_relations:
            parts.append(f"Relacion: {business_relations}")

    return " | ".join(parts)


def truncate_text_by_tokens(text: str, tokenizer: Any, max_tokens: int) -> tuple[str, bool, int]:
    """Recorta un texto respetando el tokenizer real del modelo."""
    if max_tokens <= 0:
        return "", True, 0

    tokens = _token_ids_for_text(tokenizer, text)
    if len(tokens) <= max_tokens:
        return text, False, len(tokens)

    truncated_text = tokenizer.decode(tokens[:max_tokens], skip_special_tokens=False).strip()
    return truncated_text, True, max_tokens


def prepare_listwise_document_text(
    text: str,
    tokenizer: Any,
    *,
    max_tokens_per_doc: int,
    eos_token: str = EOS_TOKEN,
    rules: SemanticPruneTextFormattingRules | None = None,
) -> tuple[str, bool, int]:
    """Trunca la parte semantica y anade el EOS exigido por E2Rank."""
    active_rules = rules or load_text_formatting_rules()
    eos_budget = max(len(_token_ids_for_text(tokenizer, eos_token)), 2)
    available_tokens = max(1, max_tokens_per_doc - eos_budget - active_rules.eos_token_budget_cushion)
    base_text, was_truncated, token_count = truncate_text_by_tokens(strip_eos(text, eos_token), tokenizer, available_tokens)
    return append_eos(base_text, eos_token), was_truncated, token_count + eos_budget


def build_listwise_prompt(
    task: str,
    query: str,
    documents: list[str],
    tokenizer: Any,
    prompt_template: str,
    num_input_docs: int = 20,
) -> str:
    top_documents = documents[:num_input_docs]
    rendered_documents = "\n".join(f"[{index}] {collapse_whitespace(document)}" for index, document in enumerate(top_documents, start=1))
    messages = [
        {
            "role": "user",
            "content": prompt_template.format(
                task=task,
                documents=rendered_documents,
                query=collapse_whitespace(query.strip()),
            ),
        }
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
