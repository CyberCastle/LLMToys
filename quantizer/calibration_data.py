#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Carga y preprocesa datasets de calibracion de manera generica."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
import re
from typing import Any, cast

import yaml

from .config import QuantizerConfig

PLACEHOLDER_RE = re.compile(r"\{([A-Za-z0-9_]+)(\?)?\}")


def _load_templates(templates_path: Path) -> dict[str, Any]:
    """Lee el YAML de plantillas del cuantizador."""

    payload = yaml.safe_load(templates_path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError("dataset_templates.yaml invalido: la raiz debe ser un mapping")
    return payload


def _select_template(dataset_name: str, templates: Mapping[str, Any]) -> tuple[str | None, dict[str, Any]]:
    """Busca la plantilla especifica mas larga cuyo prefijo coincide con el dataset."""

    matched_key = None
    matched_template: dict[str, Any] = {}
    for key, value in templates.items():
        if key == "_default" or not isinstance(value, Mapping):
            continue
        if dataset_name.startswith(key) and (matched_key is None or len(key) > len(matched_key)):
            matched_key = key
            matched_template = dict(value)
    return matched_key, matched_template


def _normalize_value(value: Any) -> str:
    """Convierte valores arbitrarios del dataset a texto seguro."""

    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, Mapping):
        return yaml.safe_dump(dict(value), sort_keys=False, allow_unicode=True).strip()
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        normalized_items = [_normalize_value(item) for item in value]
        return "\n".join(item for item in normalized_items if item)
    return str(value).strip()


def _render_template_string(template: str, example: Mapping[str, Any]) -> str:
    """Renderiza un bloque de texto soportando placeholders opcionales por linea."""

    rendered_lines: list[str] = []
    for raw_line in template.splitlines():
        matches = list(PLACEHOLDER_RE.finditer(raw_line))
        if not matches:
            if raw_line.strip():
                rendered_lines.append(raw_line.rstrip())
            continue

        skip_line = False
        rendered_line = raw_line
        for match in matches:
            field_name = match.group(1)
            is_optional = match.group(2) == "?"
            raw_value = example.get(field_name)
            value = _normalize_value(raw_value)

            if not value and is_optional:
                skip_line = True
                break
            if not value and not is_optional:
                raise ValueError(f"El ejemplo no contiene el campo requerido '{field_name}'")

            rendered_line = rendered_line.replace(match.group(0), value)

        if skip_line:
            continue

        normalized_line = rendered_line.rstrip()
        if normalized_line.strip():
            rendered_lines.append(normalized_line)

    return "\n".join(rendered_lines).strip()


def _render_messages(messages_template: Sequence[Mapping[str, Any]], example: Mapping[str, Any]) -> list[dict[str, str]]:
    """Aplica una plantilla de mensajes estilo chat a un ejemplo concreto."""

    messages: list[dict[str, str]] = []
    for message in messages_template:
        role = _normalize_value(message.get("role")) or "user"
        content_template = _normalize_value(message.get("content"))
        content = _render_template_string(content_template, example)
        if content:
            messages.append({"role": role, "content": content})
    return messages


def _apply_chat_template(tokenizer: Any, messages: Sequence[Mapping[str, str]]) -> str:
    """Renderiza mensajes usando el chat template del tokenizer si existe."""

    if hasattr(tokenizer, "apply_chat_template"):
        return tokenizer.apply_chat_template(
            list(messages),
            tokenize=False,
            add_generation_prompt=False,
        )

    return "\n".join(f"{message['role']}: {message['content']}" for message in messages)


def _tokenize_text(tokenizer: Any, text: str, max_sequence_length: int) -> dict[str, Any] | None:
    """Tokeniza una muestra y descarta las secuencias vacias."""

    encoded = tokenizer(
        text,
        add_special_tokens=False,
        truncation=True,
        max_length=max_sequence_length,
        return_attention_mask=True,
    )

    input_ids = list(encoded.get("input_ids", []))
    if not input_ids:
        return None

    attention_mask = list(encoded.get("attention_mask", [1] * len(input_ids)))
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "source_text": text,
    }


def prepare_calibration_records(
    examples: Iterable[Mapping[str, Any]],
    tokenizer: Any,
    dataset_name: str,
    max_sequence_length: int,
    templates_path: Path,
    *,
    num_calibration_samples: int | None = None,
) -> list[dict[str, Any]]:
    """Convierte ejemplos crudos a registros tokenizados listos para calibracion."""

    templates = _load_templates(templates_path)
    _, template = _select_template(dataset_name, templates)
    default_template = templates.get("_default", {}) if isinstance(templates.get("_default"), Mapping) else {}

    rendered_samples: list[dict[str, Any]] = []
    for example in examples:
        if not isinstance(example, Mapping):
            raise ValueError("Cada ejemplo del dataset de calibracion debe ser un mapping")

        if "messages_template" in template:
            messages = _render_messages(template["messages_template"], example)
            text = _apply_chat_template(tokenizer, messages)
        elif "text_template" in template:
            text = _render_template_string(_normalize_value(template["text_template"]), example)
        else:
            text_field = template.get("text_field") or default_template.get("text_field")
            if text_field and _normalize_value(example.get(text_field)):
                text = _normalize_value(example[text_field])
            elif _normalize_value(example.get("text")):
                text = _normalize_value(example["text"])
            else:
                available_columns = ", ".join(sorted(str(key) for key in example.keys()))
                raise ValueError(
                    "No se pudo resolver una plantilla para el dataset "
                    f"{dataset_name!r}. Columnas disponibles en el ejemplo: {available_columns}"
                )

        tokenized = _tokenize_text(tokenizer, text, max_sequence_length)
        if tokenized is not None:
            rendered_samples.append(tokenized)
        if num_calibration_samples is not None and len(rendered_samples) >= num_calibration_samples:
            break

    return rendered_samples


def load_calibration_data(
    tokenizer: Any,
    config: QuantizerConfig,
    templates_path: Path | None = None,
    num_calibration_samples: int | None = None,
):
    """Carga y preprocesa el dataset de calibracion aplicando la plantilla registrada."""

    try:
        from datasets import Dataset, load_dataset
    except ImportError as exc:
        raise RuntimeError("La carga de calibracion requiere el paquete 'datasets'. Instala el subproyecto tools/") from exc

    active_templates_path = templates_path or config.dataset_templates_path
    target_num_calibration_samples = config.num_calibration_samples if num_calibration_samples is None else num_calibration_samples

    try:
        if config.dataset_config_name:
            raw_dataset = load_dataset(
                config.calibration_dataset,
                config.dataset_config_name,
                split=config.calibration_split,
            )
        else:
            raw_dataset = load_dataset(config.calibration_dataset, split=config.calibration_split)
    except Exception as exc:
        raise RuntimeError(
            "No se pudo cargar el dataset de calibracion. Revisa acceso en Hugging Face,"
            " aceptación de términos y autenticación con huggingface-cli login."
        ) from exc

    shuffled_dataset = raw_dataset.shuffle(seed=42)
    candidate_count = min(
        len(shuffled_dataset),
        max(target_num_calibration_samples * 4, target_num_calibration_samples),
    )
    subset = shuffled_dataset.select(range(candidate_count)) if candidate_count < len(shuffled_dataset) else shuffled_dataset

    records = prepare_calibration_records(
        cast(Iterable[Mapping[str, Any]], subset),
        tokenizer,
        config.calibration_dataset,
        config.max_sequence_length,
        active_templates_path,
        num_calibration_samples=target_num_calibration_samples,
    )

    if not records:
        raise ValueError("El dataset de calibracion no produjo ninguna secuencia tokenizada util")

    dataset_records = [{key: value for key, value in record.items() if key != "source_text"} for record in records]
    return Dataset.from_list(dataset_records)
