#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Helpers compartidos para cargar tokenizers y medir budgets de prompts."""

from __future__ import annotations

from functools import lru_cache
from typing import Any, Callable

from transformers import AutoTokenizer

ChatPromptFallback = Callable[[Any, str, str], object]


@lru_cache(maxsize=8)
def load_tokenizer(
    tokenizer_name: str,
    revision: str | None = None,
    tokenizer_revision: str | None = None,
    trust_remote_code: bool = True,
    hf_token: str | None = None,
) -> Any:
    """Carga y cachea un tokenizer de Hugging Face con parametros estables."""

    return AutoTokenizer.from_pretrained(
        tokenizer_name,
        revision=tokenizer_revision or revision,
        trust_remote_code=trust_remote_code,
        token=hf_token,
    )


def normalize_token_count(token_ids: object, *, error_message: str) -> int:
    """Normaliza distintos formatos de salida de tokenizer a un entero de tokens."""

    token_ids_obj: Any = token_ids
    if hasattr(token_ids_obj, "get"):
        token_ids_obj = token_ids_obj.get("input_ids", [])
    if hasattr(token_ids_obj, "shape"):
        return int(token_ids_obj.shape[-1])
    if isinstance(token_ids_obj, list) and token_ids_obj and isinstance(token_ids_obj[0], list):
        return len(token_ids_obj[0])
    if isinstance(token_ids_obj, list):
        return len(token_ids_obj)
    raise ValueError(error_message)


def count_text_tokens(tokenizer: Any, text: str) -> int:
    """Cuenta tokens de texto plano sin agregar tokens especiales."""

    return len(tokenizer.encode(text, add_special_tokens=False))


def count_chat_prompt_tokens(
    tokenizer: Any,
    system_prompt: str,
    user_prompt: str,
    *,
    fallback_builder: ChatPromptFallback | None = None,
    error_message: str = "No se pudo contar los tokens del prompt.",
) -> int:
    """Cuenta tokens de un prompt de chat usando plantilla nativa o fallback controlado."""

    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    if callable(apply_chat_template):
        token_ids = apply_chat_template(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            tokenize=True,
            add_generation_prompt=True,
        )
        return normalize_token_count(token_ids, error_message=error_message)
    if fallback_builder is None:
        raise ValueError(error_message)
    fallback_token_ids = fallback_builder(tokenizer, system_prompt, user_prompt)
    return normalize_token_count(fallback_token_ids, error_message=error_message)
