#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Preflight compartido de presupuestos de tokens para prompts locales."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from llm_core.tokenizer_utils import count_chat_prompt_tokens


class PromptTooLongError(ValueError):
    """Error base cuando un prompt local excede el contexto disponible."""


def assert_prompt_fits(
    *,
    tokenizer: Any,
    system_prompt: str,
    user_prompt: str,
    max_model_len: int,
    max_tokens: int,
    safety_margin: int,
    fallback_builder: Callable[[Any, str, str], Any] | None = None,
    error_message: str,
    overflow_message: str,
) -> int:
    """Cuenta tokens de chat y falla si prompt + salida reservada no caben."""

    prompt_tokens = count_chat_prompt_tokens(
        tokenizer,
        system_prompt,
        user_prompt,
        fallback_builder=fallback_builder,
        error_message=error_message,
    )
    if prompt_tokens + max_tokens + safety_margin > max_model_len:
        raise PromptTooLongError(overflow_message.format(prompt_tokens=prompt_tokens))
    return prompt_tokens
