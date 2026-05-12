#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Pruebas unitarias para `llm_core.prompt_optimizer`."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from llm_core.prompt_optimizer import (
    ModelTokenizerSpec,
    count_prompt_tokens,
    optimize_prompt_schema,
    resolve_model_tokenizer_spec,
)
from llm_core.vllm_engine import VLLMRuntimeDefaults


@dataclass
class _FakeBatchEncoding:
    """Emula el subconjunto de `BatchEncoding` usado por el contador de tokens."""

    input_ids: list[list[int]]

    def get(self, key: str, default: object) -> object:
        """Devuelve los `input_ids` cuando se solicitan desde la prueba."""

        if key == "input_ids":
            return self.input_ids
        return default


class _FakeTokenizer:
    """Tokenizer determinista de prueba con conteo por palabras."""

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        """Convierte cada palabra en un token entero artificial."""

        del add_special_tokens
        return list(range(len(text.split())))

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> _FakeBatchEncoding:
        """Cuenta palabras del mensaje total y devuelve un `BatchEncoding` minimo."""

        del tokenize
        del add_generation_prompt
        token_count = sum(len(message["content"].split()) for message in messages) + 1
        return _FakeBatchEncoding([list(range(token_count))])


def test_resolve_model_tokenizer_spec_uses_effective_tokenizer_and_context_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    """Gemma 4 E4B debe exponer el tokenizer efectivo y el cap real de contexto."""

    monkeypatch.delenv("GEMMA4_E4B_RUNTIME_MODE", raising=False)

    spec = resolve_model_tokenizer_spec(
        "gemma4_e4b",
        runtime_defaults=VLLMRuntimeDefaults(max_model_len=4096),
    )

    assert spec.model_name == "google/gemma-4-E4B-it"
    assert spec.tokenizer_name == "Chunity/gemma-4-E4B-it-AWQ-4bit"
    assert spec.tokenizer_mode == "auto"
    assert spec.max_model_len == 2048


def test_count_prompt_tokens_handles_batch_encoding(monkeypatch: pytest.MonkeyPatch) -> None:
    """El contador debe soportar `BatchEncoding` y calcular el prompt final."""

    monkeypatch.setattr(
        "llm_core.prompt_optimizer.resolve_model_tokenizer_spec",
        lambda *args, **kwargs: ModelTokenizerSpec(
            model_name="fake/model",
            tokenizer_name="fake/tokenizer",
            revision=None,
            tokenizer_revision=None,
            trust_remote_code=True,
            tokenizer_mode="auto",
            max_model_len=64,
        ),
    )
    monkeypatch.setattr(
        "llm_core.prompt_optimizer._load_tokenizer",
        lambda **kwargs: _FakeTokenizer(),
    )

    stats = count_prompt_tokens(
        active_model="gemma4",
        system_prompt="sistema corto",
        user_prompt="usuario final",
    )

    assert stats.user_prompt_tokens == 2
    assert stats.final_prompt_tokens == 5
    assert stats.available_input_tokens == 0


def test_optimize_prompt_schema_compresses_when_needed(monkeypatch: pytest.MonkeyPatch) -> None:
    """La optimizacion debe devolver una version comprimida cuando el prompt no cabe."""

    monkeypatch.setattr(
        "llm_core.prompt_optimizer.resolve_model_tokenizer_spec",
        lambda *args, **kwargs: ModelTokenizerSpec(
            model_name="fake/model",
            tokenizer_name="fake/tokenizer",
            revision=None,
            tokenizer_revision=None,
            trust_remote_code=True,
            tokenizer_mode="auto",
            max_model_len=14,
        ),
    )
    monkeypatch.setattr(
        "llm_core.prompt_optimizer._load_tokenizer",
        lambda **kwargs: _FakeTokenizer(),
    )
    monkeypatch.setattr(
        "llm_core.prompt_optimizer._compress_schema_text",
        lambda schema_text, question, rate: "tabla id",
    )

    result = optimize_prompt_schema(
        active_model="gemma4",
        system_prompt="sistema",
        prompt_template="dialecto {dialect} esquema {schema} pregunta {question} sentinel {failure_sentinel}",
        dialect="postgres",
        schema_text="tabla hechos id entity_c fecha amount_total moneda sitio estado canal owner region pais",
        question="cuanto vendi",
        failure_sentinel="FAIL",
        safety_margin_tokens=1,
        compression_rates=(0.5,),
    )

    assert result.compressed is True
    assert result.compression_rate == pytest.approx(0.5)
    assert result.schema_text == "tabla id"
    assert result.compressed_schema_tokens < result.original_schema_tokens
    assert result.stats.final_prompt_tokens <= result.stats.available_input_tokens
