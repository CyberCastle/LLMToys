#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Utilidades para medir y comprimir prompts sobre runners vLLM.

La resolucion del tokenizer y del contexto maximo se apoya en `VLLMRuntimeDefaults`
y en el perfil del runner asociado al alias del modelo.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import os
import re
from typing import Any

from langchain_core.prompts import PromptTemplate

from llm_core.model_registry import build_runner, resolve_model_name
from llm_core.tokenizer_utils import count_chat_prompt_tokens, count_text_tokens, load_tokenizer as _load_tokenizer
from llm_core.vllm_engine import VLLMRuntimeDefaults, build_prompt

LLMLINGUA2_MODEL = "microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank"
DEFAULT_SAFETY_MARGIN_TOKENS = 256
DEFAULT_COMPRESSION_RATES: tuple[float, ...] = (
    0.85,
    0.75,
    0.67,
    0.55,
    0.45,
    0.33,
    0.25,
)
DEFAULT_FORCE_TOKENS: list[str] = ["\n", ":", "-", "_", ".", '"']
DEFAULT_MAX_SCHEMA_BLOCK_CHARS = 1200


@dataclass(frozen=True)
class ModelTokenizerSpec:
    """Describe el modelo y tokenizer usados para estimar tokens de entrada."""

    model_name: str
    tokenizer_name: str
    revision: str | None
    tokenizer_revision: str | None
    trust_remote_code: bool
    tokenizer_mode: str
    max_model_len: int


@dataclass(frozen=True)
class PromptTokenStats:
    """Resume el presupuesto de tokens observado para un prompt concreto."""

    model_name: str
    tokenizer_name: str
    user_prompt_tokens: int
    final_prompt_tokens: int
    max_model_len: int
    safety_margin_tokens: int

    @property
    def available_input_tokens(self) -> int:
        """Retorna los tokens seguros disponibles para entrada antes de generar."""

        return max(0, self.max_model_len - self.safety_margin_tokens)


@dataclass(frozen=True)
class PromptOptimizationResult:
    """Representa el resultado final del proceso de compresion del esquema."""

    schema_text: str
    compressed: bool
    compression_rate: float | None
    original_schema_tokens: int
    compressed_schema_tokens: int
    stats: PromptTokenStats


def _resolve_runtime_defaults(runtime_defaults: VLLMRuntimeDefaults | None) -> VLLMRuntimeDefaults:
    """Normaliza los defaults opcionales a una instancia usable e inmutable."""

    return runtime_defaults or VLLMRuntimeDefaults()


def resolve_model_tokenizer_spec(
    active_model: str,
    runtime_defaults: VLLMRuntimeDefaults | None = None,
) -> ModelTokenizerSpec:
    """Resuelve el tokenizer y el contexto efectivo asociados a un alias soportado."""

    resolved_defaults = _resolve_runtime_defaults(runtime_defaults)
    runner = build_runner(active_model, runtime_defaults=resolved_defaults)
    profile = runner.get_model_profile()
    canonical_model_name = resolve_model_name(active_model)
    tokenizer_name = resolved_defaults.tokenizer or profile.effective_tokenizer_name() or profile.model_name or canonical_model_name
    max_model_len = resolved_defaults.max_model_len
    if profile.max_model_len_cap is not None:
        max_model_len = min(max_model_len, profile.max_model_len_cap)

    return ModelTokenizerSpec(
        model_name=canonical_model_name,
        tokenizer_name=tokenizer_name,
        revision=resolved_defaults.revision,
        tokenizer_revision=resolved_defaults.tokenizer_revision,
        trust_remote_code=resolved_defaults.trust_remote_code,
        tokenizer_mode=resolved_defaults.tokenizer_mode,
        max_model_len=max_model_len,
    )


@lru_cache(maxsize=1)
def _get_llmlingua2_compressor() -> Any:
    """Construye de forma perezosa el compresor LLMLingua-2."""

    try:
        from llmlingua import PromptCompressor
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "La compresion de prompts requiere `llmlingua`. Ejecuta `uv sync` para instalar las dependencias del proyecto."
        ) from exc

    return PromptCompressor(
        model_name=LLMLINGUA2_MODEL,
        device_map="cpu",
        use_llmlingua2=True,
    )


def count_prompt_tokens(
    active_model: str,
    system_prompt: str,
    user_prompt: str,
    safety_margin_tokens: int = DEFAULT_SAFETY_MARGIN_TOKENS,
    runtime_defaults: VLLMRuntimeDefaults | None = None,
) -> PromptTokenStats:
    """Cuenta tokens efectivos para un alias de modelo y un prompt ya renderizado."""

    spec = resolve_model_tokenizer_spec(active_model, runtime_defaults=runtime_defaults)
    tokenizer = _load_tokenizer(
        tokenizer_name=spec.tokenizer_name,
        revision=spec.revision,
        tokenizer_revision=spec.tokenizer_revision,
        trust_remote_code=spec.trust_remote_code,
        tokenizer_mode=spec.tokenizer_mode,
        hf_token=os.environ.get("HF_TOKEN"),
    )
    return PromptTokenStats(
        model_name=spec.model_name,
        tokenizer_name=spec.tokenizer_name,
        user_prompt_tokens=count_text_tokens(tokenizer, user_prompt),
        final_prompt_tokens=count_chat_prompt_tokens(
            tokenizer,
            system_prompt,
            user_prompt,
            fallback_builder=lambda active_tokenizer, system_text, user_text: active_tokenizer.encode(
                build_prompt(active_tokenizer, system_text, user_text),
                add_special_tokens=False,
            ),
        ),
        max_model_len=spec.max_model_len,
        safety_margin_tokens=safety_margin_tokens,
    )


def _split_schema_blocks(schema_text: str) -> list[str]:
    """Separa el esquema por bloques logicos para facilitar su compresion."""

    blocks: list[str] = []
    current_block: list[str] = []

    for line in schema_text.splitlines():
        if line and not line[0].isspace():
            if current_block:
                blocks.append("\n".join(current_block).strip())
            current_block = [line]
            continue
        if current_block:
            current_block.append(line)

    if current_block:
        blocks.append("\n".join(current_block).strip())

    chunked_blocks: list[str] = []
    for block in blocks:
        chunked_blocks.extend(_chunk_schema_block(block))

    return [block for block in chunked_blocks if block]


def _chunk_schema_block(block: str, max_block_chars: int = DEFAULT_MAX_SCHEMA_BLOCK_CHARS) -> list[str]:
    """Trocea bloques largos del esquema para estabilizar la compresion."""

    if len(block) <= max_block_chars:
        return [block]

    lines = block.splitlines()
    if len(lines) <= 2:
        return [block]

    header = lines[0]
    chunks: list[str] = []
    current_chunk = [header]
    current_len = len(header)

    for line in lines[1:]:
        projected_len = current_len + len(line) + 1
        if projected_len > max_block_chars and len(current_chunk) > 1:
            chunks.append("\n".join(current_chunk).strip())
            current_chunk = [header, line]
            current_len = len(header) + len(line) + 1
            continue

        current_chunk.append(line)
        current_len = projected_len

    if current_chunk:
        chunks.append("\n".join(current_chunk).strip())

    return chunks


def _normalize_compressed_schema_text(text: str) -> str:
    """Normaliza el texto comprimido para evitar artefactos comunes del compresor."""

    normalized = text.replace("\r\n", "\n")
    normalized = re.sub(r"(?<=\w)\s*_\s*(?=\w)", "_", normalized)
    normalized = re.sub(r"(?<=\w)\s*-\s*(?=\w)", "-", normalized)
    normalized = re.sub(r"\s+:", ":", normalized)
    normalized = re.sub(r":\s+", ": ", normalized)
    normalized = re.sub(r"\n{3,}", "\n\n", normalized)
    return normalized.strip()


def _compress_schema_text(
    schema_text: str,
    question: str,
    rate: float,
    force_tokens: list[str] | None = None,
) -> str:
    """Comprime el esquema usando LLMLingua-2 y devuelve texto normalizado."""

    compressor = _get_llmlingua2_compressor()
    schema_blocks = _split_schema_blocks(schema_text)
    result = compressor.compress_prompt(
        schema_blocks,
        question=question,
        rate=rate,
        use_context_level_filter=True,
        use_token_level_filter=False,
        use_sentence_level_filter=False,
        force_tokens=force_tokens or DEFAULT_FORCE_TOKENS,
    )
    compressed_prompt_list = result.get("compressed_prompt_list")
    if isinstance(compressed_prompt_list, list) and compressed_prompt_list:
        compressed_prompt = "\n\n".join(block.strip() for block in compressed_prompt_list if isinstance(block, str) and block.strip())
    else:
        compressed_prompt = result.get("compressed_prompt", "")

    if not isinstance(compressed_prompt, str) or not compressed_prompt.strip():
        raise ValueError("LLMLingua-2 devolvio un prompt comprimido vacio o invalido.")

    return _normalize_compressed_schema_text(compressed_prompt)


def optimize_prompt_schema(
    active_model: str,
    system_prompt: str,
    prompt_template: str,
    dialect: str,
    schema_text: str,
    question: str,
    failure_sentinel: str,
    safety_margin_tokens: int = DEFAULT_SAFETY_MARGIN_TOKENS,
    compression_rates: tuple[float, ...] = DEFAULT_COMPRESSION_RATES,
    runtime_defaults: VLLMRuntimeDefaults | None = None,
) -> PromptOptimizationResult:
    """Ajusta el bloque de esquema al contexto disponible, comprimiendolo si hace falta."""

    resolved_defaults = _resolve_runtime_defaults(runtime_defaults)
    prompt = PromptTemplate.from_template(prompt_template)
    initial_user_prompt = prompt.format(
        dialect=dialect,
        schema=schema_text,
        question=question,
        failure_sentinel=failure_sentinel,
    )
    initial_stats = count_prompt_tokens(
        active_model=active_model,
        system_prompt=system_prompt,
        user_prompt=initial_user_prompt,
        safety_margin_tokens=safety_margin_tokens,
        runtime_defaults=resolved_defaults,
    )

    spec = resolve_model_tokenizer_spec(active_model, runtime_defaults=resolved_defaults)
    tokenizer = _load_tokenizer(
        tokenizer_name=spec.tokenizer_name,
        revision=spec.revision,
        tokenizer_revision=spec.tokenizer_revision,
        trust_remote_code=spec.trust_remote_code,
        tokenizer_mode=spec.tokenizer_mode,
        hf_token=os.environ.get("HF_TOKEN"),
    )
    original_schema_tokens = count_text_tokens(tokenizer, schema_text)

    if initial_stats.final_prompt_tokens <= initial_stats.available_input_tokens:
        return PromptOptimizationResult(
            schema_text=schema_text,
            compressed=False,
            compression_rate=None,
            original_schema_tokens=original_schema_tokens,
            compressed_schema_tokens=original_schema_tokens,
            stats=initial_stats,
        )

    for rate in compression_rates:
        compressed_schema_text = _compress_schema_text(schema_text, question=question, rate=rate)
        compressed_user_prompt = prompt.format(
            dialect=dialect,
            schema=compressed_schema_text,
            question=question,
            failure_sentinel=failure_sentinel,
        )
        compressed_stats = count_prompt_tokens(
            active_model=active_model,
            system_prompt=system_prompt,
            user_prompt=compressed_user_prompt,
            safety_margin_tokens=safety_margin_tokens,
            runtime_defaults=resolved_defaults,
        )
        compressed_schema_tokens = count_text_tokens(tokenizer, compressed_schema_text)
        if compressed_stats.final_prompt_tokens <= compressed_stats.available_input_tokens:
            return PromptOptimizationResult(
                schema_text=compressed_schema_text,
                compressed=True,
                compression_rate=rate,
                original_schema_tokens=original_schema_tokens,
                compressed_schema_tokens=compressed_schema_tokens,
                stats=compressed_stats,
            )

    raise ValueError(
        "No fue posible ajustar el prompt al contexto del modelo incluso despues de comprimir el esquema con LLMLingua-2. "
        f"Tokens finales: {initial_stats.final_prompt_tokens}. Limite de entrada seguro: {initial_stats.available_input_tokens}."
    )
