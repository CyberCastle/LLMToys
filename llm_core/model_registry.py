#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
model_registry.py

Registro explicito de aliases de modelos soportados por `llm_core`.

Resolver un alias no debe disparar imports pesados ni side effects de modulo;
por eso el nombre canonico del modelo vive directamente en este registro.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module

from llm_core.vllm_engine import VLLMRuntimeDefaults
from llm_core.vllm_interface import VLLMModelRunner


@dataclass(frozen=True)
class RegisteredModel:
    """Describe como construir un runner para un alias soportado."""

    config_module: str
    runner_class: str
    model_name: str


MODEL_REGISTRY: dict[str, RegisteredModel] = {
    "gemma4": RegisteredModel(
        config_module="llm_core.vllm_config_gemma4",
        runner_class="Gemma4Runner",
        model_name="google/gemma-4-26B-A4B-it",
    ),
    "gemma4_e4b": RegisteredModel(
        config_module="llm_core.vllm_config_gemma4_e4b",
        runner_class="Gemma4E4BRunner",
        model_name="google/gemma-4-E4B-it",
    ),
    "qwen3": RegisteredModel(
        config_module="llm_core.vllm_config_qwen36",
        runner_class="Qwen3Runner",
        model_name="Qwen/Qwen3-30B-A3B",
    ),
    "ministral3": RegisteredModel(
        config_module="llm_core.vllm_config_ministral3",
        runner_class="Ministral3Runner",
        model_name="mistralai/Ministral-3-8B-Reasoning-2512",
    ),
    "phi4_reasoning": RegisteredModel(
        config_module="llm_core.vllm_config_phi4_reasoning",
        runner_class="Phi4ReasoningRunner",
        model_name="microsoft/Phi-4-reasoning",
    ),
}


def format_supported_model_options() -> str:
    """Devuelve las opciones validas en formato legible para errores."""

    return ", ".join(f"'{option_name}'" for option_name in sorted(MODEL_REGISTRY))


def list_supported_models() -> tuple[str, ...]:
    """Retorna los aliases soportados ordenados alfabeticamente."""

    return tuple(sorted(MODEL_REGISTRY))


def _get_registered_model(active_model: str) -> RegisteredModel:
    """Resuelve y valida el registro asociado al alias solicitado."""

    registered_model = MODEL_REGISTRY.get(active_model)
    if registered_model is None:
        raise ValueError(f"Modelo '{active_model}' no reconocido. Opciones: {format_supported_model_options()}")
    return registered_model


def resolve_model_name(active_model: str) -> str:
    """Devuelve el identificador canonico del modelo sin importar su modulo."""

    return _get_registered_model(active_model).model_name


def build_runner(
    active_model: str,
    runtime_defaults: VLLMRuntimeDefaults | None = None,
) -> VLLMModelRunner:
    """Instancia el runner asociado al alias solicitado con defaults explicitos."""

    registered_model = _get_registered_model(active_model)

    try:
        config_module = import_module(registered_model.config_module)
    except Exception as exc:
        raise ImportError(f"No se pudo importar el modulo {registered_model.config_module!r} para el alias {active_model!r}.") from exc

    try:
        runner_candidate = getattr(config_module, registered_model.runner_class)
    except AttributeError as exc:
        raise ImportError(
            f"El alias {active_model!r} apunta a {registered_model.config_module!r}, "
            f"pero no existe la clase {registered_model.runner_class!r}."
        ) from exc

    if not isinstance(runner_candidate, type):
        raise TypeError(
            f"El registro del alias {active_model!r} no resuelve a una clase runner valida: " f"{registered_model.runner_class!r}."
        )
    if not issubclass(runner_candidate, VLLMModelRunner):
        raise TypeError(f"La clase {registered_model.runner_class!r} para el alias {active_model!r} debe extender VLLMModelRunner.")

    return runner_candidate(runtime_defaults=runtime_defaults)
