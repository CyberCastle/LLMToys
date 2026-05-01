#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Pruebas para el registro central de modelos de `llm_core`."""

from __future__ import annotations

import sys
import types

import pytest

from llm_core.model_registry import (
    MODEL_REGISTRY,
    RegisteredModel,
    build_runner,
    list_supported_models,
    resolve_model_name,
)
from llm_core.vllm_engine import VLLMConfig, VLLMRuntimeDefaults
from llm_core.vllm_interface import VLLMModelRunner


def test_list_supported_models_returns_sorted_aliases() -> None:
    """Expone los aliases soportados en orden alfabetico estable."""

    assert list_supported_models() == ("gemma4", "gemma4_e4b", "qwen3")


def test_resolve_model_name_for_gemma4_e4b() -> None:
    """Resuelve el nombre canonico sin necesidad de importar el runner."""

    assert resolve_model_name("gemma4_e4b") == "google/gemma-4-E4B-it"


def test_resolve_model_name_does_not_import_runner_module(monkeypatch: pytest.MonkeyPatch) -> None:
    """Consultar el modelo canonico no debe disparar imports pesados ni side effects."""

    monkeypatch.setattr(
        "llm_core.model_registry.import_module",
        lambda _module_name: (_ for _ in ()).throw(AssertionError("resolve_model_name no debe importar modulos")),
    )

    assert resolve_model_name("qwen3") == "Qwen/Qwen3-30B-A3B"


def test_unknown_model_error_lists_supported_models() -> None:
    """Mantiene el mensaje de error alineado con las opciones reales del runner."""

    with pytest.raises(ValueError, match="gemma4_e4b"):
        resolve_model_name("modelo_inexistente")


def test_build_runner_reports_missing_module(monkeypatch: pytest.MonkeyPatch) -> None:
    """Contextualiza los fallos cuando el modulo de config registrado no existe."""

    monkeypatch.setitem(
        MODEL_REGISTRY,
        "broken_module",
        RegisteredModel(
            config_module="llm_core.modulo_inexistente",
            runner_class="BrokenRunner",
            model_name="fake/model",
        ),
    )

    with pytest.raises(ImportError, match="broken_module"):
        build_runner("broken_module")


def test_build_runner_reports_missing_runner_class(monkeypatch: pytest.MonkeyPatch) -> None:
    """Informa cuando el registro apunta a una clase inexistente dentro del modulo."""

    module_name = "tests.fake_runner_module_missing_class"
    module = types.ModuleType(module_name)
    monkeypatch.setitem(sys.modules, module_name, module)
    monkeypatch.setitem(
        MODEL_REGISTRY,
        "broken_class",
        RegisteredModel(
            config_module=module_name,
            runner_class="MissingRunner",
            model_name="fake/model",
        ),
    )

    with pytest.raises(ImportError, match="MissingRunner"):
        build_runner("broken_class")


def test_build_runner_rejects_non_runner_class(monkeypatch: pytest.MonkeyPatch) -> None:
    """Valida que el registro solo acepte clases que extiendan `VLLMModelRunner`."""

    module_name = "tests.fake_runner_module_wrong_type"
    module = types.ModuleType(module_name)
    module.WrongRunner = object
    monkeypatch.setitem(sys.modules, module_name, module)
    monkeypatch.setitem(
        MODEL_REGISTRY,
        "wrong_type",
        RegisteredModel(
            config_module=module_name,
            runner_class="WrongRunner",
            model_name="fake/model",
        ),
    )

    with pytest.raises(TypeError, match="VLLMModelRunner"):
        build_runner("wrong_type")


def test_build_runner_supports_extension_via_runner_subclass(monkeypatch: pytest.MonkeyPatch) -> None:
    """Agregar un alias nuevo debe requerir solo registrar una subclase de runner."""

    module_name = "tests.fake_runner_extension_module"
    lifecycle_calls: list[str] = []

    class FakeRunner(VLLMModelRunner):
        """Runner ficticio para validar el contrato de extensibilidad."""

        def get_model_profile(self):  # type: ignore[override]
            raise AssertionError("La prueba usa hooks sobreescritos, no el perfil base")

        def configure(self, system_prompt: str, user_prompt: str) -> VLLMConfig:  # type: ignore[override]
            lifecycle_calls.append("configure")
            cfg = VLLMConfig(
                model="fake/model",
                gpu_memory_utilization=0.5,
                max_model_len=64,
                max_tokens=8,
                temperature=0.0,
                top_p=1.0,
                top_k=1,
                repetition_penalty=1.0,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
            )
            self.cfg = cfg
            return cfg

        def load_tokenizer(self):  # type: ignore[override]
            lifecycle_calls.append("load_tokenizer")
            return object()

        def build_prompt(self, system_prompt: str, user_prompt: str) -> str:  # type: ignore[override]
            lifecycle_calls.append("build_prompt")
            return "prompt"

        def load_model(self):  # type: ignore[override]
            lifecycle_calls.append("load_model")
            return object()

        def generate(self, prompt: str) -> list[str]:  # type: ignore[override]
            lifecycle_calls.append("generate")
            return ["resultado"]

        def cleanup(self) -> None:  # type: ignore[override]
            lifecycle_calls.append("cleanup")

    module = types.ModuleType(module_name)
    module.FakeRunner = FakeRunner
    monkeypatch.setitem(sys.modules, module_name, module)
    monkeypatch.setitem(
        MODEL_REGISTRY,
        "fake_model",
        RegisteredModel(
            config_module=module_name,
            runner_class="FakeRunner",
            model_name="fake/model",
        ),
    )

    runner = build_runner("fake_model", runtime_defaults=VLLMRuntimeDefaults(max_tokens=8))

    assert isinstance(runner, FakeRunner)
    assert runner.runtime_defaults.max_tokens == 8
    assert runner.run("Sistema", "Usuario") == ["resultado"]
    assert lifecycle_calls == [
        "configure",
        "load_tokenizer",
        "build_prompt",
        "load_model",
        "generate",
        "cleanup",
    ]
