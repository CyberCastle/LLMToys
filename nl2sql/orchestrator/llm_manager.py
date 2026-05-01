#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Selector central de modelos y liberacion de recursos del pipeline NL2SQL."""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
from typing import Literal, Protocol, cast, runtime_checkable

from llm_core.model_registry import build_runner as build_generic_runner
from llm_core.vllm_runtime_utils import release_cuda_memory
from nl2sql.config import env_str

StageName = Literal["prune", "resolver", "solver", "narrative"]
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class NarrativeLLMSettings:
    """Configuracion del generador narrativo final."""

    model_alias: str = field(default_factory=lambda: env_str("NL2SQL_NARRATIVE_MODEL_ALIAS", "gemma4_e4b"))


@runtime_checkable
class CleanableRunner(Protocol):
    """Contrato mínimo del runner narrativo para liberar recursos explícitamente."""

    def cleanup(self) -> None: ...


@runtime_checkable
class NarrativeRunner(CleanableRunner, Protocol):
    """Contrato mínimo del runner narrativo usado por la etapa final."""

    def run(self, system_prompt: str, user_prompt: str) -> list[str]: ...


class LLMManager:
    """Coordina el cambio de modelos y la liberacion de VRAM entre etapas."""

    def __init__(self, config: NarrativeLLMSettings | None = None) -> None:
        self.config = config or NarrativeLLMSettings()
        self._active_stage: StageName | None = None
        self._narrative_runner: NarrativeRunner | None = None

    def prepare_stage(self, stage: StageName) -> None:
        """Marca el cambio de etapa y libera runtimes previos si aplica."""

        if stage == "narrative" and self._active_stage == stage and self._narrative_runner is not None:
            return
        self.release()
        self._active_stage = stage

    def acquire_narrative_runner(self) -> NarrativeRunner:
        """Instancia el runner vLLM del narrador final si todavia no existe."""

        self.prepare_stage("narrative")
        if self._narrative_runner is None:
            runner = build_generic_runner(self.config.model_alias)
            self._narrative_runner = cast(NarrativeRunner, runner)
        return self._narrative_runner

    def release(self) -> None:
        """Libera runners genericos y runtimes cacheados de las etapas previas."""

        if self._narrative_runner is not None:
            cleanup = getattr(self._narrative_runner, "cleanup", None)
            if not isinstance(self._narrative_runner, CleanableRunner) and not callable(cleanup):
                raise TypeError("El runner narrativo no implementa cleanup()")
            cleanup()
            self._narrative_runner = None

        try:
            from nl2sql.semantic_prune.e2rank_engine import clear_e2rank_runtime

            clear_e2rank_runtime()
        except Exception as exc:
            logger.warning("No se pudo liberar el runtime de pruning: %s", exc)

        try:
            from nl2sql.semantic_resolver import release_semantic_resolver_runtimes

            release_semantic_resolver_runtimes()
        except Exception as exc:
            logger.warning("No se pudo liberar el runtime del resolver: %s", exc)

        release_cuda_memory()
        self._active_stage = None
