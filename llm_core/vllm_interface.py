#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
vllm_interface.py

Clase base para runners vLLM. Toda la logica comun del ciclo de vida vive
aca; cada LLM concreto aporta solo su perfil y, si hace falta, hooks puntuales
como el post-procesado de salida.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import logging
import os
from typing import Any

from llm_core.vllm_engine import (
    ModelRuntimeProfile,
    VLLMConfig,
    VLLMRuntimeDefaults,
    build_prompt as render_chat_prompt,
    build_vllm_config_from_profile,
    validate_config,
)
from llm_core.vllm_runtime_utils import (
    destroy_distributed_process_group,
    release_cuda_memory,
    resolve_fallback_max_model_len,
    should_try_stepdown_fallback,
    shutdown_vllm_engine_once,
)

logger = logging.getLogger(__name__)


class VLLMModelRunner(ABC):
    """Template method para ejecutar un LLM sobre vLLM sin duplicacion."""

    def __init__(self, runtime_defaults: VLLMRuntimeDefaults | None = None) -> None:
        self.runtime_defaults = runtime_defaults or VLLMRuntimeDefaults()
        self.cfg: VLLMConfig | None = None
        self._tokenizer: Any = None
        self._llm: Any = None
        self._shutdown_ids: set[int] = set()

    @abstractmethod
    def get_model_profile(self) -> ModelRuntimeProfile:
        """Devuelve el perfil especifico del modelo asociado al runner."""

    def _get_hf_token(self) -> str | None:
        """Resuelve el token de Hugging Face de forma perezosa y sin mutar el entorno."""

        token = os.environ.get("HF_TOKEN")
        if token is None or not token.strip():
            return None
        return token.strip()

    def configure(self, system_prompt: str, user_prompt: str) -> VLLMConfig:
        """Construye la configuracion efectiva a partir del perfil del runner."""

        cfg = build_vllm_config_from_profile(
            self.get_model_profile(),
            runtime_defaults=self.runtime_defaults,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            hf_token=self._get_hf_token(),
        )
        self.cfg = cfg
        return cfg

    def load_tokenizer(self) -> Any:
        """Carga el tokenizer efectivo del runner con Hugging Face Transformers."""

        from transformers import AutoTokenizer

        if self.cfg is None:
            raise RuntimeError("El runner debe configurarse antes de cargar el tokenizer")

        tokenizer_name = self.cfg.tokenizer or self.cfg.model
        self._tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            revision=self.cfg.tokenizer_revision,
            trust_remote_code=self.cfg.trust_remote_code,
            token=self.cfg.hf_token,
        )
        return self._tokenizer

    def build_prompt(self, system_prompt: str, user_prompt: str) -> str:
        """Construye el prompt final usando el chat template del tokenizer."""

        return render_chat_prompt(
            tokenizer=self._tokenizer,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )

    def _build_llm_instance(self) -> Any:
        """Instancia `vllm.LLM` con la configuracion efectiva del runner."""

        from vllm import LLM

        if self.cfg is None:
            raise RuntimeError("El runner debe configurarse antes de cargar el modelo")
        return LLM(**self.cfg.build_llm_kwargs())

    def load_model(self) -> Any:
        """Carga el runtime de vLLM y reintenta una vez con menos contexto si aplica."""

        if self.cfg is None:
            raise RuntimeError("El runner debe configurarse antes de cargar el modelo")

        attempted_fallback = False
        while True:
            try:
                self._llm = self._build_llm_instance()
                return self._llm
            except BaseException as exc:
                if attempted_fallback or not should_try_stepdown_fallback(exc):
                    raise
                fallback_max_model_len = resolve_fallback_max_model_len(self.cfg.max_model_len, exc)
                if fallback_max_model_len is None or fallback_max_model_len >= self.cfg.max_model_len:
                    raise
                attempted_fallback = True
                logger.warning(
                    "Reintentando %s con max_model_len=%s tras error de inicializacion del engine.",
                    self.get_model_profile().alias,
                    fallback_max_model_len,
                )
                self.cfg.max_model_len = fallback_max_model_len
                if self.cfg.max_tokens > fallback_max_model_len:
                    self.cfg.max_tokens = fallback_max_model_len
                if self.cfg.max_num_batched_tokens is not None:
                    self.cfg.max_num_batched_tokens = min(
                        self.cfg.max_num_batched_tokens,
                        fallback_max_model_len,
                    )
                release_cuda_memory()

    def _post_process_output(self, text: str) -> str:
        """Hook para ajustar salidas de modelos con formato especial."""

        return text

    def generate(self, prompt: str) -> list[str]:
        """Ejecuta la generacion y aplica el hook de post-procesado por salida."""

        if self.cfg is None or self._llm is None:
            raise RuntimeError("El modelo debe cargarse antes de generar")
        sampling_params = self.cfg.build_sampling_params()
        outputs = self._llm.generate([prompt], sampling_params)
        return [self._post_process_output(output.outputs[0].text).strip() for output in outputs]

    def cleanup(self) -> None:
        """Libera tokenizer, runtime vLLM, process group y memoria CUDA."""

        llm = self._llm
        self._llm = None
        self._tokenizer = None
        self.cfg = None

        if llm is not None:
            try:
                shutdown_vllm_engine_once(llm, self._shutdown_ids)
            except Exception as exc:
                logger.warning("No se pudo apagar el engine de vLLM limpiamente: %s", exc)

        try:
            destroy_distributed_process_group()
        except Exception as exc:
            logger.warning("No se pudo destruir el process group distribuido: %s", exc)

        release_cuda_memory()

    def run(self, system_prompt: str, user_prompt: str) -> list[str]:
        """Orquesta configurar, cargar, generar y limpiar con `try/finally`."""

        self.cfg = self.configure(system_prompt, user_prompt)
        validate_config(self.cfg)
        try:
            self._tokenizer = self.load_tokenizer()
            prompt = self.build_prompt(self.cfg.system_prompt, self.cfg.user_prompt)
            self._llm = self.load_model()
            return self.generate(prompt)
        finally:
            self.cleanup()
