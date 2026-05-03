#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Orquesta la cuantizacion AWQ y GPTQ usando llmcompressor."""

from __future__ import annotations

import gc
from pathlib import Path
import shutil
from typing import Any

from .awq_recipe import build_awq_recipe
from .calibration_data import load_calibration_data
from .config import QuantizerConfig
from .gptq_recipe import build_gptq_recipe
from .output_readme import write_quantized_output_readme

SAFE_SEQUENTIAL_TARGETS_BY_MODEL_TYPE: dict[str, list[str]] = {
    "qwen2": ["Qwen2Attention", "Qwen2MLP"],
    "qwen3": ["Qwen3Attention", "Qwen3MLP"],
    "llama": ["LlamaAttention", "LlamaMLP"],
    "mistral": ["MistralAttention", "MistralMLP"],
    "gemma2": ["Gemma2Attention", "Gemma2MLP"],
    "mixtral": ["MixtralAttention", "MixtralSparseMoeBlock"],
}


def _require_torch():
    """Importa torch con un error descriptivo si no esta disponible."""

    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("El cuantizador requiere torch instalado en tools/quantizer") from exc
    return torch


def _read_model_type(model_id: str, *, trust_remote_code_model: bool = False) -> str:
    """Lee `model_type` desde `AutoConfig` sin cargar los pesos del modelo."""

    from transformers import AutoConfig

    model_config = AutoConfig.from_pretrained(
        model_id,
        trust_remote_code=trust_remote_code_model,
    )
    model_type = getattr(model_config, "model_type", None)
    if not model_type:
        raise ValueError(f"No fue posible determinar model_type para '{model_id}'")
    return str(model_type).strip().lower()


def _build_max_memory(config: QuantizerConfig, scheme: str) -> dict[Any, str]:
    """Convierte el presupuesto de memoria a un mapping compatible con Transformers."""

    gpu_mib = max(1024, int(config.effective_max_gpu_memory_gib(scheme) * 1024))
    return {0: f"{gpu_mib}MiB", "cpu": "49152MiB"}


def _load_model(config: QuantizerConfig, scheme: str):
    """Carga modelo y tokenizer respetando el presupuesto de VRAM configurado."""

    torch = _require_torch()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("La cuantizacion requiere una GPU CUDA disponible")

    model = AutoModelForCausalLM.from_pretrained(
        config.model_id,
        dtype=torch.bfloat16,
        device_map="auto",
        max_memory=_build_max_memory(config, scheme),
        trust_remote_code=config.trust_remote_code_model,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_id,
        trust_remote_code=config.trust_remote_code_model,
    )
    if getattr(tokenizer, "pad_token", None) is None and getattr(tokenizer, "eos_token", None):
        tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def _build_recipe(scheme: str, model_type: str, config: QuantizerConfig) -> list[Any]:
    """Construye la receta de cuantizacion segun el esquema solicitado."""

    normalized_scheme = scheme.strip().lower()
    if normalized_scheme == "awq":
        return build_awq_recipe(
            model_type,
            config.awq_mappings_path,
            sequential_onloading=config.sequential_onloading_for(scheme),
        )
    if normalized_scheme == "gptq":
        return build_gptq_recipe()
    raise ValueError("El esquema debe ser 'awq' o 'gptq'")


def _cleanup_runtime(model: Any | None = None) -> None:
    """Libera memoria de Python y CUDA entre corridas de cuantizacion."""

    torch = _require_torch()
    try:
        from llmcompressor.core.session_functions import reset_session
    except ImportError:
        reset_session = None

    if reset_session is not None:
        try:
            reset_session()
        except Exception:
            pass

    if model is not None:
        del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if hasattr(torch.cuda, "ipc_collect"):
            torch.cuda.ipc_collect()


def _prepare_output_dir(output_dir: Path) -> None:
    """Limpia artefactos previos y recrea el directorio de salida del esquema."""

    if output_dir.exists():
        if not output_dir.is_dir():
            raise RuntimeError(f"La ruta de salida ya existe y no es un directorio: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def _resolve_sequential_targets(
    config: QuantizerConfig,
    model_type: str,
    scheme: str = "awq",
) -> list[str] | None:
    """Elige targets secuenciales seguros segun configuracion y arquitectura."""

    if not config.sequential_onloading_for(scheme):
        return None
    if config.sequential_targets_mode_for(scheme) == "auto":
        return None
    if config.sequential_targets_mode_for(scheme) == "explicit":
        return config.sequential_targets_for(scheme)
    return SAFE_SEQUENTIAL_TARGETS_BY_MODEL_TYPE.get(model_type, ["Linear"])


def quantize_model(config: QuantizerConfig, scheme: str) -> Path:
    """Cuantiza el modelo con AWQ o GPTQ y retorna el directorio generado."""

    try:
        from llmcompressor import oneshot
    except ImportError as exc:
        raise RuntimeError("No se encontro llmcompressor.") from exc

    model_type = _read_model_type(
        config.model_id,
        trust_remote_code_model=config.trust_remote_code_model,
    )
    output_dir = config.output_dir_for(scheme)
    _prepare_output_dir(output_dir)
    sequential_targets = _resolve_sequential_targets(config, model_type, scheme)
    requested_sample_count = config.effective_calibration_sample_count(scheme)
    sample_count = config.effective_calibration_sample_count(scheme)
    effective_gpu_memory = config.effective_max_gpu_memory_gib(scheme)
    sequential_onloading = config.sequential_onloading_for(scheme)

    print(f"[quantizer] Runtime {scheme.upper()}: " f"samples={requested_sample_count}, max_gpu_memory={effective_gpu_memory:.1f} GiB")
    if sequential_onloading:
        print(f"[quantizer] Pipeline secuencial activo con targets: {sequential_targets or 'auto'}")

    model = None
    tokenizer = None
    try:
        model, tokenizer = _load_model(config, scheme)
        calibration_dataset = load_calibration_data(
            tokenizer,
            config,
            num_calibration_samples=sample_count,
        )
        recipe = _build_recipe(scheme, model_type, config)
        sample_count = config.effective_calibration_sample_count(
            scheme,
            len(calibration_dataset),
        )

        oneshot_kwargs: dict[str, Any] = {
            "model": model,
            "tokenizer": tokenizer,
            "dataset": calibration_dataset,
            "recipe": recipe,
            "max_seq_length": config.max_sequence_length,
            "num_calibration_samples": sample_count,
            "shuffle_calibration_samples": False,
            "save_compressed": True,
            "output_dir": str(output_dir),
            "trust_remote_code_model": config.trust_remote_code_model,
            "pipeline": "independent",
        }
        if sequential_onloading:
            oneshot_kwargs["pipeline"] = "sequential"
            oneshot_kwargs["sequential_offload_device"] = "cpu"
            oneshot_kwargs["sequential_targets_per_subgraph"] = config.sequential_targets_per_subgraph_for(scheme)
            if sequential_targets is not None:
                oneshot_kwargs["sequential_targets"] = sequential_targets

        oneshot(
            **oneshot_kwargs,
        )
        tokenizer.save_pretrained(output_dir)
        write_quantized_output_readme(
            config,
            scheme,
            output_dir=output_dir,
            model_type=model_type,
            actual_sample_count=sample_count,
            resolved_sequential_targets=sequential_targets,
        )
    finally:
        cleanup_model = model
        cleanup_tokenizer = tokenizer
        model = None
        tokenizer = None
        calibration_dataset = None
        _cleanup_runtime(cleanup_model)
        del cleanup_tokenizer

    config_path = output_dir / "config.json"
    if not config_path.exists():
        raise RuntimeError(f"La cuantizacion termino sin generar config.json en {output_dir}")

    return output_dir
