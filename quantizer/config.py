#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Configuracion tipada del cuantizador generico."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import re

MODULE_DIR = Path(__file__).resolve().parent
DEFAULT_AWQ_MAPPINGS_PATH = MODULE_DIR / "assets" / "awq_mappings.yaml"
DEFAULT_DATASET_TEMPLATES_PATH = MODULE_DIR / "assets" / "dataset_templates.yaml"

# Defaults documentados. Los identificadores concretos de modelo y dataset se
# inyectan por entorno para mantener el modulo agnostico.
DEFAULT_CALIBRATION_SPLIT = "train"
DEFAULT_AWQ_NUM_CALIBRATION_SAMPLES = 128
DEFAULT_GPTQ_NUM_CALIBRATION_SAMPLES = 512
DEFAULT_MAX_SEQUENCE_LENGTH = 2048
DEFAULT_QUANTIZE_SCHEME = "both"
DEFAULT_OUTPUT_DIR = "out/quantized"
DEFAULT_AWQ_MAX_GPU_MEMORY_GIB = 13.0
DEFAULT_GPTQ_MAX_GPU_MEMORY_GIB = 10.0
DEFAULT_AWQ_SEQUENTIAL_ONLOADING = True
DEFAULT_GPTQ_SEQUENTIAL_ONLOADING = True
DEFAULT_AWQ_SEQUENTIAL_TARGETS_PER_SUBGRAPH = 1
DEFAULT_GPTQ_SEQUENTIAL_TARGETS_PER_SUBGRAPH = 1
DEFAULT_SEQUENTIAL_TARGETS_MODE = "safe"
DEFAULT_TRUST_REMOTE_CODE_MODEL = False
DEFAULT_RUN_VLLM_SMOKE_TEST = False
DEFAULT_MEMORY_PREFLIGHT_MODE = "guard"


def _parse_bool(raw_value: str | None, default: bool) -> bool:
    """Convierte strings de entorno a booleanos con un conjunto acotado."""

    if raw_value is None:
        return default

    normalized = raw_value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"Valor booleano invalido: {raw_value!r}")


def _parse_optional_path(raw_value: str | None, default_path: Path) -> Path:
    """Resuelve una ruta opcional de entorno o usa el asset bundled."""

    if raw_value is None or not raw_value.strip():
        return default_path
    return Path(raw_value).expanduser().resolve()


def _parse_memory_preflight_mode(raw_value: str | None) -> str:
    """Normaliza el modo del preflight de memoria a un conjunto acotado."""

    if raw_value is None or not raw_value.strip():
        return DEFAULT_MEMORY_PREFLIGHT_MODE

    normalized = raw_value.strip().lower()
    if normalized not in {"off", "report", "guard", "fail-fast"}:
        raise ValueError("QUANTIZER_MEMORY_PREFLIGHT_MODE debe ser 'off', 'report', 'guard' o 'fail-fast'")
    return normalized


def _parse_sequential_targets(
    raw_value: str | None,
    env_var_name: str,
) -> tuple[str, list[str] | None]:
    """Parsea la configuracion de targets secuenciales desde entorno."""

    if raw_value is None:
        return DEFAULT_SEQUENTIAL_TARGETS_MODE, None

    normalized = raw_value.strip()
    if not normalized:
        return DEFAULT_SEQUENTIAL_TARGETS_MODE, None
    normalized_lower = normalized.lower()
    if normalized_lower in {"safe", "safe-auto"}:
        return DEFAULT_SEQUENTIAL_TARGETS_MODE, None
    if normalized_lower == "auto":
        return "auto", None

    targets = [item.strip() for item in normalized.split(",") if item.strip()]
    if not targets:
        raise ValueError(f"{env_var_name} debe contener al menos un target, 'auto' o 'safe-auto'")
    return "explicit", targets


@dataclass(slots=True)
class QuantizerConfig:
    """Agrupa toda la configuracion necesaria para cuantizar un modelo."""

    model_id: str
    calibration_dataset: str
    calibration_split: str = DEFAULT_CALIBRATION_SPLIT
    dataset_config_name: str | None = None
    awq_num_calibration_samples: int = DEFAULT_AWQ_NUM_CALIBRATION_SAMPLES
    gptq_num_calibration_samples: int = DEFAULT_GPTQ_NUM_CALIBRATION_SAMPLES
    max_sequence_length: int = DEFAULT_MAX_SEQUENCE_LENGTH
    quantize_scheme: str = DEFAULT_QUANTIZE_SCHEME
    output_dir: Path = Path(DEFAULT_OUTPUT_DIR)
    awq_max_gpu_memory_gib: float = DEFAULT_AWQ_MAX_GPU_MEMORY_GIB
    gptq_max_gpu_memory_gib: float = DEFAULT_GPTQ_MAX_GPU_MEMORY_GIB
    awq_sequential_onloading: bool = DEFAULT_AWQ_SEQUENTIAL_ONLOADING
    awq_sequential_targets: list[str] | None = None
    awq_sequential_targets_mode: str = DEFAULT_SEQUENTIAL_TARGETS_MODE
    awq_sequential_targets_per_subgraph: int = DEFAULT_AWQ_SEQUENTIAL_TARGETS_PER_SUBGRAPH
    gptq_sequential_onloading: bool = DEFAULT_GPTQ_SEQUENTIAL_ONLOADING
    gptq_sequential_targets: list[str] | None = None
    gptq_sequential_targets_mode: str = DEFAULT_SEQUENTIAL_TARGETS_MODE
    gptq_sequential_targets_per_subgraph: int = DEFAULT_GPTQ_SEQUENTIAL_TARGETS_PER_SUBGRAPH
    trust_remote_code_model: bool = DEFAULT_TRUST_REMOTE_CODE_MODEL
    run_vllm_smoke_test: bool = DEFAULT_RUN_VLLM_SMOKE_TEST
    memory_preflight_mode: str = DEFAULT_MEMORY_PREFLIGHT_MODE
    awq_mappings_path: Path = DEFAULT_AWQ_MAPPINGS_PATH
    dataset_templates_path: Path = DEFAULT_DATASET_TEMPLATES_PATH

    @classmethod
    def from_env(cls) -> "QuantizerConfig":
        """Construye la configuracion leyendo exclusivamente variables de entorno."""

        model_id = os.getenv("QUANTIZER_MODEL_ID", "").strip()
        dataset_name = os.getenv("QUANTIZER_CALIBRATION_DATASET", "").strip()

        missing_vars = []
        if not model_id:
            missing_vars.append("QUANTIZER_MODEL_ID")
        if not dataset_name:
            missing_vars.append("QUANTIZER_CALIBRATION_DATASET")
        if missing_vars:
            missing_text = ", ".join(missing_vars)
            raise ValueError(f"Faltan variables requeridas del cuantizador: {missing_text}")

        quantize_scheme = os.getenv("QUANTIZE_SCHEME", DEFAULT_QUANTIZE_SCHEME).strip().lower()
        if quantize_scheme not in {"awq", "gptq", "both"}:
            raise ValueError("QUANTIZE_SCHEME debe ser 'awq', 'gptq' o 'both'")

        dataset_config_name = os.getenv("QUANTIZER_DATASET_CONFIG_NAME", "").strip() or None
        awq_sequential_targets_mode, awq_sequential_targets = _parse_sequential_targets(
            os.getenv("QUANTIZER_AWQ_SEQUENTIAL_TARGETS"),
            "QUANTIZER_AWQ_SEQUENTIAL_TARGETS",
        )
        gptq_sequential_targets_mode, gptq_sequential_targets = _parse_sequential_targets(
            os.getenv("QUANTIZER_GPTQ_SEQUENTIAL_TARGETS"),
            "QUANTIZER_GPTQ_SEQUENTIAL_TARGETS",
        )

        return cls(
            model_id=model_id,
            calibration_dataset=dataset_name,
            calibration_split=os.getenv("QUANTIZER_CALIBRATION_SPLIT", DEFAULT_CALIBRATION_SPLIT).strip() or DEFAULT_CALIBRATION_SPLIT,
            dataset_config_name=dataset_config_name,
            awq_num_calibration_samples=int(
                os.getenv(
                    "QUANTIZER_AWQ_NUM_CALIBRATION_SAMPLES",
                    str(DEFAULT_AWQ_NUM_CALIBRATION_SAMPLES),
                )
            ),
            gptq_num_calibration_samples=int(
                os.getenv(
                    "QUANTIZER_GPTQ_NUM_CALIBRATION_SAMPLES",
                    str(DEFAULT_GPTQ_NUM_CALIBRATION_SAMPLES),
                )
            ),
            max_sequence_length=int(os.getenv("QUANTIZER_MAX_SEQUENCE_LENGTH", str(DEFAULT_MAX_SEQUENCE_LENGTH))),
            quantize_scheme=quantize_scheme,
            output_dir=Path(os.getenv("QUANTIZER_OUTPUT_DIR", DEFAULT_OUTPUT_DIR)).expanduser(),
            awq_max_gpu_memory_gib=float(
                os.getenv(
                    "QUANTIZER_AWQ_MAX_GPU_MEMORY_GIB",
                    str(DEFAULT_AWQ_MAX_GPU_MEMORY_GIB),
                )
            ),
            gptq_max_gpu_memory_gib=float(
                os.getenv(
                    "QUANTIZER_GPTQ_MAX_GPU_MEMORY_GIB",
                    str(DEFAULT_GPTQ_MAX_GPU_MEMORY_GIB),
                )
            ),
            awq_sequential_onloading=_parse_bool(
                os.getenv("QUANTIZER_AWQ_SEQUENTIAL_ONLOADING"),
                DEFAULT_AWQ_SEQUENTIAL_ONLOADING,
            ),
            awq_sequential_targets=awq_sequential_targets,
            awq_sequential_targets_mode=awq_sequential_targets_mode,
            awq_sequential_targets_per_subgraph=int(
                os.getenv(
                    "QUANTIZER_AWQ_SEQUENTIAL_TARGETS_PER_SUBGRAPH",
                    str(DEFAULT_AWQ_SEQUENTIAL_TARGETS_PER_SUBGRAPH),
                )
            ),
            gptq_sequential_onloading=_parse_bool(
                os.getenv("QUANTIZER_GPTQ_SEQUENTIAL_ONLOADING"),
                DEFAULT_GPTQ_SEQUENTIAL_ONLOADING,
            ),
            gptq_sequential_targets=gptq_sequential_targets,
            gptq_sequential_targets_mode=gptq_sequential_targets_mode,
            gptq_sequential_targets_per_subgraph=int(
                os.getenv(
                    "QUANTIZER_GPTQ_SEQUENTIAL_TARGETS_PER_SUBGRAPH",
                    str(DEFAULT_GPTQ_SEQUENTIAL_TARGETS_PER_SUBGRAPH),
                )
            ),
            trust_remote_code_model=_parse_bool(
                os.getenv("QUANTIZER_TRUST_REMOTE_CODE_MODEL"),
                DEFAULT_TRUST_REMOTE_CODE_MODEL,
            ),
            run_vllm_smoke_test=_parse_bool(
                os.getenv("QUANTIZER_RUN_VLLM_SMOKE_TEST"),
                DEFAULT_RUN_VLLM_SMOKE_TEST,
            ),
            memory_preflight_mode=_parse_memory_preflight_mode(os.getenv("QUANTIZER_MEMORY_PREFLIGHT_MODE")),
            awq_mappings_path=_parse_optional_path(
                os.getenv("QUANTIZER_AWQ_MAPPINGS_PATH"),
                DEFAULT_AWQ_MAPPINGS_PATH,
            ),
            dataset_templates_path=_parse_optional_path(
                os.getenv("QUANTIZER_DATASET_TEMPLATES_PATH"),
                DEFAULT_DATASET_TEMPLATES_PATH,
            ),
        )

    def model_slug(self) -> str:
        """Deriva un slug estable a partir de la ultima parte del `model_id`."""

        tail = self.model_id.rstrip("/").split("/")[-1]
        normalized = re.sub(r"[^A-Za-z0-9._-]+", "-", tail).strip("-")
        return normalized or "modelo"

    def output_dir_for(self, scheme: str) -> Path:
        """Retorna el directorio objetivo para el esquema solicitado."""

        normalized_scheme = scheme.strip().lower()
        if normalized_scheme not in {"awq", "gptq"}:
            raise ValueError("El esquema debe ser 'awq' o 'gptq'")
        suffix = f"{self.model_slug()}-W4A16-{normalized_scheme.upper()}"
        return self.output_dir / suffix

    def effective_calibration_sample_count(
        self,
        scheme: str,
        available_samples: int | None = None,
    ) -> int:
        """Calcula cuantas muestras usar efectivamente para el esquema indicado."""

        normalized_scheme = scheme.strip().lower()
        if normalized_scheme not in {"awq", "gptq"}:
            raise ValueError("El esquema debe ser 'awq' o 'gptq'")

        if normalized_scheme == "awq":
            target_samples = self.awq_num_calibration_samples
        else:
            target_samples = self.gptq_num_calibration_samples
        if available_samples is not None:
            target_samples = min(target_samples, available_samples)

        return max(1, target_samples)

    def effective_max_gpu_memory_gib(self, scheme: str) -> float:
        """Calcula el budget efectivo de VRAM para el esquema indicado."""

        normalized_scheme = scheme.strip().lower()
        if normalized_scheme not in {"awq", "gptq"}:
            raise ValueError("El esquema debe ser 'awq' o 'gptq'")
        if normalized_scheme == "gptq":
            return self.gptq_max_gpu_memory_gib
        return self.awq_max_gpu_memory_gib

    def sequential_onloading_for(self, scheme: str) -> bool:
        """Retorna si el esquema indicado usa pipeline secuencial con offload."""

        normalized_scheme = scheme.strip().lower()
        if normalized_scheme == "awq":
            return self.awq_sequential_onloading
        if normalized_scheme == "gptq":
            return self.gptq_sequential_onloading
        raise ValueError("El esquema debe ser 'awq' o 'gptq'")

    def sequential_targets_mode_for(self, scheme: str) -> str:
        """Retorna el modo de targets secuenciales para el esquema indicado."""

        normalized_scheme = scheme.strip().lower()
        if normalized_scheme == "awq":
            return self.awq_sequential_targets_mode
        if normalized_scheme == "gptq":
            return self.gptq_sequential_targets_mode
        raise ValueError("El esquema debe ser 'awq' o 'gptq'")

    def sequential_targets_for(self, scheme: str) -> list[str] | None:
        """Retorna los targets explicitamente configurados para el esquema."""

        normalized_scheme = scheme.strip().lower()
        if normalized_scheme == "awq":
            return self.awq_sequential_targets
        if normalized_scheme == "gptq":
            return self.gptq_sequential_targets
        raise ValueError("El esquema debe ser 'awq' o 'gptq'")

    def sequential_targets_per_subgraph_for(self, scheme: str) -> int:
        """Retorna cuántos targets deben agruparse por subgrafo en el esquema."""

        normalized_scheme = scheme.strip().lower()
        if normalized_scheme == "awq":
            return self.awq_sequential_targets_per_subgraph
        if normalized_scheme == "gptq":
            return self.gptq_sequential_targets_per_subgraph
        raise ValueError("El esquema debe ser 'awq' o 'gptq'")

    def resolved_sequential_targets(self, scheme: str) -> list[str] | None:
        """Retorna solo los targets explicitamente configurados para el esquema."""

        if not self.sequential_onloading_for(scheme):
            return None
        return self.sequential_targets_for(scheme)
