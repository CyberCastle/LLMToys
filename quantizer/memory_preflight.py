#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Preflight conservador de memoria para cuantizacion AWQ y GPTQ."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import os
import platform
import shutil
import subprocess
from typing import Any, Literal

import psutil

from .config import QuantizerConfig

try:
    import resource
except ImportError:
    resource = None

BYTES_PER_GIB = 1024**3
BYTES_PER_MIB = 1024**2
DEFAULT_MODEL_LOAD_DTYPE = "bf16"
DEFAULT_RAM_SAFE_UTILIZATION = 0.77
DEFAULT_RAM_HARD_UTILIZATION = 0.90
DEFAULT_VRAM_SAFE_UTILIZATION = 0.90
DEFAULT_VRAM_HARD_UTILIZATION = 1.00
DEFAULT_SUBPROCESS_OVERHEAD_BYTES = 256 * BYTES_PER_MIB

MemoryRisk = Literal["safe", "risky", "inviable"]

RISK_ORDER: dict[MemoryRisk, int] = {
    "safe": 0,
    "risky": 1,
    "inviable": 2,
}

DTYPE_BYTES: dict[str, int] = {
    "bf16": 2,
    "bfloat16": 2,
    "fp16": 2,
    "float16": 2,
    "fp32": 4,
    "float32": 4,
}


@dataclass(frozen=True, slots=True)
class SchemeHeuristics:
    """Resume factores conservadores usados para estimar picos de memoria."""

    activation_expansion_factor: float
    cpu_activation_share: float
    gpu_activation_share: float
    ram_buffer_factor: float
    vram_buffer_factor: float
    dataset_overhead_factor: float
    shmem_factor: float
    cpu_weight_copy_factor: float
    gpu_weight_peak_factor: float
    cpu_staging_factor: float
    offload_factor: float
    vram_offload_factor: float


@dataclass(frozen=True, slots=True)
class ProcessMemoryConsumer:
    """Resume un proceso relevante para diagnosticos de presion de RAM."""

    pid: int
    name: str
    rss_bytes: int
    command_preview: str | None = None


@dataclass(slots=True)
class SystemMemorySnapshot:
    """Fotografia de la memoria RAM relevante para el proceso actual."""

    system_available_bytes: int
    system_total_bytes: int | None
    system_shared_bytes: int | None
    cgroup_available_bytes: int | None
    cgroup_limit_bytes: int | None
    cgroup_current_bytes: int | None
    cgroup_version: str | None
    cgroup_path: str | None
    cgroup_shmem_bytes: int | None
    rlimit_available_bytes: int | None
    process_rss_bytes: int
    process_vms_bytes: int
    dev_shm_available_bytes: int | None
    dev_shm_total_bytes: int | None
    top_ram_processes: list[ProcessMemoryConsumer] = field(default_factory=list)

    @property
    def effective_available_bytes(self) -> int:
        """Retorna la RAM realmente utilizable segun sistema y limites efectivos."""

        candidates = [self.system_available_bytes]
        if self.cgroup_available_bytes is not None:
            candidates.append(self.cgroup_available_bytes)
        if self.rlimit_available_bytes is not None:
            candidates.append(self.rlimit_available_bytes)
        return max(0, min(value for value in candidates if value >= 0))


@dataclass(slots=True)
class GpuMemorySnapshot:
    """Fotografia de la VRAM libre y del budget configurado para la corrida."""

    cuda_available: bool
    available_bytes: int | None
    total_bytes: int | None
    configured_budget_bytes: int | None
    source: str

    @property
    def effective_available_bytes(self) -> int:
        """Retorna la VRAM utilizable considerando disponibilidad real y budget."""

        if not self.cuda_available:
            return 0

        candidates: list[int] = []
        if self.available_bytes is not None and self.available_bytes >= 0:
            candidates.append(self.available_bytes)
        if self.configured_budget_bytes is not None and self.configured_budget_bytes >= 0:
            candidates.append(self.configured_budget_bytes)
        return max(0, min(candidates)) if candidates else 0


@dataclass(slots=True)
class ModelMemoryProfile:
    """Describe el tamaño del modelo sin materializar sus pesos en RAM/VRAM."""

    model_type: str | None
    hidden_size: int
    intermediate_size: int | None
    num_hidden_layers: int | None
    vocab_size: int | None
    parameter_count: int
    buffer_count: int
    dtype: str
    estimate_source: str

    @property
    def model_bytes(self) -> int:
        """Retorna el tamaño de pesos y buffers para el dtype indicado."""

        return (self.parameter_count + self.buffer_count) * _dtype_bytes(self.dtype)


@dataclass(slots=True)
class MemoryBreakdown:
    """Desglose de una estimacion de pico de memoria por categoria."""

    model_bytes: int
    activation_bytes: int
    internal_buffers_bytes: int
    dataset_bytes: int
    shared_memory_bytes: int
    offload_bytes: int
    subprocess_bytes: int
    base_process_bytes: int

    @property
    def total_bytes(self) -> int:
        """Retorna el total agregado de la estimacion."""

        return (
            self.model_bytes
            + self.activation_bytes
            + self.internal_buffers_bytes
            + self.dataset_bytes
            + self.shared_memory_bytes
            + self.offload_bytes
            + self.subprocess_bytes
            + self.base_process_bytes
        )


@dataclass(slots=True)
class MemoryPreflightResult:
    """Resultado final del preflight de memoria para un esquema concreto."""

    scheme: str
    status: MemoryRisk
    likely_failure: str | None
    sample_count: int
    max_sequence_length: int
    model_profile: ModelMemoryProfile
    ram_snapshot: SystemMemorySnapshot
    vram_snapshot: GpuMemorySnapshot
    ram_estimate: MemoryBreakdown
    vram_estimate: MemoryBreakdown
    risk_factors: list[str]
    notes: list[str]
    suggested_sample_count: int | None
    suggested_sequence_length: int | None


def _dtype_bytes(dtype: str) -> int:
    """Mapea un dtype textual al numero de bytes por elemento."""

    normalized = dtype.strip().lower()
    if normalized not in DTYPE_BYTES:
        raise ValueError(f"dtype no soportado para el preflight: {dtype!r}")
    return DTYPE_BYTES[normalized]


def _format_gib(raw_bytes: int | None) -> str:
    """Convierte bytes a GiB con dos decimales para diagnosticos legibles."""

    if raw_bytes is None:
        return "n/d"
    return f"{raw_bytes / BYTES_PER_GIB:.2f} GiB"


def _round_down(value: int, step: int) -> int:
    """Redondea hacia abajo manteniendo un minimo de un paso."""

    if value <= 0:
        return 0
    return max(step, (value // step) * step)


def _worst_status(*statuses: MemoryRisk) -> MemoryRisk:
    """Retorna el riesgo mas severo de un conjunto de estados."""

    return max(statuses, key=RISK_ORDER.__getitem__)


def _is_linux() -> bool:
    """Indica si el proceso corre sobre Linux para activar extras del kernel."""

    return platform.system() == "Linux"


def _read_process_memory() -> tuple[int, int]:
    """Retorna RSS y VMS del proceso actual usando `psutil`."""

    process = psutil.Process()
    memory_info = process.memory_info()
    rss_bytes = int(getattr(memory_info, "rss", 0) or 0)
    vms_bytes = int(getattr(memory_info, "vms", 0) or 0)
    return rss_bytes, vms_bytes


def _safe_read_text(path: Path) -> str | None:
    """Lee un archivo de texto sin propagar errores irrelevantes al preflight."""

    try:
        return path.read_text(encoding="utf-8").strip()
    except Exception:
        return None


def _build_command_preview(raw_value: Any, *, max_length: int = 96) -> str | None:
    """Normaliza un comando para mostrarlo de forma compacta en el reporte."""

    if isinstance(raw_value, str):
        normalized = raw_value.strip()
    elif isinstance(raw_value, (list, tuple)):
        normalized_parts = [str(item).strip() for item in raw_value if str(item).strip()]
        normalized = " ".join(normalized_parts)
    else:
        normalized = ""

    if not normalized:
        return None
    if len(normalized) <= max_length:
        return normalized
    return normalized[: max_length - 3].rstrip() + "..."


def _collect_top_ram_processes(limit: int = 5) -> list[ProcessMemoryConsumer]:
    """Retorna los procesos externos con mayor RSS para diagnosticar presion de RAM."""

    current_pid = os.getpid()
    consumers: list[ProcessMemoryConsumer] = []
    for process in psutil.process_iter(["pid", "name", "cmdline", "memory_info"]):
        try:
            process_info = process.info
            pid = int(process_info.get("pid") or 0)
            if pid <= 0 or pid == current_pid:
                continue

            memory_info = process_info.get("memory_info")
            rss_bytes = int(getattr(memory_info, "rss", 0) or 0)
            if rss_bytes <= 0:
                continue

            name = str(process_info.get("name") or "desconocido").strip() or "desconocido"
            consumers.append(
                ProcessMemoryConsumer(
                    pid=pid,
                    name=name,
                    rss_bytes=rss_bytes,
                    command_preview=_build_command_preview(process_info.get("cmdline")),
                )
            )
        except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess, TypeError, ValueError):
            continue

    consumers.sort(key=lambda consumer: consumer.rss_bytes, reverse=True)
    return consumers[:limit]


def _parse_cgroup_limit(raw_value: str | None) -> int | None:
    """Normaliza el valor de un limite de memoria de cgroup."""

    if raw_value is None or not raw_value or raw_value == "max":
        return None
    try:
        parsed = int(raw_value)
    except ValueError:
        return None
    if parsed <= 0 or parsed >= 1 << 60:
        return None
    return parsed


def _resolve_linux_cgroup_memory_files() -> tuple[str | None, str | None, Path | None, Path | None, Path | None]:
    """Ubica los archivos de memoria del cgroup actual en Linux v1 o v2."""

    if not _is_linux():
        return None, None, None, None, None

    cgroup_path = Path("/proc/self/cgroup")
    if not cgroup_path.exists():
        return None, None, None, None, None

    for raw_line in cgroup_path.read_text(encoding="utf-8").splitlines():
        try:
            _hierarchy, controllers, relative_path = raw_line.split(":", 2)
        except ValueError:
            continue
        if controllers:
            continue
        base = Path("/sys/fs/cgroup") / relative_path.lstrip("/")
        max_path = base / "memory.max"
        current_path = base / "memory.current"
        stat_path = base / "memory.stat"
        if max_path.exists() and current_path.exists():
            return "v2", relative_path, max_path, current_path, stat_path

    for raw_line in cgroup_path.read_text(encoding="utf-8").splitlines():
        try:
            _hierarchy, controllers, relative_path = raw_line.split(":", 2)
        except ValueError:
            continue
        controller_set = {item.strip() for item in controllers.split(",") if item.strip()}
        if "memory" not in controller_set:
            continue
        base = Path("/sys/fs/cgroup/memory") / relative_path.lstrip("/")
        limit_path = base / "memory.limit_in_bytes"
        current_path = base / "memory.usage_in_bytes"
        stat_path = base / "memory.stat"
        if limit_path.exists() and current_path.exists():
            return "v1", relative_path, limit_path, current_path, stat_path

    return None, None, None, None, None


def _read_linux_cgroup_memory_snapshot(
    system_total_bytes: int | None,
) -> tuple[str | None, str | None, int | None, int | None, int | None, int | None]:
    """Retorna limite, uso y shmem del cgroup actual cuando existe en Linux."""

    version, relative_path, limit_path, current_path, stat_path = _resolve_linux_cgroup_memory_files()
    if limit_path is None or current_path is None:
        return version, relative_path, None, None, None, None

    limit_bytes = _parse_cgroup_limit(_safe_read_text(limit_path))
    current_text = _safe_read_text(current_path)
    try:
        current_bytes = int(current_text) if current_text else None
    except ValueError:
        current_bytes = None

    if limit_bytes is not None and system_total_bytes is not None and limit_bytes >= system_total_bytes * 4:
        limit_bytes = None

    available_bytes = None
    if limit_bytes is not None and current_bytes is not None:
        available_bytes = max(0, limit_bytes - current_bytes)

    shmem_bytes = None
    if stat_path is not None and stat_path.exists():
        for raw_line in stat_path.read_text(encoding="utf-8").splitlines():
            if raw_line.startswith("shmem "):
                try:
                    shmem_bytes = int(raw_line.split()[1])
                except ValueError:
                    shmem_bytes = None
                break

    return version, relative_path, limit_bytes, current_bytes, available_bytes, shmem_bytes


def _read_rlimit_available_bytes(process_vms_bytes: int) -> int | None:
    """Calcula la memoria virtual restante segun `RLIMIT_AS` si aplica."""

    if resource is None:
        return None

    rlimit_as = getattr(resource, "RLIMIT_AS", None)
    rlimit_infinity = getattr(resource, "RLIM_INFINITY", None)
    if rlimit_as is None:
        return None

    try:
        soft_limit, _hard_limit = resource.getrlimit(rlimit_as)
    except (AttributeError, OSError, ValueError):
        return None
    if soft_limit == -1 or (rlimit_infinity is not None and soft_limit == rlimit_infinity) or soft_limit <= 0:
        return None
    return max(0, int(soft_limit) - process_vms_bytes)


def _detect_shared_memory_filesystem() -> tuple[int | None, int | None]:
    """Mide el filesystem de memoria compartida cuando la plataforma lo expone."""

    if not _is_linux():
        return None, None

    dev_shm_path = Path("/dev/shm")
    if not dev_shm_path.exists():
        return None, None

    usage = psutil.disk_usage(str(dev_shm_path))
    return int(usage.free), int(usage.total)


def _detect_system_memory_snapshot() -> SystemMemorySnapshot:
    """Construye la fotografia de RAM efectiva para el proceso actual."""

    memory = psutil.virtual_memory()
    system_total_bytes = int(getattr(memory, "total", 0) or 0) or None
    system_available_bytes = int(getattr(memory, "available", 0) or 0)
    system_shared_bytes = int(getattr(memory, "shared", 0) or 0) or None

    if system_available_bytes <= 0:
        raise RuntimeError("No fue posible medir la RAM disponible del sistema para el preflight")

    process_rss_bytes, process_vms_bytes = _read_process_memory()
    cgroup_version, cgroup_path, cgroup_limit_bytes, cgroup_current_bytes, cgroup_available_bytes, cgroup_shmem_bytes = (
        _read_linux_cgroup_memory_snapshot(system_total_bytes)
    )
    rlimit_available_bytes = _read_rlimit_available_bytes(process_vms_bytes)
    dev_shm_available_bytes, dev_shm_total_bytes = _detect_shared_memory_filesystem()

    return SystemMemorySnapshot(
        system_available_bytes=system_available_bytes,
        system_total_bytes=system_total_bytes,
        system_shared_bytes=system_shared_bytes,
        cgroup_available_bytes=cgroup_available_bytes,
        cgroup_limit_bytes=cgroup_limit_bytes,
        cgroup_current_bytes=cgroup_current_bytes,
        cgroup_version=cgroup_version,
        cgroup_path=cgroup_path,
        cgroup_shmem_bytes=cgroup_shmem_bytes,
        rlimit_available_bytes=rlimit_available_bytes,
        process_rss_bytes=process_rss_bytes,
        process_vms_bytes=process_vms_bytes,
        dev_shm_available_bytes=dev_shm_available_bytes,
        dev_shm_total_bytes=dev_shm_total_bytes,
        top_ram_processes=_collect_top_ram_processes(),
    )


def _detect_gpu_memory_snapshot(configured_budget_bytes: int) -> GpuMemorySnapshot:
    """Retorna la VRAM libre real a partir de CUDA o `nvidia-smi`."""

    try:
        import torch

        if torch.cuda.is_available():
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            return GpuMemorySnapshot(
                cuda_available=True,
                available_bytes=int(free_bytes),
                total_bytes=int(total_bytes),
                configured_budget_bytes=configured_budget_bytes,
                source="torch.cuda.mem_get_info",
            )
    except Exception:
        pass

    if shutil.which("nvidia-smi"):
        command = [
            "nvidia-smi",
            "--query-gpu=memory.free,memory.total",
            "--format=csv,noheader,nounits",
        ]
        visible_devices = os.getenv("CUDA_VISIBLE_DEVICES", "").strip()
        if visible_devices:
            first_gpu = visible_devices.split(",", 1)[0].strip()
            if first_gpu and first_gpu.isdigit():
                command.extend(["-i", first_gpu])
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode == 0 and completed.stdout.strip():
            first_line = completed.stdout.strip().splitlines()[0]
            fields = [item.strip() for item in first_line.split(",")]
            if len(fields) >= 2:
                try:
                    free_mib = int(fields[0])
                    total_mib = int(fields[1])
                    return GpuMemorySnapshot(
                        cuda_available=True,
                        available_bytes=free_mib * BYTES_PER_MIB,
                        total_bytes=total_mib * BYTES_PER_MIB,
                        configured_budget_bytes=configured_budget_bytes,
                        source="nvidia-smi",
                    )
                except ValueError:
                    pass

    return GpuMemorySnapshot(
        cuda_available=False,
        available_bytes=None,
        total_bytes=None,
        configured_budget_bytes=configured_budget_bytes,
        source="unavailable",
    )


def _get_config_int(config_obj: Any, *field_names: str) -> int | None:
    """Busca el primer entero util entre varios nombres de campo de config."""

    for field_name in field_names:
        raw_value = getattr(config_obj, field_name, None)
        if raw_value is None:
            continue
        try:
            parsed = int(raw_value)
        except (TypeError, ValueError):
            continue
        if parsed > 0:
            return parsed
    return None


def _count_model_tensors_from_meta_model(model_config: Any, *, trust_remote_code_model: bool) -> tuple[int, int]:
    """Cuenta parametros y buffers construyendo el modelo sobre `meta`."""

    from accelerate import init_empty_weights
    from transformers import AutoModelForCausalLM

    with init_empty_weights():
        try:
            model = AutoModelForCausalLM.from_config(
                model_config,
                trust_remote_code=trust_remote_code_model,
            )
        except TypeError:
            model = AutoModelForCausalLM.from_config(model_config)

    try:
        parameter_count = sum(int(parameter.numel()) for parameter in model.parameters())
        buffer_count = sum(int(buffer.numel()) for buffer in model.buffers())
    finally:
        del model
    return parameter_count, buffer_count


def _estimate_parameter_count_from_config(model_config: Any) -> int:
    """Aproxima el numero de parametros usando dimensiones tipicas de decoder-only LLM."""

    hidden_size = _get_config_int(model_config, "hidden_size", "d_model", "n_embd")
    intermediate_size = _get_config_int(model_config, "intermediate_size", "ffn_dim", "n_inner")
    num_hidden_layers = _get_config_int(model_config, "num_hidden_layers", "n_layer", "num_layers")
    vocab_size = _get_config_int(model_config, "vocab_size")
    num_attention_heads = _get_config_int(model_config, "num_attention_heads", "n_head")
    num_key_value_heads = _get_config_int(model_config, "num_key_value_heads", "n_head_kv")

    if hidden_size is None or num_hidden_layers is None:
        raise RuntimeError("No fue posible inferir hidden_size y num_hidden_layers para el preflight")

    if intermediate_size is None:
        intermediate_size = hidden_size * 4

    if num_attention_heads and num_key_value_heads:
        head_dim = max(1, hidden_size // num_attention_heads)
        attention_parameters = hidden_size * hidden_size + (hidden_size * head_dim * num_key_value_heads * 2) + hidden_size * hidden_size
    else:
        attention_parameters = 4 * hidden_size * hidden_size

    mlp_parameters = 3 * hidden_size * intermediate_size
    layer_norm_parameters = 4 * hidden_size
    embedding_parameters = (vocab_size or 0) * hidden_size
    tie_word_embeddings = bool(getattr(model_config, "tie_word_embeddings", True))
    lm_head_parameters = 0 if tie_word_embeddings else embedding_parameters

    return embedding_parameters + (num_hidden_layers * (attention_parameters + mlp_parameters + layer_norm_parameters)) + lm_head_parameters


def _load_model_memory_profile(config: QuantizerConfig, *, dtype: str) -> ModelMemoryProfile:
    """Resuelve un perfil de memoria del modelo sin cargar sus pesos reales."""

    try:
        from transformers import AutoConfig
    except ImportError as exc:
        raise RuntimeError("El preflight de memoria requiere transformers instalado") from exc

    model_config = AutoConfig.from_pretrained(
        config.model_id,
        trust_remote_code=config.trust_remote_code_model,
    )
    hidden_size = _get_config_int(model_config, "hidden_size", "d_model", "n_embd")
    if hidden_size is None:
        raise RuntimeError(f"No fue posible inferir hidden_size para '{config.model_id}'")

    try:
        parameter_count, buffer_count = _count_model_tensors_from_meta_model(
            model_config,
            trust_remote_code_model=config.trust_remote_code_model,
        )
        estimate_source = "meta_model"
    except Exception:
        parameter_count = _estimate_parameter_count_from_config(model_config)
        buffer_count = hidden_size * 8
        estimate_source = "config_formula"

    return ModelMemoryProfile(
        model_type=str(getattr(model_config, "model_type", "")).strip().lower() or None,
        hidden_size=hidden_size,
        intermediate_size=_get_config_int(model_config, "intermediate_size", "ffn_dim", "n_inner"),
        num_hidden_layers=_get_config_int(model_config, "num_hidden_layers", "n_layer", "num_layers"),
        vocab_size=_get_config_int(model_config, "vocab_size"),
        parameter_count=parameter_count,
        buffer_count=buffer_count,
        dtype=dtype,
        estimate_source=estimate_source,
    )


def _resolve_scheme_heuristics(scheme: str, *, sequential_onloading: bool) -> SchemeHeuristics:
    """Retorna factores conservadores segun esquema y modo de offload."""

    normalized_scheme = scheme.strip().lower()
    if normalized_scheme == "awq":
        return SchemeHeuristics(
            activation_expansion_factor=2.8 if sequential_onloading else 3.2,
            cpu_activation_share=0.50 if sequential_onloading else 0.28,
            gpu_activation_share=0.12 if sequential_onloading else 0.78,
            ram_buffer_factor=0.22 if sequential_onloading else 0.26,
            vram_buffer_factor=0.06 if sequential_onloading else 0.15,
            dataset_overhead_factor=2.0,
            shmem_factor=0.32,
            cpu_weight_copy_factor=1.15 if sequential_onloading else 1.20,
            gpu_weight_peak_factor=0.81 if sequential_onloading else 1.08,
            cpu_staging_factor=0.24 if sequential_onloading else 0.30,
            offload_factor=0.15 if sequential_onloading else 0.06,
            vram_offload_factor=0.01 if sequential_onloading else 0.04,
        )
    if normalized_scheme == "gptq":
        return SchemeHeuristics(
            activation_expansion_factor=4.0 if sequential_onloading else 4.8,
            cpu_activation_share=0.72 if sequential_onloading else 0.60,
            gpu_activation_share=0.10 if sequential_onloading else 0.48,
            ram_buffer_factor=0.28 if sequential_onloading else 0.34,
            vram_buffer_factor=0.05 if sequential_onloading else 0.12,
            dataset_overhead_factor=2.6,
            shmem_factor=0.55,
            cpu_weight_copy_factor=1.22 if sequential_onloading else 1.35,
            gpu_weight_peak_factor=0.76 if sequential_onloading else 1.05,
            cpu_staging_factor=0.26 if sequential_onloading else 0.38,
            offload_factor=0.22 if sequential_onloading else 0.30,
            vram_offload_factor=0.01 if sequential_onloading else 0.03,
        )
    raise ValueError("El esquema debe ser 'awq' o 'gptq'")


def _estimate_memory_breakdowns(
    *,
    config: QuantizerConfig,
    scheme: str,
    model_profile: ModelMemoryProfile,
    ram_snapshot: SystemMemorySnapshot,
    vram_snapshot: GpuMemorySnapshot,
    launched_via_subprocess: bool,
) -> tuple[MemoryBreakdown, MemoryBreakdown]:
    """Estima el pico de RAM y VRAM para una corrida concreta de cuantizacion."""

    normalized_scheme = scheme.strip().lower()
    sample_count = config.effective_calibration_sample_count(normalized_scheme)
    sequence_length = config.max_sequence_length
    sequential_onloading = config.sequential_onloading_for(normalized_scheme)
    heuristics = _resolve_scheme_heuristics(normalized_scheme, sequential_onloading=sequential_onloading)

    model_bytes = model_profile.model_bytes
    dtype_bytes = _dtype_bytes(model_profile.dtype)
    activation_core_bytes = sample_count * sequence_length * model_profile.hidden_size * dtype_bytes
    activation_peak_bytes = int(activation_core_bytes * heuristics.activation_expansion_factor)

    candidate_count = max(sample_count * 4, sample_count)
    dataset_raw_bytes = candidate_count * sequence_length * 2 * 8
    dataset_bytes = int(dataset_raw_bytes * heuristics.dataset_overhead_factor)
    shared_memory_bytes = int((dataset_bytes + activation_core_bytes) * heuristics.shmem_factor)

    gpu_limit_bytes = vram_snapshot.effective_available_bytes
    gpu_weight_bytes = min(model_bytes, gpu_limit_bytes)
    cpu_weight_bytes = max(0, model_bytes - gpu_weight_bytes)

    ram_model_bytes = int((cpu_weight_bytes * heuristics.cpu_weight_copy_factor) + (gpu_weight_bytes * heuristics.cpu_staging_factor))
    vram_model_bytes = int(gpu_weight_bytes * heuristics.gpu_weight_peak_factor)

    ram_activation_bytes = int(activation_peak_bytes * heuristics.cpu_activation_share)
    vram_activation_bytes = int(activation_peak_bytes * heuristics.gpu_activation_share)

    ram_internal_buffers_bytes = int(max(model_bytes * heuristics.ram_buffer_factor, activation_peak_bytes * 0.20))
    vram_internal_buffers_bytes = int(max(vram_model_bytes * heuristics.vram_buffer_factor, vram_activation_bytes * 0.20))

    ram_offload_bytes = 0
    if sequential_onloading or cpu_weight_bytes > 0 or normalized_scheme == "gptq":
        ram_offload_bytes = int(max(cpu_weight_bytes, model_bytes * heuristics.offload_factor))
    vram_offload_bytes = int(vram_model_bytes * heuristics.vram_offload_factor)

    subprocess_bytes = 0
    if launched_via_subprocess:
        subprocess_bytes = max(DEFAULT_SUBPROCESS_OVERHEAD_BYTES, int(ram_snapshot.process_rss_bytes * 0.50))

    ram_estimate = MemoryBreakdown(
        model_bytes=ram_model_bytes,
        activation_bytes=ram_activation_bytes,
        internal_buffers_bytes=ram_internal_buffers_bytes,
        dataset_bytes=dataset_bytes,
        shared_memory_bytes=shared_memory_bytes,
        offload_bytes=ram_offload_bytes,
        subprocess_bytes=subprocess_bytes,
        base_process_bytes=ram_snapshot.process_rss_bytes,
    )
    vram_estimate = MemoryBreakdown(
        model_bytes=vram_model_bytes,
        activation_bytes=vram_activation_bytes,
        internal_buffers_bytes=vram_internal_buffers_bytes,
        dataset_bytes=0,
        shared_memory_bytes=0,
        offload_bytes=vram_offload_bytes,
        subprocess_bytes=0,
        base_process_bytes=0,
    )
    return ram_estimate, vram_estimate


def _classify_resource_usage(total_bytes: int, available_bytes: int, *, safe_ratio: float, hard_ratio: float) -> MemoryRisk:
    """Clasifica un recurso comparando estimacion frente a disponibilidad efectiva."""

    if available_bytes <= 0:
        return "inviable"

    safe_budget = int(available_bytes * safe_ratio)
    hard_budget = int(available_bytes * hard_ratio)
    if total_bytes <= safe_budget:
        return "safe"
    if total_bytes <= hard_budget:
        return "risky"
    return "inviable"


def _classify_likely_failure(
    *,
    status: MemoryRisk,
    ram_status: MemoryRisk,
    vram_status: MemoryRisk,
    ram_estimate: MemoryBreakdown,
    vram_estimate: MemoryBreakdown,
    ram_snapshot: SystemMemorySnapshot,
    vram_snapshot: GpuMemorySnapshot,
) -> str | None:
    """Resume si el fallo probable seria por RAM del sistema o por VRAM."""

    if status == "safe":
        return None
    if not vram_snapshot.cuda_available:
        return "cuda_unavailable"
    if ram_status == "inviable":
        return "ram_oom_killer"
    if vram_status == "inviable" and ram_status == "safe":
        return "cuda_oom"
    if ram_status == "risky" and vram_status == "safe":
        return "ram_oom_killer"

    ram_hard_budget = max(1, int(ram_snapshot.effective_available_bytes * DEFAULT_RAM_HARD_UTILIZATION))
    vram_hard_budget = max(1, int(vram_snapshot.effective_available_bytes * DEFAULT_VRAM_HARD_UTILIZATION))
    ram_pressure = ram_estimate.total_bytes / ram_hard_budget
    vram_pressure = vram_estimate.total_bytes / vram_hard_budget
    if ram_pressure >= (vram_pressure * 0.85):
        return "ram_oom_killer"
    return "cuda_oom"


def _compute_suggestions(
    *,
    config: QuantizerConfig,
    scheme: str,
    model_profile: ModelMemoryProfile,
    ram_estimate: MemoryBreakdown,
    vram_estimate: MemoryBreakdown,
    ram_snapshot: SystemMemorySnapshot,
    vram_snapshot: GpuMemorySnapshot,
) -> tuple[int | None, int | None]:
    """Sugiere un `samples` o `seq_len` mas seguro manteniendo la configuracion base."""

    normalized_scheme = scheme.strip().lower()
    sample_count = config.effective_calibration_sample_count(normalized_scheme)
    sequence_length = config.max_sequence_length
    if sample_count <= 0 or sequence_length <= 0:
        return None, None

    heuristics = _resolve_scheme_heuristics(
        normalized_scheme,
        sequential_onloading=config.sequential_onloading_for(normalized_scheme),
    )
    dtype_bytes = _dtype_bytes(model_profile.dtype)
    activation_unit = model_profile.hidden_size * dtype_bytes
    dataset_unit = 4 * 2 * 8 * heuristics.dataset_overhead_factor
    shmem_unit = (dataset_unit + activation_unit) * heuristics.shmem_factor
    ram_variable_unit = (
        (activation_unit * heuristics.activation_expansion_factor * heuristics.cpu_activation_share)
        + (activation_unit * heuristics.activation_expansion_factor * 0.20)
        + dataset_unit
        + shmem_unit
    )
    vram_variable_unit = (activation_unit * heuristics.activation_expansion_factor * heuristics.gpu_activation_share) + (
        activation_unit * heuristics.activation_expansion_factor * 0.20
    )

    current_product = sample_count * sequence_length
    fixed_ram_bytes = max(0, int(ram_estimate.total_bytes - (ram_variable_unit * current_product)))
    fixed_vram_bytes = max(0, int(vram_estimate.total_bytes - (vram_variable_unit * current_product)))
    ram_safe_budget = int(ram_snapshot.effective_available_bytes * DEFAULT_RAM_SAFE_UTILIZATION)
    vram_safe_budget = int(vram_snapshot.effective_available_bytes * DEFAULT_VRAM_SAFE_UTILIZATION)

    candidate_limits: list[int] = []
    if ram_variable_unit > 0 and ram_safe_budget > fixed_ram_bytes:
        candidate_limits.append(int((ram_safe_budget - fixed_ram_bytes) / ram_variable_unit))
    if vram_variable_unit > 0 and vram_safe_budget > fixed_vram_bytes:
        candidate_limits.append(int((vram_safe_budget - fixed_vram_bytes) / vram_variable_unit))
    if not candidate_limits:
        return 1, _round_down(min(sequence_length, 128), 128)

    max_safe_product = max(1, min(candidate_limits))
    if max_safe_product >= current_product:
        return None, None

    suggested_samples = max(1, max_safe_product // sequence_length)
    suggested_sequence_length = max(128, max_safe_product // sample_count)
    suggested_sequence_length = min(sequence_length, _round_down(suggested_sequence_length, 128))
    return suggested_samples, suggested_sequence_length


def _collect_risk_factors(
    *,
    config: QuantizerConfig,
    scheme: str,
    model_profile: ModelMemoryProfile,
    ram_snapshot: SystemMemorySnapshot,
    vram_snapshot: GpuMemorySnapshot,
    ram_estimate: MemoryBreakdown,
    vram_estimate: MemoryBreakdown,
    launched_via_subprocess: bool,
) -> list[str]:
    """Sintetiza los principales factores de riesgo detectados por el preflight."""

    normalized_scheme = scheme.strip().lower()
    sample_count = config.effective_calibration_sample_count(normalized_scheme)
    risk_factors: list[str] = []

    if not vram_snapshot.cuda_available:
        risk_factors.append("No se detecto una GPU CUDA utilizable para la cuantizacion.")
    if ram_snapshot.cgroup_available_bytes is not None and ram_snapshot.cgroup_available_bytes < ram_snapshot.system_available_bytes:
        risk_factors.append("El cgroup/systemd reduce la RAM efectiva disponible para este proceso.")
    if ram_snapshot.dev_shm_available_bytes is not None and ram_snapshot.dev_shm_available_bytes < max(
        ram_estimate.shared_memory_bytes, 2 * BYTES_PER_GIB
    ):
        risk_factors.append("El filesystem de memoria compartida libre es bajo frente al uso estimado de shmem.")
    if ram_snapshot.system_shared_bytes is not None and ram_snapshot.system_total_bytes is not None:
        if ram_snapshot.system_shared_bytes > int(ram_snapshot.system_total_bytes * 0.15):
            risk_factors.append("El sistema ya muestra presion significativa de memoria compartida.")
    if model_profile.model_bytes > vram_snapshot.effective_available_bytes:
        risk_factors.append("El tamaño del modelo excede la VRAM efectiva y forzara offload a CPU.")
    if sample_count >= 256:
        risk_factors.append("El numero de muestras de calibracion es alto para un preflight conservador.")
    if config.max_sequence_length >= 2048:
        risk_factors.append("La longitud maxima de secuencia aumenta de forma agresiva activaciones y caches temporales.")
    if config.sequential_onloading_for(normalized_scheme):
        risk_factors.append("El pipeline secuencial reduce presion de VRAM pero incrementa copias y offload en RAM.")
    if normalized_scheme == "gptq":
        risk_factors.append("GPTQ suele elevar el pico de RAM por estadisticas internas y buffers de calibracion.")
    if launched_via_subprocess:
        risk_factors.append("El lanzamiento por subproceso agrega duplicacion temporal de memoria del interprete.")
    if ram_estimate.total_bytes > int(ram_snapshot.effective_available_bytes * DEFAULT_RAM_SAFE_UTILIZATION):
        risk_factors.append("La RAM estimada consume mas del presupuesto seguro reservado para evitar OOM killer.")
    if vram_snapshot.cuda_available and vram_estimate.total_bytes > int(
        vram_snapshot.effective_available_bytes * DEFAULT_VRAM_SAFE_UTILIZATION
    ):
        risk_factors.append("La VRAM estimada se acerca demasiado al limite utilizable de la GPU.")

    return risk_factors


def evaluate_quantization_memory_preflight(
    config: QuantizerConfig,
    scheme: str,
    *,
    launched_via_subprocess: bool = False,
    dtype: str = DEFAULT_MODEL_LOAD_DTYPE,
) -> MemoryPreflightResult:
    """Ejecuta el preflight de memoria y clasifica el riesgo de OOM."""

    normalized_scheme = scheme.strip().lower()
    if normalized_scheme not in {"awq", "gptq"}:
        raise ValueError("El esquema debe ser 'awq' o 'gptq'")

    ram_snapshot = _detect_system_memory_snapshot()
    configured_budget_bytes = int(config.effective_max_gpu_memory_gib(normalized_scheme) * BYTES_PER_GIB)
    vram_snapshot = _detect_gpu_memory_snapshot(configured_budget_bytes)
    model_profile = _load_model_memory_profile(config, dtype=dtype)
    ram_estimate, vram_estimate = _estimate_memory_breakdowns(
        config=config,
        scheme=normalized_scheme,
        model_profile=model_profile,
        ram_snapshot=ram_snapshot,
        vram_snapshot=vram_snapshot,
        launched_via_subprocess=launched_via_subprocess,
    )

    ram_status = _classify_resource_usage(
        ram_estimate.total_bytes,
        ram_snapshot.effective_available_bytes,
        safe_ratio=DEFAULT_RAM_SAFE_UTILIZATION,
        hard_ratio=DEFAULT_RAM_HARD_UTILIZATION,
    )
    vram_status = _classify_resource_usage(
        vram_estimate.total_bytes,
        vram_snapshot.effective_available_bytes,
        safe_ratio=DEFAULT_VRAM_SAFE_UTILIZATION,
        hard_ratio=DEFAULT_VRAM_HARD_UTILIZATION,
    )
    if not vram_snapshot.cuda_available:
        vram_status = "inviable"
    status = _worst_status(ram_status, vram_status)
    suggested_sample_count, suggested_sequence_length = _compute_suggestions(
        config=config,
        scheme=normalized_scheme,
        model_profile=model_profile,
        ram_estimate=ram_estimate,
        vram_estimate=vram_estimate,
        ram_snapshot=ram_snapshot,
        vram_snapshot=vram_snapshot,
    )
    likely_failure = _classify_likely_failure(
        status=status,
        ram_status=ram_status,
        vram_status=vram_status,
        ram_estimate=ram_estimate,
        vram_estimate=vram_estimate,
        ram_snapshot=ram_snapshot,
        vram_snapshot=vram_snapshot,
    )
    risk_factors = _collect_risk_factors(
        config=config,
        scheme=normalized_scheme,
        model_profile=model_profile,
        ram_snapshot=ram_snapshot,
        vram_snapshot=vram_snapshot,
        ram_estimate=ram_estimate,
        vram_estimate=vram_estimate,
        launched_via_subprocess=launched_via_subprocess,
    )

    notes = [
        f"dtype de carga estimado: {model_profile.dtype}",
        f"fuente de estimacion del modelo: {model_profile.estimate_source}",
        f"fuente de VRAM: {vram_snapshot.source}",
    ]
    if ram_snapshot.cgroup_version and ram_snapshot.cgroup_path:
        notes.append(f"cgroup detectado: {ram_snapshot.cgroup_version} {ram_snapshot.cgroup_path}")

    return MemoryPreflightResult(
        scheme=normalized_scheme,
        status=status,
        likely_failure=likely_failure,
        sample_count=config.effective_calibration_sample_count(normalized_scheme),
        max_sequence_length=config.max_sequence_length,
        model_profile=model_profile,
        ram_snapshot=ram_snapshot,
        vram_snapshot=vram_snapshot,
        ram_estimate=ram_estimate,
        vram_estimate=vram_estimate,
        risk_factors=risk_factors,
        notes=notes,
        suggested_sample_count=suggested_sample_count,
        suggested_sequence_length=suggested_sequence_length,
    )


def format_memory_preflight_report(result: MemoryPreflightResult) -> str:
    """Genera un reporte textual compacto y accionable para el usuario."""

    ram_available_text = _format_gib(result.ram_snapshot.effective_available_bytes)
    ram_system_text = _format_gib(result.ram_snapshot.system_available_bytes)
    ram_cgroup_text = _format_gib(result.ram_snapshot.cgroup_available_bytes)
    dev_shm_text = _format_gib(result.ram_snapshot.dev_shm_available_bytes)
    vram_available_text = _format_gib(result.vram_snapshot.effective_available_bytes)
    vram_free_text = _format_gib(result.vram_snapshot.available_bytes)
    vram_budget_text = _format_gib(result.vram_snapshot.configured_budget_bytes)

    ram_parts = (
        f"modelo={_format_gib(result.ram_estimate.model_bytes)}",
        f"activaciones={_format_gib(result.ram_estimate.activation_bytes)}",
        f"buffers={_format_gib(result.ram_estimate.internal_buffers_bytes)}",
        f"dataset={_format_gib(result.ram_estimate.dataset_bytes)}",
        f"shmem={_format_gib(result.ram_estimate.shared_memory_bytes)}",
        f"offload={_format_gib(result.ram_estimate.offload_bytes)}",
        f"subprocess={_format_gib(result.ram_estimate.subprocess_bytes)}",
        f"base={_format_gib(result.ram_estimate.base_process_bytes)}",
    )
    vram_parts = (
        f"modelo={_format_gib(result.vram_estimate.model_bytes)}",
        f"activaciones={_format_gib(result.vram_estimate.activation_bytes)}",
        f"buffers={_format_gib(result.vram_estimate.internal_buffers_bytes)}",
        f"offload={_format_gib(result.vram_estimate.offload_bytes)}",
    )

    report_lines = [
        f"[quantizer] Preflight {result.scheme.upper()}: {result.status.upper()}",
        (
            "[quantizer] RAM efectiva: "
            f"{ram_available_text} (sistema={ram_system_text}, cgroup={ram_cgroup_text}, shmem_fs libre={dev_shm_text})"
        ),
        ("[quantizer] RAM estimada: " f"{_format_gib(result.ram_estimate.total_bytes)} ({', '.join(ram_parts)})"),
        (
            "[quantizer] VRAM efectiva: "
            f"{vram_available_text} (libre={vram_free_text}, budget_config={vram_budget_text}, fuente={result.vram_snapshot.source})"
        ),
        ("[quantizer] VRAM estimada: " f"{_format_gib(result.vram_estimate.total_bytes)} ({', '.join(vram_parts)})"),
        (
            "[quantizer] Perfil del modelo: "
            f"params≈{result.model_profile.parameter_count:,}, hidden_size={result.model_profile.hidden_size}, "
            f"dtype={result.model_profile.dtype}, fuente={result.model_profile.estimate_source}"
        ),
    ]
    if result.risk_factors:
        report_lines.append("[quantizer] Riesgos: " + "; ".join(result.risk_factors))
    if result.likely_failure == "ram_oom_killer":
        report_lines.append("[quantizer] Fallo probable: RAM / OOM killer del sistema")
    elif result.likely_failure == "cuda_oom":
        report_lines.append("[quantizer] Fallo probable: VRAM / CUDA OOM")
    elif result.likely_failure == "cuda_unavailable":
        report_lines.append("[quantizer] Fallo probable: no hay una GPU CUDA utilizable")
    if result.likely_failure == "ram_oom_killer" and result.ram_snapshot.top_ram_processes:
        top_processes_text = []
        for consumer in result.ram_snapshot.top_ram_processes:
            process_text = f"pid={consumer.pid} {consumer.name} rss≈{_format_gib(consumer.rss_bytes)}"
            if consumer.command_preview and consumer.command_preview.lower() != consumer.name.lower():
                process_text += f" [{consumer.command_preview}]"
            top_processes_text.append(process_text)
        report_lines.append("[quantizer] Procesos con mayor uso de RAM: " + "; ".join(top_processes_text))
        report_lines.append("[quantizer] Cierra estos procesos si no son necesarios para liberar RAM antes de ejecutar el cuantizador.")
    if result.suggested_sample_count is not None or result.suggested_sequence_length is not None:
        report_lines.append(
            "[quantizer] Sugerencias conservadoras: "
            f"samples<={result.suggested_sample_count or result.sample_count}, "
            f"seq_len<={result.suggested_sequence_length or result.max_sequence_length}"
        )
    if result.notes:
        report_lines.append("[quantizer] Notas: " + "; ".join(result.notes))
    return "\n".join(report_lines)


def enforce_memory_preflight_policy(result: MemoryPreflightResult, mode: str) -> None:
    """Aplica el guardrail configurado y aborta si el riesgo excede el umbral."""

    normalized_mode = mode.strip().lower()
    if normalized_mode == "off":
        return
    if normalized_mode == "report":
        return
    if normalized_mode == "guard" and result.status != "inviable":
        return
    if normalized_mode == "fail-fast" and result.status == "safe":
        return
    raise RuntimeError(
        "El preflight de memoria detecto que la corrida no es segura con el umbral configurado.\n" + format_memory_preflight_report(result)
    )
