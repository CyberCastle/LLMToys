#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Generacion del README de salida para checkpoints cuantizados."""

from __future__ import annotations

from datetime import datetime, UTC
from functools import lru_cache
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Sequence

from .config import QuantizerConfig


def _read_package_version(package_name: str) -> str:
    """Resuelve la version instalada de un paquete o retorna `n/a`."""

    try:
        return version(package_name)
    except PackageNotFoundError:
        return "n/a"


def _markdown_table_value(raw_value: object) -> str:
    """Escapa valores para que puedan mostrarse de forma segura en tablas Markdown."""

    text = str(raw_value)
    text = text.replace("|", "\\|")
    text = text.replace("\n", "<br>")
    return text


def _format_bool(value: bool) -> str:
    """Representa booleanos como texto legible para el model card."""

    return "yes" if value else "no"


@lru_cache(maxsize=32)
def _resolve_base_model_license(model_id: str) -> str:
    """Obtiene la licencia del modelo base desde Hugging Face si esta disponible."""

    try:
        from huggingface_hub import HfApi
    except ImportError:
        return "unknown"

    try:
        model_info = HfApi().model_info(model_id)
    except Exception:
        return "unknown"

    card_data = getattr(model_info, "cardData", None)
    if card_data is not None:
        if isinstance(card_data, dict):
            license_value = card_data.get("license")
        elif hasattr(card_data, "get"):
            license_value = card_data.get("license")
        elif hasattr(card_data, "to_dict"):
            license_value = card_data.to_dict().get("license")
        else:
            license_value = getattr(card_data, "license", None)
        if isinstance(license_value, str) and license_value.strip():
            return license_value.strip()

    for tag in getattr(model_info, "tags", []) or []:
        normalized_tag = str(tag).strip()
        if normalized_tag.startswith("license:"):
            license_value = normalized_tag.split(":", 1)[1].strip()
            if license_value:
                return license_value

    return "unknown"


def _usage_model_id(output_dir: Path) -> str:
    """Construye un identificador de ejemplo para subir el checkpoint al Hub."""

    return f"your-hf-username/{output_dir.name}"


def _bits_label(scheme: str) -> str:
    """Resume el formato de bits de la cuantizacion generada."""

    del scheme
    return "4-bit weights / 16-bit activations"


def _tested_backend_label() -> str:
    """Resume el backend verificado de forma consistente por el cuantizador."""

    return "Transformers"


def _build_transformers_usage_block(config: QuantizerConfig, output_dir: Path) -> str:
    """Genera un snippet de uso base para Transformers y el Hub."""

    trust_remote_code_literal = "True" if config.trust_remote_code_model else "False"
    model_id = _usage_model_id(output_dir)
    return "\n".join(
        [
            "```python",
            "from transformers import AutoTokenizer, AutoModelForCausalLM",
            "",
            f'model_id = "{model_id}"',
            "",
            "tokenizer = AutoTokenizer.from_pretrained(model_id)",
            "model = AutoModelForCausalLM.from_pretrained(",
            "    model_id,",
            '    device_map="auto",',
            f"    trust_remote_code={trust_remote_code_literal},",
            ")",
            "```",
        ]
    )


def _requested_sequential_targets_label(config: QuantizerConfig, scheme: str) -> str:
    """Resume la configuracion declarada para los targets secuenciales."""

    mode = config.sequential_targets_mode_for(scheme)
    if mode == "auto":
        return "auto"
    if mode == "explicit":
        targets = config.sequential_targets_for(scheme) or []
        return ", ".join(targets) if targets else "explicit"
    return "safe-auto"


def _resolved_sequential_targets_label(
    config: QuantizerConfig,
    scheme: str,
    resolved_targets: list[str] | None,
) -> str:
    """Resume los targets efectivos usados durante la corrida."""

    if not config.sequential_onloading_for(scheme):
        return "disabled"
    if resolved_targets:
        return ", ".join(resolved_targets)
    if config.sequential_targets_mode_for(scheme) == "auto":
        return "automatic detection by llmcompressor"
    return "safe-auto"


def build_quantized_output_readme(
    config: QuantizerConfig,
    scheme: str,
    *,
    output_dir: Path,
    model_type: str | None,
    actual_sample_count: int,
    resolved_sequential_targets: list[str] | None,
) -> str:
    """Construye el contenido Markdown del model card en ingles."""

    normalized_scheme = scheme.strip().lower()
    if normalized_scheme not in {"awq", "gptq"}:
        raise ValueError("El esquema debe ser 'awq' o 'gptq'")

    generated_at = datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    scheme_upper = normalized_scheme.upper()
    model_title = f"{config.model_slug()} W4A16 {scheme_upper}"
    base_model_license = _resolve_base_model_license(config.model_id)
    dataset_config_name = config.dataset_config_name or "n/a"
    model_type_text = model_type or "unknown"
    max_gpu_memory_gib = f"{config.effective_max_gpu_memory_gib(normalized_scheme):.1f} GiB"
    requested_targets = _requested_sequential_targets_label(config, normalized_scheme)
    effective_targets = _resolved_sequential_targets_label(config, normalized_scheme, resolved_sequential_targets)
    usage_block = _build_transformers_usage_block(config, output_dir)

    quantization_rows = [
        ("Base model", config.model_id),
        ("Output folder", output_dir.name),
        ("Quantization scheme", scheme_upper),
        ("Weight / activation format", "W4A16"),
        ("Model architecture", model_type_text),
        ("Calibration dataset", config.calibration_dataset),
        ("Calibration split", config.calibration_split),
        ("Dataset configuration", dataset_config_name),
        ("Calibration samples used", actual_sample_count),
        ("Max sequence length", config.max_sequence_length),
        ("Max GPU memory budget", max_gpu_memory_gib),
        ("Sequential onloading", _format_bool(config.sequential_onloading_for(normalized_scheme))),
        ("Requested sequential targets", requested_targets),
        ("Effective sequential targets", effective_targets),
        ("Sequential targets per subgraph", config.sequential_targets_per_subgraph_for(normalized_scheme)),
        ("trust_remote_code", _format_bool(config.trust_remote_code_model)),
        ("Memory preflight mode", config.memory_preflight_mode),
        ("vLLM smoke test requested", _format_bool(config.run_vllm_smoke_test)),
    ]
    toolchain_rows = [
        ("Generated at (UTC)", generated_at),
        ("Runner entrypoint", "uv run quantizer/run.py"),
        ("llmcompressor", _read_package_version("llmcompressor")),
        ("transformers", _read_package_version("transformers")),
        ("torch", _read_package_version("torch")),
        ("compressed-tensors", _read_package_version("compressed-tensors")),
    ]

    def _render_table(rows: Sequence[tuple[str, object]]) -> str:
        lines = ["| Setting | Value |", "| --- | --- |"]
        for key, value in rows:
            lines.append(f"| {_markdown_table_value(key)} | {_markdown_table_value(value)} |")
        return "\n".join(lines)

    return "\n".join(
        [
            "---",
            f"license: {_markdown_table_value(base_model_license)}",
            f"base_model: {_markdown_table_value(config.model_id)}",
            "library_name: transformers",
            "pipeline_tag: text-generation",
            "tags:",
            "- text-generation",
            "- quantized",
            f"- {normalized_scheme}",
            "- w4a16",
            "- llmcompressor",
            "datasets:",
            f"- {_markdown_table_value(config.calibration_dataset)}",
            "---",
            "",
            f"# {model_title}",
            "",
            f"Quantized model derived from `{config.model_id}`.",
            "",
            (
                "This repository contains a locally generated quantized checkpoint ready to be uploaded to the "
                "Hugging Face Hub. The folder includes the quantized weights, tokenizer files, and the exact "
                "quantization settings used to produce this artifact."
            ),
            "",
            "## Format",
            "",
            f"- Quantization type: {scheme_upper}",
            f"- Bits: {_bits_label(normalized_scheme)}",
            f"- Calibration dataset: `{config.calibration_dataset}`",
            f"- Tested backend: {_tested_backend_label()}",
            "",
            "## Usage",
            "",
            usage_block,
            "",
            "## About LLMToys",
            "",
            ("This quantizer is hosted in the LLMToys repository: " "https://github.com/CyberCastle/LLMToys."),
            (
                "LLMToys is a collection of practical LLM tools and experiments maintained in a single codebase. "
                "It groups reusable components for local model execution, quantization workflows, runtime tuning, "
                "and structured generation pipelines such as natural-language-to-SQL."
            ),
            "",
            "## Quantization Configuration",
            "",
            _render_table(quantization_rows),
            "",
            "## Toolchain",
            "",
            _render_table(toolchain_rows),
            "",
            "## Notes",
            "",
            "- This README is generated automatically by the quantizer so the artifact keeps its execution context.",
            "- Review the original base model license and any upstream usage restrictions before publishing this checkpoint.",
            "- If you rerun the quantizer with different settings, regenerate and upload the full output directory again.",
            "",
        ]
    )


def write_quantized_output_readme(
    config: QuantizerConfig,
    scheme: str,
    *,
    output_dir: Path,
    model_type: str | None,
    actual_sample_count: int,
    resolved_sequential_targets: list[str] | None,
) -> Path:
    """Escribe el model card generado junto al checkpoint cuantizado."""

    readme_path = output_dir / "README.md"
    content = build_quantized_output_readme(
        config,
        scheme,
        output_dir=output_dir,
        model_type=model_type,
        actual_sample_count=actual_sample_count,
        resolved_sequential_targets=resolved_sequential_targets,
    )
    readme_path.write_text(content, encoding="utf-8")
    return readme_path
