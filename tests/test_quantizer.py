#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest
import yaml

from quantizer.awq_recipe import build_awq_recipe
from quantizer.calibration_data import prepare_calibration_records
from quantizer.config import QuantizerConfig
from quantizer.gptq_recipe import build_gptq_recipe
from quantizer.memory_preflight import (
    GpuMemorySnapshot,
    MemoryBreakdown,
    MemoryPreflightResult,
    ModelMemoryProfile,
    ProcessMemoryConsumer,
    SystemMemorySnapshot,
    enforce_memory_preflight_policy,
    evaluate_quantization_memory_preflight,
    format_memory_preflight_report,
)
from quantizer.quantizer import _cleanup_runtime, _prepare_output_dir, _resolve_sequential_targets, quantize_model


class DummyTokenizer:
    """Tokenizer sintético para validar el preprocesamiento sin descargar modelos."""

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        assert tokenize is False
        assert add_generation_prompt is False
        return "\n".join(f"{message['role']}: {message['content']}" for message in messages)

    def __call__(
        self,
        text,
        *,
        add_special_tokens=False,
        truncation=True,
        max_length=None,
        return_attention_mask=True,
    ):
        del add_special_tokens, truncation, return_attention_mask
        token_count = len(text.split())
        if max_length is not None:
            token_count = min(token_count, max_length)
        input_ids = list(range(1, token_count + 1))
        return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids)}


def _write_yaml(tmp_path: Path, name: str, payload: dict) -> Path:
    yaml_path = tmp_path / name
    yaml_path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return yaml_path


def test_config_from_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("QUANTIZER_MODEL_ID", "org/modelo-prueba")
    monkeypatch.setenv("QUANTIZER_CALIBRATION_DATASET", "org/dataset-prueba")
    monkeypatch.setenv("QUANTIZER_CALIBRATION_SPLIT", "validation")
    monkeypatch.setenv("QUANTIZER_AWQ_NUM_CALIBRATION_SAMPLES", "24")
    monkeypatch.setenv("QUANTIZER_GPTQ_NUM_CALIBRATION_SAMPLES", "32")
    monkeypatch.setenv("QUANTIZER_MAX_SEQUENCE_LENGTH", "1024")
    monkeypatch.setenv("QUANTIZE_SCHEME", "gptq")
    monkeypatch.setenv("QUANTIZER_OUTPUT_DIR", str(tmp_path / "salida"))
    monkeypatch.setenv("QUANTIZER_AWQ_MAX_GPU_MEMORY_GIB", "12.5")
    monkeypatch.setenv("QUANTIZER_GPTQ_MAX_GPU_MEMORY_GIB", "10.5")
    monkeypatch.setenv("QUANTIZER_AWQ_SEQUENTIAL_ONLOADING", "true")
    monkeypatch.setenv("QUANTIZER_GPTQ_SEQUENTIAL_ONLOADING", "false")
    monkeypatch.setenv("QUANTIZER_TRUST_REMOTE_CODE_MODEL", "true")
    monkeypatch.setenv("QUANTIZER_RUN_VLLM_SMOKE_TEST", "true")
    monkeypatch.setenv("QUANTIZER_MEMORY_PREFLIGHT_MODE", "fail-fast")

    config = QuantizerConfig.from_env()

    assert config.model_id == "org/modelo-prueba"
    assert config.calibration_dataset == "org/dataset-prueba"
    assert config.calibration_split == "validation"
    assert config.awq_num_calibration_samples == 24
    assert config.gptq_num_calibration_samples == 32
    assert config.max_sequence_length == 1024
    assert config.quantize_scheme == "gptq"
    assert config.output_dir == tmp_path / "salida"
    assert config.awq_max_gpu_memory_gib == 12.5
    assert config.gptq_max_gpu_memory_gib == 10.5
    assert config.awq_sequential_onloading is True
    assert config.gptq_sequential_onloading is False
    assert config.resolved_sequential_targets("gptq") is None
    assert config.trust_remote_code_model is True
    assert config.run_vllm_smoke_test is True
    assert config.memory_preflight_mode == "fail-fast"


def test_config_rejects_invalid_memory_preflight_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QUANTIZER_MODEL_ID", "org/modelo-prueba")
    monkeypatch.setenv("QUANTIZER_CALIBRATION_DATASET", "org/dataset-prueba")
    monkeypatch.setenv("QUANTIZER_MEMORY_PREFLIGHT_MODE", "desconocido")

    with pytest.raises(ValueError, match="QUANTIZER_MEMORY_PREFLIGHT_MODE"):
        QuantizerConfig.from_env()


def test_config_default_sequential_targets(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QUANTIZER_MODEL_ID", "org/modelo-prueba")
    monkeypatch.setenv("QUANTIZER_CALIBRATION_DATASET", "org/dataset-prueba")
    monkeypatch.delenv("QUANTIZER_AWQ_SEQUENTIAL_TARGETS", raising=False)
    monkeypatch.delenv("QUANTIZER_GPTQ_SEQUENTIAL_TARGETS", raising=False)
    monkeypatch.delenv("QUANTIZER_AWQ_SEQUENTIAL_ONLOADING", raising=False)
    monkeypatch.delenv("QUANTIZER_GPTQ_SEQUENTIAL_ONLOADING", raising=False)

    config = QuantizerConfig.from_env()

    assert config.awq_sequential_onloading is True
    assert config.gptq_sequential_onloading is True
    assert config.awq_sequential_targets_mode == "safe"
    assert config.gptq_sequential_targets_mode == "safe"
    assert config.resolved_sequential_targets("awq") is None
    assert config.resolved_sequential_targets("gptq") is None


def test_effective_calibration_sample_count_caps_awq_only() -> None:
    config = QuantizerConfig(
        model_id="org/modelo-prueba",
        calibration_dataset="org/dataset-prueba",
        awq_num_calibration_samples=128,
        gptq_num_calibration_samples=512,
    )

    assert config.effective_calibration_sample_count("awq") == 128
    assert config.effective_calibration_sample_count("awq", available_samples=64) == 64
    assert config.effective_calibration_sample_count("gptq") == 512


def test_effective_gpu_memory_budget_is_scheme_specific() -> None:
    config = QuantizerConfig(
        model_id="org/modelo-prueba",
        calibration_dataset="org/dataset-prueba",
        awq_max_gpu_memory_gib=13.0,
        gptq_max_gpu_memory_gib=10.0,
    )

    assert config.effective_max_gpu_memory_gib("awq") == 13.0
    assert config.effective_max_gpu_memory_gib("gptq") == 10.0


def test_config_auto_sequential_targets(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QUANTIZER_MODEL_ID", "org/modelo-prueba")
    monkeypatch.setenv("QUANTIZER_CALIBRATION_DATASET", "org/dataset-prueba")
    monkeypatch.setenv("QUANTIZER_AWQ_SEQUENTIAL_TARGETS", "auto")
    monkeypatch.setenv("QUANTIZER_GPTQ_SEQUENTIAL_TARGETS", "auto")

    config = QuantizerConfig.from_env()

    assert config.awq_sequential_targets_mode == "auto"
    assert config.gptq_sequential_targets_mode == "auto"
    assert config.resolved_sequential_targets("awq") is None
    assert config.resolved_sequential_targets("gptq") is None


def test_config_safe_auto_alias_for_targets(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QUANTIZER_MODEL_ID", "org/modelo-prueba")
    monkeypatch.setenv("QUANTIZER_CALIBRATION_DATASET", "org/dataset-prueba")
    monkeypatch.setenv("QUANTIZER_AWQ_SEQUENTIAL_TARGETS", "safe-auto")
    monkeypatch.setenv("QUANTIZER_GPTQ_SEQUENTIAL_TARGETS", "safe-auto")

    config = QuantizerConfig.from_env()

    assert config.awq_sequential_targets_mode == "safe"
    assert config.gptq_sequential_targets_mode == "safe"
    assert config.resolved_sequential_targets("awq") is None
    assert config.resolved_sequential_targets("gptq") is None


def test_config_explicit_sequential_targets(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QUANTIZER_MODEL_ID", "org/modelo-prueba")
    monkeypatch.setenv("QUANTIZER_CALIBRATION_DATASET", "org/dataset-prueba")
    monkeypatch.setenv("QUANTIZER_AWQ_SEQUENTIAL_TARGETS", "Qwen2Attention,Qwen2MLP")
    monkeypatch.setenv("QUANTIZER_GPTQ_SEQUENTIAL_TARGETS", "Linear")

    config = QuantizerConfig.from_env()

    assert config.awq_sequential_targets_mode == "explicit"
    assert config.gptq_sequential_targets_mode == "explicit"
    assert config.resolved_sequential_targets("awq") == ["Qwen2Attention", "Qwen2MLP"]
    assert config.resolved_sequential_targets("gptq") == ["Linear"]


def test_safe_qwen2_sequential_targets() -> None:
    config = QuantizerConfig(
        model_id="org/modelo-prueba",
        calibration_dataset="org/dataset-prueba",
    )

    assert _resolve_sequential_targets(config, "qwen2", "awq") == ["Qwen2Attention", "Qwen2MLP"]
    assert _resolve_sequential_targets(config, "desconocido", "awq") == ["Linear"]
    assert _resolve_sequential_targets(config, "qwen2", "gptq") == ["Qwen2Attention", "Qwen2MLP"]


def test_cleanup_runtime_resets_session_and_cuda_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    class DummyCuda:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def empty_cache() -> None:
            calls.append("empty_cache")

        @staticmethod
        def ipc_collect() -> None:
            calls.append("ipc_collect")

    import llmcompressor.core.session_functions as session_functions

    monkeypatch.setattr(session_functions, "reset_session", lambda: calls.append("reset_session"))
    monkeypatch.setattr(
        "quantizer.quantizer._require_torch",
        lambda: SimpleNamespace(cuda=DummyCuda()),
    )

    _cleanup_runtime(object())

    assert calls == ["reset_session", "empty_cache", "ipc_collect"]


def test_prepare_output_dir_removes_previous_artifacts(tmp_path: Path) -> None:
    output_dir = tmp_path / "salida-awq"
    output_dir.mkdir(parents=True)
    (output_dir / "stale.json").write_text("viejo", encoding="utf-8")
    nested_dir = output_dir / "nested"
    nested_dir.mkdir()
    (nested_dir / "artifact.bin").write_text("obsoleto", encoding="utf-8")

    _prepare_output_dir(output_dir)

    assert output_dir.exists()
    assert output_dir.is_dir()
    assert list(output_dir.iterdir()) == []


def test_prepare_output_dir_rejects_existing_file(tmp_path: Path) -> None:
    output_file = tmp_path / "salida-awq"
    output_file.write_text("no es carpeta", encoding="utf-8")

    with pytest.raises(RuntimeError, match="no es un directorio"):
        _prepare_output_dir(output_file)


def test_quantize_model_writes_english_hf_readme(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    config = QuantizerConfig(
        model_id="org/modelo-prueba",
        calibration_dataset="org/dataset-prueba",
        calibration_split="validation",
        awq_num_calibration_samples=24,
        max_sequence_length=1024,
        output_dir=tmp_path,
        awq_max_gpu_memory_gib=12.5,
        run_vllm_smoke_test=True,
        memory_preflight_mode="report",
    )

    oneshot_calls: list[dict[str, object]] = []

    monkeypatch.setattr("quantizer.output_readme._resolve_base_model_license", lambda model_id: "apache-2.0")

    class DummyTokenizerForSave:
        def save_pretrained(self, output_dir: Path) -> None:
            target_dir = Path(output_dir)
            (target_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
            (target_dir / "tokenizer_config.json").write_text("{}", encoding="utf-8")

    def _fake_oneshot(**kwargs) -> None:
        oneshot_calls.append(kwargs)
        target_dir = Path(str(kwargs["output_dir"]))
        (target_dir / "config.json").write_text("{}", encoding="utf-8")
        (target_dir / "model.safetensors").write_text("weights", encoding="utf-8")

    monkeypatch.setitem(sys.modules, "llmcompressor", SimpleNamespace(oneshot=_fake_oneshot))
    monkeypatch.setattr("quantizer.quantizer._read_model_type", lambda *args, **kwargs: "qwen2")
    monkeypatch.setattr(
        "quantizer.quantizer._load_model",
        lambda *args, **kwargs: (object(), DummyTokenizerForSave()),
    )
    monkeypatch.setattr(
        "quantizer.quantizer.load_calibration_data",
        lambda tokenizer, config, num_calibration_samples: [
            {"input_ids": [1, 2, 3]},
            {"input_ids": [4, 5]},
        ],
    )
    monkeypatch.setattr("quantizer.quantizer._build_recipe", lambda *args, **kwargs: ["dummy-recipe"])
    monkeypatch.setattr("quantizer.quantizer._cleanup_runtime", lambda *args, **kwargs: None)

    output_dir = quantize_model(config, "awq")

    assert len(oneshot_calls) == 1
    assert output_dir == tmp_path / "modelo-prueba-W4A16-AWQ"

    readme_text = (output_dir / "README.md").read_text(encoding="utf-8")
    assert "license: apache-2.0" in readme_text
    assert "# modelo-prueba W4A16 AWQ" in readme_text
    assert "Quantized model derived from `org/modelo-prueba`." in readme_text
    assert "This repository contains a locally generated quantized checkpoint ready to be uploaded to the Hugging Face Hub." in readme_text
    assert "## Format" in readme_text
    assert "- Quantization type: AWQ" in readme_text
    assert "- Bits: 4-bit weights / 16-bit activations" in readme_text
    assert "- Calibration dataset: `org/dataset-prueba`" in readme_text
    assert "- Tested backend: Transformers" in readme_text
    assert "## Usage" in readme_text
    assert 'model_id = "your-hf-username/modelo-prueba-W4A16-AWQ"' in readme_text
    assert "trust_remote_code=False" in readme_text
    assert "## About LLMToys" in readme_text
    assert "https://github.com/CyberCastle/LLMToys" in readme_text
    assert "LLMToys is a collection of practical LLM tools and experiments maintained in a single codebase." in readme_text
    assert "| Base model | org/modelo-prueba |" in readme_text
    assert "| Calibration dataset | org/dataset-prueba |" in readme_text
    assert "| Calibration split | validation |" in readme_text
    assert "| Calibration samples used | 2 |" in readme_text
    assert "| Max GPU memory budget | 12.5 GiB |" in readme_text
    assert "| Effective sequential targets | Qwen2Attention, Qwen2MLP |" in readme_text
    assert "| vLLM smoke test requested | yes |" in readme_text
    assert "## Toolchain" in readme_text
    assert "Review the original base model license" in readme_text


def test_output_dir_derivation(tmp_path: Path) -> None:
    config = QuantizerConfig(
        model_id="org/modelo-prueba",
        calibration_dataset="org/dataset-prueba",
        output_dir=tmp_path,
    )

    assert config.output_dir_for("awq") == tmp_path / "modelo-prueba-W4A16-AWQ"
    assert config.output_dir_for("gptq") == tmp_path / "modelo-prueba-W4A16-GPTQ"


def test_model_slug() -> None:
    config = QuantizerConfig(model_id="org/modelo-prueba-7b", calibration_dataset="org/dataset")
    assert config.model_slug() == "modelo-prueba-7b"


def test_awq_recipe_structure() -> None:
    recipe = build_awq_recipe("qwen2")

    assert len(recipe) == 2
    assert recipe[0].__class__.__name__ == "AWQModifier"
    assert recipe[1].__class__.__name__ == "QuantizationModifier"
    assert getattr(recipe[1], "scheme") == "W4A16_ASYM"
    assert getattr(recipe[1], "ignore") == ["lm_head"]
    assert getattr(recipe[0], "mappings")


def test_awq_recipe_unknown_arch() -> None:
    with pytest.raises(ValueError, match="Arquitecturas soportadas"):
        build_awq_recipe("arquitectura-inexistente")


def test_gptq_recipe_structure() -> None:
    recipe = build_gptq_recipe()

    assert len(recipe) == 1
    assert recipe[0].__class__.__name__ == "GPTQModifier"
    assert getattr(recipe[0], "scheme") == "W4A16"
    assert getattr(recipe[0], "ignore") == ["lm_head"]


def test_calibration_data_template_resolution(tmp_path: Path) -> None:
    templates_path = _write_yaml(
        tmp_path,
        "templates.yaml",
        {
            "birdsql/bird-critic": {
                "text_template": (
                    "Dialecto: {dialect}\n" "Pregunta: {query}\n" "SQL con problema: {issue_sql}\n" "SQL corregido: {clean_up_sql?}"
                )
            },
            "_default": {"text_field": "text"},
        },
    )

    records = prepare_calibration_records(
        [
            {
                "dialect": "PostgreSQL",
                "query": "cuantos registros?",
                "issue_sql": ["SELECT 1", "SELECT 2"],
                "clean_up_sql": [],
            },
            {
                "dialect": "SQLite",
                "query": "sin sql corregido",
                "issue_sql": ["SELECT 3"],
                "clean_up_sql": [],
            },
        ],
        DummyTokenizer(),
        "birdsql/bird-critic-1.0-open",
        128,
        templates_path,
    )

    assert len(records) == 2
    assert "Pregunta: cuantos registros?" in records[0]["source_text"]
    assert "SELECT 1\nSELECT 2" in records[0]["source_text"]
    assert "SQL corregido:" not in records[1]["source_text"]
    assert records[0]["input_ids"]


def test_calibration_data_default_text_field(tmp_path: Path) -> None:
    templates_path = _write_yaml(
        tmp_path,
        "templates.yaml",
        {
            "_default": {"text_field": "text"},
        },
    )

    records = prepare_calibration_records(
        [{"text": "hola mundo"}],
        DummyTokenizer(),
        "dataset/desconocido",
        128,
        templates_path,
    )

    assert len(records) == 1
    assert records[0]["source_text"] == "hola mundo"


def test_calibration_data_unknown_dataset_no_text_field(tmp_path: Path) -> None:
    templates_path = _write_yaml(tmp_path, "templates.yaml", {})

    with pytest.raises(ValueError, match="Columnas disponibles en el ejemplo: prompt"):
        prepare_calibration_records(
            [{"prompt": "hola"}],
            DummyTokenizer(),
            "dataset/desconocido",
            128,
            templates_path,
        )


def test_runner_direct_execution_uses_package_imports() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["QUANTIZER_MODEL_ID"] = "org/modelo-prueba"
    env["QUANTIZER_CALIBRATION_DATASET"] = "org/dataset-prueba"
    env["QUANTIZE_SCHEME"] = "invalido"

    completed = subprocess.run(
        [sys.executable, str(repo_root / "quantizer" / "run.py")],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode != 0
    assert "QUANTIZE_SCHEME debe ser 'awq', 'gptq' o 'both'" in completed.stderr
    assert "attempted relative import with no known parent package" not in completed.stderr


def test_memory_preflight_marks_ram_oom_as_inviable(monkeypatch: pytest.MonkeyPatch) -> None:
    config = QuantizerConfig(
        model_id="org/modelo-prueba",
        calibration_dataset="org/dataset-prueba",
        awq_num_calibration_samples=256,
        max_sequence_length=2048,
        awq_max_gpu_memory_gib=13.0,
    )

    monkeypatch.setattr(
        "quantizer.memory_preflight._detect_system_memory_snapshot",
        lambda: SystemMemorySnapshot(
            system_available_bytes=16 * 1024**3,
            system_total_bytes=64 * 1024**3,
            system_shared_bytes=10 * 1024**3,
            cgroup_available_bytes=16 * 1024**3,
            cgroup_limit_bytes=32 * 1024**3,
            cgroup_current_bytes=16 * 1024**3,
            cgroup_version="v2",
            cgroup_path="/user.slice",
            cgroup_shmem_bytes=8 * 1024**3,
            rlimit_available_bytes=None,
            process_rss_bytes=512 * 1024**2,
            process_vms_bytes=2 * 1024**3,
            dev_shm_available_bytes=2 * 1024**3,
            dev_shm_total_bytes=8 * 1024**3,
        ),
    )
    monkeypatch.setattr(
        "quantizer.memory_preflight._detect_gpu_memory_snapshot",
        lambda configured_budget_bytes: GpuMemorySnapshot(
            cuda_available=True,
            available_bytes=14 * 1024**3,
            total_bytes=16 * 1024**3,
            configured_budget_bytes=configured_budget_bytes,
            source="torch.cuda.mem_get_info",
        ),
    )
    monkeypatch.setattr(
        "quantizer.memory_preflight._load_model_memory_profile",
        lambda _config, *, dtype: ModelMemoryProfile(
            model_type="qwen2",
            hidden_size=4096,
            intermediate_size=11008,
            num_hidden_layers=32,
            vocab_size=151936,
            parameter_count=7_000_000_000,
            buffer_count=0,
            dtype=dtype,
            estimate_source="meta_model",
        ),
    )

    result = evaluate_quantization_memory_preflight(config, "awq")

    assert result.status == "inviable"
    assert result.likely_failure == "ram_oom_killer"
    assert result.suggested_sample_count is not None
    assert any("OOM killer" in line for line in [format_memory_preflight_report(result)])


def test_memory_preflight_can_be_safe(monkeypatch: pytest.MonkeyPatch) -> None:
    config = QuantizerConfig(
        model_id="org/modelo-prueba",
        calibration_dataset="org/dataset-prueba",
        awq_num_calibration_samples=16,
        max_sequence_length=1024,
        awq_max_gpu_memory_gib=13.0,
    )

    monkeypatch.setattr(
        "quantizer.memory_preflight._detect_system_memory_snapshot",
        lambda: SystemMemorySnapshot(
            system_available_bytes=64 * 1024**3,
            system_total_bytes=64 * 1024**3,
            system_shared_bytes=2 * 1024**3,
            cgroup_available_bytes=None,
            cgroup_limit_bytes=None,
            cgroup_current_bytes=None,
            cgroup_version=None,
            cgroup_path=None,
            cgroup_shmem_bytes=None,
            rlimit_available_bytes=None,
            process_rss_bytes=256 * 1024**2,
            process_vms_bytes=2 * 1024**3,
            dev_shm_available_bytes=32 * 1024**3,
            dev_shm_total_bytes=32 * 1024**3,
        ),
    )
    monkeypatch.setattr(
        "quantizer.memory_preflight._detect_gpu_memory_snapshot",
        lambda configured_budget_bytes: GpuMemorySnapshot(
            cuda_available=True,
            available_bytes=15 * 1024**3,
            total_bytes=16 * 1024**3,
            configured_budget_bytes=configured_budget_bytes,
            source="torch.cuda.mem_get_info",
        ),
    )
    monkeypatch.setattr(
        "quantizer.memory_preflight._load_model_memory_profile",
        lambda _config, *, dtype: ModelMemoryProfile(
            model_type="qwen2",
            hidden_size=2048,
            intermediate_size=5632,
            num_hidden_layers=24,
            vocab_size=151936,
            parameter_count=1_000_000_000,
            buffer_count=0,
            dtype=dtype,
            estimate_source="meta_model",
        ),
    )

    result = evaluate_quantization_memory_preflight(config, "awq")

    assert result.status == "safe"
    assert result.likely_failure is None


def test_memory_preflight_real_awq_success_case_is_not_blocked(monkeypatch: pytest.MonkeyPatch) -> None:
    config = QuantizerConfig(
        model_id="XGenerationLab/XiYanSQL-QwenCoder-7B-2504",
        calibration_dataset="birdsql/bird-critic-1.0-open",
        awq_num_calibration_samples=256,
        max_sequence_length=2048,
        awq_max_gpu_memory_gib=12.0,
        awq_sequential_onloading=True,
    )

    monkeypatch.setattr(
        "quantizer.memory_preflight._detect_system_memory_snapshot",
        lambda: SystemMemorySnapshot(
            system_available_bytes=23 * 1024**3,
            system_total_bytes=64 * 1024**3,
            system_shared_bytes=3 * 1024**3,
            cgroup_available_bytes=None,
            cgroup_limit_bytes=None,
            cgroup_current_bytes=None,
            cgroup_version=None,
            cgroup_path=None,
            cgroup_shmem_bytes=None,
            rlimit_available_bytes=None,
            process_rss_bytes=int(0.88 * 1024**3),
            process_vms_bytes=3 * 1024**3,
            dev_shm_available_bytes=int(15.5 * 1024**3),
            dev_shm_total_bytes=16 * 1024**3,
        ),
    )
    monkeypatch.setattr(
        "quantizer.memory_preflight._detect_gpu_memory_snapshot",
        lambda configured_budget_bytes: GpuMemorySnapshot(
            cuda_available=True,
            available_bytes=int(15.42 * 1024**3),
            total_bytes=16 * 1024**3,
            configured_budget_bytes=configured_budget_bytes,
            source="torch.cuda.mem_get_info",
        ),
    )
    monkeypatch.setattr(
        "quantizer.memory_preflight._load_model_memory_profile",
        lambda _config, *, dtype: ModelMemoryProfile(
            model_type="qwen2",
            hidden_size=3584,
            intermediate_size=18944,
            num_hidden_layers=28,
            vocab_size=151936,
            parameter_count=7_615_616_512,
            buffer_count=0,
            dtype=dtype,
            estimate_source="meta_model",
        ),
    )

    result = evaluate_quantization_memory_preflight(config, "awq")

    assert result.status == "risky"
    assert result.likely_failure == "ram_oom_killer"


def test_system_memory_snapshot_is_multiplatform(monkeypatch: pytest.MonkeyPatch) -> None:
    import quantizer.memory_preflight as memory_preflight

    class DummyProcess:
        def memory_info(self) -> SimpleNamespace:
            return SimpleNamespace(rss=512 * 1024**2, vms=3 * 1024**3)

    monkeypatch.setattr(memory_preflight.platform, "system", lambda: "Windows")
    monkeypatch.setattr(
        memory_preflight.psutil,
        "virtual_memory",
        lambda: SimpleNamespace(total=64 * 1024**3, available=24 * 1024**3),
    )
    monkeypatch.setattr(memory_preflight.psutil, "Process", lambda: DummyProcess())
    monkeypatch.setattr(memory_preflight.psutil, "process_iter", lambda *_args, **_kwargs: [])
    monkeypatch.setattr(
        memory_preflight.psutil,
        "disk_usage",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("No debe consultar /dev/shm fuera de Linux")),
    )
    monkeypatch.setattr(memory_preflight, "resource", None)

    snapshot = memory_preflight._detect_system_memory_snapshot()

    assert snapshot.system_available_bytes == 24 * 1024**3
    assert snapshot.process_rss_bytes == 512 * 1024**2
    assert snapshot.cgroup_available_bytes is None
    assert snapshot.cgroup_limit_bytes is None
    assert snapshot.dev_shm_available_bytes is None
    assert snapshot.rlimit_available_bytes is None
    assert snapshot.top_ram_processes == []


def test_memory_preflight_report_lists_top_ram_processes() -> None:
    result = MemoryPreflightResult(
        scheme="awq",
        status="inviable",
        likely_failure="ram_oom_killer",
        sample_count=128,
        max_sequence_length=2048,
        model_profile=ModelMemoryProfile(
            model_type="qwen2",
            hidden_size=4096,
            intermediate_size=11008,
            num_hidden_layers=32,
            vocab_size=151936,
            parameter_count=7_000_000_000,
            buffer_count=0,
            dtype="bf16",
            estimate_source="meta_model",
        ),
        ram_snapshot=SystemMemorySnapshot(
            system_available_bytes=16 * 1024**3,
            system_total_bytes=64 * 1024**3,
            system_shared_bytes=4 * 1024**3,
            cgroup_available_bytes=None,
            cgroup_limit_bytes=None,
            cgroup_current_bytes=None,
            cgroup_version=None,
            cgroup_path=None,
            cgroup_shmem_bytes=None,
            rlimit_available_bytes=None,
            process_rss_bytes=256 * 1024**2,
            process_vms_bytes=2 * 1024**3,
            dev_shm_available_bytes=8 * 1024**3,
            dev_shm_total_bytes=16 * 1024**3,
            top_ram_processes=[
                ProcessMemoryConsumer(pid=111, name="code", rss_bytes=2 * 1024**3, command_preview="/usr/share/code/code"),
                ProcessMemoryConsumer(pid=222, name="firefox", rss_bytes=int(1.5 * 1024**3), command_preview="/usr/lib/firefox/firefox"),
            ],
        ),
        vram_snapshot=GpuMemorySnapshot(
            cuda_available=True,
            available_bytes=12 * 1024**3,
            total_bytes=16 * 1024**3,
            configured_budget_bytes=12 * 1024**3,
            source="torch.cuda.mem_get_info",
        ),
        ram_estimate=MemoryBreakdown(
            model_bytes=8 * 1024**3,
            activation_bytes=10 * 1024**3,
            internal_buffers_bytes=3 * 1024**3,
            dataset_bytes=1 * 1024**3,
            shared_memory_bytes=2 * 1024**3,
            offload_bytes=2 * 1024**3,
            subprocess_bytes=0,
            base_process_bytes=256 * 1024**2,
        ),
        vram_estimate=MemoryBreakdown(
            model_bytes=10 * 1024**3,
            activation_bytes=1 * 1024**3,
            internal_buffers_bytes=1 * 1024**3,
            dataset_bytes=0,
            shared_memory_bytes=0,
            offload_bytes=0,
            subprocess_bytes=0,
            base_process_bytes=0,
        ),
        risk_factors=["La RAM estimada consume mas del presupuesto seguro."],
        notes=["fuente de VRAM: torch.cuda.mem_get_info"],
        suggested_sample_count=64,
        suggested_sequence_length=1024,
    )

    report = format_memory_preflight_report(result)

    assert "Procesos con mayor uso de RAM" in report
    assert "pid=111 code" in report
    assert "pid=222 firefox" in report
    assert "Cierra estos procesos" in report


def test_fail_fast_mode_blocks_risky_preflight() -> None:
    result = MemoryPreflightResult(
        scheme="awq",
        status="risky",
        likely_failure="ram_oom_killer",
        sample_count=128,
        max_sequence_length=2048,
        model_profile=ModelMemoryProfile(
            model_type="qwen2",
            hidden_size=4096,
            intermediate_size=11008,
            num_hidden_layers=32,
            vocab_size=151936,
            parameter_count=7_000_000_000,
            buffer_count=0,
            dtype="bf16",
            estimate_source="meta_model",
        ),
        ram_snapshot=SystemMemorySnapshot(
            system_available_bytes=32 * 1024**3,
            system_total_bytes=64 * 1024**3,
            system_shared_bytes=4 * 1024**3,
            cgroup_available_bytes=None,
            cgroup_limit_bytes=None,
            cgroup_current_bytes=None,
            cgroup_version=None,
            cgroup_path=None,
            cgroup_shmem_bytes=None,
            rlimit_available_bytes=None,
            process_rss_bytes=256 * 1024**2,
            process_vms_bytes=2 * 1024**3,
            dev_shm_available_bytes=8 * 1024**3,
            dev_shm_total_bytes=16 * 1024**3,
        ),
        vram_snapshot=GpuMemorySnapshot(
            cuda_available=True,
            available_bytes=14 * 1024**3,
            total_bytes=16 * 1024**3,
            configured_budget_bytes=13 * 1024**3,
            source="torch.cuda.mem_get_info",
        ),
        ram_estimate=MemoryBreakdown(
            model_bytes=8 * 1024**3,
            activation_bytes=10 * 1024**3,
            internal_buffers_bytes=3 * 1024**3,
            dataset_bytes=1 * 1024**3,
            shared_memory_bytes=2 * 1024**3,
            offload_bytes=2 * 1024**3,
            subprocess_bytes=0,
            base_process_bytes=256 * 1024**2,
        ),
        vram_estimate=MemoryBreakdown(
            model_bytes=11 * 1024**3,
            activation_bytes=2 * 1024**3,
            internal_buffers_bytes=1 * 1024**3,
            dataset_bytes=0,
            shared_memory_bytes=0,
            offload_bytes=0,
            subprocess_bytes=0,
            base_process_bytes=0,
        ),
        risk_factors=["La RAM estimada consume mas del presupuesto seguro."],
        notes=["fuente de VRAM: torch.cuda.mem_get_info"],
        suggested_sample_count=64,
        suggested_sequence_length=1536,
    )

    enforce_memory_preflight_policy(result, "guard")
    with pytest.raises(RuntimeError, match="preflight de memoria"):
        enforce_memory_preflight_policy(result, "fail-fast")


def test_runner_uses_isolated_processes_for_both(monkeypatch: pytest.MonkeyPatch) -> None:
    import quantizer.run as run_module

    calls: list[tuple[list[str], bool, str]] = []
    dummy_config = SimpleNamespace(
        quantize_scheme="both",
        run_vllm_smoke_test=False,
        model_id="org/modelo-prueba",
        memory_preflight_mode="off",
    )

    monkeypatch.setattr(run_module, "load_dotenv", lambda: None)
    monkeypatch.setattr(run_module, "QuantizerConfig", SimpleNamespace(from_env=lambda: dummy_config))
    monkeypatch.setattr(
        run_module,
        "quantize_model",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("No debe cuantizar en el proceso padre")),
    )
    monkeypatch.setattr(
        run_module,
        "smoke_test",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("No debe ejecutar smoke_test en el proceso padre")),
    )

    def _fake_run(command: list[str], check: bool, env: dict[str, str]) -> SimpleNamespace:
        calls.append((command, check, env["QUANTIZE_SCHEME"]))
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(run_module.subprocess, "run", _fake_run)

    run_module.main()

    assert [scheme for _, _, scheme in calls] == ["awq", "gptq"]
    assert all(command[0] == sys.executable for command, _, _ in calls)
