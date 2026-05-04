#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import pytest

from llm_core.vllm_runtime_utils import resolve_gpu_utilization_from_available_memory
from nl2sql.sql_solver_generator.runtime import resolve_initial_solver_runtime_settings
from nl2sql.sql_solver_generator.sql_generator import (
    PromptTooLongError,
    require_solver_model_name,
    preflight_prompt_size,
    resolve_solver_runtime_retry,
    select_generation_prompt_variant,
)
from tests.generic_domain import (
    build_semantic_plan_mapping_active_a_per_c,
    generic_schema_tables,
    generic_semantic_contract,
)


class _FakeTokenizer:
    def __init__(self, prompt_tokens: int) -> None:
        self.prompt_tokens = prompt_tokens

    def apply_chat_template(self, messages, tokenize, add_generation_prompt):
        assert tokenize is True
        assert add_generation_prompt is True
        assert len(messages) == 2
        return {"input_ids": [0] * self.prompt_tokens}


def test_solver_rejects_blank_model_name() -> None:
    with pytest.raises(ValueError, match="SQL_SOLVER_MODEL no puede estar vacio"):
        require_solver_model_name("   ")


def test_prompt_preflight_accepts_prompt_with_reserved_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "nl2sql.sql_solver_generator.runtime.load_solver_tokenizer",
        lambda _model_name: _FakeTokenizer(prompt_tokens=7700),
    )

    prompt_tokens = preflight_prompt_size(
        model_name="fake-model",
        system_prompt="system",
        user_prompt="user",
        max_model_len=8192,
        max_tokens=384,
    )

    assert prompt_tokens == 7700


def test_prompt_preflight_raises_when_prompt_exceeds_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "nl2sql.sql_solver_generator.runtime.load_solver_tokenizer",
        lambda _model_name: _FakeTokenizer(prompt_tokens=7745),
    )

    with pytest.raises(PromptTooLongError, match="max_model_len=8192"):
        preflight_prompt_size(
            model_name="fake-model",
            system_prompt="system",
            user_prompt="user",
            max_model_len=8192,
            max_tokens=384,
        )


def test_gpu_utilization_se_ajusta_a_la_vram_libre_real() -> None:
    adjusted = resolve_gpu_utilization_from_available_memory(
        0.90,
        free_memory_gib=10.28,
        total_memory_gib=15.61,
    )

    assert adjusted == pytest.approx((10.28 - 0.25) / 15.61, rel=1e-4)


def test_retry_runtime_interpreta_error_de_vram_de_vllm() -> None:
    retry = resolve_solver_runtime_retry(
        ValueError(
            "Free memory on device cuda:0 (10.28/15.61 GiB) on startup is less than desired GPU memory utilization (0.9, 14.05 GiB). Decrease GPU memory utilization or reduce GPU memory used by other processes."
        ),
        current_gpu_memory_utilization=0.90,
        enforce_eager=False,
        cpu_offload_gb=2.0,
    )

    assert retry is not None
    gpu_memory_utilization, enforce_eager, cpu_offload_gb = retry
    assert gpu_memory_utilization < 0.90
    assert enforce_eager is True
    assert cpu_offload_gb == 3.0


def test_runtime_inicial_permite_override_del_piso_minimo_de_offload(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SQL_SOLVER_MIN_CPU_OFFLOAD_GB", "0.0")
    monkeypatch.setattr("nl2sql.sql_solver_generator.runtime.torch.cuda.is_available", lambda: False)

    gpu_memory_utilization, enforce_eager, cpu_offload_gb = resolve_initial_solver_runtime_settings(
        gpu_memory_utilization=0.90,
        enforce_eager=True,
        cpu_offload_gb=0.0,
    )

    assert gpu_memory_utilization == pytest.approx(0.90)
    assert enforce_eager is True
    assert cpu_offload_gb == pytest.approx(0.0)


def test_solver_prompt_variant_reduce_few_shots_si_preflight_falla(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured_prompts: list[str] = []

    def _fake_preflight(_model_name, _system_prompt, user_prompt, _max_model_len, _max_tokens):
        captured_prompts.append(user_prompt)
        if len(captured_prompts) == 1:
            raise PromptTooLongError("prompt demasiado largo")
        return 1400

    monkeypatch.setattr(
        "nl2sql.sql_solver_generator.sql_generator.preflight_prompt_size",
        _fake_preflight,
    )

    system_prompt, user_prompt, diagnostics = select_generation_prompt_variant(
        semantic_plan=build_semantic_plan_mapping_active_a_per_c(),
        pruned_schema=generic_schema_tables(),
        semantic_rules=generic_semantic_contract(),
        business_rules_summary="- rule: none",
        prompts={
            "spec_generation": {
                "system_prompt": "dialecto=$dialect contrato=$generation_contract_json",
                "user_prompt_template": "plan=$semantic_plan_yaml\nesquema=$pruned_schema_yaml\nshape=$sql_shape_yaml\nexamples=$few_shot_examples_yaml\nreglas=$business_rules_summary",
            }
        },
        dialect_name="tsql",
        model_name="fake-model",
        max_model_len=2048,
        max_tokens=384,
    )

    assert system_prompt
    assert user_prompt
    assert diagnostics["prompt_variant"] == "rich_with_one_example"
    assert diagnostics["prompt_tokens"] == 1400
    assert len(captured_prompts) == 2
    assert captured_prompts[0] != captured_prompts[1]
