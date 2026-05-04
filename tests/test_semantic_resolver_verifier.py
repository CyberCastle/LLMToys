#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import sys
from types import ModuleType
from types import SimpleNamespace

from nl2sql.semantic_resolver.verification import (
    SemanticVerificationResult,
    VerifierPromptTooLongError,
    classify_semantic_verification,
    run_local_verifier_chat,
    validate_verification_payload_json,
    verify_compiled_plan,
)
from tests.generic_domain import (
    build_compiled_plan_active_a_per_c,
    generic_semantic_contract,
)


def test_validate_verification_payload_json_rechaza_payload_invalido() -> None:
    try:
        validate_verification_payload_json('{"is_semantically_aligned": true, "extra": 1}')
    except ValueError as exc:
        assert "contrato JSON" in str(exc)
    else:
        raise AssertionError("Se esperaba ValueError para payload invalido del verificador")


def test_run_local_verifier_chat_reenvia_enforce_eager_a_vllm(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class _FakeSamplingParams:
        def __init__(self, **kwargs) -> None:
            captured["sampling_kwargs"] = kwargs

    class _FakeLLM:
        def __init__(self, **kwargs) -> None:
            captured["llm_kwargs"] = kwargs

        def chat(self, messages, sampling_params):
            captured["messages"] = messages
            captured["sampling_params"] = sampling_params
            return [
                SimpleNamespace(
                    outputs=[
                        SimpleNamespace(
                            text='{"is_semantically_aligned": true}',
                            token_ids=[1, 2, 3],
                            finish_reason="stop",
                        )
                    ]
                )
            ]

    fake_vllm_module = ModuleType("vllm")
    fake_vllm_module.LLM = _FakeLLM
    fake_vllm_module.SamplingParams = _FakeSamplingParams

    monkeypatch.setitem(sys.modules, "vllm", fake_vllm_module)
    monkeypatch.setattr("nl2sql.semantic_resolver.verification.release_local_llm", lambda _llm: None)

    raw_text, diagnostics = run_local_verifier_chat(
        model_name="fake-verifier",
        system_prompt="sistema",
        user_prompt="usuario",
        temperature=0.0,
        max_model_len=2048,
        max_tokens=128,
        dtype="auto",
        gpu_memory_utilization=0.82,
        enforce_eager=False,
        cpu_offload_gb=3.0,
    )

    assert raw_text == '{"is_semantically_aligned": true}'
    assert diagnostics["finish_reason"] == "stop"
    assert diagnostics["generated_tokens"] == 3
    assert captured["llm_kwargs"]["enforce_eager"] is False
    assert captured["llm_kwargs"]["cpu_offload_gb"] == 3.0
    assert captured["messages"] == [
        {"role": "system", "content": "sistema"},
        {"role": "user", "content": "usuario"},
    ]


def test_verify_compiled_plan_inyecta_ejemplos_curados_en_prompt(monkeypatch) -> None:
    captured: dict[str, str] = {}

    def fake_run_local_verifier_chat(**kwargs):
        captured["system_prompt"] = kwargs["system_prompt"]
        captured["user_prompt"] = kwargs["user_prompt"]
        captured["dtype"] = kwargs["dtype"]
        return (
            '{"is_semantically_aligned": false, "missing_filters": ["status_a.name = Active"], "confidence": 0.35, "rationale": "Falta el filtro de estado"}',
            {
                "finish_reason": "stop",
                "generated_tokens": 42,
                "wall_time_seconds": 0.2,
                "model_name": "fake-verifier",
            },
        )

    monkeypatch.setattr(
        "nl2sql.semantic_resolver.verification.run_local_verifier_chat",
        fake_run_local_verifier_chat,
    )
    monkeypatch.setattr(
        "nl2sql.semantic_resolver.verification._preflight_verifier_prompt_size",
        lambda *_args, **_kwargs: 182,
    )

    config = SimpleNamespace(
        verifier_system_prompt="contrato={verification_contract_json}",
        verifier_user_prompt_template="pregunta={query}\nplan={compiled_plan_yaml}\nejemplos={few_shot_examples_yaml}",
        verifier_few_shot_limit=2,
        verifier_model="fake-verifier",
        verifier_dtype="float16",
        verifier_temperature=0.0,
        verifier_max_model_len=1024,
        verifier_max_tokens=128,
        verifier_gpu_memory_utilization=0.5,
        verifier_enforce_eager=True,
        verifier_cpu_offload_gb=0.0,
    )

    compiled_plan = build_compiled_plan_active_a_per_c()
    verification, diagnostics = verify_compiled_plan(
        query=compiled_plan.query,
        compiled_plan=compiled_plan,
        pruned_schema={"entity_a": {"columns": [{"name": "id"}, {"name": "status_a_id"}]}},
        semantic_rules=generic_semantic_contract(),
        config=config,
    )

    assert verification.is_semantically_aligned is False
    assert verification.missing_filters == ["status_a.name = Active"]
    assert diagnostics["model_name"] == "fake-verifier"
    assert diagnostics["prompt_variant"] == "rich_with_examples"
    assert diagnostics["prompt_tokens"] == 182
    assert "promedio de entidades_a con estado activo por entidad_c" in captured["user_prompt"]
    assert "metric_count_a_active" in captured["user_prompt"]
    assert "metric_score_trace" not in captured["user_prompt"]
    assert captured["dtype"] == "float16"


def test_verify_compiled_plan_reduce_prompt_cuando_preflight_excede_contexto(
    monkeypatch,
) -> None:
    captured_prompts: list[str] = []

    def _fake_preflight(_model_name, _system_prompt, user_prompt, _max_model_len, _max_tokens, **_kwargs):
        captured_prompts.append(user_prompt)
        if len(captured_prompts) == 1:
            raise VerifierPromptTooLongError("prompt demasiado largo")
        return 121

    def fake_run_local_verifier_chat(**kwargs):
        return (
            '{"is_semantically_aligned": true, "missing_filters": [], "confidence": 0.91, "rationale": "ok"}',
            {
                "finish_reason": "stop",
                "generated_tokens": 18,
                "wall_time_seconds": 0.1,
                "model_name": "fake-verifier",
            },
        )

    monkeypatch.setattr(
        "nl2sql.semantic_resolver.verification._preflight_verifier_prompt_size",
        _fake_preflight,
    )
    monkeypatch.setattr(
        "nl2sql.semantic_resolver.verification.run_local_verifier_chat",
        fake_run_local_verifier_chat,
    )

    config = SimpleNamespace(
        verifier_system_prompt="contrato={verification_contract_json}",
        verifier_user_prompt_template="pregunta={query}\nplan={compiled_plan_yaml}\nesquema={pruned_schema_yaml}\nejemplos={few_shot_examples_yaml}",
        verifier_few_shot_limit=2,
        verifier_model="fake-verifier",
        verifier_dtype="auto",
        verifier_temperature=0.0,
        verifier_max_model_len=256,
        verifier_max_tokens=64,
        verifier_gpu_memory_utilization=0.5,
        verifier_enforce_eager=True,
        verifier_cpu_offload_gb=0.0,
    )

    compiled_plan = build_compiled_plan_active_a_per_c()
    verification, diagnostics = verify_compiled_plan(
        query=compiled_plan.query,
        compiled_plan=compiled_plan,
        pruned_schema={
            "entity_a": {"columns": [{"name": "id"}, {"name": "status_a_id"}]},
            "entity_c": {"columns": [{"name": "id"}, {"name": "display_name"}]},
        },
        semantic_rules=generic_semantic_contract(),
        config=config,
    )

    assert verification.is_semantically_aligned is True
    assert diagnostics["prompt_variant"] == "rich_without_examples"
    assert diagnostics["prompt_tokens"] == 121
    assert len(captured_prompts) == 2
    assert "metric_count_a_active" in captured_prompts[0]
    assert captured_prompts[0] != captured_prompts[1]
    assert "ejemplos=[]" in captured_prompts[1]


def test_classify_semantic_verification_warns_for_recoverable_mismatch() -> None:
    issue = classify_semantic_verification(
        SemanticVerificationResult(
            is_semantically_aligned=False,
            failure_class="recoverable_semantic_mismatch",
            repairability="high",
            wrong_metric="metric_count_c",
            suggested_measure="metric_count_a",
            confidence=0.42,
        )
    )

    assert issue is not None
    assert issue.severity == "warning"
    assert issue.code == "semantic_verification_failed"


def test_classify_semantic_verification_blocks_non_recoverable_failure() -> None:
    issue = classify_semantic_verification(
        SemanticVerificationResult(
            is_semantically_aligned=False,
            failure_class="insufficient_context",
            repairability="none",
            blocking_reason="faltan activos minimos",
            confidence=0.12,
        )
    )

    assert issue is not None
    assert issue.severity == "error"
    assert issue.message == "faltan activos minimos"
