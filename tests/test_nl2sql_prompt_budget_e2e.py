#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from types import SimpleNamespace

import pytest

from llm_core.prompt_optimizer import ModelTokenizerSpec
from nl2sql.orchestrator import NL2SQLConfig
from nl2sql.orchestrator.stages.narrative_stage import select_narrative_prompt_variant
from nl2sql.semantic_resolver.verification import (
    load_verification_rules,
    select_verifier_prompt_variant,
)
from nl2sql.sql_solver_generator.sql_generator import (
    load_generation_tuning_rules,
    select_generation_prompt_variant,
)
from tests.generic_domain import (
    build_compiled_plan_active_a_per_c,
    build_semantic_plan_mapping_active_a_per_c,
    generic_schema_tables,
    generic_semantic_contract,
)


class _WordTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[str]:
        tokens = [token for token in str(text).replace("\n", " ").split(" ") if token]
        if add_special_tokens:
            return ["<bos>", *tokens, "<eos>"]
        return tokens

    def apply_chat_template(self, messages, tokenize: bool, add_generation_prompt: bool):
        assert tokenize is True
        tokens: list[str] = []
        for message in messages:
            tokens.extend(self.encode(message["content"]))
        if add_generation_prompt:
            tokens.append("<assistant>")
        return {"input_ids": tokens}


def _large_rows() -> list[dict[str, object]]:
    return [
        {
            "entity_c": f"Entity C {index}",
            "avg_value": float(index),
            "comment": "resultado extendido para inflar el preview narrativo y verificar degradacion segura del prompt",
            "segment": "segment_alpha",
            "region": "north_zone",
        }
        for index in range(1, 26)
    ]


def test_large_artifacts_fit_safe_context_budgets_across_llm_stages(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenizer = _WordTokenizer()
    semantic_rules = generic_semantic_contract()
    compiled_plan = build_compiled_plan_active_a_per_c()
    semantic_plan = build_semantic_plan_mapping_active_a_per_c()
    pruned_schema = generic_schema_tables()

    monkeypatch.setattr(
        "nl2sql.semantic_resolver.verification._load_verifier_tokenizer",
        lambda *_args, **_kwargs: tokenizer,
    )
    monkeypatch.setattr(
        "nl2sql.sql_solver_generator.runtime.load_solver_tokenizer",
        lambda *_args, **_kwargs: tokenizer,
    )
    monkeypatch.setattr("llm_core.prompt_optimizer._load_tokenizer", lambda *_args, **_kwargs: tokenizer)
    monkeypatch.setattr(
        "llm_core.prompt_optimizer.resolve_model_tokenizer_spec",
        lambda *_args, **_kwargs: ModelTokenizerSpec(
            model_name="fake-narrative-model",
            tokenizer_name="fake-narrative-tokenizer",
            revision=None,
            tokenizer_revision=None,
            trust_remote_code=True,
            tokenizer_mode="auto",
            max_model_len=512,
        ),
    )

    verifier_config = SimpleNamespace(
        verifier_system_prompt="contrato={verification_contract_json}",
        verifier_user_prompt_template="pregunta={query}\nplan={compiled_plan_yaml}\nesquema={pruned_schema_yaml}\nejemplos={few_shot_examples_yaml}",
        verifier_few_shot_limit=3,
        verifier_model="fake-verifier",
        verifier_max_model_len=384,
        verifier_max_tokens=32,
    )
    _verifier_system, _verifier_user, verifier_diagnostics = select_verifier_prompt_variant(
        query=compiled_plan.query,
        compiled_plan=compiled_plan,
        pruned_schema=pruned_schema,
        semantic_rules=semantic_rules,
        config=verifier_config,
    )
    verifier_rules = load_verification_rules()
    assert (
        verifier_diagnostics["prompt_tokens"] + verifier_config.verifier_max_tokens + verifier_rules.prompt_token_safety_margin
        <= verifier_config.verifier_max_model_len
    )

    _solver_system, _solver_user, solver_diagnostics = select_generation_prompt_variant(
        semantic_plan=semantic_plan,
        pruned_schema=pruned_schema,
        semantic_rules=semantic_rules,
        business_rules_summary="- no_mixed_amounts: guardrail\n- use_declared_join_path: canonical_path",
        prompts={
            "spec_generation": {
                "system_prompt": "dialecto=$dialect contrato=$generation_contract_json",
                "user_prompt_template": "plan=$semantic_plan_yaml\nesquema=$pruned_schema_yaml\nshape=$sql_shape_yaml\nreglas=$business_rules_summary\nejemplos=$few_shot_examples_yaml",
            }
        },
        dialect_name="tsql",
        model_name="fake-solver",
        max_model_len=1024,
        max_tokens=64,
    )
    solver_rules = load_generation_tuning_rules()
    assert solver_diagnostics["prompt_tokens"] + 64 + solver_rules.prompt_token_safety_margin <= 1024

    _narrative_system, _narrative_user, narrative_diagnostics = select_narrative_prompt_variant(
        prompts={
            "system": "eres analista del dominio generico y resumes resultados sin inventar metricas",
            "user_template": "Pregunta: {query}\nSQL ejecutado: {sql}\nResultado ({row_count} filas{truncated_marker}):\n{rows_preview}\nRedacta la respuesta final.",
        },
        query=compiled_plan.query,
        sql="SELECT entity_c, AVG(metric_value) AS avg_value FROM giant_fact_table GROUP BY entity_c ORDER BY entity_c, avg_value",
        row_count=25,
        truncated=False,
        rows=_large_rows(),
        config=NL2SQLConfig(rows_preview_limit=25),
    )
    prompt_stats = narrative_diagnostics["prompt_stats"]
    assert prompt_stats["final_prompt_tokens"] <= prompt_stats["max_model_len"] - prompt_stats["safety_margin_tokens"]
