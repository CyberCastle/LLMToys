#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import pytest

from nl2sql.sql_solver_generator.sql_generator import build_sampling_kwargs, load_generation_tuning_rules
from nl2sql.sql_solver_generator.prompt_contract import validate_generation_payload_json


class _FakeTokenizer:
    eos_token_id = 42


def test_sampling_kwargs_include_end_marker_and_eos_stop() -> None:
    kwargs = build_sampling_kwargs(tokenizer=_FakeTokenizer(), temperature=0.0, max_tokens=384)
    tuning_rules = load_generation_tuning_rules()

    assert kwargs["stop"] == [tuning_rules.end_of_output_marker]
    assert kwargs["stop_token_ids"] == [42]
    assert kwargs["max_tokens"] == 384
    assert kwargs["skip_special_tokens"] is True


def test_prompt_contract_accepts_only_final_sql_output() -> None:
    payload = validate_generation_payload_json('{"final_sql": "SELECT 1"}')

    assert payload.final_sql == "SELECT 1"


def test_prompt_contract_rejects_legacy_spec_and_metadata_sections() -> None:
    with pytest.raises(ValueError, match="contrato JSON"):
        validate_generation_payload_json(
            '{"SQLQuerySpec": {"query_type": "scalar_metric", "base_entity": "entity_c", "base_table": "entity_c"}, "final_sql": "SELECT 1", "metadata": {}}'
        )


def test_json_parser_accepts_final_sql_string() -> None:
    payload = validate_generation_payload_json('{"final_sql": "SELECT 1"}')

    assert payload.final_sql == "SELECT 1"
