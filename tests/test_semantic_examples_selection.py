#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path

from nl2sql.utils.semantic_contract import SemanticContract
from nl2sql.utils.semantic_examples import select_relevant_semantic_examples


def _contract_with_examples() -> SemanticContract:
    payload = {
        "semantic_contract": {
            "business_invariants": {},
            "retrieval_heuristics": {
                "semantic_examples": [
                    {"question": "clientes activos", "metrics": ["metric_a"]},
                    {"question": "clientes", "metrics": ["metric_b"]},
                ]
            },
            "sql_safety": {},
        }
    }
    return SemanticContract.model_validate(payload["semantic_contract"])


def test_select_relevant_semantic_examples_usa_bonos_desde_config_unificada(tmp_path: Path) -> None:
    config_path = tmp_path / "nl2sql_config.yaml"
    config_path.write_text(
        "semantic_resolver:\n"
        "  compiler_rules:\n"
        "    semantic_example_selection:\n"
        "      exact_match_bonus: 0.0\n"
        "      containment_bonus: 10.0\n",
        encoding="utf-8",
    )

    selected = select_relevant_semantic_examples(
        _contract_with_examples(),
        "clientes activos premium",
        limit=1,
        config_path=config_path,
    )

    assert selected[0]["question"] == "clientes activos"
