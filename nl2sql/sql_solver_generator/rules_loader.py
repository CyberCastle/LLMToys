#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from nl2sql.config import (
    SolverFilterValueRules,
    load_sql_solver_filter_value_rules,
    load_sql_solver_prompt_rules,
)
from nl2sql.utils.decision_models import StrictModel
from nl2sql.utils.semantic_contract import SemanticContract, load_semantic_contract

from .business_rules import BusinessRule


class SolverRules(StrictModel):
    """Reglas de seguridad SQL derivadas del contrato semantico."""

    forbidden_keywords: frozenset[str]
    hard_join_blacklist: tuple["HardJoinRule", ...] = ()


class HardJoinRule(StrictModel):
    """Restriccion declarativa para joins directos no autorizados."""

    table: str
    reason: str = ""


def load_solver_rules(source: str | Path | Mapping[str, Any] | SemanticContract) -> SolverRules:
    contract = load_semantic_contract(source)
    safety = contract.sql_safety.execution_safety
    hard_join_blacklist = []
    for raw_item in contract.sql_safety.hard_join_blacklist:
        if not isinstance(raw_item, Mapping):
            continue
        table_name = raw_item.get("table")
        if not isinstance(table_name, str) or not table_name.strip():
            continue
        reason = raw_item.get("reason")
        hard_join_blacklist.append(
            HardJoinRule(
                table=table_name.strip(),
                reason=str(reason).strip() if isinstance(reason, str) else "",
            )
        )
    return SolverRules(
        forbidden_keywords=frozenset(str(keyword).upper() for keyword in safety.get("forbidden_keywords", [])),
        hard_join_blacklist=tuple(hard_join_blacklist),
    )


def load_solver_prompts(path: str) -> dict:
    prompts = load_sql_solver_prompt_rules(Path(path).expanduser().resolve())
    return prompts.spec_generation.model_dump(mode="python")


def load_filter_value_rules(path: str) -> SolverFilterValueRules:
    """Devuelve las reglas lexicas del solver ya validadas en `nl2sql.config`."""

    return load_sql_solver_filter_value_rules(Path(path).expanduser().resolve())


def load_business_rules(source: str | Path | Mapping[str, Any] | SemanticContract) -> list[BusinessRule]:
    """Carga reglas SQL declarativas desde `semantic_contract.sql_safety`."""

    contract = load_semantic_contract(source)
    raw_items = contract.sql_safety.semantic_sql_business_rules

    rules: list[BusinessRule] = []
    for item in raw_items:
        rules.append(
            BusinessRule(
                id=str(item.get("id", "rule")),
                type=str(item.get("type", "unknown")),
                params={key: value for key, value in item.items() if key not in {"id", "type"}},
            )
        )
    return rules
