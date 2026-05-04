#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path
import time
from typing import Any, Mapping

from pydantic import Field, ValidationError
import yaml

from llm_core.tokenizer_utils import load_tokenizer as _load_verifier_tokenizer
from nl2sql.config import (
    SemanticResolverVerificationRules,
    load_semantic_resolver_verification_rules,
    resolve_nl2sql_config_path,
)
from nl2sql.utils.decision_models import StrictModel
from nl2sql.utils.prompt_budget import (
    PromptTooLongError as SharedPromptTooLongError,
    assert_prompt_fits,
)
from nl2sql.utils.semantic_contract import SemanticContract
from nl2sql.utils.vllm_runtime import build_local_llm, release_local_llm
from nl2sql.utils.semantic_examples import (
    compact_semantic_examples_for_prompt,
    render_semantic_examples_for_prompt,
    select_relevant_semantic_examples,
)
from nl2sql.utils.sql_identifiers import JOIN_EDGE_RE, TABLE_COLUMN_RE
from nl2sql.utils.text_utils import truncate_text

from .plan_model import CompiledSemanticPlan


def _resolve_verification_rules_path(config: object | None = None) -> Path:
    if config is not None:
        raw_path = getattr(config, "prompts_path", None) or getattr(config, "compiler_rules_path", None)
        if raw_path is not None:
            return Path(raw_path).expanduser().resolve()
    return resolve_nl2sql_config_path()


@lru_cache(maxsize=8)
def load_verification_rules(
    path: str | Path | None = None,
) -> SemanticResolverVerificationRules:
    """Devuelve las reglas tipadas del verificador semantico."""

    resolved_path = Path(path).expanduser().resolve() if path is not None else _resolve_verification_rules_path()
    return load_semantic_resolver_verification_rules(resolved_path)


class SemanticVerificationResult(StrictModel):
    """Revision estructurada de alineacion entre pregunta y plan compilado."""

    is_semantically_aligned: bool
    failure_class: str | None = None
    repairability: str = "none"
    missing_filters: list[str] = Field(default_factory=list)
    wrong_metric: str | None = None
    suggested_measure: str | None = None
    suggested_join_tables: list[str] = Field(default_factory=list)
    suggested_plan_delta: dict[str, Any] = Field(default_factory=dict)
    blocking_reason: str | None = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    rationale: str = Field(default="", max_length=800)


class VerifierPromptTooLongError(SharedPromptTooLongError):
    """Error de preflight cuando el prompt del verificador no cabe en contexto."""


def render_verification_contract_json() -> str:
    """Expone el esquema JSON minimo que el verificador debe respetar."""

    compact_schema = {
        "type": "object",
        "required": ["is_semantically_aligned"],
        "properties": {
            "is_semantically_aligned": {"type": "boolean"},
            "failure_class": {"type": ["string", "null"]},
            "repairability": {"type": "string"},
            "missing_filters": {"type": "array", "items": {"type": "string"}},
            "wrong_metric": {"type": ["string", "null"]},
            "suggested_measure": {"type": ["string", "null"]},
            "suggested_join_tables": {"type": "array", "items": {"type": "string"}},
            "suggested_plan_delta": {"type": "object"},
            "blocking_reason": {"type": ["string", "null"]},
            "confidence": {"type": "number"},
            "rationale": {"type": "string"},
        },
        "additionalProperties": False,
    }
    return json.dumps(compact_schema, ensure_ascii=False, indent=2)


def validate_verification_payload_json(raw_text: str) -> SemanticVerificationResult:
    """Valida el JSON emitido por el verificador local."""

    candidate = raw_text.strip()
    if not candidate:
        raise ValueError("La salida del verificador semantico esta vacia")
    try:
        return SemanticVerificationResult.model_validate_json(candidate)
    except ValidationError as exc:
        raise ValueError("La salida del verificador semantico no cumple el contrato JSON esperado") from exc


def classify_semantic_verification(result: SemanticVerificationResult):
    """Convierte la verificacion semantica en una decision compartida del pipeline."""

    from nl2sql.utils.decision_models import DecisionIssue

    if result.is_semantically_aligned:
        return None

    rules = load_verification_rules()
    repairability = result.repairability.strip().lower()
    failure_class = (result.failure_class or "").strip().lower()
    blocking = failure_class in rules.non_recoverable_failure_classes or repairability not in rules.recoverable_repairabilities
    return DecisionIssue(
        stage="semantic_verification",
        code="semantic_verification_failed",
        severity="error" if blocking else "warning",
        message=(
            result.blocking_reason.strip()
            if isinstance(result.blocking_reason, str) and result.blocking_reason.strip()
            else "El plan semantico no esta alineado con la pregunta original."
        ),
        context=result.model_dump(mode="python"),
    )


def _preflight_verifier_prompt_size(
    model_name: str,
    system_prompt: str,
    user_prompt: str,
    max_model_len: int,
    max_tokens: int,
    rules: SemanticResolverVerificationRules | None = None,
) -> int:
    active_rules = rules or load_verification_rules()
    tokenizer = _load_verifier_tokenizer(model_name)
    try:
        return assert_prompt_fits(
            tokenizer=tokenizer,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            max_model_len=max_model_len,
            max_tokens=max_tokens,
            safety_margin=active_rules.prompt_token_safety_margin,
            overflow_message=(
                "prompt_tokens={prompt_tokens} + max_tokens="
                f"{max_tokens} + safety={active_rules.prompt_token_safety_margin} > max_model_len={max_model_len}; "
                "reduce el payload del verificador."
            ),
            fallback_builder=lambda active_tokenizer, system_text, user_text: active_tokenizer(
                f"{system_text}\n\n{user_text}",
                return_tensors=None,
            ).get("input_ids", []),
            error_message="No se pudo contar los tokens del prompt del verificador semantico",
        )
    except SharedPromptTooLongError as exc:
        raise VerifierPromptTooLongError(str(exc)) from exc


def _extract_compiled_plan_hints(
    compiled_plan: CompiledSemanticPlan,
) -> tuple[list[str], dict[str, set[str]]]:
    relevant_tables: list[str] = []
    relevant_columns: dict[str, set[str]] = {}

    def _add_table(table_name: str | None) -> None:
        if not isinstance(table_name, str) or not table_name.strip():
            return
        normalized = table_name.strip()
        if normalized not in relevant_tables:
            relevant_tables.append(normalized)

    def _add_column_ref(column_ref: str | None) -> None:
        if not isinstance(column_ref, str) or "." not in column_ref:
            return
        table_name, column_name = column_ref.split(".", 1)
        if not table_name or not column_name:
            return
        _add_table(table_name)
        relevant_columns.setdefault(table_name, set()).add(column_name)

    _add_table(compiled_plan.base_entity)
    _add_column_ref(compiled_plan.grain)

    if compiled_plan.measure is not None:
        _add_table(compiled_plan.measure.source_table)
        for table_name, column_name in TABLE_COLUMN_RE.findall(compiled_plan.measure.formula):
            _add_table(table_name)
            relevant_columns.setdefault(table_name, set()).add(column_name)

    for group_field in compiled_plan.group_by:
        _add_column_ref(group_field)
    for group_field in compiled_plan.base_group_by:
        _add_column_ref(group_field)
    for selected_filter in compiled_plan.selected_filters:
        _add_column_ref(selected_filter.field)
    if compiled_plan.time_filter is not None:
        _add_column_ref(compiled_plan.time_filter.field)

    for join_expression in compiled_plan.join_path:
        match = JOIN_EDGE_RE.match(join_expression)
        if match is None:
            continue
        left_table = match.group("left_table")
        right_table = match.group("right_table")
        _add_table(left_table)
        _add_table(right_table)
        relevant_columns.setdefault(left_table, set()).add(match.group("left_column"))
        relevant_columns.setdefault(right_table, set()).add(match.group("right_column"))

    for table_name in compiled_plan.required_tables:
        _add_table(table_name)

    return relevant_tables, relevant_columns


def _compact_compiled_plan_for_verifier(
    compiled_plan: CompiledSemanticPlan,
    *,
    include_warnings: bool,
    minimal: bool,
    rules: SemanticResolverVerificationRules,
) -> dict[str, Any]:
    compact_plan: dict[str, Any] = {
        "query": compiled_plan.query,
        "semantic_model": compiled_plan.semantic_model,
        "intent": compiled_plan.intent,
        "base_entity": compiled_plan.base_entity,
        "grain": compiled_plan.grain,
        "group_by": compiled_plan.group_by,
        "required_tables": compiled_plan.required_tables,
    }

    if compiled_plan.measure is not None:
        compact_plan["measure"] = {
            "name": compiled_plan.measure.name,
            "source_table": compiled_plan.measure.source_table,
            "formula": truncate_text(
                compiled_plan.measure.formula,
                (rules.measure_formula_chars_minimal if minimal else rules.measure_formula_chars_rich),
            ),
        }
    if compiled_plan.time_filter is not None:
        compact_plan["time_filter"] = {
            "field": compiled_plan.time_filter.field,
            "operator": compiled_plan.time_filter.operator,
            "value": compiled_plan.time_filter.value,
        }
    if compiled_plan.selected_filters:
        compact_plan["selected_filters"] = [
            {
                "name": selected_filter.name,
                "field": selected_filter.field,
                "operator": selected_filter.operator,
                "value": selected_filter.value,
            }
            for selected_filter in compiled_plan.selected_filters
        ]
    if compiled_plan.post_aggregation is not None:
        compact_plan["post_aggregation"] = {
            "function": compiled_plan.post_aggregation.function,
            "over": compiled_plan.post_aggregation.over,
        }
    if compiled_plan.join_path:
        compact_plan["join_path"] = compiled_plan.join_path[: rules.max_join_edges]
    if compiled_plan.join_path_hint and not minimal:
        compact_plan["join_path_hint"] = compiled_plan.join_path_hint
    if compiled_plan.derived_metric_ref and not minimal:
        compact_plan["derived_metric_ref"] = compiled_plan.derived_metric_ref
    if compiled_plan.population_scope and not minimal:
        compact_plan["population_scope"] = compiled_plan.population_scope
    if compiled_plan.base_group_by:
        compact_plan["base_group_by"] = compiled_plan.base_group_by
    if compiled_plan.intermediate_alias and not minimal:
        compact_plan["intermediate_alias"] = compiled_plan.intermediate_alias
    if include_warnings and compiled_plan.warnings:
        compact_plan["warnings"] = compiled_plan.warnings[: rules.max_warnings]
    return compact_plan


def _extract_schema_columns(raw_table: Mapping[str, Any]) -> list[str]:
    raw_columns = raw_table.get("columns")
    if isinstance(raw_columns, list):
        return [str(column.get("name")) for column in raw_columns if isinstance(column, Mapping) and column.get("name")]
    if isinstance(raw_columns, Mapping):
        return [str(column_name) for column_name in raw_columns]
    return []


def _compact_pruned_schema_for_verifier(
    pruned_schema: Mapping[str, Any],
    compiled_plan: CompiledSemanticPlan,
    *,
    tight: bool,
    minimal: bool,
    rules: SemanticResolverVerificationRules,
) -> dict[str, Any]:
    compact_schema: dict[str, Any] = {}
    relevant_tables, relevant_columns = _extract_compiled_plan_hints(compiled_plan)
    table_candidates = relevant_tables or [str(table_name) for table_name in pruned_schema]

    if minimal:
        max_tables = rules.max_tables_minimal
        max_columns = rules.max_columns_per_table_minimal
    elif tight:
        max_tables = rules.max_tables_tight
        max_columns = rules.max_columns_per_table_tight
    else:
        max_tables = rules.max_tables_rich
        max_columns = rules.max_columns_per_table_rich

    for table_name in table_candidates[:max_tables]:
        raw_table = pruned_schema.get(table_name)
        if not isinstance(raw_table, Mapping):
            continue

        raw_column_names = _extract_schema_columns(raw_table)
        priority_columns = [column for column in raw_column_names if column in relevant_columns.get(table_name, set())]
        fallback_columns = [column for column in raw_column_names if column not in priority_columns]
        selected_columns = (priority_columns + fallback_columns)[:max_columns]

        payload: dict[str, Any] = {"columns": selected_columns}
        if not tight and not minimal:
            serialized_fks: list[str] = []
            for raw_fk in raw_table.get("foreign_keys", []) or []:
                if not isinstance(raw_fk, Mapping):
                    continue
                col = str(raw_fk.get("col") or "")
                ref_table = str(raw_fk.get("ref_table") or "")
                ref_col = str(raw_fk.get("ref_col") or "")
                if not col or not ref_table or not ref_col:
                    continue
                if col in selected_columns or ref_table in table_candidates:
                    serialized_fks.append(f"{col} -> {ref_table}.{ref_col}")
            if serialized_fks:
                payload["foreign_keys"] = serialized_fks[: rules.schema_foreign_key_limit_rich]
        compact_schema[str(table_name)] = payload
    return compact_schema


def _render_verifier_prompts(
    *,
    query: str,
    compiled_plan: CompiledSemanticPlan,
    pruned_schema: Mapping[str, Any],
    semantic_rules: SemanticContract,
    config,
    include_examples: bool,
    tight_schema: bool,
    include_warnings: bool,
    minimal_plan: bool,
) -> tuple[str, str]:
    rules = getattr(config, "verification_rules", None) or load_verification_rules(_resolve_verification_rules_path(config))
    selected_examples: list[dict[str, Any]] = []
    if include_examples:
        selected_examples = compact_semantic_examples_for_prompt(
            select_relevant_semantic_examples(
                semantic_rules,
                query,
                limit=max(0, int(config.verifier_few_shot_limit)),
                selection_rules=getattr(
                    getattr(config, "compiler_rules", None),
                    "semantic_example_selection",
                    None,
                ),
            ),
            question_chars=rules.example_question_chars,
            metric_limit=rules.example_metric_limit,
            dimension_limit=rules.example_dimension_limit,
        )

    system_prompt = config.verifier_system_prompt.format(
        verification_contract_json=render_verification_contract_json(),
    )
    user_prompt = config.verifier_user_prompt_template.format(
        query=query,
        compiled_plan_yaml=yaml.safe_dump(
            _compact_compiled_plan_for_verifier(
                compiled_plan,
                include_warnings=include_warnings,
                minimal=minimal_plan,
                rules=rules,
            ),
            sort_keys=False,
            allow_unicode=True,
        ),
        pruned_schema_yaml=yaml.safe_dump(
            _compact_pruned_schema_for_verifier(
                pruned_schema,
                compiled_plan,
                tight=tight_schema,
                minimal=minimal_plan,
                rules=rules,
            ),
            sort_keys=False,
            allow_unicode=True,
        ),
        few_shot_examples_yaml=render_semantic_examples_for_prompt(selected_examples),
    )
    return system_prompt, user_prompt


def select_verifier_prompt_variant(
    *,
    query: str,
    compiled_plan: CompiledSemanticPlan,
    pruned_schema: Mapping[str, Any],
    semantic_rules: SemanticContract,
    config,
) -> tuple[str, str, dict[str, Any]]:
    rules = getattr(config, "verification_rules", None) or load_verification_rules(_resolve_verification_rules_path(config))
    variants = (
        {
            "name": "rich_with_examples",
            "include_examples": True,
            "tight_schema": False,
            "include_warnings": True,
            "minimal_plan": False,
        },
        {
            "name": "rich_without_examples",
            "include_examples": False,
            "tight_schema": False,
            "include_warnings": True,
            "minimal_plan": False,
        },
        {
            "name": "tight_without_examples",
            "include_examples": False,
            "tight_schema": True,
            "include_warnings": False,
            "minimal_plan": False,
        },
        {
            "name": "minimal_without_examples",
            "include_examples": False,
            "tight_schema": True,
            "include_warnings": False,
            "minimal_plan": True,
        },
    )
    last_error: VerifierPromptTooLongError | None = None

    for variant in variants:
        system_prompt, user_prompt = _render_verifier_prompts(
            query=query,
            compiled_plan=compiled_plan,
            pruned_schema=pruned_schema,
            semantic_rules=semantic_rules,
            config=config,
            include_examples=bool(variant["include_examples"]),
            tight_schema=bool(variant["tight_schema"]),
            include_warnings=bool(variant["include_warnings"]),
            minimal_plan=bool(variant["minimal_plan"]),
        )
        try:
            prompt_tokens = _preflight_verifier_prompt_size(
                config.verifier_model,
                system_prompt,
                user_prompt,
                config.verifier_max_model_len,
                config.verifier_max_tokens,
                rules=rules,
            )
            return (
                system_prompt,
                user_prompt,
                {
                    "prompt_variant": variant["name"],
                    "prompt_tokens": prompt_tokens,
                },
            )
        except VerifierPromptTooLongError as exc:
            last_error = exc

    if last_error is not None:
        raise last_error
    raise VerifierPromptTooLongError("No fue posible construir un prompt valido para el verificador semantico")


def run_local_verifier_chat(
    *,
    model_name: str,
    system_prompt: str,
    user_prompt: str,
    temperature: float,
    max_model_len: int,
    max_tokens: int,
    dtype: str,
    gpu_memory_utilization: float,
    enforce_eager: bool,
    cpu_offload_gb: float,
) -> tuple[str, dict[str, Any]]:
    """Ejecuta el verificador local con vLLM y devuelve texto crudo + diagnosticos."""

    from vllm import LLM, SamplingParams  # noqa: PLC0415 - import perezoso

    llm = None
    try:
        llm = build_local_llm(
            LLM,
            model=model_name,
            dtype=dtype,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=True,
            enforce_eager=enforce_eager,
            cpu_offload_gb=cpu_offload_gb,
        )
        sampling = SamplingParams(temperature=temperature, max_tokens=max_tokens)
        started = time.perf_counter()
        outputs = llm.chat(
            [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            sampling_params=sampling,
        )
        completion = outputs[0].outputs[0]
        return completion.text.strip(), {
            "finish_reason": str(getattr(completion, "finish_reason", "")),
            "generated_tokens": len(getattr(completion, "token_ids", []) or []),
            "wall_time_seconds": time.perf_counter() - started,
            "model_name": model_name,
        }
    finally:
        if llm is not None:
            release_local_llm(llm)
            llm = None


def verify_compiled_plan(
    *,
    query: str,
    compiled_plan: CompiledSemanticPlan,
    pruned_schema: Mapping[str, Any],
    semantic_rules: SemanticContract,
    config,
) -> tuple[SemanticVerificationResult, dict[str, Any]]:
    """Ejecuta la verificacion semantica local del plan compilado."""

    system_prompt, user_prompt, prompt_diagnostics = select_verifier_prompt_variant(
        query=query,
        compiled_plan=compiled_plan,
        pruned_schema=pruned_schema,
        semantic_rules=semantic_rules,
        config=config,
    )
    raw_text, diagnostics = run_local_verifier_chat(
        model_name=config.verifier_model,
        system_prompt=system_prompt,
        user_prompt=user_prompt,
        temperature=config.verifier_temperature,
        max_model_len=config.verifier_max_model_len,
        max_tokens=config.verifier_max_tokens,
        dtype=config.verifier_dtype,
        gpu_memory_utilization=config.verifier_gpu_memory_utilization,
        enforce_eager=config.verifier_enforce_eager,
        cpu_offload_gb=config.verifier_cpu_offload_gb,
    )
    diagnostics.update(prompt_diagnostics)
    return validate_verification_payload_json(raw_text), diagnostics
