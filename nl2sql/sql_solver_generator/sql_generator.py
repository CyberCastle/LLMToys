#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Generacion SQL local con contratos Pydantic estrictos.

Etapas LLM: un modelo local produce solo JSON valido para SQLGenerationPayload,
apoyado por few-shot curado cargado desde semantic_contract.
Etapas deterministas: preflight de contexto, construccion de SQLQuerySpec,
compactacion de schema, guardrails de runtime y validacion posterior aseguran
que el LLM no reemplace la seguridad SQL ni la normalizacion por dialecto.
"""

from __future__ import annotations

import logging
from pathlib import Path
import re
from string import Template
from typing import Any, Mapping

import yaml

from nl2sql.utils.normalization import normalize_text_for_matching
from nl2sql.utils.semantic_filters import (
    iter_semantic_filter_payloads_from_plan,
    load_selected_filters_from_compiled_plan,
    resolve_semantic_filter_selections,
)
from nl2sql.utils.semantic_contract import SemanticContract
from nl2sql.utils.vllm_runtime import build_local_llm, release_local_llm
from nl2sql.utils.semantic_examples import (
    render_semantic_examples_for_prompt,
    select_relevant_semantic_examples,
)
from nl2sql.utils.sql_identifiers import TABLE_COLUMN_RE, TABLE_REFERENCE_RE

from .prompt_contract import (
    render_generation_contract_json,
    validate_generation_payload_json,
)
from .config import resolve_filter_value_rules_path
from .payload_compaction import (
    compact_solver_semantic_context as _compact_solver_semantic_context,
    drop_empty as _drop_empty,
)
from .query_shape import classify_query_shape
from .rules_loader import SolverFilterValueRules, load_filter_value_rules
from .runtime import (
    PromptTooLongError,
    build_sampling_kwargs,
    load_generation_tuning_rules,
    load_solver_tokenizer,
    preflight_prompt_size,
    require_solver_model_name,
    resolve_initial_solver_runtime_settings as _resolve_initial_solver_runtime_settings,
    resolve_solver_runtime_retry,
    run_local_llm_chat as _run_local_llm_chat,
)
from .spec_model import (
    SQLDimension,
    SQLFilter,
    SQLJoinEdge,
    SQLJoinPlan,
    SQLMetric,
    SQLQuerySpec,
    SQLTimeFilter,
)

DEBUG_RAW_OUTPUT_PATH: str | None = None
logger = logging.getLogger(__name__)


def _write_debug_output(
    *,
    text: str,
    finish_reason: str,
    prompt_tokens: int,
    generated_tokens: int,
    wall_time_seconds: float,
) -> None:
    debug_path = DEBUG_RAW_OUTPUT_PATH
    if not debug_path:
        return
    path = Path(debug_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        (
            f"# finish_reason={finish_reason}\n"
            f"# prompt_tokens={prompt_tokens}\n"
            f"# generated_tokens={generated_tokens}\n"
            f"# wall_time_seconds={wall_time_seconds:.3f}\n"
            f"{text}"
        ),
        encoding="utf-8",
    )


def _compact_examples_for_solver(
    examples: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    tuning_rules = load_generation_tuning_rules()
    compact_examples: list[dict[str, Any]] = []
    for raw_example in examples:
        compact_example: dict[str, Any] = {}
        question = raw_example.get("question")
        if isinstance(question, str) and question.strip():
            compact_example["question"] = question.strip()
        model_name = raw_example.get("model")
        if isinstance(model_name, str) and model_name.strip():
            compact_example["model"] = model_name.strip()
        metrics = raw_example.get("metrics")
        if isinstance(metrics, list):
            compact_example["metrics"] = [str(item) for item in metrics[: tuning_rules.example_metric_limit] if str(item).strip()]
        dimensions = raw_example.get("dimensions")
        if isinstance(dimensions, list):
            compact_example["dimensions"] = [str(item) for item in dimensions[: tuning_rules.example_dimension_limit] if str(item).strip()]
        filters = raw_example.get("filters")
        if isinstance(filters, list):
            compact_example["filters"] = [item for item in filters[: tuning_rules.example_filter_limit] if isinstance(item, dict)]
        expected_join_path = raw_example.get("expected_join_path")
        if isinstance(expected_join_path, str) and expected_join_path.strip():
            compact_example["expected_join_path"] = expected_join_path.strip()
        if compact_example:
            compact_examples.append(compact_example)
    return compact_examples


def _load_default_filter_value_rules() -> SolverFilterValueRules:
    return load_filter_value_rules(str(resolve_filter_value_rules_path()))


def _render_generation_prompts(
    *,
    semantic_plan: Mapping[str, Any],
    pruned_schema: Mapping[str, Any],
    semantic_rules: SemanticContract,
    business_rules_summary: str,
    prompts: Mapping[str, Any],
    dialect_name: str,
    few_shot_limit: int,
    minimal_payload: bool,
    filter_value_rules: SolverFilterValueRules,
) -> tuple[str, str]:
    prompt_section = prompts.get("spec_generation")
    if not isinstance(prompt_section, Mapping):
        raise ValueError("Falta la seccion 'sql_solver.prompts.spec_generation' en la configuracion de NL2SQL")

    system_template = prompt_section.get("system_prompt")
    user_template = prompt_section.get("user_prompt_template")
    if not isinstance(system_template, str) or not isinstance(user_template, str):
        raise ValueError("Los prompts del solver deben contener system_prompt y user_prompt_template")

    prompt_semantic_plan, prompt_pruned_schema = prepare_prompt_payloads(
        semantic_plan=semantic_plan,
        pruned_schema=pruned_schema,
        minimal=minimal_payload,
        filter_value_rules=filter_value_rules,
    )
    compiled_plan = semantic_plan.get("compiled_plan", {})
    query = str(compiled_plan.get("query") or semantic_plan.get("query") or "")
    selected_examples = _compact_examples_for_solver(select_relevant_semantic_examples(semantic_rules, query, limit=max(0, few_shot_limit)))

    system_prompt = Template(system_template).safe_substitute(
        dialect=dialect_name,
        generation_contract_json=render_generation_contract_json(),
    )
    user_prompt = Template(user_template).safe_substitute(
        dialect=dialect_name,
        semantic_plan_yaml=yaml.safe_dump(prompt_semantic_plan, sort_keys=False, allow_unicode=True),
        pruned_schema_yaml=yaml.safe_dump(prompt_pruned_schema, sort_keys=False, allow_unicode=True),
        sql_shape_yaml=yaml.safe_dump(
            _build_sql_shape_guidance(semantic_plan, dialect_name, filter_value_rules=filter_value_rules),
            sort_keys=False,
            allow_unicode=True,
        ),
        business_rules_summary=business_rules_summary,
        few_shot_examples_yaml=render_semantic_examples_for_prompt(selected_examples),
        generation_contract_json=render_generation_contract_json(),
    )
    return system_prompt, user_prompt


def select_generation_prompt_variant(
    *,
    semantic_plan: Mapping[str, Any],
    pruned_schema: Mapping[str, Any],
    semantic_rules: SemanticContract,
    business_rules_summary: str,
    prompts: Mapping[str, Any],
    dialect_name: str,
    model_name: str,
    max_model_len: int,
    max_tokens: int,
    filter_value_rules: SolverFilterValueRules | None = None,
) -> tuple[str, str, dict[str, Any]]:
    tuning_rules = load_generation_tuning_rules()
    variants = (
        {
            "name": "rich_with_examples",
            "few_shot_limit": tuning_rules.default_few_shot_limit,
            "minimal_payload": False,
        },
        {
            "name": "rich_with_one_example",
            "few_shot_limit": tuning_rules.lean_few_shot_limit,
            "minimal_payload": False,
        },
        {
            "name": "rich_without_examples",
            "few_shot_limit": 0,
            "minimal_payload": False,
        },
        {
            "name": "minimal_without_examples",
            "few_shot_limit": 0,
            "minimal_payload": True,
        },
    )
    last_error: PromptTooLongError | None = None

    for variant in variants:
        system_prompt, user_prompt = _render_generation_prompts(
            semantic_plan=semantic_plan,
            pruned_schema=pruned_schema,
            semantic_rules=semantic_rules,
            business_rules_summary=business_rules_summary,
            prompts=prompts,
            dialect_name=dialect_name,
            few_shot_limit=int(variant["few_shot_limit"]),
            minimal_payload=bool(variant["minimal_payload"]),
            filter_value_rules=filter_value_rules,
        )
        try:
            prompt_tokens = preflight_prompt_size(model_name, system_prompt, user_prompt, max_model_len, max_tokens)
            return (
                system_prompt,
                user_prompt,
                {
                    "prompt_variant": variant["name"],
                    "prompt_tokens": prompt_tokens,
                },
            )
        except PromptTooLongError as exc:
            last_error = exc

    if last_error is not None:
        raise last_error
    raise PromptTooLongError("No fue posible construir un prompt valido para el solver SQL")


def _parse_join_edge(join_expression: str) -> SQLJoinEdge | None:
    match = re.match(
        r"^\s*(?P<left_table>\w+)\.(?P<left_column>\w+)\s*=\s*(?P<right_table>\w+)\.(?P<right_column>\w+)\s*$",
        join_expression,
    )
    if match is None:
        return None
    return SQLJoinEdge(
        left_table=match.group("left_table"),
        left_column=match.group("left_column"),
        right_table=match.group("right_table"),
        right_column=match.group("right_column"),
        join_type="inner",
    )


def _source_to_dimension(source: str) -> SQLDimension:
    table_name, _, column_name = source.partition(".")
    return SQLDimension(
        name=column_name or source,
        source=source,
        entity=table_name,
        type="id" if column_name == "id" else "string",
    )


def _resolve_time_filter_payload(compiled_plan: Mapping[str, Any], dialect_name: str) -> SQLTimeFilter | None:
    time_filter = compiled_plan.get("time_filter")
    if not isinstance(time_filter, Mapping):
        return None
    resolved_expression = ""
    resolved_expressions = time_filter.get("resolved_expressions")
    if isinstance(resolved_expressions, Mapping):
        resolved_expression = str(resolved_expressions.get(dialect_name, "") or "")
    return SQLTimeFilter(
        field=str(time_filter.get("field", "")),
        operator=str(time_filter.get("operator", ">=")),
        value=str(time_filter.get("value", "")),
        resolved_expression=resolved_expression,
    )


def _resolve_semantic_filter_payloads(
    semantic_plan: Mapping[str, Any],
    *,
    filter_value_rules: SolverFilterValueRules | None = None,
) -> list[SQLFilter]:
    """Materializa filtros semanticos cuyo valor aparece explicitamente en la pregunta."""

    active_rules = filter_value_rules or _load_default_filter_value_rules()
    compiled_plan = semantic_plan.get("compiled_plan", {})
    compiled = compiled_plan if isinstance(compiled_plan, Mapping) else {}
    compiled_selected_filters = load_selected_filters_from_compiled_plan(compiled)
    if compiled_selected_filters:
        return [
            SQLFilter(
                name=selected_filter.name,
                field=selected_filter.field,
                operator=selected_filter.operator,
                value=selected_filter.value,
                source=selected_filter.source,
            )
            for selected_filter in compiled_selected_filters
        ]

    query = str(compiled.get("query") or semantic_plan.get("query") or "")
    if not query.strip():
        return []

    grouped_fields = {
        normalize_text_for_matching(str(field_name))
        for field_name in [
            *(compiled.get("group_by", []) or []),
            *(compiled.get("base_group_by", []) or []),
            *(compiled.get("final_group_by", []) or []),
        ]
        if str(field_name).strip()
    }

    return [
        SQLFilter(
            name=selected_filter.name,
            field=selected_filter.field,
            operator=selected_filter.operator,
            value=selected_filter.value,
            source=selected_filter.source,
        )
        for selected_filter in resolve_semantic_filter_selections(
            query,
            iter_semantic_filter_payloads_from_plan(semantic_plan),
            grouped_fields=grouped_fields,
            filter_value_rules=active_rules,
        )
    ]


def _resolve_post_aggregation(compiled_plan: Mapping[str, Any]) -> str:
    post_aggregation = compiled_plan.get("post_aggregation")
    if isinstance(post_aggregation, Mapping):
        return str(post_aggregation.get("function", "none") or "none")
    if isinstance(post_aggregation, str) and post_aggregation:
        return post_aggregation
    return "none"


def _coerce_positive_int(value: object) -> int | None:
    if isinstance(value, bool):
        return None
    if not isinstance(value, (int, float, str)):
        return None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _resolve_query_limit(compiled_plan: Mapping[str, Any], query_type: str) -> int | None:
    tuning_rules = load_generation_tuning_rules()
    explicit_limit = _coerce_positive_int(compiled_plan.get("limit"))
    if explicit_limit is not None:
        return explicit_limit
    ranking = compiled_plan.get("ranking")
    if isinstance(ranking, Mapping):
        ranking_limit = _coerce_positive_int(ranking.get("limit"))
        if ranking_limit is not None:
            return ranking_limit
    if query_type == "ranking":
        return tuning_rules.ranking_default_limit
    if query_type == "detail_listing":
        return tuning_rules.detail_listing_default_limit
    return None


def _build_sql_shape_guidance(
    semantic_plan: Mapping[str, Any],
    dialect_name: str,
    *,
    filter_value_rules: SolverFilterValueRules | None = None,
) -> dict[str, Any]:
    compiled_plan_raw = semantic_plan.get("compiled_plan", {})
    compiled_plan = compiled_plan_raw if isinstance(compiled_plan_raw, Mapping) else {}
    measure_raw = compiled_plan.get("measure", {})
    measure = measure_raw if isinstance(measure_raw, Mapping) else {}
    query_type = classify_query_shape(compiled_plan)
    post_aggregation = _resolve_post_aggregation(compiled_plan)
    row_limit = _resolve_query_limit(compiled_plan, query_type)
    group_by = [str(item) for item in compiled_plan.get("group_by", []) or []]
    base_group_by = [str(item) for item in compiled_plan.get("base_group_by", group_by) or []]
    metric_alias = str(compiled_plan.get("intermediate_alias") or measure.get("name") or "grouped_measure")
    shape: dict[str, Any] = {
        "query_type": query_type,
        "dialect": dialect_name,
        "base_metric_formula": measure.get("formula"),
        "base_group_by": base_group_by,
        "final_group_by": [str(item) for item in compiled_plan.get("final_group_by", []) or []],
        "post_aggregation": post_aggregation,
    }
    selected_filters = _resolve_semantic_filter_payloads(semantic_plan, filter_value_rules=filter_value_rules)
    if selected_filters:
        shape["filters"] = [filter_obj.model_dump(mode="python") for filter_obj in selected_filters]
    if row_limit is not None:
        shape["row_limit"] = row_limit
    if query_type == "derived_metric" and post_aggregation != "none":
        shape["required_shape"] = {
            "inner_query": {
                "metric_formula": measure.get("formula"),
                "metric_alias": metric_alias,
                "group_by": base_group_by,
                "rule": "la CTE/subquery interna debe producir una fila por cada valor de base_group_by",
            },
            "outer_query": {
                "select": f"{post_aggregation.upper()}(CAST({metric_alias} AS DECIMAL(18,2)))",
                "rule": "si final_group_by es [], no proyectar columnas de base_group_by en el SELECT externo",
            },
            "forbidden_shape": "es invalido aplicar la agregacion externa sobre una subquery interna que no tenga GROUP BY base_group_by cuando base_group_by no es []",
        }
    if query_type == "detail_listing" and row_limit is not None:
        shape["required_shape"] = {
            "limit": row_limit,
            "rule": "si query_type es detail_listing, la consulta final debe limitar filas usando TOP/LIMIT segun el dialecto",
        }
    ranking = compiled_plan.get("ranking")
    if query_type == "ranking" and isinstance(ranking, Mapping):
        shape["ranking"] = {
            "direction": str(ranking.get("direction") or "desc"),
            "limit": row_limit,
        }
        shape["required_shape"] = {
            "group_by": group_by,
            "order_by": {
                "metric": str(measure.get("name") or "metric"),
                "direction": str(ranking.get("direction") or "desc"),
            },
            "limit": row_limit,
            "rule": "si query_type es ranking, la consulta final debe ordenar por la metrica agregada en la direccion declarada y limitar a row_limit",
        }
    return _drop_empty(shape)


def build_query_spec_from_plan(
    semantic_plan: Mapping[str, Any],
    dialect_name: str,
    *,
    filter_value_rules: SolverFilterValueRules | None = None,
) -> SQLQuerySpec:
    compiled_plan_raw = semantic_plan.get("compiled_plan", {})
    compiled_plan = compiled_plan_raw if isinstance(compiled_plan_raw, Mapping) else {}
    query_type = classify_query_shape(compiled_plan)
    measure_raw = compiled_plan.get("measure", {})
    measure = measure_raw if isinstance(measure_raw, Mapping) else {}
    base_entity = str(compiled_plan.get("base_entity", "") or "")
    base_table = str(measure.get("source_table") or base_entity)
    post_aggregation = _resolve_post_aggregation(compiled_plan)
    group_by = [str(item) for item in compiled_plan.get("group_by", []) or []]
    base_group_by = [str(item) for item in compiled_plan.get("base_group_by", group_by) or []]
    time_filter = _resolve_time_filter_payload(compiled_plan, dialect_name)
    selected_filters: list[SQLFilter] = _resolve_semantic_filter_payloads(
        semantic_plan,
        filter_value_rules=filter_value_rules,
    )
    if time_filter is not None:
        selected_filters.append(
            SQLFilter(
                name="time_filter",
                field=time_filter.field,
                operator=time_filter.operator,
                value=time_filter.resolved_expression or time_filter.value,
                source="semantic_filter",
            )
        )

    joins = [edge for edge in (_parse_join_edge(str(item)) for item in compiled_plan.get("join_path", []) or []) if edge]
    join_plan: list[SQLJoinPlan] = []
    if joins:
        join_plan.append(
            SQLJoinPlan(
                path_name=str(compiled_plan.get("join_path_hint") or "semantic_join_path"),
                source="semantic_join_paths",
                joins=joins,
            )
        )

    metric_name = str(measure.get("name") or compiled_plan.get("derived_metric_ref") or "metric")
    metric_formula = str(measure.get("formula", "") or "")
    row_limit = _resolve_query_limit(compiled_plan, query_type)
    selected_metrics: list[SQLMetric] = []
    if query_type != "detail_listing":
        selected_metrics.append(
            SQLMetric(
                name=metric_name,
                formula=metric_formula,
                entity=base_entity,
                aggregation_level=("derived_two_level" if query_type == "derived_metric" else "grouped"),
            )
        )

    return SQLQuerySpec(
        query_type=query_type,
        dialect=dialect_name,  # type: ignore[arg-type]
        base_entity=base_entity,
        base_table=base_table,
        selected_metrics=selected_metrics,
        selected_dimensions=[_source_to_dimension(source) for source in group_by],
        selected_filters=selected_filters,
        time_filter=time_filter,
        join_plan=join_plan,
        base_group_by=base_group_by,
        final_group_by=[str(item) for item in compiled_plan.get("final_group_by", []) or []],
        post_aggregation=post_aggregation,  # type: ignore[arg-type]
        limit=row_limit,
        warnings=[str(item) for item in compiled_plan.get("warnings", []) or []],
    )


def generate_spec_and_sql(
    *,
    semantic_plan: Mapping[str, Any],
    pruned_schema: Mapping[str, Any],
    semantic_rules: SemanticContract,
    business_rules_summary: str,
    prompts: Mapping[str, Any],
    model_name: str,
    dialect_name: str,
    max_model_len: int,
    max_tokens: int,
    temperature: float,
    dtype: str,
    gpu_memory_utilization: float,
    enforce_eager: bool = True,
    cpu_offload_gb: float = 0.0,
    swap_space_gb: float = 4.0,
    filter_value_rules_path: str | None = None,
    filter_value_rules: SolverFilterValueRules | None = None,
) -> tuple[SQLQuerySpec, str, dict[str, Any]]:
    """Invoca al generador SQL configurado y valida el contrato JSON local."""

    from vllm import LLM, SamplingParams  # noqa: PLC0415 - import perezoso

    model_name = require_solver_model_name(model_name)
    active_filter_value_rules = filter_value_rules or load_filter_value_rules(
        filter_value_rules_path or str(resolve_filter_value_rules_path())
    )

    system_prompt, user_prompt, prompt_diagnostics = select_generation_prompt_variant(
        semantic_plan=semantic_plan,
        pruned_schema=pruned_schema,
        semantic_rules=semantic_rules,
        business_rules_summary=business_rules_summary,
        prompts=prompts,
        dialect_name=dialect_name,
        model_name=model_name,
        max_model_len=max_model_len,
        max_tokens=max_tokens,
        filter_value_rules=active_filter_value_rules,
    )
    spec = build_query_spec_from_plan(semantic_plan, dialect_name, filter_value_rules=active_filter_value_rules)
    prompt_tokens = int(prompt_diagnostics["prompt_tokens"])
    tokenizer = load_solver_tokenizer(model_name)
    sampling_kwargs = build_sampling_kwargs(tokenizer=tokenizer, temperature=temperature, max_tokens=max_tokens)

    (
        effective_gpu_memory_utilization,
        effective_enforce_eager,
        effective_cpu_offload_gb,
    ) = _resolve_initial_solver_runtime_settings(
        gpu_memory_utilization=gpu_memory_utilization,
        enforce_eager=enforce_eager,
        cpu_offload_gb=cpu_offload_gb,
    )

    for runtime_attempt in range(2):
        llm = None
        try:
            llm = build_local_llm(
                LLM,
                model=model_name,
                max_model_len=max_model_len,
                gpu_memory_utilization=effective_gpu_memory_utilization,
                dtype=dtype,
                trust_remote_code=True,
                enforce_eager=effective_enforce_eager,
                cpu_offload_gb=effective_cpu_offload_gb,
                swap_space=swap_space_gb,
            )
            sampling = SamplingParams(**sampling_kwargs)
            text, diagnostics = _run_local_llm_chat(
                llm=llm,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                sampling=sampling,
            )
            finish_reason = diagnostics["finish_reason"] or "?"
            generated_tokens = int(diagnostics["generated_tokens"])
            elapsed = float(diagnostics["wall_time_seconds"])
            logger.warning(
                "sql_solver.generation finish_reason=%s prompt_tokens=%d generated_tokens=%d wall_time=%.1fs",
                finish_reason,
                prompt_tokens,
                generated_tokens,
                elapsed,
            )
            _write_debug_output(
                text=text,
                finish_reason=finish_reason,
                prompt_tokens=prompt_tokens,
                generated_tokens=generated_tokens,
                wall_time_seconds=elapsed,
            )
            if finish_reason == "length":
                logger.error("sql_solver.generation truncado por max_tokens; ajusta prompt o salida reservada")
                raise RuntimeError("sql_solver_generation_truncated_by_max_tokens")
            payload = validate_generation_payload_json(text)
            return (
                spec,
                payload.final_sql,
                {
                    "finish_reason": finish_reason,
                    "prompt_tokens": prompt_tokens,
                    "prompt_variant": str(prompt_diagnostics.get("prompt_variant", "")),
                    "generated_tokens": generated_tokens,
                    "wall_time_seconds": elapsed,
                },
            )
        except Exception as exc:
            retry_settings = None
            if runtime_attempt == 0:
                retry_settings = resolve_solver_runtime_retry(
                    exc,
                    current_gpu_memory_utilization=effective_gpu_memory_utilization,
                    enforce_eager=effective_enforce_eager,
                    cpu_offload_gb=effective_cpu_offload_gb,
                )
            if retry_settings is None:
                raise
            (
                effective_gpu_memory_utilization,
                effective_enforce_eager,
                effective_cpu_offload_gb,
            ) = retry_settings
            logger.warning(
                "sql_solver.runtime reintenta con gpu_memory_utilization=%.4f enforce_eager=%s cpu_offload_gb=%.1f tras error de arranque: %s",
                effective_gpu_memory_utilization,
                effective_enforce_eager,
                effective_cpu_offload_gb,
                exc,
            )
        finally:
            release_local_llm(llm)
            llm = None

    raise RuntimeError("El solver SQL agoto sus reintentos de runtime sin generar salida")


def prepare_prompt_payloads(
    *,
    semantic_plan: Mapping[str, Any],
    pruned_schema: Mapping[str, Any],
    minimal: bool = False,
    filter_value_rules: SolverFilterValueRules | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    compact_plan = _compact_semantic_plan(
        semantic_plan,
        minimal=minimal,
        filter_value_rules=filter_value_rules,
    )
    required_tables = resolve_required_tables(compact_plan, pruned_schema)
    compact_schema = _compact_pruned_schema(pruned_schema, required_tables, compact_plan, minimal=minimal)
    return compact_plan, compact_schema


def _compact_semantic_plan(
    semantic_plan: Mapping[str, Any],
    *,
    minimal: bool,
    filter_value_rules: SolverFilterValueRules | None = None,
) -> dict[str, Any]:
    compiled_plan_raw = semantic_plan.get("compiled_plan", {})
    compiled_plan = compiled_plan_raw if isinstance(compiled_plan_raw, Mapping) else {}

    compact_compiled = _drop_empty(
        {
            "query": compiled_plan.get("query"),
            "semantic_model": None if minimal else compiled_plan.get("semantic_model"),
            "intent": compiled_plan.get("intent"),
            "base_entity": compiled_plan.get("base_entity"),
            "grain": compiled_plan.get("grain"),
            "measure": compiled_plan.get("measure"),
            "group_by": compiled_plan.get("group_by"),
            "final_group_by": compiled_plan.get("final_group_by"),
            "time_filter": compiled_plan.get("time_filter"),
            "ranking": compiled_plan.get("ranking"),
            "post_aggregation": compiled_plan.get("post_aggregation"),
            "selected_filters": [
                filter_obj.model_dump(mode="python")
                for filter_obj in _resolve_semantic_filter_payloads(
                    semantic_plan,
                    filter_value_rules=filter_value_rules,
                )
            ],
            "join_path": compiled_plan.get("join_path"),
            "required_tables": compiled_plan.get("required_tables"),
            "confidence": compiled_plan.get("confidence"),
            "warnings": None if minimal else compiled_plan.get("warnings"),
            "join_path_hint": None if minimal else compiled_plan.get("join_path_hint"),
            "derived_metric_ref": (None if minimal else compiled_plan.get("derived_metric_ref")),
            "population_scope": (None if minimal else compiled_plan.get("population_scope")),
            "base_group_by": compiled_plan.get("base_group_by"),
            "intermediate_alias": (None if minimal else compiled_plan.get("intermediate_alias")),
            "solver_semantic_context": _compact_solver_semantic_context(compiled_plan, minimal=minimal),
        }
    )
    return {"compiled_plan": compact_compiled}


def resolve_required_tables(compact_plan: Mapping[str, Any], pruned_schema: Mapping[str, Any]) -> list[str]:
    compiled_raw = compact_plan.get("compiled_plan", {})
    compiled = compiled_raw if isinstance(compiled_raw, Mapping) else {}
    explicit_required_tables = "required_tables" in compiled
    raw_tables = compiled.get("required_tables")
    if raw_tables is None:
        raw_tables = _infer_required_tables_from_compiled_plan(compiled)
    required_tables: list[str] = []
    required_table_set: set[str] = set()
    if isinstance(raw_tables, list):
        for table_name in raw_tables:
            normalized = str(table_name)
            if normalized in pruned_schema and normalized not in required_table_set:
                required_tables.append(normalized)
                required_table_set.add(normalized)
    if required_tables:
        return required_tables
    if explicit_required_tables and isinstance(raw_tables, list):
        return []
    return [str(table_name) for table_name in pruned_schema]


def _infer_required_tables_from_compiled_plan(compiled: Mapping[str, Any]) -> list[str]:
    """Deriva tablas minimas desde el plan cuando no trae required_tables."""

    inferred_tables: list[str] = []
    seen_tables: set[str] = set()

    def _add_table(table_name: str) -> None:
        if not table_name or table_name in seen_tables:
            return
        seen_tables.add(table_name)
        inferred_tables.append(table_name)

    def _add_ref(reference: object) -> None:
        if not isinstance(reference, str) or "." not in reference:
            return
        _add_table(reference.split(".", 1)[0].strip())

    measure = compiled.get("measure")
    if isinstance(measure, Mapping):
        source_table = measure.get("source_table")
        if isinstance(source_table, str):
            _add_table(source_table)
        formula = str(measure.get("formula") or "")
        for table_name in TABLE_REFERENCE_RE.findall(formula):
            _add_table(str(table_name))

    grain = compiled.get("grain")
    _add_ref(grain)
    for group_field in compiled.get("group_by", []) or []:
        _add_ref(group_field)
    for group_field in compiled.get("base_group_by", []) or []:
        _add_ref(group_field)

    for filter_obj in compiled.get("selected_filters", []) or []:
        if isinstance(filter_obj, Mapping):
            _add_ref(filter_obj.get("field"))

    time_filter = compiled.get("time_filter")
    if isinstance(time_filter, Mapping):
        _add_ref(time_filter.get("field"))

    for join_edge in compiled.get("join_path", []) or []:
        if not isinstance(join_edge, str) or "=" not in join_edge:
            continue
        left_side, right_side = (side.strip() for side in join_edge.split("=", 1))
        _add_ref(left_side)
        _add_ref(right_side)

    return inferred_tables


def _extract_relevant_schema_columns(
    compact_plan: Mapping[str, Any],
) -> dict[str, set[str]]:
    compiled_raw = compact_plan.get("compiled_plan", {})
    compiled = compiled_raw if isinstance(compiled_raw, Mapping) else {}
    relevant_columns: dict[str, set[str]] = {}

    def _add_ref(reference: object) -> None:
        if not isinstance(reference, str) or "." not in reference:
            return
        table_name, column_name = reference.split(".", 1)
        if not table_name or not column_name:
            return
        relevant_columns.setdefault(table_name, set()).add(column_name)

    for reference in (compiled.get("grain"),):
        _add_ref(reference)
    for field_name in compiled.get("group_by", []) or []:
        _add_ref(field_name)
    for field_name in compiled.get("base_group_by", []) or []:
        _add_ref(field_name)

    for filter_obj in compiled.get("selected_filters", []) or []:
        if isinstance(filter_obj, Mapping):
            _add_ref(filter_obj.get("field"))

    time_filter = compiled.get("time_filter")
    if isinstance(time_filter, Mapping):
        _add_ref(time_filter.get("field"))

    measure = compiled.get("measure")
    if isinstance(measure, Mapping):
        source_table = measure.get("source_table")
        if isinstance(source_table, str) and source_table:
            relevant_columns.setdefault(source_table, set()).add("id")
        formula = str(measure.get("formula") or "")
        for table_name, column_name in TABLE_COLUMN_RE.findall(formula):
            relevant_columns.setdefault(table_name, set()).add(column_name)

    for join_edge in compiled.get("join_path", []) or []:
        if not isinstance(join_edge, str) or "=" not in join_edge:
            continue
        left_side, right_side = (side.strip() for side in join_edge.split("=", 1))
        _add_ref(left_side)
        _add_ref(right_side)

    return relevant_columns


def _extract_schema_columns(
    raw_table: Mapping[str, Any],
    column_types: Mapping[str, Any],
) -> list[tuple[str, str]]:
    columns_raw = raw_table.get("columns", {})
    extracted: list[tuple[str, str]] = []
    if isinstance(columns_raw, Mapping):
        for column_name, column_meta in columns_raw.items():
            type_name = column_types.get(column_name)
            if type_name is None and isinstance(column_meta, Mapping):
                type_name = column_meta.get("type")
            extracted.append((str(column_name), str(type_name or "")))
        return extracted
    if isinstance(columns_raw, list):
        for column_meta in columns_raw:
            if not isinstance(column_meta, Mapping):
                continue
            column_name = column_meta.get("name")
            if not column_name:
                continue
            type_name = column_types.get(column_name)
            if type_name is None:
                type_name = column_meta.get("type")
            extracted.append((str(column_name), str(type_name or "")))
    return extracted


def _compact_pruned_schema(
    pruned_schema: Mapping[str, Any],
    required_tables: list[str],
    compact_plan: Mapping[str, Any],
    *,
    minimal: bool,
) -> dict[str, Any]:
    tuning_rules = load_generation_tuning_rules()
    compact_schema: dict[str, Any] = {}
    required_table_set = set(required_tables)
    relevant_columns = _extract_relevant_schema_columns(compact_plan)
    max_columns_per_table = tuning_rules.max_columns_per_table_minimal if minimal else tuning_rules.max_columns_per_table_rich

    for table_name in required_tables:
        raw_table = pruned_schema.get(table_name, {})
        if not isinstance(raw_table, Mapping):
            continue

        column_types_raw = raw_table.get("column_types", {})
        column_types = column_types_raw if isinstance(column_types_raw, Mapping) else {}
        compact_columns: dict[str, str] = {}
        extracted_columns = _extract_schema_columns(raw_table, column_types)
        priority_columns = [column for column in extracted_columns if column[0] in relevant_columns.get(table_name, set())]
        fallback_columns = [column for column in extracted_columns if column[0] not in relevant_columns.get(table_name, set())]
        for column_name, type_name in (priority_columns + fallback_columns)[:max_columns_per_table]:
            compact_columns[column_name] = type_name

        foreign_keys: list[dict[str, str]] = []
        for foreign_key in raw_table.get("foreign_keys", []) or []:
            if not isinstance(foreign_key, Mapping):
                continue
            ref_table = str(foreign_key.get("ref_table", ""))
            if ref_table and ref_table not in required_table_set:
                continue
            foreign_keys.append(
                {
                    "column": str(foreign_key.get("column") or foreign_key.get("col") or ""),
                    "ref_table": ref_table,
                    "ref_col": str(foreign_key.get("ref_col", "")),
                }
            )

        compact_schema[table_name] = _drop_empty(
            {
                "role": None if minimal else raw_table.get("role"),
                "columns": compact_columns,
                "primary_keys": ([] if minimal else [str(item) for item in raw_table.get("primary_keys", []) or []]),
                "foreign_keys": foreign_keys,
            }
        )

    return compact_schema
