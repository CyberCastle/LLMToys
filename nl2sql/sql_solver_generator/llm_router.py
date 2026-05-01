#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Any, Mapping

from pydantic import Field

from nl2sql.utils.decision_models import DecisionIssue, StrictModel

from .config import SolverConfig
from .sql_generator import PromptTooLongError, generate_spec_and_sql, load_generation_tuning_rules
from .spec_model import SQLQuerySpec

NON_RECOVERABLE_EXCEPTIONS = (
    ImportError,
    ModuleNotFoundError,
    FileNotFoundError,
    KeyError,
    IndexError,
)


class LlmAttempt(StrictModel):
    """Resultado de una corrida del generador LLM antes de normalizar SQL."""

    model_name: str
    spec: SQLQuerySpec | None
    sql: str | None
    generation_diagnostics: dict[str, Any] = Field(default_factory=dict)
    issues: list[DecisionIssue] = Field(default_factory=list)
    attempts: int = 0


def _is_non_recoverable(exc: BaseException) -> bool:
    return isinstance(exc, NON_RECOVERABLE_EXCEPTIONS)


def _is_terminal_request_error(exc: BaseException) -> bool:
    message = str(exc).lower()
    return isinstance(exc, PromptTooLongError) or exc.__class__.__name__ == "VLLMValidationError" or "maximum context length" in message


def _generation_error_issue(exc: BaseException) -> DecisionIssue:
    return DecisionIssue(
        stage="sql_generation",
        code="model_raised",
        severity="error",
        message=str(exc),
        context={"error_type": exc.__class__.__name__},
    )


def _augment_business_rules_summary(
    base_summary: str,
    issues: list[DecisionIssue],
    *,
    tuning_rules=None,
) -> str:
    issue_codes = {issue.code for issue in issues if issue.severity == "error"}
    retry_rules: list[str] = []
    active_tuning_rules = tuning_rules or load_generation_tuning_rules()

    for retry_issue_codes, guidance in active_tuning_rules.retry_rules:
        if retry_issue_codes & issue_codes:
            retry_rules.append(guidance)

    if not retry_rules:
        return base_summary

    sections: list[str] = []
    if base_summary.strip():
        sections.append(base_summary.strip())
    sections.append("Additional retry rules:")
    sections.extend(retry_rules)
    return "\n".join(sections)


class LlmRouter:
    def __init__(self, config: SolverConfig):
        self.cfg = config

    def run(
        self,
        *,
        semantic_plan: Mapping[str, Any],
        pruned_schema: Mapping[str, Any],
        semantic_rules,
        business_rules_summary: str,
        prompts: Mapping[str, Any],
        dialect,
        validator,
    ) -> LlmAttempt:
        last_attempt: LlmAttempt | None = None
        retry_business_rules_summary = business_rules_summary
        for attempt_index in range(self.cfg.max_retries + 1):
            try:
                spec, sql, generation_diagnostics = self._invoke(
                    model_name=self.cfg.model,
                    semantic_plan=semantic_plan,
                    pruned_schema=pruned_schema,
                    semantic_rules=semantic_rules,
                    business_rules_summary=retry_business_rules_summary,
                    prompts=prompts,
                    dialect=dialect,
                )
            except Exception as exc:  # noqa: BLE001
                last_attempt = LlmAttempt(
                    model_name=self.cfg.model,
                    spec=None,
                    sql=None,
                    issues=[_generation_error_issue(exc)],
                    attempts=attempt_index + 1,
                )
                if _is_terminal_request_error(exc) or _is_non_recoverable(exc):
                    return last_attempt
                continue

            sql = _apply_runtime_sql_guards(spec, sql, dialect)
            issues = validator.validate_full(spec, sql)
            last_attempt = LlmAttempt(
                model_name=self.cfg.model,
                spec=spec,
                sql=sql,
                generation_diagnostics=generation_diagnostics,
                issues=issues,
                attempts=attempt_index + 1,
            )
            if not issues:
                return last_attempt
            retry_business_rules_summary = _augment_business_rules_summary(
                business_rules_summary,
                issues,
                tuning_rules=self.cfg.generation_tuning,
            )

        return last_attempt or LlmAttempt(model_name=self.cfg.model, spec=None, sql=None, attempts=0)

    def _invoke(
        self,
        *,
        model_name: str,
        semantic_plan: Mapping[str, Any],
        pruned_schema: Mapping[str, Any],
        semantic_rules,
        business_rules_summary: str,
        prompts: Mapping[str, Any],
        dialect,
    ) -> tuple[SQLQuerySpec, str, dict[str, Any]]:
        return generate_spec_and_sql(
            semantic_plan=semantic_plan,
            pruned_schema=pruned_schema,
            semantic_rules=semantic_rules,
            business_rules_summary=business_rules_summary,
            prompts=prompts,
            model_name=model_name,
            dialect_name=dialect.name,
            max_model_len=self.cfg.max_model_len,
            max_tokens=self.cfg.max_tokens,
            temperature=self.cfg.temperature,
            dtype=self.cfg.llm_dtype,
            gpu_memory_utilization=self.cfg.gpu_memory_utilization,
            enforce_eager=self.cfg.enforce_eager,
            cpu_offload_gb=self.cfg.cpu_offload_gb,
            swap_space_gb=self.cfg.swap_space_gb,
            filter_value_rules_path=str(self.cfg.filter_value_rules_path),
            filter_value_rules=self.cfg.filter_value_rules,
        )


def _apply_runtime_sql_guards(spec: SQLQuerySpec, sql: str, dialect) -> str:
    if spec.query_type == "detail_listing" and spec.limit is not None:
        return dialect.render_row_limit(sql, spec.limit)
    return sql
