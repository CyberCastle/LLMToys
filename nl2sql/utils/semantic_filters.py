#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Resolucion compartida de filtros semanticos para NL2SQL.

El resolver necesita saber que campos actuan realmente como filtros para poder
extender el `join_path` antes de invocar al solver. El solver, a su vez,
necesita la misma decision para construir `selected_filters` sin desalinearse
del plan compilado. Este modulo concentra esa logica en un punto comun.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import re
from typing import Any, Iterable, Mapping

from nl2sql.config import SolverFilterValueRules, load_sql_solver_filter_value_rules
from nl2sql.utils.normalization import normalize_text_for_matching


@dataclass(frozen=True)
class SemanticFilterSelection:
    """Filtro semantico resuelto de forma explicita desde la pregunta."""

    name: str
    field: str
    operator: str
    value: str
    source: str = "semantic_filter"


@lru_cache(maxsize=8)
def load_default_filter_value_rules() -> SolverFilterValueRules:
    """Carga las reglas tipadas de extraccion de valores de filtro."""

    return load_sql_solver_filter_value_rules()


def iter_semantic_filter_payloads_from_plan(
    semantic_plan: Mapping[str, Any],
) -> list[tuple[str, Mapping[str, Any]]]:
    """Extrae filtros semanticos desde un `semantic_plan` serializado."""

    retrieved = semantic_plan.get("retrieved_candidates")
    if not isinstance(retrieved, Mapping):
        return []
    assets_by_kind = retrieved.get("assets_by_kind")
    if not isinstance(assets_by_kind, Mapping):
        return []
    raw_filters = assets_by_kind.get("semantic_filters")
    if not isinstance(raw_filters, list):
        return []

    filters: list[tuple[str, Mapping[str, Any]]] = []
    for raw_filter in raw_filters:
        if not isinstance(raw_filter, Mapping):
            continue
        raw_asset = raw_filter.get("asset")
        if isinstance(raw_asset, Mapping):
            raw_payload = raw_asset.get("payload")
            if isinstance(raw_payload, Mapping):
                name = str(raw_asset.get("name") or raw_payload.get("name") or "semantic_filter")
                filters.append((name, raw_payload))
            continue
        if "field" in raw_filter:
            name = str(raw_filter.get("name") or "semantic_filter")
            filters.append((name, raw_filter))
    return filters


def iter_semantic_filter_payloads_from_assets(
    filter_assets: Iterable[object],
) -> list[tuple[str, Mapping[str, Any]]]:
    """Extrae filtros semanticos desde `MatchedAsset` aceptados por el resolver."""

    filters: list[tuple[str, Mapping[str, Any]]] = []
    for raw_filter in filter_assets:
        asset = getattr(raw_filter, "asset", None)
        payload = getattr(asset, "payload", None)
        name = getattr(asset, "name", None)
        if isinstance(payload, Mapping):
            filters.append((str(name or payload.get("name") or "semantic_filter"), payload))
    return filters


def load_selected_filters_from_compiled_plan(
    compiled_plan: Mapping[str, Any],
) -> list[SemanticFilterSelection]:
    """Reconstruye `selected_filters` desde un plan compilado serializado."""

    raw_filters = compiled_plan.get("selected_filters")
    if not isinstance(raw_filters, list):
        return []

    selected_filters: list[SemanticFilterSelection] = []
    for raw_filter in raw_filters:
        if not isinstance(raw_filter, Mapping):
            continue
        field = str(raw_filter.get("field") or "").strip()
        value = str(raw_filter.get("value") or "").strip()
        if not field or not value:
            continue
        selected_filters.append(
            SemanticFilterSelection(
                name=str(raw_filter.get("name") or "semantic_filter"),
                field=field,
                operator=str(raw_filter.get("operator") or "="),
                value=value,
                source=str(raw_filter.get("source") or "semantic_filter"),
            )
        )
    return selected_filters


def normalize_filter_operator(operator: object) -> str:
    """Normaliza alias declarativos de operador a su forma SQL canonica."""

    normalized = str(operator or "equals").strip().lower()
    aliases = {
        "eq": "=",
        "equal": "=",
        "equals": "=",
        "igual": "=",
        "igual_a": "=",
    }
    return aliases.get(normalized, normalized or "=")


def semantic_filter_aliases(name: str, payload: Mapping[str, Any]) -> list[str]:
    """Construye alias textuales con los que se busca un filtro en la query."""

    aliases: list[str] = []
    for candidate in (name, payload.get("name")):
        if isinstance(candidate, str) and candidate.strip():
            aliases.append(candidate.replace("_", " ").strip())
    raw_synonyms = payload.get("synonyms")
    if isinstance(raw_synonyms, list):
        aliases.extend(str(item).strip() for item in raw_synonyms if str(item).strip())
    field = payload.get("field")
    if isinstance(field, str) and "." in field:
        aliases.append(field.rsplit(".", 1)[1].replace("_", " "))

    deduped: list[str] = []
    seen_aliases: set[str] = set()
    for alias in aliases:
        normalized_alias = re.sub(r"\s+", " ", alias.strip().lower())
        if not normalized_alias or normalized_alias in seen_aliases:
            continue
        seen_aliases.add(normalized_alias)
        deduped.append(alias.strip())
    deduped.sort(key=len, reverse=True)
    return deduped


def strip_filter_value(value: str) -> str:
    """Limpia puntuacion exterior conservando codigos internos."""

    return value.strip().strip("'\"").rstrip(".,;)")


@lru_cache(maxsize=128)
def compile_filter_alias_patterns(
    aliases: tuple[str, ...],
    leading_connectors: tuple[str, ...],
    separator_patterns: tuple[str, ...],
    bare_value_pattern: str,
) -> tuple[re.Pattern[str], ...]:
    """Compila patrones reutilizables para extraer literales de filtros."""

    normalized_connectors = [re.escape(connector).replace(r"\ ", r"\s+") for connector in leading_connectors]
    leading_connector_pattern = rf"(?:{'|'.join(normalized_connectors)})\s+" if normalized_connectors else ""
    separator_pattern = "|".join(separator_patterns) if separator_patterns else r"=|:"
    filter_value_pattern = rf"(?:'(?P<single>[^']+)'|\"(?P<double>[^\"]+)\"|(?P<bare>{bare_value_pattern}))"
    patterns: list[re.Pattern[str]] = []
    for alias in aliases:
        alias_pattern = re.escape(alias).replace(r"\ ", r"\s+")
        patterns.append(
            re.compile(
                rf"(?<!\w)(?:{leading_connector_pattern})?{alias_pattern}(?!\w)"
                rf"\s*(?:(?P<separator>{separator_pattern})\s*)?{filter_value_pattern}",
                re.IGNORECASE,
            )
        )
    return tuple(patterns)


def is_spurious_bare_filter_value(
    value: str,
    *,
    filter_value_rules: SolverFilterValueRules | None = None,
) -> bool:
    """Descarta tokens funcionales que no representan un literal util."""

    active_rules = filter_value_rules or load_default_filter_value_rules()
    normalized_value = normalize_text_for_matching(value)
    return not normalized_value or normalized_value in active_rules.stop_tokens


def extract_filter_value_from_query(
    query: str,
    aliases: list[str],
    *,
    allow_implicit_bare_value: bool = True,
    filter_value_rules: SolverFilterValueRules | None = None,
) -> str | None:
    """Busca el literal que sigue a un alias de filtro en la query."""

    active_rules = filter_value_rules or load_default_filter_value_rules()
    patterns = compile_filter_alias_patterns(
        tuple(aliases),
        active_rules.leading_connectors,
        active_rules.separator_patterns,
        active_rules.bare_value_pattern,
    )

    for pattern in patterns:
        match = pattern.search(query)
        if match is None:
            continue
        quoted_value = match.group("single") or match.group("double") or ""
        if quoted_value:
            value = strip_filter_value(quoted_value)
            if value:
                return value
            continue

        bare_value = strip_filter_value(match.group("bare") or "")
        if not bare_value or is_spurious_bare_filter_value(bare_value, filter_value_rules=filter_value_rules):
            continue

        separator = str(match.group("separator") or "").strip()
        if not allow_implicit_bare_value and not separator:
            continue

        if bare_value:
            return bare_value
    return None


def resolve_semantic_filter_selections(
    query: str,
    raw_filters: Iterable[tuple[str, Mapping[str, Any]]],
    *,
    grouped_fields: Iterable[str] = (),
    filter_value_rules: SolverFilterValueRules | None = None,
) -> list[SemanticFilterSelection]:
    """Materializa filtros cuyo valor aparece explicitamente en la pregunta."""

    active_rules = filter_value_rules or load_default_filter_value_rules()
    if not query.strip():
        return []

    normalized_grouped_fields = {normalize_text_for_matching(str(field_name)) for field_name in grouped_fields if str(field_name).strip()}

    resolved_filters: list[SemanticFilterSelection] = []
    seen_filters: set[tuple[str, str, str]] = set()
    for name, payload in raw_filters:
        field = payload.get("field")
        if not isinstance(field, str) or not field.strip():
            continue
        normalized_field = normalize_text_for_matching(field)
        value = extract_filter_value_from_query(
            query,
            semantic_filter_aliases(name, payload),
            allow_implicit_bare_value=normalized_field not in normalized_grouped_fields,
            filter_value_rules=active_rules,
        )
        if value is None:
            continue
        operator = normalize_filter_operator(payload.get("operator"))
        dedupe_key = (field.strip().lower(), operator, value)
        if dedupe_key in seen_filters:
            continue
        seen_filters.add(dedupe_key)
        resolved_filters.append(
            SemanticFilterSelection(
                name=name,
                field=field.strip(),
                operator=operator,
                value=value,
                source="semantic_filter",
            )
        )
    return resolved_filters
