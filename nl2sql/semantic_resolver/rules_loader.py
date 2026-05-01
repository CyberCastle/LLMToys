#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

from nl2sql.config import CompilerRules, SynonymScoringRules, load_semantic_resolver_compiler_rules as load_compiler_rules_from_config
from nl2sql.utils.normalization import slugify_identifier
from nl2sql.utils.semantic_contract import SemanticContract, load_semantic_contract, select_semantic_sections

from .assets import SemanticAsset


@lru_cache(maxsize=8)
def load_compiler_rules(rules_path: str, semantic_rules_path: str | None = None) -> CompilerRules:
    """Devuelve las reglas tipadas del compilador ya validadas en `nl2sql.config`."""

    return load_compiler_rules_from_config(Path(rules_path).expanduser().resolve(), semantic_rules_path)


@dataclass(frozen=True)
class SemanticJoinPath:
    """Ruta canonica entre entidades definida en semantic_rules.yaml.

    Se usa tanto en schema pruning (para inyectar como bridge autoritativo)
    como en el plan_compiler (para preferir la ruta oficial por sobre
    cualquier camino FK generico).
    """

    name: str
    from_entity: str
    to_entity: str
    # Secuencia original "tabla.col = tabla.col" que define cada join del path.
    path: tuple[str, ...]
    description: str | None = None


@dataclass(frozen=True)
class SemanticDerivedMetric:
    """Metrica de dos niveles declarada en semantic_derived_metrics.

    Los componentes se capturan por separado para permitir que el solver SQL
    genere CTE/subquery: base_measure + base_group_by en el primer nivel y
    post_aggregation en el segundo nivel. `join_path_hint` enlaza con una
    `SemanticJoinPath` por nombre para reforzar la ruta correcta.
    """

    name: str
    base_measure: str
    base_group_by: tuple[str, ...]
    post_aggregation: str
    description: str | None = None
    synonyms: tuple[str, ...] = ()
    join_path_hint: str | None = None


def _resolve_asset_name(section: str, row: dict[str, object], index: int, *, dict_key: str | None = None) -> str:
    for key in ("name", "id", "question", "entity", "field", "from"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    if dict_key:
        return dict_key
    return f"{section}_{index}"


def _normalize_dict_section(section: str, raw_section: dict[object, object]) -> list[dict[str, object]]:
    normalized_rows: list[dict[str, object]] = []
    for raw_key, raw_value in raw_section.items():
        key = str(raw_key)
        if isinstance(raw_value, dict):
            payload = dict(raw_value)
            payload.setdefault("name", key)
        elif isinstance(raw_value, list):
            payload = {"entity": key, "synonyms": list(raw_value)}
        else:
            payload = {"name": key, "value": raw_value}
        normalized_rows.append(payload)
    return normalized_rows


def build_reference_maps(assets: list[SemanticAsset]) -> tuple[dict[str, str], dict[str, set[str]]]:
    """Construye indices minimos para validar compatibilidad sin importar otros modulos."""

    entity_to_table: dict[str, str] = {}
    model_to_tables: dict[str, set[str]] = {}

    for asset in assets:
        if asset.kind.startswith("semantic_entities"):
            source_table = asset.payload.get("source_table")
            if isinstance(source_table, str) and source_table:
                entity_to_table[asset.name] = source_table

        if asset.kind.startswith("semantic_models"):
            core_tables = asset.payload.get("core_tables")
            if isinstance(core_tables, list):
                model_to_tables[asset.name] = {str(table) for table in core_tables if str(table).strip()}

    return entity_to_table, model_to_tables


def load_semantic_rules(source: str | Path | Mapping[str, object] | SemanticContract, sections: tuple[str, ...]) -> list[SemanticAsset]:
    """Lee un YAML heterogeneo y lo lleva a una lista uniforme de activos semanticos."""

    contract = load_semantic_contract(source)
    raw_data = select_semantic_sections(contract, sections)

    assets: list[SemanticAsset] = []
    for section in sections:
        raw_section = raw_data.get(section)
        if isinstance(raw_section, list):
            rows = raw_section
        elif isinstance(raw_section, dict):
            rows = _normalize_dict_section(section, raw_section)
        else:
            continue

        for index, raw_row in enumerate(rows):
            if not isinstance(raw_row, dict):
                continue

            row = dict(raw_row)
            name = _resolve_asset_name(section, row, index)
            # El asset_id usa seccion::nombre para mantener trazabilidad y permitir
            # caps o agrupaciones por familia semantica sin depender del YAML original.
            asset_id = f"{section}::{slugify_identifier(name)}"
            assets.append(
                SemanticAsset(
                    asset_id=asset_id,
                    kind=section,
                    name=name,
                    payload=row,
                )
            )

    return assets


def load_semantic_join_paths(source: str | Path | Mapping[str, object] | SemanticContract) -> tuple[SemanticJoinPath, ...]:
    """Carga la seccion ``semantic_join_paths`` del YAML de reglas.

    El resultado se cachea por path para no re-parsear el YAML entero en cada
    compilacion de plan. Rutas con campos faltantes o invalidos se descartan
    silenciosamente: su ausencia equivale a no tener hint.
    """

    if isinstance(source, (str, Path)):
        return _load_semantic_join_paths_cached(str(Path(source).expanduser().resolve()))
    return _build_semantic_join_paths(load_semantic_contract(source))


@lru_cache(maxsize=8)
def _load_semantic_join_paths_cached(path: str) -> tuple[SemanticJoinPath, ...]:
    """Carga rutas semanticas cacheando unicamente por path resuelto."""

    return _build_semantic_join_paths(load_semantic_contract(path))


def _build_semantic_join_paths(contract: SemanticContract) -> tuple[SemanticJoinPath, ...]:
    """Materializa semantic_join_paths desde un contrato ya cargado."""

    raw_section = contract.business_invariants.semantic_join_paths
    if not isinstance(raw_section, list):
        return ()

    join_paths: list[SemanticJoinPath] = []
    for raw_row in raw_section:
        if not isinstance(raw_row, dict):
            continue
        name = raw_row.get("name")
        from_entity = raw_row.get("from_entity")
        to_entity = raw_row.get("to_entity")
        raw_path = raw_row.get("path")
        if not (isinstance(name, str) and isinstance(from_entity, str) and isinstance(to_entity, str) and isinstance(raw_path, list)):
            continue
        normalized_path = tuple(str(edge).strip() for edge in raw_path if isinstance(edge, str) and edge.strip())
        if not normalized_path:
            continue
        description = raw_row.get("description")
        join_paths.append(
            SemanticJoinPath(
                name=name.strip(),
                from_entity=from_entity.strip(),
                to_entity=to_entity.strip(),
                path=normalized_path,
                description=description.strip() if isinstance(description, str) and description.strip() else None,
            )
        )
    return tuple(join_paths)


def load_semantic_derived_metrics(source: str | Path | Mapping[str, object] | SemanticContract) -> tuple[SemanticDerivedMetric, ...]:
    """Carga la seccion ``semantic_derived_metrics`` del YAML de reglas."""

    if isinstance(source, (str, Path)):
        return _load_semantic_derived_metrics_cached(str(Path(source).expanduser().resolve()))
    return _build_semantic_derived_metrics(load_semantic_contract(source))


@lru_cache(maxsize=8)
def _load_semantic_derived_metrics_cached(path: str) -> tuple[SemanticDerivedMetric, ...]:
    """Carga metricas derivadas cacheando unicamente por path resuelto."""

    return _build_semantic_derived_metrics(load_semantic_contract(path))


def _build_semantic_derived_metrics(contract: SemanticContract) -> tuple[SemanticDerivedMetric, ...]:
    """Materializa semantic_derived_metrics desde un contrato ya cargado."""

    raw_section = contract.business_invariants.semantic_derived_metrics
    if not isinstance(raw_section, list):
        return ()

    metrics: list[SemanticDerivedMetric] = []
    for raw_row in raw_section:
        if not isinstance(raw_row, dict):
            continue
        name = raw_row.get("name")
        base_measure = raw_row.get("base_measure")
        raw_base_group_by = raw_row.get("base_group_by")
        post_aggregation = raw_row.get("post_aggregation")
        if not (isinstance(name, str) and isinstance(base_measure, str) and isinstance(raw_base_group_by, list)):
            continue
        base_group_by = tuple(str(item).strip() for item in raw_base_group_by if isinstance(item, str) and item.strip())
        if not base_group_by:
            continue
        description = raw_row.get("description")
        join_path_hint = raw_row.get("join_path_hint")
        raw_synonyms = raw_row.get("synonyms")
        synonyms = (
            tuple(str(item).strip() for item in raw_synonyms if isinstance(item, str) and item.strip())
            if isinstance(raw_synonyms, list)
            else ()
        )
        metrics.append(
            SemanticDerivedMetric(
                name=name.strip(),
                base_measure=base_measure.strip(),
                base_group_by=base_group_by,
                post_aggregation=str(post_aggregation).strip() if isinstance(post_aggregation, str) else "",
                description=description.strip() if isinstance(description, str) and description.strip() else None,
                synonyms=synonyms,
                join_path_hint=join_path_hint.strip() if isinstance(join_path_hint, str) and join_path_hint.strip() else None,
            )
        )
    return tuple(metrics)
