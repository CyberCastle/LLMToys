#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Resolver semantico NL2SQL.

Etapas LLM: embedding, rerank y verificacion semantica local revisan activos y
alineacion pregunta-plan usando contratos JSON estrictos.
Etapas deterministas: expansion por ejemplos, scoring ponderado, compatibilidad
de schema y compilacion del plan fijan el artefacto oficial consumido por el
solver SQL. Ninguna de estas etapas genera SQL ni reemplaza los guardrails
posteriores del pipeline.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Mapping

from llm_core.vllm_runtime_utils import release_cuda_memory
from nl2sql.utils.collections import dedupe_preserve_order
from nl2sql.utils.decision_models import DecisionIssue, dedupe_decision_issues
from nl2sql.utils.semantic_contract import load_semantic_contract

from .assets import MatchedAsset, SemanticAsset, SemanticPlan
from .compatibility import score_compatibility
from .config import SemanticResolverConfig
from .dialects.base import ResolverDialect
from .dialects.registry import get_resolver_dialect
from .embedding_stage import (
    clear_embedding_runtime,
    embed_assets_cached,
    embed_query,
    get_embedding_runtime,
)
from .plan_compiler import compile_semantic_plan
from .plan_repair import repair_compiled_plan
from .rerank_stage import (
    clear_reranker_runtime,
    get_reranker_runtime,
    rerank_candidates,
)
from .rules_loader import build_reference_maps, load_compiler_rules, load_semantic_rules
from .synonym_logic import compute_synonym_boost, resolve_query_synonyms
from .text_formatting import build_rerank_document_text, format_asset_text
from .verification import classify_semantic_verification, verify_compiled_plan

RerankedCandidate = tuple[SemanticAsset, float, float]


def release_semantic_resolver_runtimes() -> None:
    """Libera embedding, reranker y cache CUDA del semantic resolver."""

    clear_embedding_runtime()
    clear_reranker_runtime()
    release_cuda_memory()


def _extract_pruned_tables(pruned_schema: object) -> dict[str, object]:
    if not isinstance(pruned_schema, Mapping):
        return {}

    nested_pruned_schema = pruned_schema.get("pruned_schema")
    if isinstance(nested_pruned_schema, Mapping):
        return {str(name): value for name, value in nested_pruned_schema.items()}

    nested_tables = pruned_schema.get("tables")
    if isinstance(nested_tables, Mapping):
        return {str(name): value for name, value in nested_tables.items()}

    direct_tables = {str(name): value for name, value in pruned_schema.items() if isinstance(value, Mapping)}
    return direct_tables


def _build_empty_plan(query: str, pruned_tables: dict[str, object], *, error: str) -> SemanticPlan:
    return SemanticPlan(
        query=query,
        assets_by_kind={},
        all_assets=[],
        pruned_tables=tuple(sorted(pruned_tables)),
        diagnostics={"error": error, "num_assets_total": 0, "num_accepted": 0},
    )


def _attach_compiled_plan(
    plan: SemanticPlan,
    *,
    query: str,
    config: SemanticResolverConfig,
    pruned_schema: Mapping[str, object],
    semantic_contract,
    dialect: ResolverDialect | None = None,
) -> SemanticPlan:
    if not config.enable_plan_compiler:
        return plan

    compiled_plan = compile_semantic_plan(
        plan,
        query,
        config=config,
        pruned_schema=pruned_schema,
        dialect=dialect,
    )
    diagnostics = dict(plan.diagnostics)
    blocking_compilation_issue = any(issue.severity == "error" for issue in compiled_plan.issues)

    if config.enable_semantic_verifier and not blocking_compilation_issue:
        current_plan = compiled_plan
        issues = list(compiled_plan.issues)
        warnings = list(compiled_plan.warnings)
        verification_history: list[dict[str, object]] = []
        repair_history: list[dict[str, object]] = []
        verifier_diagnostics: dict[str, object] = {}
        max_repair_attempts = max(0, int(config.max_semantic_repair_attempts))

        for repair_attempt in range(max_repair_attempts + 1):
            verification, verifier_diagnostics = verify_compiled_plan(
                query=query,
                compiled_plan=current_plan,
                pruned_schema=pruned_schema,
                semantic_rules=semantic_contract,
                config=config,
            )
            issue = classify_semantic_verification(verification)
            verification_history.append(
                {
                    "attempt": repair_attempt,
                    "verification": verification.model_dump(mode="python"),
                    "diagnostics": dict(verifier_diagnostics),
                    "issue": issue.model_dump(mode="python") if issue is not None else None,
                }
            )
            current_plan = replace(current_plan, verification=verification)
            if issue is None:
                break

            issues.append(issue)
            if issue.severity == "error" or repair_attempt >= max_repair_attempts:
                break

            repaired_plan = repair_compiled_plan(
                compiled_plan=current_plan,
                verification=verification,
                suggested_delta=verification.suggested_plan_delta,
            )
            if repaired_plan == current_plan:
                warnings.append("semantic_repair_not_applied")
                break

            previous_selected_index = -1
            next_selected_index = -1
            if current_plan.candidate_plan_set is not None:
                previous_selected_index = current_plan.candidate_plan_set.selected_index
            if repaired_plan.candidate_plan_set is not None:
                next_selected_index = repaired_plan.candidate_plan_set.selected_index
            repair_history.append(
                {
                    "attempt": repair_attempt + 1,
                    "selected_index_before": previous_selected_index,
                    "selected_index_after": next_selected_index,
                    "suggested_plan_delta": dict(verification.suggested_plan_delta),
                    "suggested_measure": verification.suggested_measure,
                    "suggested_join_tables": list(verification.suggested_join_tables),
                }
            )
            current_plan = repaired_plan

        diagnostics.update(
            {
                "semantic_verifier_model": verifier_diagnostics.get("model_name"),
                "semantic_verifier_finish_reason": verifier_diagnostics.get("finish_reason"),
                "semantic_verifier_generated_tokens": verifier_diagnostics.get("generated_tokens"),
                "semantic_verifier_wall_time_seconds": verifier_diagnostics.get("wall_time_seconds"),
                "semantic_verification_history": verification_history,
                "semantic_repair_history": repair_history,
            }
        )
        if repair_history:
            warnings.append("semantic_repair_applied")

        compiled_plan = replace(
            current_plan,
            warnings=dedupe_preserve_order(warnings),
            issues=dedupe_decision_issues(issues),
        )
    elif blocking_compilation_issue:
        diagnostics["semantic_verifier_skipped"] = "blocking_compilation_issue"

    return replace(plan, compiled_plan=compiled_plan, diagnostics=diagnostics)


def _retrieve_top_k_with_boosts(
    assets: list[SemanticAsset],
    *,
    asset_embeddings,
    query_embedding,
    compiler_rules,
    synonym_resolution,
    entity_to_table: dict[str, str],
    model_to_tables: dict[str, set[str]],
    top_k: int,
    min_score: float,
    enable_synonym_score_boost: bool,
    synonym_direct_boost: float,
    synonym_related_boost: float,
) -> list[tuple[SemanticAsset, float, float]]:
    if asset_embeddings.size == 0 or not assets:
        return []

    raw_scores = asset_embeddings @ query_embedding
    scored_assets: list[tuple[SemanticAsset, float, float]] = []

    for asset_index, asset in enumerate(assets):
        raw_score = float(raw_scores[asset_index])
        boost = 0.0
        if enable_synonym_score_boost:
            boost = compute_synonym_boost(
                asset,
                synonym_resolution,
                entity_to_table=entity_to_table,
                model_to_tables=model_to_tables,
                direct_boost=synonym_direct_boost,
                related_boost=synonym_related_boost,
                scoring_rules=compiler_rules.synonym_scoring,
            )

        effective_score = raw_score + boost
        if effective_score < min_score:
            continue
        scored_assets.append((asset, raw_score, effective_score))

    scored_assets.sort(key=lambda item: item[2], reverse=True)
    return scored_assets[:top_k]


def _payload_string_list(value: object) -> tuple[str, ...]:
    """Normaliza listas de referencias declaradas en ejemplos semanticos."""

    if not isinstance(value, list):
        return ()
    return tuple(str(item).strip() for item in value if isinstance(item, str) and item.strip())


def _asset_index_by_kind_and_name(
    assets: list[SemanticAsset],
) -> dict[tuple[str, str], SemanticAsset]:
    """Construye un indice exacto por tipo y nombre de activo semantico."""

    return {(asset.kind, asset.name): asset for asset in assets}


def _append_candidate_once(
    candidates: list[RerankedCandidate],
    seen_asset_ids: set[str],
    asset: SemanticAsset | None,
    embedding_score: float,
    rerank_score: float,
) -> bool:
    """Agrega un candidato heredando score solo si aun no esta en la lista."""

    if asset is None or asset.asset_id in seen_asset_ids:
        return False
    candidates.append((asset, embedding_score, rerank_score))
    seen_asset_ids.add(asset.asset_id)
    return True


def _append_asset_and_entity_from_reference(
    *,
    candidates: list[RerankedCandidate],
    seen_asset_ids: set[str],
    asset_index: dict[tuple[str, str], SemanticAsset],
    kind: str,
    name: str,
    embedding_score: float,
    rerank_score: float,
) -> list[dict[str, str]]:
    """Agrega un activo referenciado por ejemplo y su entidad de negocio."""

    added_assets: list[dict[str, str]] = []
    referenced_asset = asset_index.get((kind, name))
    if _append_candidate_once(candidates, seen_asset_ids, referenced_asset, embedding_score, rerank_score):
        added_assets.append({"kind": kind, "name": name})
    if referenced_asset is None:
        return added_assets
    entity_name = referenced_asset.payload.get("entity")
    if isinstance(entity_name, str) and entity_name.strip():
        entity_asset = asset_index.get(("semantic_entities", entity_name.strip()))
        if _append_candidate_once(candidates, seen_asset_ids, entity_asset, embedding_score, rerank_score):
            added_assets.append({"kind": "semantic_entities", "name": entity_name.strip()})
    return added_assets


def expand_reranked_candidates_from_examples(
    reranked_candidates: list[RerankedCandidate],
    assets: list[SemanticAsset],
    *,
    inherited_embedding_weight: float,
    inherited_rerank_weight: float,
) -> tuple[list[RerankedCandidate], list[dict[str, object]]]:
    """Inyecta activos citados por ejemplos semanticos ya recuperados.

    Los ejemplos del YAML funcionan como casos curados de negocio. Si un ejemplo
    entra al top-k, sus metricas, dimensiones y modelo declarados son candidatos
    confiables aunque el embedding de cada activo individual haya quedado fuera
    por poco. Esto evita perder filtros declarativos de estado en consultas
    agregadas similares.
    """

    if not reranked_candidates:
        return [], []

    expanded_candidates = list(reranked_candidates)
    seen_asset_ids = {asset.asset_id for asset, _embedding_score, _rerank_score in expanded_candidates}
    asset_index = _asset_index_by_kind_and_name(assets)
    example_expansion_trace: list[dict[str, object]] = []

    for example_asset, embedding_score, rerank_score in reranked_candidates:
        if example_asset.kind != "semantic_examples":
            continue

        inherited_embedding_score = embedding_score * inherited_embedding_weight
        inherited_rerank_score = rerank_score * inherited_rerank_weight

        for metric_name in _payload_string_list(example_asset.payload.get("metrics")):
            added_assets = _append_asset_and_entity_from_reference(
                candidates=expanded_candidates,
                seen_asset_ids=seen_asset_ids,
                asset_index=asset_index,
                kind="semantic_metrics",
                name=metric_name,
                embedding_score=inherited_embedding_score,
                rerank_score=inherited_rerank_score,
            )
            if added_assets:
                example_expansion_trace.append(
                    {
                        "example": example_asset.name,
                        "reference_kind": "semantic_metrics",
                        "reference_name": metric_name,
                        "embedding_score": inherited_embedding_score,
                        "rerank_score": inherited_rerank_score,
                        "added_assets": added_assets,
                    }
                )
        for dimension_name in _payload_string_list(example_asset.payload.get("dimensions")):
            added_assets = _append_asset_and_entity_from_reference(
                candidates=expanded_candidates,
                seen_asset_ids=seen_asset_ids,
                asset_index=asset_index,
                kind="semantic_dimensions",
                name=dimension_name,
                embedding_score=inherited_embedding_score,
                rerank_score=inherited_rerank_score,
            )
            if added_assets:
                example_expansion_trace.append(
                    {
                        "example": example_asset.name,
                        "reference_kind": "semantic_dimensions",
                        "reference_name": dimension_name,
                        "embedding_score": inherited_embedding_score,
                        "rerank_score": inherited_rerank_score,
                        "added_assets": added_assets,
                    }
                )

        model_name = example_asset.payload.get("model")
        if isinstance(model_name, str) and model_name.strip():
            model_asset = asset_index.get(("semantic_models", model_name.strip()))
            if _append_candidate_once(
                expanded_candidates,
                seen_asset_ids,
                model_asset,
                inherited_embedding_score,
                inherited_rerank_score,
            ):
                example_expansion_trace.append(
                    {
                        "example": example_asset.name,
                        "reference_kind": "semantic_models",
                        "reference_name": model_name.strip(),
                        "embedding_score": inherited_embedding_score,
                        "rerank_score": inherited_rerank_score,
                        "added_assets": [{"kind": "semantic_models", "name": model_name.strip()}],
                    }
                )

    return expanded_candidates, example_expansion_trace


def run_semantic_resolver(
    query: str,
    pruned_schema: dict[str, object] | Mapping[str, object],
    config: SemanticResolverConfig,
    *,
    dialect: ResolverDialect | None = None,
) -> SemanticPlan:
    """Ejecuta el pipeline completo del semantic resolver en cinco pasos.

    El parametro ``dialect`` se inyecta al ``plan_compiler`` para materializar
    expresiones SQL precomputadas en :class:`PlanTimeFilter`. Si no se provee,
    se instancia a partir de ``config.dialect`` (que respeta la variable de
    entorno ``SQL_DIALECT``). Mantener este valor opcional preserva la
    autonomia del modulo: cualquier caller agnostico puede pasar ``None`` y
    obtener un plan sin expresiones especializadas.
    """

    if dialect is None and getattr(config, "dialect", None):
        dialect = get_resolver_dialect(config.dialect)

    normalized_pruned_schema = _extract_pruned_tables(pruned_schema)
    semantic_contract = config.semantic_contract or load_semantic_contract(config.rules_path)
    assets = load_semantic_rules(semantic_contract, config.sections)
    compiler_rules = config.compiler_rules or load_compiler_rules(str(config.compiler_rules_path), config.rules_path)
    if not assets:
        empty_plan = _build_empty_plan(query, normalized_pruned_schema, error="no_assets")
        return _attach_compiled_plan(
            empty_plan,
            query=query,
            config=config,
            pruned_schema=normalized_pruned_schema,
            semantic_contract=semantic_contract,
            dialect=dialect,
        )

    # 1) Cargar activos semanticos y construir sus textos indexables.
    asset_texts = {asset.asset_id: format_asset_text(asset, formatting_rules=compiler_rules.asset_text_formatting) for asset in assets}
    entity_to_table, model_to_tables = build_reference_maps(assets)
    synonym_resolution = resolve_query_synonyms(
        query,
        assets,
        entity_to_table=entity_to_table,
        model_to_tables=model_to_tables,
        max_entities=config.synonym_query_expansion_max_entities,
        enable_query_expansion=config.enable_synonym_query_expansion,
        scoring_rules=compiler_rules.synonym_scoring,
    )

    embedding_runtime = get_embedding_runtime(
        config.embedding_model,
        dtype=config.dtype,
        max_model_len=config.max_model_len,
        gpu_memory_utilization=config.gpu_memory_utilization_embed,
        tensor_parallel_size=config.tensor_parallel_size,
        trust_remote_code=config.trust_remote_code,
    )
    query_embedding = embed_query(
        embedding_runtime.llm,
        synonym_resolution.retrieval_query,
        config.query_instruction,
        config.embedding_query_template,
    )
    asset_embeddings, embedding_cache_stats = embed_assets_cached(
        embedding_runtime.llm,
        assets,
        [asset_texts[asset.asset_id] for asset in assets],
        model=config.embedding_model,
        cache_dir=config.embedding_cache_dir,
        enable_cache=config.enable_embedding_cache,
    )

    # 2) Recuperar candidatos top-k con similitud coseno y un refuerzo explicito
    # para activos alineados con sinonimos detectados en la query.
    top_k_candidates = _retrieve_top_k_with_boosts(
        assets,
        asset_embeddings=asset_embeddings,
        query_embedding=query_embedding,
        compiler_rules=compiler_rules,
        synonym_resolution=synonym_resolution,
        entity_to_table=entity_to_table,
        model_to_tables=model_to_tables,
        top_k=config.top_k_retrieval,
        min_score=config.min_embedding_score,
        enable_synonym_score_boost=config.enable_synonym_score_boost,
        synonym_direct_boost=config.synonym_direct_boost,
        synonym_related_boost=config.synonym_related_boost,
    )
    if not top_k_candidates:
        empty_plan = SemanticPlan(
            query=query,
            assets_by_kind={},
            all_assets=[],
            pruned_tables=tuple(sorted(normalized_pruned_schema)),
            diagnostics={
                "num_assets_total": len(assets),
                "num_after_retrieval": 0,
                "num_after_rerank": 0,
                "num_after_example_expansion": 0,
                "num_accepted": 0,
                "embedding_max_model_len": embedding_runtime.effective_max_model_len,
                "embedding_cache_enabled": embedding_cache_stats.enabled,
                "embedding_cache_hits": embedding_cache_stats.hits,
                "embedding_cache_misses": embedding_cache_stats.misses,
                "embedding_cache_path": str(embedding_cache_stats.cache_path),
                "reranker_max_model_len": None,
                "retrieval_query": synonym_resolution.retrieval_query,
                "synonym_entities_detected": list(synonym_resolution.matched_entities),
            },
        )
        return _attach_compiled_plan(
            empty_plan,
            query=query,
            config=config,
            pruned_schema=normalized_pruned_schema,
            semantic_contract=semantic_contract,
            dialect=dialect,
        )

    if config.sequential_engines:
        clear_embedding_runtime()
        release_cuda_memory()

    # 3) Rerank top-k con Qwen3-Reranker siguiendo la plantilla oficial yes/no.
    reranker_runtime = get_reranker_runtime(
        config.reranker_model,
        dtype=config.dtype,
        max_model_len=config.max_model_len,
        gpu_memory_utilization=config.gpu_memory_utilization_rerank,
        tensor_parallel_size=config.tensor_parallel_size,
        trust_remote_code=config.trust_remote_code,
    )
    rerank_inputs = [
        (
            asset,
            build_rerank_document_text(
                asset,
                max_chars=config.rerank_max_document_chars,
                formatting_rules=compiler_rules.asset_text_formatting,
            ),
            effective_embedding_score,
        )
        for asset, _raw_embedding_score, effective_embedding_score in top_k_candidates
    ]
    reranked_candidates = rerank_candidates(
        reranker_runtime,
        query=query,
        instruction=config.reranker_instruction,
        system_prompt=config.rerank_system_prompt,
        user_prompt_template=config.rerank_user_prompt_template,
        candidates=rerank_inputs,
        batch_size=config.rerank_batch_size,
        max_document_chars=config.rerank_max_document_chars,
        logprobs=config.rerank_logprobs,
        prompt_token_margin=config.rerank_prompt_token_margin,
    )
    reranked_candidates.sort(key=lambda item: (item[2], item[1]), reverse=True)
    reranked_candidates = reranked_candidates[: config.top_k_rerank]
    num_after_rerank = len(reranked_candidates)
    reranked_candidates, example_expansion_trace = expand_reranked_candidates_from_examples(
        reranked_candidates,
        assets,
        inherited_embedding_weight=compiler_rules.example_expansion.inherited_embedding_weight,
        inherited_rerank_weight=compiler_rules.example_expansion.inherited_rerank_weight,
    )

    if config.sequential_engines:
        clear_reranker_runtime()
        release_cuda_memory()

    # 4) Validar compatibilidad con el esquema podado de entrada.
    available_tables = set(normalized_pruned_schema)
    matched_assets: list[MatchedAsset] = []
    for asset, embedding_score, rerank_score in reranked_candidates:
        compatibility_score, compatible_tables, rejected_reason = score_compatibility(
            asset,
            entity_to_table=entity_to_table,
            model_to_tables=model_to_tables,
            available_tables=available_tables,
        )

        if rejected_reason is None and available_tables and compatibility_score < config.compatibility_min_score:
            rejected_reason = "below_compatibility_threshold"
        if rerank_score < config.min_rerank_score:
            rejected_reason = rejected_reason or "below_rerank_threshold"

        matched_assets.append(
            MatchedAsset(
                asset=asset,
                embedding_score=embedding_score,
                rerank_score=rerank_score,
                compatibility_score=compatibility_score,
                compatible_tables=compatible_tables,
                rejected_reason=rejected_reason,
            )
        )

    # 5) Construir el SemanticPlan final con caps por tipo de activo.
    assets_by_kind: dict[str, list[MatchedAsset]] = {}
    for matched_asset in matched_assets:
        if matched_asset.rejected_reason is not None:
            continue
        assets_by_kind.setdefault(matched_asset.asset.kind, []).append(matched_asset)

    for kind, items in assets_by_kind.items():
        items.sort(
            key=lambda item: (
                item.rerank_score,
                item.compatibility_score,
                item.embedding_score,
            ),
            reverse=True,
        )
        assets_by_kind[kind] = items[: config.per_kind_caps.get(kind, 5)]

    semantic_plan = SemanticPlan(
        query=query,
        assets_by_kind=assets_by_kind,
        all_assets=matched_assets,
        pruned_tables=tuple(sorted(available_tables)),
        diagnostics={
            "num_assets_total": len(assets),
            "num_after_retrieval": len(top_k_candidates),
            "num_after_rerank": num_after_rerank,
            "num_after_example_expansion": len(reranked_candidates),
            "num_accepted": sum(1 for item in matched_assets if item.rejected_reason is None),
            "embedding_max_model_len": embedding_runtime.effective_max_model_len,
            "embedding_cache_enabled": embedding_cache_stats.enabled,
            "embedding_cache_hits": embedding_cache_stats.hits,
            "embedding_cache_misses": embedding_cache_stats.misses,
            "embedding_cache_path": str(embedding_cache_stats.cache_path),
            "reranker_max_model_len": reranker_runtime.effective_max_model_len,
            "retrieval_query": synonym_resolution.retrieval_query,
            "synonym_entities_detected": list(synonym_resolution.matched_entities),
            "synonym_tables_detected": list(synonym_resolution.matched_tables),
            "synonym_models_detected": list(synonym_resolution.matched_models),
            "min_embedding_score": config.min_embedding_score,
            "min_rerank_score": config.min_rerank_score,
            "compatibility_min_score": config.compatibility_min_score,
            "enable_synonym_query_expansion": config.enable_synonym_query_expansion,
            "enable_synonym_score_boost": config.enable_synonym_score_boost,
            "top_k_retrieval": config.top_k_retrieval,
            "top_k_rerank": config.top_k_rerank,
            "sequential_engines": config.sequential_engines,
            "rules_path": config.rules_path,
            "example_expansion_trace": example_expansion_trace,
        },
    )
    return _attach_compiled_plan(
        semantic_plan,
        query=query,
        config=config,
        pruned_schema=normalized_pruned_schema,
        semantic_contract=semantic_contract,
        dialect=dialect,
    )
