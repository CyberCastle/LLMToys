#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path

import yaml

from nl2sql.config import DEFAULT_NL2SQL_CONFIG_PATH
from nl2sql.semantic_prune.config import SemanticSchemaPruningConfig
from nl2sql.semantic_resolver.config import SemanticResolverConfig


def _write_config(tmp_path: Path, payload: dict[str, object], file_name: str = "nl2sql_config.yaml") -> Path:
    config_path = tmp_path / file_name
    config_path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return config_path


def test_semantic_prune_config_loads_prompts_from_yaml(tmp_path: Path) -> None:
    prompts_path = _write_config(
        tmp_path,
        {
            "semantic_prune": {
                "prompts": {
                    "embedding": {
                        "task_instruction": "recupera tablas y columnas relevantes para una pregunta operacional",
                        "query_template": "Tarea: {task}\nConsulta: {query}",
                    },
                    "listwise_rerank": {
                        "task_instruction": "ordena los documentos segun su relevancia para la pregunta",
                        "prompt_template": "Tarea: {task}\nCandidatos:\n{documents}\nConsulta:{query}",
                    },
                }
            }
        },
    )

    config = SemanticSchemaPruningConfig(query="registros por entidad_c", prompts_path=prompts_path)

    assert config.embedding_task_instruction == "recupera tablas y columnas relevantes para una pregunta operacional"
    assert config.embedding_query_template == "Tarea: {task}\nConsulta: {query}"
    assert config.rerank_task_instruction == "ordena los documentos segun su relevancia para la pregunta"
    assert config.listwise_prompt_template == "Tarea: {task}\nCandidatos:\n{documents}\nConsulta:{query}"


def test_semantic_prune_config_preserves_explicit_prompt_override(tmp_path: Path) -> None:
    prompts_path = _write_config(
        tmp_path,
        {
            "semantic_prune": {
                "prompts": {
                    "embedding": {
                        "task_instruction": "valor en yaml",
                        "query_template": "Plantilla YAML {query}",
                    },
                    "listwise_rerank": {
                        "task_instruction": "rerank yaml",
                        "prompt_template": "Prompt YAML {documents}",
                    },
                }
            }
        },
    )

    config = SemanticSchemaPruningConfig(
        query="registros por entidad_c",
        prompts_path=prompts_path,
        embedding_task_instruction="override explicito",
    )

    assert config.embedding_task_instruction == "override explicito"
    assert config.embedding_query_template == "Plantilla YAML {query}"


def test_semantic_resolver_config_loads_prompts_from_yaml(tmp_path: Path) -> None:
    prompts_path = _write_config(
        tmp_path,
        {
            "semantic_resolver": {
                "prompts": {
                    "embedding": {
                        "query_instruction": "recupera activos semanticos para responder una pregunta de negocio",
                        "query_template": "Instruccion: {instruction}\nConsulta:{query}",
                    },
                    "rerank": {
                        "instruction": "decide si el activo semantico ayuda a responder la pregunta",
                        "system_prompt": "solo responde yes o no segun relevancia de negocio",
                        "user_prompt_template": "<Pregunta>{query}</Pregunta>\n<Activo>{document}</Activo>\n<Instruccion>{instruction}</Instruccion>",
                    },
                    "verifier": {
                        "system_prompt": "devuelve json valido para verificacion",
                        "user_prompt_template": "pregunta={query}\nplan={compiled_plan_yaml}\nesquema={pruned_schema_yaml}\nejemplos={few_shot_examples_yaml}",
                    },
                }
            }
        },
    )

    config = SemanticResolverConfig(prompts_path=prompts_path)

    assert config.query_instruction == "recupera activos semanticos para responder una pregunta de negocio"
    assert config.embedding_query_template == "Instruccion: {instruction}\nConsulta:{query}"
    assert config.reranker_instruction == "decide si el activo semantico ayuda a responder la pregunta"
    assert config.rerank_system_prompt == "solo responde yes o no segun relevancia de negocio"
    assert (
        config.rerank_user_prompt_template
        == "<Pregunta>{query}</Pregunta>\n<Activo>{document}</Activo>\n<Instruccion>{instruction}</Instruccion>"
    )
    assert config.verifier_system_prompt == "devuelve json valido para verificacion"
    assert config.verifier_user_prompt_template.startswith("pregunta={query}")


def test_semantic_resolver_config_preserves_explicit_prompt_override(tmp_path: Path) -> None:
    prompts_path = _write_config(
        tmp_path,
        {
            "semantic_resolver": {
                "prompts": {
                    "embedding": {
                        "query_instruction": "valor yaml",
                        "query_template": "YAML {instruction} {query}",
                    },
                    "rerank": {
                        "instruction": "valor rerank yaml",
                        "system_prompt": "system yaml",
                        "user_prompt_template": "user yaml {document}",
                    },
                    "verifier": {
                        "system_prompt": "verifier yaml",
                        "user_prompt_template": "verifier user {query}",
                    },
                }
            }
        },
    )

    config = SemanticResolverConfig(
        prompts_path=prompts_path,
        reranker_instruction="override explicito",
    )

    assert config.reranker_instruction == "override explicito"
    assert config.query_instruction == "valor yaml"
    assert config.rerank_system_prompt == "system yaml"
    assert config.verifier_system_prompt == "verifier yaml"


def test_semantic_resolver_config_loads_runtime_tuning_defaults_from_yaml(tmp_path: Path, monkeypatch) -> None:
    config_payload = yaml.safe_load(DEFAULT_NL2SQL_CONFIG_PATH.read_text(encoding="utf-8"))
    config_payload["semantic_resolver"]["runtime_tuning"]["default_post_aggregation_function"] = "sum"
    config_path = _write_config(tmp_path, config_payload)

    monkeypatch.setenv("NL2SQL_CONFIG_PATH", str(config_path))

    config = SemanticResolverConfig()

    assert config.default_post_aggregation_function == "sum"
