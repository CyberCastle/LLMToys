#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

import run_semantic_schema_pruning as prune_runner
from nl2sql.semantic_prune import SemanticSchemaPruningConfig
from nl2sql.semantic_prune.schema_pruning import build_semantic_schema_pruning_result


def test_build_semantic_schema_pruning_result_requires_explicit_schema() -> None:
    with pytest.raises(ValueError, match="schema explicito"):
        build_semantic_schema_pruning_result(
            SemanticSchemaPruningConfig(query="registros por entidad_c"),
        )


def test_runner_loads_db_schema_yaml_and_normalizes_it(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    schema_path = tmp_path / "db_schema.yaml"
    schema_path.write_text(
        yaml.safe_dump(
            {
                "entity_c": {
                    "description": "Entidad agrupadora.",
                    "columns": [
                        {"name": "id", "type": "BIGINT"},
                        {"name": "display_name", "type": "VARCHAR", "description": "Etiqueta visible"},
                    ],
                    "primary_keys": ["id"],
                    "foreign_keys": [],
                }
            },
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(prune_runner, "DB_SCHEMA_PATH", str(schema_path))

    schema, source = prune_runner._load_runtime_schema()

    assert source == "yaml"
    assert schema == {
        "entity_c": {
            "description": "Entidad agrupadora.",
            "columns": [("id", "BIGINT"), ("display_name", "VARCHAR")],
            "column_descriptions": {"display_name": "Etiqueta visible"},
            "primary_keys": ["id"],
            "foreign_keys": [],
        }
    }


def test_runner_reflects_db_when_yaml_is_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    missing_schema_path = tmp_path / "missing_db_schema.yaml"
    reflected_schema = {
        "entity_a": {
            "description": "Registro operativo.",
            "columns": [("id", "BIGINT")],
            "primary_keys": ["id"],
            "foreign_keys": [],
        }
    }
    saved_payload: dict[str, object] = {}

    def fake_get_db_schema() -> dict[str, object]:
        return reflected_schema

    def fake_save_schema_to_yaml_file(schema: dict[str, object], filepath: str) -> None:
        saved_payload["schema"] = schema
        saved_payload["filepath"] = filepath

    monkeypatch.setattr(prune_runner, "DB_SCHEMA_PATH", str(missing_schema_path))
    monkeypatch.setattr(prune_runner, "get_db_schema", fake_get_db_schema)
    monkeypatch.setattr(prune_runner, "save_schema_to_yaml_file", fake_save_schema_to_yaml_file)

    schema, source = prune_runner._load_runtime_schema()

    assert source == "database_reflection"
    assert schema == reflected_schema
    assert saved_payload == {
        "schema": reflected_schema,
        "filepath": str(missing_schema_path),
    }


def test_runner_persists_query_plus_pruned_schema_payload(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output_path = tmp_path / "semantic_pruned_schema.yaml"
    monkeypatch.setattr(prune_runner, "SEMANTIC_PRUNED_SCHEMA_OUTPUT_PATH", str(output_path))

    config = SemanticSchemaPruningConfig(query="registros por entidad_c")
    persisted_path = prune_runner._persist_pruned_schema_result(
        config,
        {
            "retrieval_query": "registros por entidad_c | pistas: entidad_c registros",
            "pruned_schema": {
                "entity_c": {
                    "columns": [{"name": "id", "type": "BIGINT"}],
                    "primary_keys": ["id"],
                    "foreign_keys": [],
                }
            },
        },
    )

    assert persisted_path == output_path
    payload = yaml.safe_load(output_path.read_text(encoding="utf-8"))
    assert payload["query"] == "registros por entidad_c"
    assert payload["retrieval_query"] == "registros por entidad_c | pistas: entidad_c registros"
    assert payload["pruned_schema"]["entity_c"]["columns"][0]["name"] == "id"
