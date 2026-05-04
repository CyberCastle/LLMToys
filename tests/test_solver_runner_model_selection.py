#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Pruebas para la seleccion de modelo del solver en los runners NL2SQL."""

from __future__ import annotations

from pathlib import Path

import pytest

import run_nl2sql
import run_sql_solver
from nl2sql.sql_solver_generator.config import DEFAULT_MODEL as DEFAULT_SQL_SOLVER_MODEL

RUNNER_MODULES = (run_nl2sql, run_sql_solver)


@pytest.mark.parametrize("runner_module", RUNNER_MODULES, ids=lambda module: module.__name__)
def test_runner_keeps_default_solver_repo_id(runner_module: object) -> None:
    """Mantiene el repo id canonico cuando se usa el modelo por defecto."""

    resolved_model = runner_module._normalize_solver_model_reference(DEFAULT_SQL_SOLVER_MODEL)

    assert resolved_model == DEFAULT_SQL_SOLVER_MODEL


@pytest.mark.parametrize("runner_module", RUNNER_MODULES, ids=lambda module: module.__name__)
def test_runner_resolves_existing_local_checkpoint_path(tmp_path: Path, runner_module: object) -> None:
    """Resuelve a path absoluto cuando el modelo apunta a un checkpoint local existente."""

    model_dir = tmp_path / "quantized" / "XiYanSQL-QwenCoder-7B-2504-W4A16-AWQ"
    model_dir.mkdir(parents=True)

    resolved_model = runner_module._normalize_solver_model_reference(str(model_dir))

    assert resolved_model == str(model_dir.resolve())


def test_run_nl2sql_cleanup_preserves_quantized_checkpoints(tmp_path: Path) -> None:
    """La limpieza del batch no debe borrar modelos cuantizados alojados en `out/quantized`."""

    out_dir = tmp_path / "out"
    quantized_model_dir = out_dir / "quantized" / "XiYanSQL-QwenCoder-7B-2504-W4A16-AWQ"
    stale_dir = out_dir / "01-query-vieja"
    stale_file = out_dir / "nl2sql_batch_summary.yaml"

    quantized_model_dir.mkdir(parents=True)
    stale_dir.mkdir(parents=True)
    stale_file.write_text("queries: []\n", encoding="utf-8")

    preserved_entries = run_nl2sql._resolve_preserved_output_entries(
        str(out_dir),
        str(quantized_model_dir),
        "XGenerationLab/XiYanSQL-QwenCoder-7B-2504",
    )
    cleaned_out_dir = run_nl2sql._clean_output_dir(str(out_dir), preserved_entries=preserved_entries)

    assert cleaned_out_dir == out_dir
    assert preserved_entries == {"quantized"}
    assert quantized_model_dir.exists()
    assert not stale_dir.exists()
    assert not stale_file.exists()
