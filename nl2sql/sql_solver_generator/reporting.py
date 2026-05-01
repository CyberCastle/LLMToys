#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import yaml

from .contracts import SolverOutput


def render_solver_result(result: SolverOutput) -> None:
    print("=" * 72)
    print("SQL SOLVER RESULT")
    print("=" * 72)
    print(f"\nModel used : {result.metadata.model_used}")
    print(f"Dialect    : {result.metadata.dialect}")
    print(f"Attempts   : {result.metadata.attempts}")
    print(f"Finish     : {result.metadata.finish_reason or '?'}")
    print(f"Prompt tok : {result.metadata.prompt_tokens}")
    print(f"Output tok : {result.metadata.generated_tokens}")
    print(f"Wall time  : {result.metadata.wall_time_seconds:.1f}s")
    print("\n-- SQL --")
    print(result.sql_final or "(vacio)")
    print("\n-- SQLQuerySpec --")
    print(yaml.safe_dump(result.sql_query_spec.model_dump(mode="python"), sort_keys=False, allow_unicode=True))
    print("\n-- Warnings --")
    for warning in result.warnings:
        print(f"  * {warning}")
    print("\n-- Issues --")
    for issue in result.issues:
        print(f"  ! {issue.code}: {issue.message}")
