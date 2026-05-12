#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Pruebas unitarias para los helpers de entorno compartidos de `llm_core`."""

from __future__ import annotations

from pathlib import Path

from llm_core.env import env_path


def test_env_path_returns_path_default_when_env_is_missing(monkeypatch) -> None:
    """El helper debe preservar defaults `Path` cuando la variable no existe."""

    monkeypatch.delenv("TEST_ENV_PATH", raising=False)

    assert env_path("TEST_ENV_PATH", Path(".cache/example")) == Path(".cache/example")


def test_env_path_trims_and_expands_user_values(monkeypatch) -> None:
    """La ruta leida del entorno debe recortar espacios y expandir `~`."""

    monkeypatch.setenv("TEST_ENV_PATH", "  ~/cache/example  ")

    assert env_path("TEST_ENV_PATH", Path("unused")) == Path("~/cache/example").expanduser()
