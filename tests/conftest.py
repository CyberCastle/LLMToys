#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Bootstrap de imports para la suite local de pruebas."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from tests.generic_domain import (
    generic_schema_tables,
    generic_semantic_contract_payload,
    load_generic_domain_fixture,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]

if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


@pytest.fixture
def generic_domain_fixture() -> dict[str, object]:
    """Expone el fixture YAML del dominio generico para la suite."""

    return load_generic_domain_fixture()


@pytest.fixture
def generic_domain_contract_payload() -> dict[str, object]:
    """Entrega el contrato semantico generico con copia aislada."""

    return generic_semantic_contract_payload()


@pytest.fixture
def generic_domain_schema_tables() -> dict[str, object]:
    """Entrega el mini esquema generico con copia aislada."""

    return generic_schema_tables()
