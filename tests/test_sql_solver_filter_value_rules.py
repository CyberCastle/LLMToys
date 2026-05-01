#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path

from nl2sql.sql_solver_generator.rules_loader import load_filter_value_rules


def test_load_filter_value_rules_normaliza_stop_tokens_desde_yaml(tmp_path: Path) -> None:
    rules_path = tmp_path / "nl2sql_config.yaml"
    rules_path.write_text(
        "sql_solver:\n"
        "  filter_value_rules:\n"
        "    stop_tokens:\n"
        "      - En\n"
        "      - último\n"
        "      - today\n"
        "    leading_connectors:\n"
        "      - con\n"
        "      - de\n"
        "    separator_patterns:\n"
        '      - "="\n'
        '      - ":"\n'
        '      - "es\\\\s+"\n'
        '    bare_value_pattern: "[A-Za-z0-9][A-Za-z0-9_.:/-]*"\n',
        encoding="utf-8",
    )

    rules = load_filter_value_rules(str(rules_path))

    assert rules.stop_tokens == frozenset({"en", "ultimo", "today"})
    assert rules.leading_connectors == ("con", "de")
    assert rules.separator_patterns == ("=", ":", "es\\s+")
