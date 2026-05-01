#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path

from nl2sql.semantic_prune.schema_logic import load_query_signal_rules


def test_load_query_signal_rules_acepta_config_unificada_personalizada(tmp_path: Path) -> None:
    config_path = tmp_path / "nl2sql_config.yaml"
    config_path.write_text(
        "semantic_prune:\n"
        "  query_signal_rules:\n"
        "    groupby_dimension_patterns:\n"
        '      - "\\\\bpor\\\\s+([a-z0-9_\\\\s]+)"\n'
        "    query_term_stopwords:\n"
        "      - de\n"
        "    query_intent_noise_terms:\n"
        "      - cual\n"
        "    query_temporal_terms:\n"
        "      - fecha\n"
        "    query_aggregation_terms:\n"
        "      - promedio\n"
        "    temporal_column_name_terms:\n"
        "      - fecha\n"
        "    lookup_descriptor_terms:\n"
        "      - nombre\n"
        "    documental_terms:\n"
        "      - repositorio_interno\n"
        "    query_enrichment_temporal_hints:\n"
        "      'ultimo\\s+ano': \"fecha year\"\n"
        "    query_enrichment_aggregation_hints:\n"
        '      promedio: "avg average"\n',
        encoding="utf-8",
    )

    rules = load_query_signal_rules(str(config_path))

    assert "repositorio_interno" in rules.documental_terms
