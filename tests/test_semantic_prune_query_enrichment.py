#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from pathlib import Path

from nl2sql.semantic_prune.query_enrichment import enrich_query_for_retrieval


def test_enrich_query_for_retrieval_usa_hints_desde_config_unificada(tmp_path: Path) -> None:
    config_path = tmp_path / "nl2sql_config.yaml"
    config_path.write_text(
        "semantic_prune:\n"
        "  query_signal_rules:\n"
        '    groupby_dimension_patterns:\n      - "\\\\bpor\\\\s+([a-z0-9_\\\\s]+)"\n'
        "    query_term_stopwords:\n      - de\n"
        "    query_intent_noise_terms:\n      - lista\n"
        "    query_temporal_terms:\n      - fecha\n"
        "    query_aggregation_terms:\n      - promedio\n"
        "    temporal_column_name_terms:\n      - fecha\n"
        "    lookup_descriptor_terms:\n      - nombre\n"
        "    documental_terms:\n      - archivo\n"
        "    query_enrichment_temporal_hints:\n"
        "      'ultimo\\s+ano': \"fecha_personalizada\"\n"
        "    query_enrichment_aggregation_hints:\n"
        '      promedio: "avg_custom"\n',
        encoding="utf-8",
    )

    enriched = enrich_query_for_retrieval("promedio del ultimo ano", config_path=config_path)

    assert "fecha_personalizada" in enriched
    assert "avg_custom" in enriched
