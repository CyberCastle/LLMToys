#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from nl2sql.sql_solver_generator.query_shape import classify_query_shape


def test_classify_scalar_metric() -> None:
    assert classify_query_shape({"intent": "simple_metric", "group_by": []}) == "scalar_metric"


def test_classify_derived_metric() -> None:
    assert classify_query_shape({"intent": "post_aggregated_metric", "group_by": ["entity_c.id"]}) == "derived_metric"


def test_classify_ranking() -> None:
    assert classify_query_shape({"intent": "ranking", "group_by": ["entity_c.id"]}) == "ranking"


def test_classify_lookup_as_detail_listing() -> None:
    assert classify_query_shape({"intent": "lookup", "group_by": []}) == "detail_listing"


def test_classify_lookup_with_measure_as_scalar_metric() -> None:
    assert (
        classify_query_shape(
            {
                "intent": "lookup",
                "measure": {
                    "name": "metric_balance",
                    "formula": "sum(entity_a.amount_total)",
                },
            }
        )
        == "scalar_metric"
    )
