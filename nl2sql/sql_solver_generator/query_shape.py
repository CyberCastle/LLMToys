#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Mapping

from .spec_model import QueryType


def classify_query_shape(compiled_plan: Mapping[str, object]) -> QueryType:
    intent = str(compiled_plan.get("intent", "") or "")
    if intent == "ranking":
        return "ranking"
    if intent == "post_aggregated_metric" or compiled_plan.get("derived_metric_ref"):
        return "derived_metric"
    if intent == "lookup":
        measure = compiled_plan.get("measure")
        if isinstance(measure, Mapping) and (measure.get("formula") or measure.get("name")):
            return "scalar_metric"
        return "detail_listing"
    group_by = compiled_plan.get("group_by") or []
    if isinstance(group_by, list) and group_by:
        return "grouped_metric"
    return "scalar_metric"
