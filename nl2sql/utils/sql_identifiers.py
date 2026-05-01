#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Patrones SQL compartidos para referencias `tabla.columna`."""

from __future__ import annotations

import re

TABLE_COLUMN_RE = re.compile(r"\b([a-z_][a-z0-9_]*)\.([a-z_][a-z0-9_]*)\b", re.IGNORECASE)
TABLE_REFERENCE_RE = re.compile(r"\b([a-z_][a-z0-9_]*)\.[a-z_][a-z0-9_]*\b", re.IGNORECASE)
JOIN_EDGE_RE = re.compile(r"^\s*(?P<left_table>\w+)\.(?P<left_column>\w+)\s*=\s*" r"(?P<right_table>\w+)\.(?P<right_column>\w+)\s*$")
