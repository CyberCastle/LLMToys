#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Operaciones vectoriales compartidas para embeddings del pipeline NL2SQL."""

from __future__ import annotations

import numpy as np


def normalize_vector(vector: np.ndarray) -> np.ndarray:
    """Normaliza un vector L2 preservando `float32` y evitando copias inútiles."""

    vector = vector.astype(np.float32, copy=False)
    norm = np.linalg.norm(vector)
    return vector if norm == 0.0 else vector / norm


def normalize_matrix(matrix: np.ndarray) -> np.ndarray:
    """Normaliza una matriz fila a fila para similitud coseno vía producto punto."""

    matrix = matrix.astype(np.float32, copy=False)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return matrix / norms
