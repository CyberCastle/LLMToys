"""API publica del pruning semantico basado en E2Rank.

Este modulo es deliberadamente agnostico al dialecto SQL: trata los tipos de
columna y los nombres de identificadores como strings opacos para fines de
embedding y reranking. Cualquier extension futura que requiera distinguir
motores de base de datos (p. ej. para senales de pseudo-query) debe
encapsularse detras de una interfaz analoga a ``semantic_resolver.dialects``
y nunca acoplar el codigo de pruning a una sintaxis especifica.
"""

from __future__ import annotations

from .config import DEFAULT_MODEL, EmbeddingDType, SemanticSchemaPruningConfig
from .e2rank_engine import clear_e2rank_runtime
from .reporting import persist_pruned_schema
from .schema_pruning import SemanticSchemaPruningResult, build_semantic_schema_pruning_result, run_semantic_schema_pruning

__all__ = [
    "DEFAULT_MODEL",
    "EmbeddingDType",
    "SemanticSchemaPruningConfig",
    "SemanticSchemaPruningResult",
    "build_semantic_schema_pruning_result",
    "clear_e2rank_runtime",
    "persist_pruned_schema",
    "run_semantic_schema_pruning",
]
