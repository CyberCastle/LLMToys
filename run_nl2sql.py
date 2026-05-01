#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run_nl2sql.py

Runner plano del orquestador NL2SQL en modo batch. Encadena
prune -> resolver -> solver -> ejecucion SQLAlchemy -> narrativa
Gemma-4-E4B-it-AWQ procesando todas las peticiones por fase, de modo que
cada etapa LLM se ejecute para todas las peticiones antes de avanzar a la
siguiente. Cada peticion persiste sus artefactos intermedios y finales en
una subcarpeta dedicada bajo `out/` y un resumen consolidado se escribe en
`out/nl2sql_batch_summary.yaml`.

Configuracion: variables de entorno (.env) + constantes ALL_CAPS abajo.
NO usa argparse ni ningun parser de CLI (ver AGENTS.md).
"""

from __future__ import annotations

import os
import re
import shutil
import unicodedata
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
import yaml

load_dotenv()

from nl2sql.config import load_nl2sql_runtime_bundle
from nl2sql.orchestrator import NL2SQLConfig, NL2SQLRequest, run_nl2sql_batch
from nl2sql.orchestrator.reporting import render_nl2sql_response, serialize_nl2sql_response

# =====================================================================
# INPUTS
# =====================================================================
# Lista de preguntas de usuario a procesar en una sola corrida del pipeline.
# Cada entrada se ejecuta en su propia subcarpeta dentro de OUT_DIR y todas
# avanzan juntas por cada etapa LLM para no recargar modelos por peticion.
QUERIES: list[str] = [
    # "Lista el top 5 de clientes con mas OTs en el ultimo ano",
    # "cual es el promedio de ordenes por cliente en el ultimo ano?",
    # "cual es el promedio de cotizaciones perdidas por cliente en el ultimo ano?",
    # "cual es el promedio de OTs en ejecucion por cliente en el ultimo ano?",
    # "cuantas OTs estan en ejecucion?",
    # "cual es el stock disponible del producto con codigo unico AA-STG-080A",
    "cual es el top 10 de productos con mas OTs en el ultimo ano?",
]

# =====================================================================
# CONFIGURACION
# =====================================================================
DB_SCHEMA_PATH: str = os.getenv(
    "DB_SCHEMA_PATH", "schema-docs/db_schema.yaml"
)  # Path al YAML con el esquema operativo de la base de datos.
SEMANTIC_RULES_PATH: str = os.getenv(
    "SEMANTIC_RULES_PATH", "schema-docs/semantic_rules.yaml"
)  # Path al YAML con las reglas semanticas del dominio.
CATALOGOS_PATH: str = os.getenv(
    "CATALOGOS_PATH", "schema-docs/catalogos-embebidos.yml"
)  # Path al YAML con catalogos embebidos para resolver filtros.
OUT_DIR: str = os.getenv("NL2SQL_OUT_DIR", "out")  # Directorio base donde se persisten los artefactos del batch NL2SQL.
DIALECT: str = os.getenv("NL2SQL_DIALECT", "tsql")  # tsql | postgres
EXECUTION_SQL_OPTIMIZATION_ENABLED: bool = False  # Si True, optimiza el SQL final con sqlglot antes de ejecutarlo.

# =====================================================================
# SALIDAS
# =====================================================================
# Nombre del YAML por peticion dentro de su subcarpeta.
PER_QUERY_RESPONSE_FILENAME: str = "nl2sql_response.yaml"
# Nombre del YAML consolidado del batch en la raiz de OUT_DIR.
BATCH_SUMMARY_FILENAME: str = "nl2sql_batch_summary.yaml"
# Longitud maxima del slug usado como nombre de subcarpeta.
QUERY_SLUG_MAX_LEN: int = 60


def _clean_output_dir(output_dir: str) -> Path:
    """Elimina y recrea el directorio de salida antes de ejecutar el batch."""

    output_path = Path(output_dir)
    if output_path.exists():
        if output_path.is_dir():
            shutil.rmtree(output_path)
        else:
            output_path.unlink()
    output_path.mkdir(parents=True, exist_ok=True)
    return output_path


def _slugify_query(text: str, *, max_len: int = QUERY_SLUG_MAX_LEN) -> str:
    """Normaliza una pregunta a un slug seguro para nombres de carpeta."""

    # Pasar a ASCII descomponiendo acentos y descartando caracteres no ASCII.
    normalized = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")
    # Reemplazar cualquier caracter no alfanumerico por un guion.
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", normalized).strip("-").lower()
    if not slug:
        return "query"
    return slug[:max_len].rstrip("-") or "query"


def _build_requests(queries: list[str], base_out: Path) -> list[NL2SQLRequest]:
    """Construye las peticiones del batch, asignando una subcarpeta por query."""

    requests: list[NL2SQLRequest] = []
    for index, query in enumerate(queries, start=1):
        slug = _slugify_query(query)
        per_query_dir = base_out / f"{index:02d}-{slug}"
        per_query_dir.mkdir(parents=True, exist_ok=True)
        requests.append(
            NL2SQLRequest(
                query=query,
                db_schema_path=DB_SCHEMA_PATH,
                semantic_rules_path=SEMANTIC_RULES_PATH,
                catalogos_path=CATALOGOS_PATH,
                out_dir=str(per_query_dir),
                dialect=DIALECT,  # type: ignore[arg-type]
            )
        )
    return requests


def main() -> None:
    """Ejecuta el pipeline NL2SQL en modo batch sobre todas las QUERIES."""

    if not QUERIES:
        raise ValueError("La lista QUERIES esta vacia: agregue al menos una pregunta.")

    base_out = _clean_output_dir(OUT_DIR)
    runtime_bundle = load_nl2sql_runtime_bundle(semantic_rules_path=SEMANTIC_RULES_PATH)
    requests = _build_requests(QUERIES, base_out)
    config = NL2SQLConfig(
        runtime_bundle=runtime_bundle,
        execution_sql_optimization_enabled=EXECUTION_SQL_OPTIMIZATION_ENABLED,
    )

    responses = run_nl2sql_batch(requests, config)

    batch_summary: list[dict[str, Any]] = []
    for request, response in zip(requests, responses):
        per_query_dir = Path(request.out_dir)
        # Render por consola del resultado de cada peticion del batch.
        render_nl2sql_response(response, rows_preview_limit=config.rows_preview_limit)
        per_query_path = per_query_dir / PER_QUERY_RESPONSE_FILENAME
        per_query_path.write_text(
            yaml.safe_dump(serialize_nl2sql_response(response), sort_keys=False, allow_unicode=True),
            encoding="utf-8",
        )
        batch_summary.append(
            {
                "query": response.query,
                "status": response.status,
                "out_dir": str(per_query_dir),
                "response_path": str(per_query_path),
                "row_count": response.row_count,
                "truncated": response.truncated,
                "issues": [issue.model_dump(mode="python") for issue in response.issues],
                "warnings": list(response.warnings),
            }
        )
        print(f"\nNL2SQL_RESPONSE_YAML: {per_query_path}")

    summary_path = base_out / BATCH_SUMMARY_FILENAME
    summary_path.write_text(
        yaml.safe_dump({"queries": batch_summary}, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    print(f"\nNL2SQL_BATCH_SUMMARY: {summary_path}")


if __name__ == "__main__":
    main()
