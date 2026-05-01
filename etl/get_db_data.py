#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Extrae esquema + datos de la BD a un JSONL legible por un LLM."""

from __future__ import annotations

import json
import math
import sys
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any
from uuid import UUID

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Connection, Engine

# El script vive en etl/, pero los modulos compartidos estan en la raiz del repo.
# Insertamos la raiz explicitamente para que `uv run etl/get_db_data.py` funcione
# igual que el resto de runners aunque se ejecute por ruta.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from etl.inspect_db import (
    EXCLUDED_TABLES,
    _require_database_url,
    _simplify_type,
    get_db_schema,
)

load_dotenv(PROJECT_ROOT / ".env")

# =====================================================================
# CONFIGURACION EDITABLE
# =====================================================================
# Ruta del archivo final que consumira el LLM con esquema y filas serializadas.
OUTPUT_PATH = "etl/db_dump.jsonl"

# Limite duro de filas por tabla para evitar dumps inmanejables; usar None para export completo.
MAX_ROWS_PER_TABLE: int | None = 20_000

# Tamano de cada bloque leido desde la BD para balancear memoria y velocidad de extraccion.
CHUNK_SIZE = 5_000

# Maximo de caracteres por valor textual antes de truncarlo en el dump.
MAX_VALUE_CHARS = 5_000

# Si es False, omite columnas binarias pesadas que normalmente no aportan contexto a un LLM.
INCLUDE_BINARY_COLUMNS = False

# Permite excluir tablas adicionales sin tocar la lista base compartida en inspect_db.py.
EXTRA_EXCLUDED_TABLES: list[str] = []

# Numero de lineas iniciales que se revalidan al final para detectar JSONL roto o cabecera invalida.
VALIDATION_LINE_LIMIT = 128

# Sufijo visible que indica al LLM que un valor fue recortado por longitud.
TRUNCATION_MARKER = "...[truncated]"
# =====================================================================


def _resolve_output_path(output_path: str) -> Path:
    """Resuelve rutas relativas desde la raiz del repositorio para que el destino sea estable."""
    candidate = Path(output_path)
    if candidate.is_absolute():
        return candidate
    return PROJECT_ROOT / candidate


def _format_datetime_utc(value: datetime) -> str:
    """Normaliza datetimes a UTC para que el dump use una sola convencion temporal."""
    if value.tzinfo is None:
        normalized = value.replace(tzinfo=timezone.utc)
    else:
        normalized = value.astimezone(timezone.utc)
    return normalized.isoformat().replace("+00:00", "Z")


def _format_datetime_value(value: datetime) -> str:
    """Serializa el datetime sin inventar zona horaria ni alterar la hora extraida."""
    if value.tzinfo is None:
        return value.isoformat()
    return value.isoformat()


def _truncate_text(value: str, max_chars: int = MAX_VALUE_CHARS) -> str:
    """Recorta textos largos para evitar que el dump reviente el contexto del LLM."""
    if max_chars <= 0 or len(value) <= max_chars:
        return value
    if max_chars <= len(TRUNCATION_MARKER):
        return TRUNCATION_MARKER[:max_chars]
    return value[: max_chars - len(TRUNCATION_MARKER)] + TRUNCATION_MARKER


def _is_missing_value(value: Any) -> bool:
    """Detecta nulos de Python, pandas y numpy sin romper con contenedores complejos."""
    if value is None:
        return True
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _is_binary_type(column_type: object) -> bool:
    """Marca columnas binarias que aportan poco valor a un LLM y suelen ser muy pesadas."""
    normalized_type = _simplify_type(column_type).upper()
    return any(token in normalized_type for token in ("BINARY", "IMAGE", "BLOB"))


def _normalize_value(value: Any) -> Any:
    """Convierte tipos de BD/pandas a JSON simple y legible para un LLM."""
    if isinstance(value, np.generic):
        value = value.item()

    if _is_missing_value(value):
        return None

    if isinstance(value, pd.Timestamp):
        value = value.to_pydatetime()

    if isinstance(value, datetime):
        return _format_datetime_value(value)

    if isinstance(value, date):
        return value.isoformat()

    if isinstance(value, time):
        return value.isoformat()

    if isinstance(value, timedelta):
        return str(value)

    if isinstance(value, Decimal):
        if value == value.to_integral_value():
            return int(value)
        return format(value, "f")

    if isinstance(value, UUID):
        return str(value)

    if isinstance(value, (bytes, bytearray, memoryview)):
        if not INCLUDE_BINARY_COLUMNS:
            return None
        return _truncate_text(bytes(value).hex())

    if isinstance(value, str):
        return _truncate_text(value)

    if isinstance(value, dict):
        return {str(key): _normalize_value(item) for key, item in value.items()}

    if isinstance(value, (list, tuple, set)):
        return [_normalize_value(item) for item in value]

    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value

    if isinstance(value, (bool, int)):
        return value

    return _truncate_text(str(value))


def _split_table_name(table_name: str) -> tuple[str | None, str]:
    """Separa schema.table cuando SQLAlchemy haya devuelto el nombre calificado."""
    if "." not in table_name:
        return None, table_name
    schema_name, base_table_name = table_name.split(".", 1)
    return schema_name, base_table_name


def _quote_identifier(engine: Engine, identifier: str) -> str:
    """Cita identificadores segun el dialecto activo para evitar SQL invalido."""
    return engine.dialect.identifier_preparer.quote(identifier)


def _quote_table_name(engine: Engine, table_name: str) -> str:
    """Cita tablas soportando nombres calificados por esquema."""
    schema_name, base_table_name = _split_table_name(table_name)
    if schema_name:
        return f"{_quote_identifier(engine, schema_name)}.{_quote_identifier(engine, base_table_name)}"
    return _quote_identifier(engine, base_table_name)


def _build_export_schema(
    raw_schema: dict[str, Any],
    excluded_tables: set[str],
) -> tuple[dict[str, Any], dict[str, list[str]]]:
    """Filtra tablas/columnas no exportables y deja un esquema coherente para el dump."""
    export_schema: dict[str, Any] = {}
    omitted_columns_by_table: dict[str, list[str]] = {}

    for table_name, raw_info in raw_schema.items():
        if table_name in excluded_tables or not isinstance(raw_info, dict):
            continue

        raw_columns = raw_info.get("columns", [])
        raw_column_descriptions = raw_info.get("column_descriptions", {})
        raw_primary_keys = raw_info.get("primary_keys", [])
        raw_foreign_keys = raw_info.get("foreign_keys", [])

        included_columns: list[tuple[str, str]] = []
        included_column_names: set[str] = set()
        omitted_columns: list[str] = []

        for raw_column in raw_columns:
            if not isinstance(raw_column, (tuple, list)) or len(raw_column) != 2:
                continue
            column_name = str(raw_column[0])
            column_type = _simplify_type(raw_column[1])
            if not INCLUDE_BINARY_COLUMNS and _is_binary_type(column_type):
                omitted_columns.append(column_name)
                continue
            included_columns.append((column_name, column_type))
            included_column_names.add(column_name)

        if isinstance(raw_column_descriptions, dict):
            column_descriptions = {
                str(column_name): str(description)
                for column_name, description in raw_column_descriptions.items()
                if str(column_name) in included_column_names and isinstance(description, str)
            }
        else:
            column_descriptions = {}

        if isinstance(raw_primary_keys, list):
            primary_keys = [str(column_name) for column_name in raw_primary_keys if str(column_name) in included_column_names]
        else:
            primary_keys = []

        if isinstance(raw_foreign_keys, list):
            foreign_keys = [
                {
                    "col": str(foreign_key["col"]),
                    "ref_table": str(foreign_key["ref_table"]),
                    "ref_col": str(foreign_key["ref_col"]),
                }
                for foreign_key in raw_foreign_keys
                if isinstance(foreign_key, dict) and str(foreign_key.get("col")) in included_column_names
            ]
        else:
            foreign_keys = []

        export_schema[str(table_name)] = {
            "description": raw_info.get("description"),
            "columns": included_columns,
            "column_descriptions": column_descriptions,
            "primary_keys": primary_keys,
            "foreign_keys": foreign_keys,
        }

        if omitted_columns:
            omitted_columns_by_table[str(table_name)] = omitted_columns

    return export_schema, omitted_columns_by_table


def _write_json_line(file_handle, payload: dict[str, Any]) -> None:
    """Centraliza la escritura JSONL para mantener el archivo consistente."""
    file_handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _serialize_schema_for_header(schema: dict[str, Any]) -> dict[str, Any]:
    """Transforma el esquema interno a un objeto JSON legible sin strings multilínea escapados."""
    serialized_schema: dict[str, Any] = {}

    for table_name, table_info in schema.items():
        raw_columns = table_info.get("columns", [])
        raw_column_descriptions = table_info.get("column_descriptions", {})
        column_descriptions = raw_column_descriptions if isinstance(raw_column_descriptions, dict) else {}

        serialized_columns = []
        for column_name, column_type in raw_columns:
            serialized_column = {
                "name": str(column_name),
                "type": str(column_type),
            }
            column_description = column_descriptions.get(column_name)
            if isinstance(column_description, str) and column_description.strip():
                serialized_column["description"] = column_description.strip()
            serialized_columns.append(serialized_column)

        serialized_table: dict[str, Any] = {
            "columns": serialized_columns,
            "primary_keys": [str(column_name) for column_name in table_info.get("primary_keys", [])],
            "foreign_keys": [
                {
                    "col": str(foreign_key["col"]),
                    "ref_table": str(foreign_key["ref_table"]),
                    "ref_col": str(foreign_key["ref_col"]),
                }
                for foreign_key in table_info.get("foreign_keys", [])
            ],
        }

        table_description = table_info.get("description")
        if isinstance(table_description, str) and table_description.strip():
            serialized_table["description"] = table_description.strip()

        serialized_schema[str(table_name)] = serialized_table

    return serialized_schema


def _write_header(
    file_handle,
    engine: Engine,
    schema: dict[str, Any],
    excluded_tables: set[str],
    omitted_columns_by_table: dict[str, list[str]],
) -> None:
    """Escribe una cabecera unica con esquema, limites y metadatos del dump."""
    serialized_schema = _serialize_schema_for_header(schema)
    header: dict[str, Any] = {
        "type": "schema",
        "generated_at": _format_datetime_utc(datetime.now(timezone.utc)),
        "database_url_dialect": engine.dialect.name,
        "table_count": len(schema),
        "excluded_tables": sorted(excluded_tables),
        "include_binary_columns": INCLUDE_BINARY_COLUMNS,
        "max_rows_per_table": MAX_ROWS_PER_TABLE,
        "chunk_size": CHUNK_SIZE,
        "max_value_chars": MAX_VALUE_CHARS,
        "schema": serialized_schema,
    }
    if omitted_columns_by_table:
        header["omitted_binary_columns"] = omitted_columns_by_table
    _write_json_line(file_handle, header)


def _count_rows(connection: Connection, engine: Engine, table_name: str) -> int:
    """Obtiene el tamano de la tabla para informar al LLM y detectar truncados."""
    statement = text(f"SELECT COUNT(*) AS row_count FROM {_quote_table_name(engine, table_name)}")
    return int(connection.execute(statement).scalar_one())


def _build_select_statement(engine: Engine, table_name: str, column_names: list[str]):
    """Construye el SELECT citado usando solo las columnas realmente exportables."""
    quoted_columns = ", ".join(_quote_identifier(engine, column_name) for column_name in column_names)
    quoted_table_name = _quote_table_name(engine, table_name)
    return text(f"SELECT {quoted_columns} FROM {quoted_table_name}")


def _iter_table_rows(
    connection: Connection,
    engine: Engine,
    table_name: str,
    column_names: list[str],
):
    """Lee filas por chunks con pandas para no cargar tablas completas en memoria."""
    if not column_names:
        return

    statement = _build_select_statement(engine, table_name, column_names)
    stream_connection = connection.execution_options(stream_results=True)

    for chunk in pd.read_sql(statement, stream_connection, chunksize=CHUNK_SIZE):
        chunk_column_names = [str(column_name) for column_name in chunk.columns.tolist()]
        for row in chunk.itertuples(index=False, name=None):
            yield dict(zip(chunk_column_names, row, strict=False))


def _validate_output_file(output_path: Path, max_lines: int = VALIDATION_LINE_LIMIT) -> None:
    """Verifica que el dump se pueda leer linea a linea con json.loads."""
    line_count = 0
    with output_path.open("r", encoding="utf-8") as file_handle:
        for line_count, line in enumerate(file_handle, start=1):
            payload = json.loads(line)
            if line_count == 1 and payload.get("type") != "schema":
                raise RuntimeError("La primera linea del dump debe ser el registro de esquema.")
            if line_count >= max_lines:
                break

    if line_count == 0:
        raise RuntimeError("El dump generado esta vacio.")


def main() -> None:
    database_url = _require_database_url()
    output_path = _resolve_output_path(OUTPUT_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    excluded_tables = set(EXCLUDED_TABLES) | set(EXTRA_EXCLUDED_TABLES)
    raw_schema = get_db_schema()
    export_schema, omitted_columns_by_table = _build_export_schema(raw_schema, excluded_tables)

    engine = create_engine(database_url)
    exported_rows = 0
    truncated_tables = 0
    table_errors = 0

    try:
        with output_path.open("w", encoding="utf-8") as file_handle:
            _write_header(
                file_handle=file_handle,
                engine=engine,
                schema=export_schema,
                excluded_tables=excluded_tables,
                omitted_columns_by_table=omitted_columns_by_table,
            )

            for table_name, table_info in export_schema.items():
                raw_columns = table_info.get("columns", [])
                column_names = [column_name for column_name, _ in raw_columns]
                excluded_binary_columns = omitted_columns_by_table.get(table_name, [])

                try:
                    with engine.connect() as connection:
                        row_count = _count_rows(connection, engine, table_name)

                        table_header: dict[str, Any] = {
                            "type": "table_header",
                            "table": table_name,
                            "row_count_estimate": row_count,
                            "included_columns": column_names,
                        }
                        if excluded_binary_columns:
                            table_header["excluded_binary_columns"] = excluded_binary_columns
                        if MAX_ROWS_PER_TABLE is not None:
                            table_header["row_limit"] = MAX_ROWS_PER_TABLE
                        _write_json_line(file_handle, table_header)

                        if not column_names:
                            _write_json_line(
                                file_handle,
                                {
                                    "type": "table_skipped",
                                    "table": table_name,
                                    "reason": "all_columns_filtered_as_binary",
                                },
                            )
                            continue

                        rows_written_for_table = 0
                        for row in _iter_table_rows(connection, engine, table_name, column_names):
                            if MAX_ROWS_PER_TABLE is not None and rows_written_for_table >= MAX_ROWS_PER_TABLE:
                                _write_json_line(
                                    file_handle,
                                    {
                                        "type": "truncated",
                                        "table": table_name,
                                        "rows_written": rows_written_for_table,
                                        "row_count_estimate": row_count,
                                    },
                                )
                                truncated_tables += 1
                                break

                            normalized_row = {str(column_name): _normalize_value(value) for column_name, value in row.items()}
                            _write_json_line(
                                file_handle,
                                {
                                    "type": "row",
                                    "table": table_name,
                                    "data": normalized_row,
                                },
                            )
                            rows_written_for_table += 1
                            exported_rows += 1

                except Exception as exc:
                    table_errors += 1
                    _write_json_line(
                        file_handle,
                        {
                            "type": "table_error",
                            "table": table_name,
                            "error": _truncate_text(str(exc)),
                        },
                    )
    finally:
        engine.dispose()

    _validate_output_file(output_path)

    print(f"Dump generado en: {output_path}")
    print(
        f"tablas={len(export_schema)} filas={exported_rows} bytes={output_path.stat().st_size} "
        f"truncadas={truncated_tables} errores={table_errors}"
    )


if __name__ == "__main__":
    main()
