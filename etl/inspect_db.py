from copy import deepcopy
import os
import re
import yaml
from dotenv import load_dotenv

load_dotenv()

DEFAULT_SCHEMA_OUTPUT_PATH = os.getenv("DB_SCHEMA_PATH", "out/db_schema.yaml")


import numpy as np
import pandas as pd

from sqlalchemy import MetaData, create_engine, inspect, text

# Tablas excluidas del export a YAML
EXCLUDED_TABLES: list[str] = [
    "DATABASECHANGELOG",
    "DATABASECHANGELOGLOCK",
    "jhi_authority",
    "jhi_persistent_audit_event",
    "jhi_persistent_audit_evt_data",
    "jhi_persistent_token",
    "jhi_user",
    "jhi_user_authority",
    "jv_commit",
    "jv_commit_property",
    "jv_global_id",
    "jv_snapshot",
]

_SCHEMA_CACHE_BY_URL: dict[str, dict[str, object]] = {}


def _as_plain_string(value: object) -> str:
    return str(value)


def _as_optional_string(value: object | None) -> str | None:
    """Normalize nullable metadata values to stripped strings."""
    if value is None:
        return None
    normalized_value = str(value).strip()
    return normalized_value or None


def _get_effective_schema_name(raw_schema_name: object | None, default_schema_name: object | None) -> str:
    """Resolve the schema used to key comment lookups."""
    explicit_schema_name = _as_optional_string(raw_schema_name)
    if explicit_schema_name:
        return explicit_schema_name
    default_name = _as_optional_string(default_schema_name)
    return default_name or ""


def _simplify_type(col_type: object) -> str:
    """Strip length, precision, collation and other DB-specific details from a column type."""
    type_str = str(col_type)
    type_str = re.sub(r"\(.*?\)", "", type_str)
    type_str = re.sub(r"\s+(COLLATE|CHARACTER\s+SET|CHARSET|USING|WITH\s+TIME\s+ZONE)\s+\S+", "", type_str, flags=re.IGNORECASE)
    type_str = re.sub(r"\s+(UNSIGNED|SIGNED|ZEROFILL|AUTO_INCREMENT|IDENTITY)", "", type_str, flags=re.IGNORECASE)
    return type_str.strip()


def _get_mssql_extended_property_comments(engine) -> tuple[dict[tuple[str, str], str], dict[tuple[str, str, str], str]]:
    """Read table and column comments from SQL Server MS_Description properties."""
    table_comments: dict[tuple[str, str], str] = {}
    column_comments: dict[tuple[str, str, str], str] = {}

    table_comment_query = text("""
        SELECT
            schemas.name AS schema_name,
            tables.name AS table_name,
            CAST(properties.value AS NVARCHAR(4000)) AS description
        FROM sys.tables AS tables
        INNER JOIN sys.schemas AS schemas
            ON schemas.schema_id = tables.schema_id
        LEFT JOIN sys.extended_properties AS properties
            ON properties.major_id = tables.object_id
            AND properties.minor_id = 0
            AND properties.name = N'MS_Description';
        """)
    column_comment_query = text("""
        SELECT
            schemas.name AS schema_name,
            tables.name AS table_name,
            columns.name AS column_name,
            CAST(properties.value AS NVARCHAR(4000)) AS description
        FROM sys.tables AS tables
        INNER JOIN sys.schemas AS schemas
            ON schemas.schema_id = tables.schema_id
        INNER JOIN sys.columns AS columns
            ON columns.object_id = tables.object_id
        LEFT JOIN sys.extended_properties AS properties
            ON properties.major_id = columns.object_id
            AND properties.minor_id = columns.column_id
            AND properties.name = N'MS_Description';
        """)

    with engine.connect() as connection:
        for row in connection.execute(table_comment_query).mappings():
            schema_name = _get_effective_schema_name(row.get("schema_name"), None)
            table_name = _as_optional_string(row.get("table_name"))
            description = _as_optional_string(row.get("description"))
            if table_name and description:
                table_comments[(schema_name, table_name)] = description

        for row in connection.execute(column_comment_query).mappings():
            schema_name = _get_effective_schema_name(row.get("schema_name"), None)
            table_name = _as_optional_string(row.get("table_name"))
            column_name = _as_optional_string(row.get("column_name"))
            description = _as_optional_string(row.get("description"))
            if table_name and column_name and description:
                column_comments[(schema_name, table_name, column_name)] = description

    return table_comments, column_comments


def _get_generic_reflection_comments(engine, metadata: MetaData) -> tuple[dict[tuple[str, str], str], dict[tuple[str, str, str], str]]:
    """Fallback for dialects that expose comments through SQLAlchemy reflection."""
    inspector = inspect(engine)
    default_schema_name = getattr(inspector, "default_schema_name", None)
    table_comments: dict[tuple[str, str], str] = {}
    column_comments: dict[tuple[str, str, str], str] = {}

    for table in metadata.tables.values():
        effective_schema_name = _get_effective_schema_name(table.schema, default_schema_name)
        normalized_table_name = _as_plain_string(table.name)

        table_comment = _as_optional_string(getattr(table, "comment", None))
        if not table_comment:
            try:
                reflected_table_comment = inspector.get_table_comment(table.name, schema=table.schema)
            except NotImplementedError:
                reflected_table_comment = None
            if isinstance(reflected_table_comment, dict):
                table_comment = _as_optional_string(reflected_table_comment.get("text"))

        if table_comment:
            table_comments[(effective_schema_name, normalized_table_name)] = table_comment

        try:
            reflected_columns = inspector.get_columns(table.name, schema=table.schema)
        except NotImplementedError:
            reflected_columns = []

        if reflected_columns:
            for column_info in reflected_columns:
                if not isinstance(column_info, dict):
                    continue
                column_name = _as_optional_string(column_info.get("name"))
                column_comment = _as_optional_string(column_info.get("comment"))
                if column_name and column_comment:
                    column_comments[(effective_schema_name, normalized_table_name, column_name)] = column_comment
            continue

        for column in table.columns:
            column_comment = _as_optional_string(getattr(column, "comment", None))
            if column_comment:
                column_comments[(effective_schema_name, normalized_table_name, _as_plain_string(column.name))] = column_comment

    return table_comments, column_comments


def _get_schema_comments(engine, metadata: MetaData) -> tuple[dict[tuple[str, str], str], dict[tuple[str, str, str], str]]:
    """Pick the most reliable comment extraction strategy for the active dialect."""
    if engine.dialect.name.lower() == "mssql":
        return _get_mssql_extended_property_comments(engine)
    return _get_generic_reflection_comments(engine, metadata)


def _require_database_url() -> str:
    database_url = os.getenv("DATABASE_URL")
    if not isinstance(database_url, str) or not database_url.strip():
        raise RuntimeError("DATABASE_URL no esta configurado en el entorno.")
    return database_url


def _reflect_db_schema(database_url: str) -> dict[str, object]:
    """Refleja el esquema directamente desde la base de datos."""
    engine = create_engine(database_url)
    metadata = MetaData()
    try:
        metadata.reflect(bind=engine)
        default_schema_name = getattr(inspect(engine), "default_schema_name", None)
        table_comments, column_comments = _get_schema_comments(engine, metadata)

        schema = {}
        for table_name, table in metadata.tables.items():
            normalized_table_name = _as_plain_string(table_name)
            # metadata.tables can be schema-qualified; table.name keeps the base table identifier.
            normalized_base_table_name = _as_plain_string(table.name)
            if normalized_base_table_name in EXCLUDED_TABLES:
                continue
            effective_schema_name = _get_effective_schema_name(table.schema, default_schema_name)
            pk_cols = [_as_plain_string(col.name) for col in table.primary_key.columns]
            fk_list = [
                {
                    "col": _as_plain_string(fk.parent.name),
                    "ref_table": _as_plain_string(fk.column.table.name),
                    "ref_col": _as_plain_string(fk.column.name),
                }
                for fk in table.foreign_keys
            ]

            # Keep descriptions separate so existing consumers of the (name, type) tuples still work.
            column_descriptions = {}
            for column in table.columns:
                normalized_column_name = _as_plain_string(column.name)
                column_comment = column_comments.get((effective_schema_name, normalized_base_table_name, normalized_column_name))
                if column_comment:
                    column_descriptions[normalized_column_name] = column_comment

            schema[normalized_table_name] = {
                "description": table_comments.get((effective_schema_name, normalized_base_table_name)),
                "columns": [(_as_plain_string(col.name), _simplify_type(col.type)) for col in table.columns],
                "column_descriptions": column_descriptions,
                "primary_keys": pk_cols,
                "foreign_keys": fk_list,
            }

        return schema
    finally:
        engine.dispose()


def get_db_schema(*, force_refresh: bool = False) -> dict[str, object]:
    """Entrega el esquema cacheado y permite forzar una nueva reflexión.

    El cache se separa por `DATABASE_URL` para evitar mezclar esquemas cuando un
    mismo proceso trabaja contra distintas bases.
    """
    database_url = _require_database_url()

    if not force_refresh:
        cached_schema = _SCHEMA_CACHE_BY_URL.get(database_url)
        if cached_schema is not None:
            return deepcopy(cached_schema)

    reflected_schema = _reflect_db_schema(database_url)
    _SCHEMA_CACHE_BY_URL[database_url] = reflected_schema
    return deepcopy(reflected_schema)


def export_schema_to_yaml(schema):
    prompt_safe_schema = {}
    for table_name, info in schema.items():
        rendered_columns = []
        raw_column_descriptions = info.get("column_descriptions", {}) if isinstance(info, dict) else {}
        column_descriptions = raw_column_descriptions if isinstance(raw_column_descriptions, dict) else {}

        for name, dtype in info["columns"]:
            rendered_column = {"name": _as_plain_string(name), "type": _as_plain_string(dtype)}
            column_description = column_descriptions.get(name)
            if isinstance(column_description, str) and column_description.strip():
                rendered_column["description"] = column_description.strip()
            rendered_columns.append(rendered_column)

        rendered_table = {
            "columns": rendered_columns,
            "primary_keys": [_as_plain_string(column_name) for column_name in info["primary_keys"]],
            "foreign_keys": [
                {
                    "col": _as_plain_string(fk["col"]),
                    "ref_table": _as_plain_string(fk["ref_table"]),
                    "ref_col": _as_plain_string(fk["ref_col"]),
                }
                for fk in info["foreign_keys"]
            ],
        }

        table_description = info.get("description") if isinstance(info, dict) else None
        if isinstance(table_description, str) and table_description.strip():
            rendered_table["description"] = table_description.strip()

        prompt_safe_schema[_as_plain_string(table_name)] = rendered_table
    return yaml.safe_dump(prompt_safe_schema, sort_keys=True, allow_unicode=True)


def print_schema_yaml(schema):
    print(export_schema_to_yaml(schema).rstrip())


def save_schema_to_yaml_file(schema, filepath: str):
    output_path = os.path.abspath(filepath)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(export_schema_to_yaml(schema))


def print_db_schema(schema):
    for table, info in schema.items():
        print(f"Table: {table}")
        table_description = info.get("description") if isinstance(info, dict) else None
        if isinstance(table_description, str) and table_description:
            print(f"  Description : {table_description}")

        raw_column_descriptions = info.get("column_descriptions", {}) if isinstance(info, dict) else {}
        column_descriptions = raw_column_descriptions if isinstance(raw_column_descriptions, dict) else {}
        rendered_columns = []
        for name, dtype in info["columns"]:
            rendered_column = f"{name} ({dtype})"
            column_description = column_descriptions.get(name)
            if isinstance(column_description, str) and column_description.strip():
                rendered_column = f"{rendered_column}: {column_description.strip()}"
            rendered_columns.append(rendered_column)

        print(f"  Columns     : {', '.join(rendered_columns)}")
        print(f"  Primary Keys: {', '.join(info['primary_keys']) or 'none'}")
        if info["foreign_keys"]:
            for fk in info["foreign_keys"]:
                print(f"  FK          : {fk['col']} -> {fk['ref_table']}.{fk['ref_col']}")
        else:
            print(f"  FK          : none")
        print("-" * 40)


# =========================
# MAIN
# =========================

if __name__ == "__main__":

    schema = get_db_schema()
    save_schema_to_yaml_file(schema, DEFAULT_SCHEMA_OUTPUT_PATH)
    print_schema_yaml(schema)
    print(f"\nDB_SCHEMA_YAML: {DEFAULT_SCHEMA_OUTPUT_PATH}")
