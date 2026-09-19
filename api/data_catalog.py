"""Safe metadata catalog and bounded table previews for the web portal."""

from __future__ import annotations

from collections import defaultdict
import re
from typing import Any

from sqlalchemy import inspect, text

from storage.chromadb.schema_metadata import SCHEMA_FIELDS
from storage.mysql.client import get_connection


def _allowed_table(table_name: str, settings) -> str:
    if not re.fullmatch(r"[A-Za-z0-9_]+", table_name):
        raise KeyError(table_name)
    if table_name not in settings.SQL_ALLOWED_TABLES:
        raise KeyError(table_name)
    return table_name


def _business_metadata() -> dict[str, dict[str, dict[str, Any]]]:
    grouped: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for field in SCHEMA_FIELDS:
        grouped[field["table_name"]][field["column_name"]] = field
    return dict(grouped)


def _table_schema(
    table_name: str, settings, connection, inspector
) -> dict[str, Any]:
    table_name = _allowed_table(table_name, settings)
    business = _business_metadata().get(table_name, {})
    columns = inspector.get_columns(table_name)
    row_count = int(connection.execute(
        text(f"SELECT COUNT(*) FROM `{table_name}`")
    ).scalar_one())
    fields = []
    for column in columns:
        metadata = business.get(column["name"], {})
        fields.append({
            "name": column["name"],
            "type": str(column["type"]),
            "nullable": bool(column.get("nullable", True)),
            "business_term": metadata.get("business_term", ""),
            "aliases": metadata.get("aliases", []),
        })
    return {
        "table_name": table_name,
        "row_count": row_count,
        "column_count": len(fields),
        "fields": fields,
    }


def table_schema(table_name: str, settings) -> dict[str, Any]:
    with get_connection() as connection:
        return _table_schema(table_name, settings, connection, inspect(connection))


def catalog(settings) -> list[dict[str, Any]]:
    with get_connection() as connection:
        inspector = inspect(connection)
        return [
            _table_schema(table, settings, connection, inspector)
            for table in settings.SQL_ALLOWED_TABLES
        ]


def preview(table_name: str, settings, limit: int = 20) -> dict[str, Any]:
    table_name = _allowed_table(table_name, settings)
    limit = max(1, min(limit, 50))
    with get_connection() as connection:
        result = connection.execute(
            text(f"SELECT * FROM `{table_name}` LIMIT :limit"), {"limit": limit}
        )
        columns = list(result.keys())
        rows = [dict(zip(columns, row)) for row in result.fetchall()]
    return {
        "table_name": table_name,
        "columns": columns,
        "rows": rows,
        "count": len(rows),
        "limit": limit,
    }
