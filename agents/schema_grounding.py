"""Deterministic grounding for business terms with one exact schema meaning."""

from __future__ import annotations

import re
from typing import Any, Dict, Iterable, List, Tuple

from storage.chromadb.schema_metadata import SCHEMA_FIELDS


# Only include phrases whose meaning is unambiguous in the current schema.
# Ambiguous terms such as "退货率" are intentionally left to contextual retrieval.
EXACT_TERM_FIELDS: Tuple[Tuple[str, str, str], ...] = (
    ("会员等级", "customers", "membership_tier"),
    ("会员级别", "customers", "membership_tier"),
    ("vip等级", "customers", "membership_tier"),
    ("获客渠道", "customers", "acquisition_channel"),
    ("注册日期", "customers", "registration_date"),
    ("客户流失", "customers", "churned"),
    ("流失客户", "customers", "churned"),
    ("最近购买", "customers", "days_since_last_purchase"),
    ("订单状态", "orders", "order_status"),
    ("订单评分", "orders", "customer_rating"),
    ("客户评分", "orders", "customer_rating"),
    ("月度营收", "monthly_revenue", "revenue_usd"),
    ("月营收", "monthly_revenue", "revenue_usd"),
    ("月度退货率", "monthly_revenue", "return_rate"),
    ("商品退货率", "product_summary", "return_rate"),
    ("商品评分", "product_summary", "avg_rating"),
)


def find_required_fields(user_query: str) -> List[Dict[str, Any]]:
    """Return exact schema fields explicitly named by an unambiguous term."""
    query = user_query.lower()
    field_map = {
        (field["table_name"], field["column_name"]): field
        for field in SCHEMA_FIELDS
    }
    required: List[Dict[str, Any]] = []
    seen = set()

    for phrase, table_name, column_name in EXACT_TERM_FIELDS:
        key = (table_name, column_name)
        if phrase in query and key not in seen and key in field_map:
            field = field_map[key]
            required.append({
                "table_name": table_name,
                "column_name": column_name,
                "dtype": field.get("dtype", ""),
                "business_term": field.get("business_term", ""),
                "aliases": field.get("aliases", []),
                "relevance": "required",
                "grounded_by": phrase,
            })
            seen.add(key)

    return required


def merge_required_fields(
    selected_fields: Iterable[Dict[str, Any]], user_query: str
) -> List[Dict[str, Any]]:
    """Ensure deterministic anchors survive vector retrieval and LLM reranking."""
    merged = [dict(item) for item in selected_fields]
    positions = {
        (item.get("table_name"), item.get("column_name")): index
        for index, item in enumerate(merged)
    }

    for required in find_required_fields(user_query):
        key = (required["table_name"], required["column_name"])
        if key in positions:
            merged[positions[key]].update(required)
        else:
            positions[key] = len(merged)
            merged.append(required)

    return merged


def missing_required_columns(
    sql: str, selected_fields: Iterable[Dict[str, Any]]
) -> List[str]:
    """Return required identifiers that a generated SQL statement ignored."""
    missing = []
    sql_lower = sql.lower()
    for item in selected_fields:
        if item.get("relevance") != "required":
            continue
        column = str(item.get("column_name", "")).lower()
        if column and not re.search(rf"\b{re.escape(column)}\b", sql_lower):
            missing.append(f"{item.get('table_name')}.{column}")
    return missing


def invalid_qualified_columns(sql: str) -> List[str]:
    """Validate alias.column ownership against the authoritative schema metadata."""
    table_columns: Dict[str, set[str]] = {}
    column_tables: Dict[str, List[str]] = {}
    for field in SCHEMA_FIELDS:
        table = field["table_name"].lower()
        column = field["column_name"].lower()
        table_columns.setdefault(table, set()).add(column)
        column_tables.setdefault(column, []).append(table)

    aliases: Dict[str, str] = {}
    reserved = {
        "where", "group", "order", "limit", "left", "right", "inner",
        "outer", "join", "on", "having", "union", "cross",
    }
    source_pattern = re.compile(
        r"\b(?:from|join)\s+([a-z_]\w*)(?:\s+(?:as\s+)?([a-z_]\w*))?",
        re.IGNORECASE,
    )
    for match in source_pattern.finditer(sql):
        table = match.group(1).lower()
        alias = (match.group(2) or table).lower()
        if alias in reserved:
            alias = table
        aliases[table] = table
        aliases[alias] = table

    invalid = []
    for alias, column in re.findall(
        r"\b([a-z_]\w*)\.([a-z_]\w*)\b", sql, flags=re.IGNORECASE
    ):
        table = aliases.get(alias.lower())
        column_lower = column.lower()
        if not table or table not in table_columns:
            continue
        if column_lower not in table_columns[table]:
            owners = column_tables.get(column_lower, [])
            if owners:
                invalid.append(
                    f"{alias}.{column} 不属于 {table}；该字段属于 "
                    + ", ".join(f"{owner}.{column_lower}" for owner in owners)
                )
            else:
                invalid.append(f"{alias}.{column} 不存在于已知 Schema")

    return list(dict.fromkeys(invalid))


def percentage_scale_issue(user_query: str, sql: str) -> str | None:
    """Require business percentages to use the human-readable 0-100 scale."""
    if not any(term in user_query.lower() for term in ("占比", "百分比")):
        return None
    if re.search(r"(?:\*\s*100(?:\.0+)?|100(?:\.0+)?\s*\*)", sql):
        return None
    return "用户要求占比/百分比，结果必须乘以100并按0-100百分数口径输出"
