"""AST-based SQL safety and resource-bound enforcement."""

from __future__ import annotations

import re
from typing import Iterable

from sqlglot import exp, parse
from sqlglot.errors import ParseError


SIDE_EFFECT_FUNCTIONS = {
    "benchmark",
    "get_lock",
    "is_free_lock",
    "load_file",
    "master_pos_wait",
    "release_lock",
    "sleep",
}


def validate_read_only_sql(
    sql: str,
    *,
    allowed_tables: Iterable[str] | None = None,
    dialect: str = "mysql",
) -> tuple[bool, str]:
    if not sql or not sql.strip():
        return False, "SQL 为空"

    try:
        statements = parse(sql, read=dialect)
    except ParseError as exc:
        return False, f"SQL AST 解析失败: {exc.errors[0].get('description', 'syntax error')}"

    if len(statements) != 1:
        return False, "仅允许单条只读查询"
    tree = statements[0]
    if tree is None or not isinstance(tree, exp.Query):
        return False, "仅允许只读 SELECT/CTE 查询"

    forbidden_types = (
        exp.Insert,
        exp.Update,
        exp.Delete,
        exp.Create,
        exp.Drop,
        exp.Alter,
        exp.Command,
        exp.Transaction,
        exp.Merge,
        exp.Copy,
        exp.Into,
    )
    if any(tree.find(node_type) is not None for node_type in forbidden_types):
        return False, "SQL AST 中包含写入、导出或管理操作，仅允许只读查询"

    # AST function names cover both known functions and anonymous MySQL calls.
    for function in tree.find_all(exp.Func):
        function_name = function.sql_name().lower()
        if isinstance(function, exp.Anonymous):
            function_name = function.name.lower()
        if function_name in SIDE_EFFECT_FUNCTIONS:
            return False, f"禁止调用可能产生副作用的函数: {function_name}"

    # Keep a lexical backstop for MySQL extensions that a lenient parser may
    # preserve as generic commands or comments.
    normalized = re.sub(r"/\*.*?\*/|--[^\n]*|#[^\n]*", " ", sql, flags=re.DOTALL)
    if re.search(r"\bINTO\s+(?:OUTFILE|DUMPFILE)\b", normalized, re.IGNORECASE):
        return False, "禁止将查询结果写入文件"

    if allowed_tables is not None:
        allowed = {name.lower() for name in allowed_tables}
        cte_names = {cte.alias_or_name.lower() for cte in tree.find_all(exp.CTE)}
        referenced = {
            table.name.lower()
            for table in tree.find_all(exp.Table)
            if table.name and table.name.lower() not in cte_names
        }
        denied = sorted(referenced - allowed)
        if denied:
            return False, "查询引用了未授权数据表: " + ", ".join(denied)

    return True, "OK"


def enforce_row_limit(sql: str, max_rows: int, dialect: str = "mysql") -> str:
    """Return a semantically equivalent query capped at ``max_rows`` rows."""
    if max_rows < 1:
        raise ValueError("max_rows must be positive")
    statements = parse(sql, read=dialect)
    if len(statements) != 1 or not isinstance(statements[0], exp.Query):
        raise ValueError("仅允许单条只读 SELECT/CTE 查询")

    tree = statements[0]
    limit = tree.args.get("limit")
    existing_limit: int | None = None
    if limit is not None and isinstance(limit.expression, exp.Literal):
        try:
            existing_limit = int(limit.expression.this)
        except (TypeError, ValueError):
            existing_limit = None
    if existing_limit is None or existing_limit > max_rows:
        tree = tree.limit(max_rows)
    return tree.sql(dialect=dialect)
