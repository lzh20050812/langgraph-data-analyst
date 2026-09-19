"""Deterministic SQL/request, result, and report evidence validation."""

from __future__ import annotations

import math
from numbers import Number
import re
from typing import Any

from sqlglot import exp, parse_one
from sqlglot.errors import ParseError

from agents.analysis_request import METRIC_CATALOG
from storage.chromadb.schema_metadata import SCHEMA_RELATIONSHIPS


DIMENSION_COLUMNS = {
    "country": {"country"},
    "membership_tier": {"membership_tier"},
    "acquisition_channel": {"acquisition_channel"},
    "time": {"year", "month", "quarter", "order_date", "opened_at"},
    "device": {"preferred_device", "device_used"},
    "channel": {"channel"},
    "priority": {"priority"},
    "status": {"status", "order_status"},
    "team": {"team"},
}

METRIC_RESULT_COLUMNS = {
    "revenue": {"revenue", "revenue_usd", "total_revenue"},
    "customer_count": {"customer_count", "cnt", "count"},
    "avg_order_value": {"avg_order_value", "avg_aov", "aov"},
    "paid_amount": {"paid_amount", "total_amount", "total_amount_usd", "revenue"},
    "return_rate": {"return_rate", "return_rate_pct"},
    "churn_rate": {"churn_rate", "churn_rate_pct"},
    "ticket_count": {"ticket_count", "cnt", "count"},
    "avg_resolution_hours": {"avg_resolution_hours", "avg_hours", "resolution_hours"},
    "satisfaction_score": {"satisfaction_score", "avg_satisfaction_score", "csat"},
}


def _check(code: str, status: str, message: str, **details: Any) -> dict[str, Any]:
    return {"code": code, "status": status, "message": message, "details": details}


def _source_tables(tree: exp.Expression) -> list[str]:
    ctes = {cte.alias_or_name.lower() for cte in tree.find_all(exp.CTE)}
    return sorted({
        table.name.lower() for table in tree.find_all(exp.Table)
        if table.name and table.name.lower() not in ctes
    })


def _selected_columns(tree: exp.Expression) -> set[str]:
    return {column.name.lower() for column in tree.find_all(exp.Column) if column.name}


def _grouped_columns(tree: exp.Expression) -> set[str]:
    group = tree.args.get("group")
    if group is None:
        return set()
    return {
        column.name.lower()
        for expression in group.expressions
        for column in expression.find_all(exp.Column)
        if column.name
    } | {
        expression.name.lower()
        for expression in group.expressions
        if isinstance(expression, exp.Column) and expression.name
    }


def _result_columns(rows: list[dict[str, Any]]) -> set[str]:
    return {str(column).lower() for row in rows[:1] for column in row}


def _has_aggregate(sql_lower: str, function: str, column: str) -> bool:
    return bool(re.search(
        rf"\b{function}\s*\([^)]*\b{re.escape(column)}\b[^)]*\)", sql_lower
    ))


def _metric_sql_issue(metric_id: str, sql_lower: str, tables: set[str]) -> str | None:
    definition = METRIC_CATALOG[metric_id]
    if definition.source_table == "unavailable":
        return "; ".join(definition.limitations)
    if definition.source_table not in tables:
        return f"指标 {definition.name} 应来自 {definition.source_table}"
    formula = definition.formula.lower()
    if formula.startswith("sum("):
        column = definition.required_fields[0]
        if not _has_aggregate(sql_lower, "sum", column):
            return f"指标 {definition.name} 必须按 SUM({column}) 计算"
    elif formula.startswith("avg("):
        column = definition.required_fields[0]
        if not _has_aggregate(sql_lower, "avg", column):
            return f"指标 {definition.name} 必须按 AVG({column}) 计算"
    elif formula == "count(*)" and "count(" not in sql_lower:
        return f"指标 {definition.name} 必须使用 COUNT 聚合"
    return None


def _join_fanout_issues(
    metrics: list[str], tables: set[str], sql_lower: str
) -> list[str]:
    issues = []
    normalized_sql = re.sub(r"[\s`]", "", sql_lower)
    for relationship in SCHEMA_RELATIONSHIPS:
        if not {relationship["left_table"], relationship["right_table"]}.issubset(tables):
            continue
        one_side = relationship["right_table"]
        for metric_id in metrics:
            definition = METRIC_CATALOG[metric_id]
            if definition.source_table != one_side:
                continue
            if metric_id == "customer_count" and not re.search(
                r"count\(distinct(?:[a-z_]\w*\.)?customer_id\)", normalized_sql
            ):
                issues.append(
                    f"{one_side} 位于一对多关联的一侧，客户计数必须对 customer_id 去重或先聚合"
                )
            elif any(
                _has_aggregate(sql_lower, function, column)
                for column in definition.required_fields
                for function in ("sum", "avg")
            ) and "select distinct" not in sql_lower:
                issues.append(
                    f"聚合 {one_side} 字段可能被 {relationship['left_table']} 多行重复放大"
                )
    return issues


def validate_query_evidence(
    *,
    sql: str | None,
    analysis_request: dict[str, Any] | None,
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Validate SQL semantics and materialized results against the request."""
    request = analysis_request or {}
    checks: list[dict[str, Any]] = []
    if not sql:
        return {
            "status": "failed", "checks": [_check("sql.present", "failed", "缺少 SQL")]
        }
    try:
        tree = parse_one(sql, read="mysql")
    except ParseError as exc:
        return {
            "status": "failed",
            "checks": [_check("sql.parse", "failed", f"SQL 无法解析: {exc}")],
        }
    tables = set(_source_tables(tree))
    columns = _selected_columns(tree)
    result_columns = _result_columns(rows)
    metrics = list(request.get("metrics") or [])
    dimensions = list(request.get("dimensions") or [])
    if not metrics:
        checks.append(_check(
            "request.metrics", "unverifiable", "请求没有受控指标，无法证明业务口径"
        ))
    for metric_id in metrics:
        issue = _metric_sql_issue(metric_id, sql.lower(), tables)
        checks.append(_check(
            f"metric.{metric_id}", "failed" if issue else "verified",
            issue or f"指标 {metric_id} 的来源与聚合函数一致",
        ))
        expected_outputs = METRIC_RESULT_COLUMNS.get(metric_id, {metric_id})
        if rows:
            output_present = bool(expected_outputs & result_columns)
            checks.append(_check(
                f"result.metric.{metric_id}",
                "verified" if output_present else "failed",
                f"结果包含指标 {metric_id}"
                if output_present else f"结果缺少可识别的指标列 {metric_id}",
                expected=sorted(expected_outputs),
            ))
    grouped_columns = _grouped_columns(tree)
    has_aggregate = any(
        tree.find(kind) is not None for kind in (exp.Sum, exp.Avg, exp.Count)
    )
    for dimension in dimensions:
        candidates = DIMENSION_COLUMNS.get(dimension, {dimension})
        present = bool(candidates & result_columns) if rows else bool(candidates & columns)
        checks.append(_check(
            f"dimension.{dimension}", "verified" if present else "failed",
            f"维度 {dimension} 已保留" if present else f"SQL/结果缺少维度 {dimension}",
        ))
        if has_aggregate:
            grouped = bool(candidates & grouped_columns)
            checks.append(_check(
                f"grain.{dimension}", "verified" if grouped else "failed",
                f"聚合粒度包含维度 {dimension}"
                if grouped else f"聚合查询未按维度 {dimension} 分组",
            ))
    sql_lower = sql.lower()
    literals = {str(literal.this).lower() for literal in tree.find_all(exp.Literal)}
    for country in (request.get("filters") or {}).get("country", []):
        present = country.lower() in literals or country.lower() in sql_lower
        checks.append(_check(
            "filter.country", "verified" if present else "failed",
            f"国家筛选 {country} 已保留" if present else f"SQL 丢失国家筛选 {country}",
        ))
    for year in (request.get("time_scope") or {}).get("years", []):
        present = str(year) in sql_lower
        checks.append(_check(
            "filter.year", "verified" if present else "failed",
            f"年份 {year} 已保留" if present else f"SQL 丢失年份 {year}",
        ))
    requested_limit = request.get("limit")
    if requested_limit:
        limit = tree.args.get("limit")
        actual = None
        if limit is not None and isinstance(limit.expression, exp.Literal):
            actual = int(limit.expression.this)
        checks.append(_check(
            "result.limit", "verified" if actual == requested_limit else "failed",
            f"LIMIT {requested_limit} 已保留" if actual == requested_limit
            else f"请求 LIMIT {requested_limit}，SQL 实际为 {actual}",
        ))
    for rule in request.get("sort") or []:
        order = tree.args.get("order")
        ordered = list(order.expressions) if order is not None else []
        expected_desc = rule.get("direction", "desc") == "desc"
        direction_ok = bool(ordered) and bool(ordered[0].args.get("desc")) == expected_desc
        checks.append(_check(
            "result.sort", "verified" if direction_ok else "failed",
            f"排序方向 {rule.get('direction', 'desc')} 已保留"
            if direction_ok else "SQL 丢失请求的排序或方向不一致",
        ))
        candidates = {rule.get("field", "").lower()} | METRIC_RESULT_COLUMNS.get(
            rule.get("field", ""), set()
        )
        result_field = next((field for field in candidates if field in result_columns), None)
        if rows and result_field:
            values = [row.get(result_field) for row in rows]
            numeric = [float(value) for value in values if isinstance(value, Number)]
            expected = sorted(numeric, reverse=expected_desc)
            checks.append(_check(
                "result.order", "verified" if numeric == expected else "failed",
                "结果顺序与请求一致" if numeric == expected else "结果行顺序与请求不一致",
                field=result_field,
            ))
    for message in _join_fanout_issues(metrics, tables, sql_lower):
        checks.append(_check("join.fanout", "failed", message))
    if not rows:
        checks.append(_check(
            "result.non_empty", "unverifiable", "查询结果为空；条件存在但无法验证数值事实"
        ))
    else:
        checks.append(_check("result.non_empty", "verified", f"返回 {len(rows)} 行"))
        null_columns = sorted({
            column for row in rows for column, value in row.items() if value is None
        })
        if null_columns:
            checks.append(_check(
                "result.nulls", "unverifiable",
                "结果包含 NULL，相关事实不能全部验证", columns=null_columns,
            ))
    statuses = {item["status"] for item in checks}
    status = "failed" if "failed" in statuses else (
        "unverifiable" if "unverifiable" in statuses else "verified"
    )
    return {"status": status, "checks": checks, "source_tables": sorted(tables)}


def _unit_for_column(column: str) -> str | None:
    lowered = column.lower()
    if any(token in lowered for token in ("usd", "amount", "revenue", "spend", "aov")):
        return "USD"
    if "rate" in lowered or "pct" in lowered:
        return "%" if "pct" in lowered else "ratio"
    if "hours" in lowered:
        return "hours"
    if any(token in lowered for token in ("count", "cnt", "orders", "customers")):
        return "count"
    return None


def build_facts(
    rows: list[dict[str, Any]],
    metric_catalog_version: str | None,
    analysis_request: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    request = analysis_request or {}
    dimension_columns = {
        column
        for dimension in request.get("dimensions") or []
        for column in DIMENSION_COLUMNS.get(dimension, {dimension})
    }
    dimension_columns.update({"year", "month", "quarter"})
    facts = []
    for row_index, row in enumerate(rows[:50]):
        dimensions = {
            key: value for key, value in row.items()
            if (
                not isinstance(value, Number)
                or isinstance(value, bool)
                or key.lower() in dimension_columns
                or key.lower().endswith("_id")
            )
        }
        for column, value in row.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, Number)
                or column.lower() in dimension_columns
                or column.lower().endswith("_id")
            ):
                continue
            facts.append({
                "fact_id": f"F{len(facts) + 1}", "kind": "query_value",
                "label": column, "value": value, "unit": _unit_for_column(column),
                "metric_catalog_version": metric_catalog_version,
                "source_column": column, "row_index": row_index,
                "dimensions": dimensions,
                "calculation": "materialized SQL result",
                "verification_status": "verified",
            })
    return facts


def validate_report_claims(report: str, evidence: dict[str, Any]) -> dict[str, Any]:
    """Validate supported numeric report claims and evidence citations."""
    facts = {item["fact_id"]: item for item in evidence.get("facts") or []}
    checks = []
    factual_lines = 0
    requested_years = {
        int(year) for year in (
            (evidence.get("analysis_request") or {}).get("time_scope") or {}
        ).get("years", [])
    }
    fact_years = {
        int(value)
        for fact in facts.values()
        for key, value in (fact.get("dimensions") or {}).items()
        if key.lower() == "year" and str(value).isdigit()
    }
    allowed_years = requested_years | fact_years
    dimension_values = {
        str(value): fact["fact_id"]
        for fact in facts.values()
        for value in (fact.get("dimensions") or {}).values()
        if value is not None
    }
    for line in (part.strip() for part in report.splitlines() if part.strip()):
        years = {
            int(value) for value in re.findall(r"(?<!\d)(?:19|20)\d{2}(?!\d)", line)
        }
        invalid_years = sorted(years - allowed_years)
        if invalid_years:
            checks.append(_check(
                "report.period", "failed", "报告出现请求范围外的年份",
                years=invalid_years, line=line,
            ))
        claim_text = re.sub(r"^\s*\d+[.)、]\s*", "", line)
        claim_text = re.sub(r"\[F\d+\]", "", claim_text)
        claim_text = re.sub(r"(?<!\d)(?:19|20)\d{2}(?!\d)", "", claim_text)
        numbers = [float(value.replace(",", "")) for value in re.findall(
            r"(?<![A-Za-z\d])[+-]?\d+(?:,\d{3})*(?:\.\d+)?", claim_text
        )]
        if not numbers:
            continue
        # Ignore Markdown section numbering and priority labels without claims.
        if re.fullmatch(r"#{1,6}\s+.*", line) and len(numbers) == 1:
            continue
        factual_lines += 1
        citations = re.findall(r"\[(F\d+)\]", line)
        if not citations:
            checks.append(_check(
                "report.citation", "failed", "含数值的报告行缺少事实引用", line=line
            ))
            continue
        unknown = [citation for citation in citations if citation not in facts]
        if unknown:
            checks.append(_check(
                "report.citation", "failed", "报告引用了不存在的事实", citations=unknown
            ))
            continue
        fact_values = [float(facts[citation]["value"]) for citation in citations]
        matches = all(
            any(
                math.isclose(number, fact, rel_tol=1e-6, abs_tol=1e-6)
                or math.isclose(number, fact * 100, rel_tol=1e-6, abs_tol=1e-6)
                for fact in fact_values
            )
            for number in numbers
        )
        checks.append(_check(
            "report.numeric_claim", "verified" if matches else "failed",
            "数值与引用事实一致" if matches else "数值与引用事实不一致",
            line=line, citations=citations,
        ))
        mentioned_dimensions = {
            value for value in dimension_values
            if value and value in line
        }
        cited_dimensions = {
            str(value)
            for citation in citations
            for value in (facts[citation].get("dimensions") or {}).values()
        }
        if mentioned_dimensions and not mentioned_dimensions.issubset(cited_dimensions):
            checks.append(_check(
                "report.dimension", "failed", "报告名称与引用事实的维度不一致",
                mentioned=sorted(mentioned_dimensions),
                cited=sorted(cited_dimensions), line=line,
            ))
        cited_facts = [facts[citation] for citation in citations]
        peer_facts = [
            item for item in facts.values()
            if any(
                item.get("label") == cited.get("label")
                and item.get("unit") == cited.get("unit")
                for cited in cited_facts
            )
        ]
        if ("最高" in line or "最大" in line) and peer_facts:
            maximum = max(float(item["value"]) for item in peer_facts)
            if not any(math.isclose(value, maximum) for value in fact_values):
                checks.append(_check(
                    "report.ranking", "failed", "最高/最大结论未引用最大事实", line=line
                ))
        if ("最低" in line or "最小" in line) and peer_facts:
            minimum = min(float(item["value"]) for item in peer_facts)
            if not any(math.isclose(value, minimum) for value in fact_values):
                checks.append(_check(
                    "report.ranking", "failed", "最低/最小结论未引用最小事实", line=line
                ))
    if factual_lines == 0:
        checks.append(_check(
            "report.facts", "unverifiable", "报告没有可程序验证的数值事实"
        ))
    statuses = {item["status"] for item in checks}
    status = "failed" if "failed" in statuses else (
        "unverifiable" if "unverifiable" in statuses else "verified"
    )
    return {"status": status, "checks": checks, "supported_fact_types": [
        "materialized numeric query values with [F#] citations"
    ]}
