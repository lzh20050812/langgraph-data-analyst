"""Evidence bundle helpers shared by workflow nodes."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from agents.contracts import validate_evidence
from agents.evidence_validation import build_facts, validate_query_evidence


def build_evidence_bundle(
    *,
    user_query: str,
    task_plan: Dict[str, Any],
    sql: Optional[str],
    rows: Optional[Iterable[Dict[str, Any]]],
) -> Dict[str, Any]:
    materialized_rows: List[Dict[str, Any]] = list(rows or [])
    columns = list(materialized_rows[0].keys()) if materialized_rows else []
    analysis_request = task_plan.get("analysis_request") or {}
    metric_catalog_version = analysis_request.get("metric_catalog_version")
    validation = validate_query_evidence(
        sql=sql, analysis_request=analysis_request, rows=materialized_rows
    )
    limitations = [
        item["message"] for item in validation.get("checks", [])
        if item["status"] == "unverifiable"
    ]
    return validate_evidence({
        "user_query": user_query,
        "task_plan": task_plan,
        "sql": sql,
        "columns": columns,
        "rows": materialized_rows,
        "row_count": len(materialized_rows),
        "analysis_request": analysis_request,
        "metric_catalog_version": metric_catalog_version,
        "source_tables": validation.get("source_tables", []),
        "facts": build_facts(
            materialized_rows, metric_catalog_version, analysis_request
        ),
        "validation": validation,
        "query_contract": task_plan.get("query_contract") or {},
        "provenance": {
            "sql": "runtime_executed",
            "request_schema_version": analysis_request.get("schema_version"),
            "metric_catalog_version": metric_catalog_version,
        },
        "limitations": limitations,
        "report_validation": {},
        "data_quality": {},
        "analysis_results": {},
        "prediction_results": {},
    })


def update_evidence(
    evidence: Optional[Dict[str, Any]], **updates: Any
) -> Dict[str, Any]:
    """Return the same bundle after applying explicit node outputs."""
    bundle = evidence if evidence is not None else {}
    bundle.update(updates)
    return bundle
