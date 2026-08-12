"""Evidence bundle helpers shared by workflow nodes."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional


def build_evidence_bundle(
    *,
    user_query: str,
    task_plan: Dict[str, Any],
    sql: Optional[str],
    rows: Optional[Iterable[Dict[str, Any]]],
) -> Dict[str, Any]:
    materialized_rows: List[Dict[str, Any]] = list(rows or [])
    columns = list(materialized_rows[0].keys()) if materialized_rows else []
    return {
        "user_query": user_query,
        "task_plan": task_plan,
        "sql": sql,
        "columns": columns,
        "rows": materialized_rows,
        "row_count": len(materialized_rows),
        "data_quality": {},
        "analysis_results": {},
        "prediction_results": {},
    }


def update_evidence(
    evidence: Optional[Dict[str, Any]], **updates: Any
) -> Dict[str, Any]:
    """Return the same bundle after applying explicit node outputs."""
    bundle = evidence if evidence is not None else {}
    bundle.update(updates)
    return bundle
