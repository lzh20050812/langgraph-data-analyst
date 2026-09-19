"""Versioned dataset audit and cross-capability metric helpers for stage five."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
import statistics
from typing import Any, Iterable

from sqlglot import exp, parse_one
from sqlglot.errors import ParseError

from agents.business_semantics import build_query_contract
from storage.chromadb.schema_metadata import SCHEMA_FIELDS


ROOT = Path(__file__).resolve().parents[2]
CATALOG_PATH = ROOT / "evaluation/data/benchmark_catalog_v2026_09.json"
VALID_USAGES = {
    "development_regression", "validation_and_frozen_test",
    "frozen_test", "transfer_test",
}
VALID_TASK_TYPES = {"sql_query", "analysis", "prediction", "mixed"}
VALID_EVIDENCE_STATUSES = {"verified", "failed", "unverifiable"}


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _hash(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _issue(dataset_id: str, sample_id: Any, code: str, message: str) -> dict[str, Any]:
    return {
        "dataset_id": dataset_id, "sample_id": sample_id,
        "code": code, "message": message,
    }


def _known_schema() -> tuple[set[str], set[str]]:
    tables = {item["table_name"] for item in SCHEMA_FIELDS}
    fields = {f"{item['table_name']}.{item['column_name']}" for item in SCHEMA_FIELDS}
    return tables, fields


def _audit_sql(
    dataset_id: str, sample_id: Any, sql: str, known_tables: set[str]
) -> list[dict[str, Any]]:
    issues: list[dict[str, Any]] = []
    try:
        tree = parse_one(sql, read="mysql")
    except ParseError as exc:
        return [_issue(dataset_id, sample_id, "sql_parse", str(exc))]
    if not isinstance(tree, exp.Query):
        issues.append(_issue(dataset_id, sample_id, "sql_read_only", "expected SQL is not a query"))
    ctes = {cte.alias_or_name.lower() for cte in tree.find_all(exp.CTE)}
    referenced = {
        table.name for table in tree.find_all(exp.Table)
        if table.name and table.name.lower() not in ctes
    }
    unknown = sorted(referenced - known_tables)
    if unknown:
        issues.append(_issue(
            dataset_id, sample_id, "known_tables",
            "unknown expected SQL tables: " + ", ".join(unknown),
        ))
    return issues


def audit_catalog(catalog_path: Path = CATALOG_PATH) -> dict[str, Any]:
    """Audit dataset identity, hashes, labels, splits, and parseable references."""
    catalog = _read_json(catalog_path)
    issues: list[dict[str, Any]] = []
    dataset_results = []
    known_tables, known_fields = _known_schema()
    total = 0

    for spec in catalog.get("datasets", []):
        dataset_id = spec.get("dataset_id", "unknown")
        path = ROOT / spec.get("path", "")
        if spec.get("usage") not in VALID_USAGES:
            issues.append(_issue(dataset_id, None, "usage", "unknown dataset usage"))
        if not path.is_file():
            issues.append(_issue(dataset_id, None, "missing_file", str(path)))
            continue
        actual_hash = _hash(path)
        if actual_hash != spec.get("sha256"):
            issues.append(_issue(dataset_id, None, "sha256", "dataset hash mismatch"))
        rows = _read_json(path)
        if not isinstance(rows, list):
            issues.append(_issue(dataset_id, None, "format", "dataset root must be a list"))
            continue
        total += len(rows)
        if len(rows) != spec.get("samples"):
            issues.append(_issue(dataset_id, None, "sample_count", "catalog count mismatch"))

        ids = [row.get("id", index + 1) for index, row in enumerate(rows)]
        if len({str(value) for value in ids}) != len(ids):
            issues.append(_issue(dataset_id, None, "unique_id", "duplicate sample ids"))
        for index, row in enumerate(rows):
            sample_id = row.get("id", index + 1)
            if (
                not str(row.get("query", "")).strip()
                and "turns" not in row
                and not str(row.get("description", "")).strip()
            ):
                issues.append(_issue(
                    dataset_id, sample_id, "required_fields",
                    "missing query/turns/description",
                ))
            for table in row.get("expected_tables", []):
                if table not in known_tables:
                    issues.append(_issue(dataset_id, sample_id, "known_tables", table))
            for field in row.get("expected_fields", []):
                if field not in known_fields:
                    issues.append(_issue(dataset_id, sample_id, "known_fields", field))
            if row.get("expected_sql"):
                issues.extend(_audit_sql(
                    dataset_id, sample_id, row["expected_sql"], known_tables
                ))
            if "task_type" in row and row["task_type"] not in VALID_TASK_TYPES:
                issues.append(_issue(dataset_id, sample_id, "task_type", row["task_type"]))
            if row.get("expected_recipe"):
                contract = build_query_contract(row["query"])
                if not contract or contract.recipe_id != row["expected_recipe"]:
                    issues.append(_issue(
                        dataset_id, sample_id, "known_recipe",
                        f"expected {row['expected_recipe']}, got {getattr(contract, 'recipe_id', None)}",
                    ))
            if "turns" in row and (not row["turns"] or not row.get("expected_final")):
                issues.append(_issue(dataset_id, sample_id, "turn_expectations", "missing turns/final expectation"))
            if "expected_status" in row and row["expected_status"] not in VALID_EVIDENCE_STATUSES:
                issues.append(_issue(dataset_id, sample_id, "expected_status", row["expected_status"]))
            if row.get("source_test"):
                test_path = ROOT / row["source_test"].split("::", 1)[0]
                if not test_path.is_file():
                    issues.append(_issue(dataset_id, sample_id, "source_test", str(test_path)))

        split = spec.get("split") or {}
        if split:
            validation = {str(value) for value in split.get("validation_ids", [])}
            frozen = {str(value) for value in split.get("frozen_test_ids", [])}
            all_ids = {str(value) for value in ids}
            if validation & frozen or validation | frozen != all_ids:
                issues.append(_issue(
                    dataset_id, None, "disjoint_split",
                    "validation and frozen test ids must be disjoint and exhaustive",
                ))
        dataset_results.append({
            "dataset_id": dataset_id, "samples": len(rows),
            "usage": spec.get("usage"), "sha256": actual_hash,
            "status": "passed" if not any(
                item["dataset_id"] == dataset_id for item in issues
            ) else "failed",
        })

    if total != catalog.get("total_samples"):
        issues.append(_issue("catalog", None, "total_samples", f"expected {catalog.get('total_samples')}, got {total}"))
    return {
        "catalog_version": catalog.get("catalog_version"),
        "total_samples": total,
        "dataset_count": len(dataset_results),
        "status": "passed" if not issues else "failed",
        "datasets": dataset_results,
        "issues": issues,
    }


def validate_sample_record(record: dict[str, Any]) -> list[str]:
    """Validate the minimum per-sample artifact required for fair comparisons."""
    required = {
        "sample_id", "dataset_id", "split", "architecture", "success",
        "duration_ms", "llm_calls", "tool_calls", "prompt_tokens",
        "completion_tokens", "total_tokens", "estimated_cost_usd",
        "error", "failure_category", "output", "config", "versions",
    }
    missing = sorted(required - record.keys())
    errors = [f"missing field: {field}" for field in missing]
    for field in ("duration_ms", "llm_calls", "tool_calls", "prompt_tokens", "completion_tokens", "total_tokens", "estimated_cost_usd"):
        value = record.get(field)
        if value is not None and (not isinstance(value, (int, float)) or value < 0):
            errors.append(f"{field} must be non-negative or null")
    if record.get("success") is False and not record.get("failure_category"):
        errors.append("failed records require failure_category")
    return errors


def _rate(records: Iterable[dict[str, Any]], field: str) -> dict[str, Any]:
    values = [record[field] for record in records if isinstance(record.get(field), bool)]
    return {
        "value": round(sum(values) / len(values), 6) if values else None,
        "n": len(values),
    }


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute independent capability metrics without treating missing labels as zero."""
    durations = sorted(float(item["duration_ms"]) for item in records if item.get("duration_ms") is not None)
    percentile = lambda fraction: (
        durations[min(len(durations) - 1, max(0, int(len(durations) * fraction + 0.999999) - 1))]
        if durations else None
    )
    return {
        "samples": len(records),
        "query": {
            "business_answer_accuracy": _rate(records, "answer_correct"),
            "condition_retention_rate": _rate(records, "conditions_retained"),
            "execution_success_rate": _rate(records, "execution_success"),
            "repair_success_rate": _rate(records, "repair_success"),
        },
        "retrieval": {
            "table_recall": _rate(records, "table_recall_pass"),
            "field_recall": _rate(records, "field_recall_pass"),
            "relationship_coverage": _rate(records, "relationship_covered"),
        },
        "multiturn": {
            "task_completion_rate": _rate(records, "task_completed"),
            "inheritance_correct_rate": _rate(records, "inheritance_correct"),
            "clarification_correct_rate": _rate(records, "clarification_correct"),
        },
        "report": {
            "numeric_consistency_rate": _rate(records, "numeric_consistent"),
            "citation_correct_rate": _rate(records, "citations_correct"),
            "unsupported_claim_free_rate": _rate(records, "unsupported_claim_free"),
        },
        "engineering": {
            "latency_p50_ms": round(statistics.median(durations), 3) if durations else None,
            "latency_p95_ms": round(percentile(0.95), 3) if durations else None,
            "llm_calls": sum(item.get("llm_calls") or 0 for item in records),
            "tool_calls": sum(item.get("tool_calls") or 0 for item in records),
            "total_tokens": sum(item.get("total_tokens") or 0 for item in records),
            "estimated_cost_usd": round(sum(item.get("estimated_cost_usd") or 0 for item in records), 6),
            "recovery_success_rate": _rate(records, "recovery_success"),
        },
    }
