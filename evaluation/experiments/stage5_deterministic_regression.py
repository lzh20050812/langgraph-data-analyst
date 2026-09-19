"""Fast, no-LLM stage-five regression for datasets, retrieval, and multi-turn state."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
from time import perf_counter
from typing import Any

from agents.analysis_request import build_analysis_request, inherit_analysis_request
from agents.business_semantics import build_query_contract
from agents.schema_retrieval import complete_relationship_paths, lexical_schema_search
from evaluation.framework.stage5 import (
    ROOT, audit_catalog, summarize_records, validate_sample_record,
)
from storage.chromadb.schema_metadata import SCHEMA_CATALOG_VERSION


DEFAULT_OUTPUT = ROOT / "evaluation/optimization_results/stage5_20260918"


def _load(relative: str) -> list[dict[str, Any]]:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def _git_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
            encoding="utf-8", errors="replace", stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unavailable"


def _record(
    *, sample_id: Any, dataset_id: str, split: str,
    success: bool, duration_ms: float, output: dict[str, Any],
    failure_category: str | None = None, **metrics: Any,
) -> dict[str, Any]:
    record = {
        "sample_id": str(sample_id), "dataset_id": dataset_id,
        "split": split, "architecture": "deterministic_control",
        "success": success, "duration_ms": round(duration_ms, 3),
        "llm_calls": 0, "tool_calls": 1,
        "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
        "estimated_cost_usd": 0.0, "error": None if success else failure_category,
        "failure_category": None if success else failure_category,
        "output": output,
        "config": {"network": False, "paid_model": False, "random_seed": 42},
        "versions": {
            "protocol": "stage5-deterministic-v1",
            "schema_catalog": SCHEMA_CATALOG_VERSION,
            "git_commit": _git_revision(),
        },
    }
    record.update(metrics)
    errors = validate_sample_record(record)
    if errors:
        raise ValueError(f"invalid sample record {sample_id}: {errors}")
    return record


def _nested(value: dict[str, Any], dotted: str) -> Any:
    current: Any = value
    for part in dotted.split("."):
        if not isinstance(current, dict):
            return None
        current = current.get(part)
    return current


def run() -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    audit = audit_catalog()

    for index, sample in enumerate(_load(
        "evaluation/data/sql_semantics/paraphrase_queries.json"
    ), start=1):
        started = perf_counter()
        contract = build_query_contract(sample["query"])
        actual = contract.recipe_id if contract else None
        success = actual == sample["expected_recipe"]
        records.append(_record(
            sample_id=f"P{index}", dataset_id="semantic-paraphrase-v1",
            split="development_regression", success=success,
            duration_ms=(perf_counter() - started) * 1000,
            output={"expected_recipe": sample["expected_recipe"], "actual_recipe": actual},
            failure_category=None if success else "semantic_recipe_mismatch",
            answer_correct=success, conditions_retained=success,
            execution_success=None, repair_success=None,
        ))

    for sample in _load("evaluation/data/rag_schema/support_queries.json"):
        started = perf_counter()
        selected = lexical_schema_search(
            sample["query"], top_k=10, data_source="support_ops"
        )
        completed = complete_relationship_paths(selected, data_source="support_ops")
        fields = {f"{item['table_name']}.{item['column_name']}" for item in completed}
        tables = {item["table_name"] for item in completed}
        field_pass = set(sample["expected_fields"]).issubset(fields)
        table_pass = set(sample["expected_tables"]).issubset(tables)
        relationship_pass = None
        if sample.get("requires_join"):
            relationship_pass = {
                "support_agents.agent_id", "support_tickets.agent_id"
            }.issubset(fields)
        success = field_pass and table_pass and relationship_pass is not False
        records.append(_record(
            sample_id=sample["id"], dataset_id="schema-support-transfer-v1",
            split="transfer_test", success=success,
            duration_ms=(perf_counter() - started) * 1000,
            output={"retrieved_fields": sorted(fields), "retrieved_tables": sorted(tables)},
            failure_category=None if success else "retrieval_miss",
            table_recall_pass=table_pass, field_recall_pass=field_pass,
            relationship_covered=relationship_pass,
        ))

    for sample in _load("evaluation/data/multiturn/test_cases.json"):
        started = perf_counter()
        effective = None
        changes = []
        for turn in sample["turns"]:
            current = build_analysis_request(turn)
            effective, turn_changes = inherit_analysis_request(effective, current, turn)
            changes.extend(turn_changes)
        mismatches = {
            key: {"expected": expected, "actual": _nested(effective or {}, key)}
            for key, expected in sample["expected_final"].items()
            if _nested(effective or {}, key) != expected
        }
        expected_issue = sample.get("expected_issue_contains")
        issues = (effective or {}).get("unsupported_conditions") or []
        issue_ok = not expected_issue or any(expected_issue in issue for issue in issues)
        success = not mismatches and issue_ok
        records.append(_record(
            sample_id=sample["id"], dataset_id="multiturn-v1",
            split="development_regression", success=success,
            duration_ms=(perf_counter() - started) * 1000,
            output={
                "effective_request": effective, "changes": changes,
                "mismatches": mismatches, "expected_issue_found": issue_ok,
            },
            failure_category=None if success else "multiturn_state_mismatch",
            task_completed=success, inheritance_correct=not mismatches,
            clarification_correct=issue_ok if expected_issue else None,
        ))

    failure_cases = _load("evaluation/data/failure_cases/evidence_cases.json")
    result = {
        "protocol": "stage5-deterministic-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "No LLM calls; deterministic CI and dataset audit only.",
        "dataset_audit": audit,
        "failure_case_library": {
            "samples": len(failure_cases),
            "categories": sorted({item["category"] for item in failure_cases}),
            "status": "passed" if audit["status"] == "passed" else "failed",
        },
        "metrics": summarize_records(records),
        "records": records,
        "success": audit["status"] == "passed" and all(item["success"] for item in records),
    }
    return result


def _report(result: dict[str, Any]) -> str:
    metrics = result["metrics"]
    return f"""# Stage 5 deterministic regression

- Protocol: `{result['protocol']}`
- Dataset audit: **{result['dataset_audit']['status']}** ({result['dataset_audit']['total_samples']} samples)
- Per-sample deterministic runs: **{metrics['samples']}**
- Overall status: **{'passed' if result['success'] else 'failed'}**
- Paid LLM calls: **0**

## Metrics

- Semantic answer accuracy: `{metrics['query']['business_answer_accuracy']['value']}` (`n={metrics['query']['business_answer_accuracy']['n']}`)
- Support transfer table recall pass rate: `{metrics['retrieval']['table_recall']['value']}`
- Support transfer field recall pass rate: `{metrics['retrieval']['field_recall']['value']}`
- Relationship coverage: `{metrics['retrieval']['relationship_coverage']['value']}`
- Multi-turn completion: `{metrics['multiturn']['task_completion_rate']['value']}`
- Multi-turn inheritance correctness: `{metrics['multiturn']['inheritance_correct_rate']['value']}`
- Latency P50/P95: `{metrics['engineering']['latency_p50_ms']}` / `{metrics['engineering']['latency_p95_ms']}` ms

## Boundary

This fast run validates labels, deterministic semantic routing, transfer retrieval, and multi-turn state. It is not a substitute for the paid, same-model architecture comparison and makes no claim about current LLM accuracy.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true", help="Do not write artifacts")
    args = parser.parse_args()
    result = run()
    print(json.dumps({
        "success": result["success"], "audit": result["dataset_audit"]["status"],
        "samples": result["dataset_audit"]["total_samples"],
        "deterministic_records": len(result["records"]),
    }, ensure_ascii=False, indent=2))
    if not args.check:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "results.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (args.output_dir / "report.md").write_text(_report(result), encoding="utf-8")
    if not result["success"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
