"""Same-model architecture comparison with explicit paid-run authorization.

Default invocation is a dry run. External model calls require ``--execute`` and
the exact confirmation string plus sample, call, token, and cost ceilings.
"""

from __future__ import annotations

import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
from typing import Any
from unittest.mock import patch

from agents.run_context import run_hooks
from agents.business_semantics import build_query_contract
from config.settings import get_settings
from evaluation.experiments import run_formal_comparison as legacy
from evaluation.experiments.run_v4_evaluation import run_v4_experiment
from evaluation.framework.stage5 import ROOT, summarize_records, validate_sample_record
from storage.chromadb.schema_metadata import get_schema_fields


DATASET = ROOT / "evaluation/data/multi_agent/test_queries.json"
OUTPUT_ROOT = ROOT / "evaluation/results/architecture_comparison_v2"
CONFIRMATION = "CONFIRM_PAID_EVALUATION"
ARCHITECTURES = ("direct_sql", "single_agent_tools", "controlled_workflow")


class BudgetExceeded(RuntimeError):
    pass


def _git_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True,
            encoding="utf-8", errors="replace", stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unavailable"


def _full_schema_context() -> str:
    return "\n".join(
        f"{item['table_name']}.{item['column_name']} ({item.get('dtype', '')})"
        f" — {item.get('business_term', '')}"
        for item in get_schema_fields("ai_analytics")
    )


class BudgetTracker:
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    @property
    def cost(self) -> float:
        return (
            self.prompt_tokens * self.args.input_cost_per_million
            + self.completion_tokens * self.args.output_cost_per_million
        ) / 1_000_000

    def sink(self, event: dict[str, Any]) -> None:
        if event.get("type") == "llm_call_started":
            self.calls += 1
        elif event.get("type") == "llm_call_completed":
            self.prompt_tokens += int(event.get("prompt_tokens") or 0)
            self.completion_tokens += int(event.get("completion_tokens") or 0)
        if (
            self.calls > self.args.max_total_calls
            or self.total_tokens > self.args.max_total_tokens
            or self.cost > self.args.max_cost_usd
        ):
            raise BudgetExceeded("evaluation budget exhausted")


def validate_execution_authorization(args: argparse.Namespace) -> None:
    if not args.execute:
        return
    if args.confirm != CONFIRMATION:
        raise ValueError(f"paid run requires --confirm {CONFIRMATION}")
    positive = {
        "max_samples": args.max_samples,
        "max_total_calls": args.max_total_calls,
        "max_total_tokens": args.max_total_tokens,
        "max_cost_usd": args.max_cost_usd,
    }
    invalid = [name for name, value in positive.items() if value is None or value <= 0]
    if invalid:
        raise ValueError("paid run requires positive caps: " + ", ".join(invalid))
    if args.max_samples > 50:
        raise ValueError("max_samples cannot exceed the fixed 50-sample suite")
    if args.input_cost_per_million <= 0 or args.output_cost_per_million <= 0:
        raise ValueError("paid run requires positive input/output token prices")


def build_protocol(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "protocol": "architecture-comparison-v2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": str(DATASET.relative_to(ROOT)).replace("\\", "/"),
        "max_samples": args.max_samples,
        "model": args.model,
        "temperature": 0.0,
        "architectures": {
            "direct_sql": "one-shot SQL with the full authorized schema and one DB execution",
            "single_agent_tools": "one LLM agent with scoped schema retrieval and DB execution",
            "controlled_workflow": "current planned workflow with the same model and data permissions",
        },
        "shared_permissions": {
            "data_source": "same MySQL snapshot",
            "tables": list(get_settings().SQL_ALLOWED_TABLES),
            "database": "read-only SELECT/CTE with identical timeout and row cap",
            "schema": "same public catalog; delivery differs by architecture",
        },
        "budget": {
            "max_total_calls": args.max_total_calls,
            "max_total_tokens": args.max_total_tokens,
            "max_cost_usd": args.max_cost_usd,
            "input_cost_per_million": args.input_cost_per_million,
            "output_cost_per_million": args.output_cost_per_million,
        },
        "success_rules": {
            "sql_query": "executed result equals expected SQL result",
            "other": "execution succeeds; report quality requires separate blinded human review",
        },
        "git_commit": _git_revision(),
        "historical_results_reused": False,
    }


def _run_one(sample: dict[str, Any], architecture: str) -> dict[str, Any]:
    if architecture == "direct_sql":
        context = _full_schema_context()
        with patch.object(legacy, "_schema_context", lambda _query: context):
            return legacy.run_baseline([sample], "rag_llm", verbose=False)["details"][0]
    if architecture == "single_agent_tools":
        return legacy.run_baseline([sample], "rag_llm", verbose=False)["details"][0]
    return run_v4_experiment([sample], verbose=False)["details"][0]


def _failure_category(detail: dict[str, Any]) -> str | None:
    if detail.get("success"):
        return None
    if detail.get("error"):
        return "runtime_error"
    if detail.get("execution_success") is False:
        return "execution_failure"
    if detail.get("result_correct") is False:
        return "answer_mismatch"
    return "task_incomplete"


def _cohort(sample: dict[str, Any]) -> str:
    return "known_template" if build_query_contract(sample["query"]) else "unfamiliar"


def run_comparison(args: argparse.Namespace) -> dict[str, Any]:
    tests = json.loads(DATASET.read_text(encoding="utf-8"))[:args.max_samples]
    settings = get_settings()
    settings.LLM_MODEL = args.model
    settings.LLM_FALLBACK_MODEL = ""
    tracker = BudgetTracker(args)
    records: list[dict[str, Any]] = []

    for architecture in ARCHITECTURES:
        for sample in tests:
            before = (tracker.calls, tracker.prompt_tokens, tracker.completion_tokens, tracker.cost)
            with run_hooks(event_sink=tracker.sink):
                detail = _run_one(sample, architecture)
            llm_calls = tracker.calls - before[0]
            prompt_tokens = tracker.prompt_tokens - before[1]
            completion_tokens = tracker.completion_tokens - before[2]
            cost = tracker.cost - before[3]
            success = bool(detail.get("success"))
            record = {
                "sample_id": str(sample["id"]), "dataset_id": "multi-agent-v1",
                "split": "frozen_test", "architecture": architecture,
                "cohort": _cohort(sample),
                "success": success,
                "duration_ms": round(float(detail.get("duration_seconds") or 0) * 1000, 3),
                "llm_calls": llm_calls,
                "tool_calls": int(detail.get("agent_call_count") or 0) + int(bool(detail.get("execution_success"))),
                "prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
                "estimated_cost_usd": round(cost, 8),
                "error": detail.get("error"),
                "failure_category": _failure_category(detail),
                "output": {
                    "generated_sql": detail.get("generated_sql") or detail.get("sql"),
                    "sql_generation_output": str(
                        detail.get("sql_generation_output") or ""
                    )[:4000] or None,
                    "result_correct": detail.get("result_correct"),
                    "execution_success": detail.get("execution_success"),
                    "report_validation": detail.get("report_validation"),
                },
                "config": {
                    "model": args.model, "temperature": 0.0,
                    "allowed_tables": list(settings.SQL_ALLOWED_TABLES),
                    "max_result_rows": settings.SQL_MAX_RESULT_ROWS,
                    "sql_timeout_ms": settings.SQL_EXECUTION_TIMEOUT_MS,
                },
                "versions": {"protocol": "architecture-comparison-v2", "git_commit": _git_revision()},
                "answer_correct": (
                    bool(detail.get("result_correct"))
                    if sample.get("expected_sql") else None
                ),
                "execution_success": detail.get("execution_success"),
            }
            errors = validate_sample_record(record)
            if errors:
                raise ValueError(f"invalid record {architecture}/{sample['id']}: {errors}")
            records.append(record)

    return {
        "protocol": build_protocol(args),
        "budget_usage": {
            "llm_calls": tracker.calls, "prompt_tokens": tracker.prompt_tokens,
            "completion_tokens": tracker.completion_tokens,
            "total_tokens": tracker.total_tokens,
            "estimated_cost_usd": round(tracker.cost, 8),
        },
        "by_architecture": {
            architecture: summarize_records([
                item for item in records if item["architecture"] == architecture
            ]) for architecture in ARCHITECTURES
        },
        "records": records,
        "limitations": [
            "Non-SQL report quality is not reduced to report length and requires blinded human review.",
            "A single paid run does not establish variance; repeat seeds/runs must be reported separately.",
        ],
    }


def _markdown_report(result: dict[str, Any]) -> str:
    lines = [
        "# Architecture Comparison V2", "",
        f"- Model: `{result['protocol']['model']}`",
        f"- Samples per architecture: **{result['protocol']['max_samples']}**",
        f"- LLM calls: **{result['budget_usage']['llm_calls']}**",
        f"- Total tokens: **{result['budget_usage']['total_tokens']}**",
        f"- Peak-price estimated cost: **${result['budget_usage']['estimated_cost_usd']:.6f}**",
        "", "## Results", "",
        "| Architecture | Answer accuracy | Execution success | P50 ms | P95 ms | Tokens | Estimated cost |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for architecture in ARCHITECTURES:
        metrics = result["by_architecture"][architecture]
        answer = metrics["query"]["business_answer_accuracy"]
        execution = metrics["query"]["execution_success_rate"]
        engineering = metrics["engineering"]
        lines.append(
            f"| {architecture} | {answer['value']} (n={answer['n']}) | "
            f"{execution['value']} (n={execution['n']}) | "
            f"{engineering['latency_p50_ms']} | {engineering['latency_p95_ms']} | "
            f"{engineering['total_tokens']} | ${engineering['estimated_cost_usd']:.6f} |"
        )
    cohort_counts: dict[str, int] = {}
    for item in result["records"]:
        cohort_counts[item["cohort"]] = cohort_counts.get(item["cohort"], 0) + 1
    lines.extend([
        "", "## Cohort and limitations", "",
        f"- Record cohorts: `{json.dumps(cohort_counts, ensure_ascii=False)}`.",
        "- Strict answer accuracy requires result-set equivalence with the reference SQL; extra reasonable columns still count as a mismatch.",
        "- This five-sample pilot consists of known deterministic business templates. It demonstrates template-path behavior, not unfamiliar-query generalization.",
        "- Report, multi-turn, and recovery metrics have `n=0` in this paid pilot and must not be inferred.",
        "- This is one run. No variance or statistical-significance claim is made.",
    ])
    return "\n".join(lines) + "\n"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--confirm", default="")
    parser.add_argument("--model", default=get_settings().LLM_MODEL)
    parser.add_argument("--max-samples", type=int, default=5)
    parser.add_argument("--max-total-calls", type=int, default=0)
    parser.add_argument("--max-total-tokens", type=int, default=0)
    parser.add_argument("--max-cost-usd", type=float, default=0.0)
    parser.add_argument("--input-cost-per-million", type=float, default=0.0)
    parser.add_argument("--output-cost-per-million", type=float, default=0.0)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    return parser


def main() -> None:
    args = _parser().parse_args()
    validate_execution_authorization(args)
    protocol = build_protocol(args)
    if not args.execute:
        print(json.dumps({"mode": "dry-run", **protocol}, ensure_ascii=False, indent=2))
        return
    result = run_comparison(args)
    run_dir = args.output_dir / datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    (run_dir / "report.md").write_text(_markdown_report(result), encoding="utf-8")
    print(f"saved: {run_dir}")


if __name__ == "__main__":
    main()
