"""同一数据快照、同一任务集、同一成功口径的正式对照实验。

三组方案：
1. single_llm：单个 LLM 一次生成 SQL，数据库只负责执行；
2. rag_llm：检索相关 Schema 后，由单个 LLM 生成 SQL并执行；
3. full_multi_agent：项目的完整按需多 Agent 流程。

SQL 任务只有结果与标准 SQL 等价才算成功。其余任务必须真实查询成功并生成
基于查询结果的报告。报告质量另由统一的 LLM-as-Judge 评分，不混入 TCS。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from agents.llm import chat
from evaluation.experiments.run_v4_evaluation import run_v4_experiment
from evaluation.framework.result import load_json, save_json
from evaluation.metrics.multi_agent_metrics_v2 import evaluate_all_reports_v2
from evaluation.metrics.text2sql_metrics import compare_result_sets
from storage.db_adapter import get_available_adapter


ROOT = Path(__file__).resolve().parents[2]
TEST_DATA = ROOT / "evaluation" / "data" / "multi_agent" / "test_queries.json"
OUTPUT_ROOT = ROOT / "evaluation" / "results" / "formal_comparison"


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_revision() -> Dict[str, Any]:
    def run(*args: str) -> str:
        try:
            return subprocess.check_output(
                ["git", *args], cwd=ROOT, text=True, encoding="utf-8",
                errors="replace", stderr=subprocess.DEVNULL,
            ).strip()
        except Exception:
            return "unavailable"

    return {
        "commit": run("rev-parse", "HEAD"),
        "dirty": bool(run("status", "--porcelain")),
    }


def create_manifest(run_dir: Path, tests: List[Dict[str, Any]]) -> Dict[str, Any]:
    from config.settings import get_settings

    settings = get_settings()
    tracked = [
        TEST_DATA,
        ROOT / "requirements.txt",
        ROOT / "requirements-dev.txt",
        ROOT / "agents" / "business_semantics.py",
        ROOT / "agents" / "sql_agent.py",
        ROOT / "agents" / "schema_grounding.py",
        ROOT / "agents" / "task_planning.py",
        ROOT / "storage" / "mysql" / "client.py",
        ROOT / "evaluation" / "experiments" / "run_formal_comparison.py",
        ROOT / "data" / "raw" / "customers.csv",
        ROOT / "data" / "raw" / "orders.csv",
        ROOT / "data" / "raw" / "monthly_revenue.csv",
        ROOT / "data" / "raw" / "product_summary.csv",
        ROOT / "data" / "processed" / "customers_cleaned.csv",
        ROOT / "data" / "chromadb" / "chroma.sqlite3",
        ROOT / "storage" / "chromadb" / "schema_metadata.py",
    ]
    files = {
        str(path.relative_to(ROOT)): {"sha256": _sha256(path), "bytes": path.stat().st_size}
        for path in tracked if path.exists()
    }
    manifest = {
        "protocol_version": "formal-comparison-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "test_count": len(tests),
        "test_ids": [t["id"] for t in tests],
        "task_success_rules": {
            "sql_query": "generated SQL executes and result equals expected_sql result",
            "analysis": "generated SQL executes and grounded report has >=100 characters",
            "prediction": "generated SQL executes and grounded prediction report has >=100 characters",
            "mixed": "generated SQL executes and grounded report has >=100 characters",
        },
        "schemes": {
            "single_llm": "one-shot SQL generation without schema retrieval; DB executes read-only SQL",
            "rag_llm": "schema retrieval + one-shot SQL generation; DB executes read-only SQL",
            "full_multi_agent": "problem-driven project pipeline with specialist tools",
        },
        "experiment_flags": {
            "business_semantics_enabled": settings.BUSINESS_SEMANTICS_ENABLED,
            "sql_max_retries": settings.SQL_MAX_RETRIES,
            "mysql_select_timeout_ms": 60000,
        },
        "git": _git_revision(),
        "files": files,
    }
    save_json(manifest, run_dir / "manifest.json")
    return manifest


def _schema_context(query: str) -> str:
    from storage.chromadb.embedder import get_embedder

    rows = get_embedder().search(query, top_k=20)
    lines: List[str] = []
    seen = set()
    for row in rows:
        key = (row.get("table_name"), row.get("column_name"))
        if key in seen:
            continue
        seen.add(key)
        lines.append(
            f"{key[0]}.{key[1]} ({row.get('dtype', '')})"
            f" — {row.get('business_term', '')}"
        )
    return "\n".join(lines)


def _extract_sql(text: str) -> Optional[str]:
    fenced = re.search(r"```(?:sql)?\s*(.*?)```", text, re.I | re.S)
    candidate = fenced.group(1).strip() if fenced else text.strip()
    start = re.search(r"\b(SELECT|WITH)\b", candidate, re.I)
    if not start:
        return None
    candidate = candidate[start.start():].strip()
    # 移除代码块外说明；只允许一条只读语句。
    candidate = candidate.split("```", 1)[0].strip()
    if ";" in candidate:
        candidate = candidate.split(";", 1)[0].strip() + ";"
    if re.search(r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|TRUNCATE|CREATE)\b", candidate, re.I):
        return None
    return candidate


def _compact_result(value: Any, limit: int = 12000) -> str:
    text = json.dumps(value, ensure_ascii=False, default=str)
    return text if len(text) <= limit else text[:limit] + "…"


def _generate_sql(test: Dict[str, Any], mode: str) -> tuple[str, Optional[str]]:
    context = _schema_context(test["query"]) if mode == "rag_llm" else ""
    schema_note = (
        f"\n检索到的数据库字段如下：\n{context}\n只能使用这些字段。"
        if context else
        "\n你没有 Schema 检索能力，请仅根据问题推断可能的表和字段。"
    )
    prompt = f"""为下列电商数据任务生成一条 MySQL 8 只读查询。
任务类型：{test.get('task_type')}
用户问题：{test['query']}
{schema_note}

要求：
- 只输出一个 ```sql 代码块，不要解释；
- 查询应返回回答或进一步分析该任务所需的真实数据；
- 百分比统一使用 0 到 100；
- 禁止写操作，禁止虚构字段。
"""
    raw = chat(
        [{"role": "system", "content": "你是严谨的 MySQL 数据分析师。"},
         {"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=1200,
    )
    return raw, _extract_sql(raw)


def _generate_grounded_report(test: Dict[str, Any], sql: str, result: Any) -> str:
    prompt = f"""请只依据给定查询结果回答用户，不能补造数据库中不存在的数值。

用户问题：{test['query']}
任务类型：{test.get('task_type')}
已执行 SQL：{sql}
真实查询结果：{_compact_result(result)}

请用中文给出简洁、结构化的结论。分析/预测/综合任务应包含关键数据、解释和可执行建议；
预测任务要明确区分历史事实与估计，不得把估计伪装成数据库事实。
"""
    return chat(
        [{"role": "system", "content": "你是数据证据优先的电商分析师。"},
         {"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=1800,
    )


def run_baseline(tests: List[Dict[str, Any]], mode: str, verbose: bool = True) -> Dict[str, Any]:
    adapter = get_available_adapter()
    details: List[Dict[str, Any]] = []
    started = time.time()

    for index, test in enumerate(tests, 1):
        item_started = time.time()
        detail: Dict[str, Any] = {
            "id": test["id"], "query": test["query"],
            "task_type": test.get("task_type", "sql_query"), "mode": mode,
            "expected_sql": test.get("expected_sql"),
            "expected_insights": test.get("expected_insights", []),
            "has_db_access": True,
            "generated_sql": None, "execution_success": False,
            "result_correct": None, "query_result": None,
            "llm_output": None, "report": None, "success": False,
            "error": None, "llm_call_count": 0,
            "agent_call_count": 1 if mode == "rag_llm" else 0,
        }
        try:
            raw, sql = _generate_sql(test, mode)
            detail["sql_generation_output"] = raw
            detail["llm_call_count"] = 1
            detail["generated_sql"] = sql
            if not sql:
                raise ValueError("未提取到只读 SQL")
            result = adapter.execute_sql(sql)
            detail["execution_success"] = True
            detail["query_result"] = result

            expected_sql = test.get("expected_sql")
            if expected_sql:
                expected_result = adapter.execute_sql(expected_sql)
                detail["result_correct"] = compare_result_sets(result, expected_result)

            if detail["task_type"] == "sql_query":
                detail["llm_output"] = _compact_result(result)
                detail["success"] = (
                    detail["result_correct"] is True if expected_sql
                    else detail["execution_success"]
                )
            else:
                report = _generate_grounded_report(test, sql, result)
                detail["llm_call_count"] = 2
                detail["llm_output"] = report
                detail["report"] = report
                detail["success"] = detail["execution_success"] and len(report.strip()) >= 100
        except Exception as exc:
            detail["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
        detail["duration_seconds"] = round(time.time() - item_started, 3)
        details.append(detail)
        if verbose:
            status = "OK" if detail["success"] else "FAIL"
            print(f"[{mode} {index}/{len(tests)}] {status} {detail['duration_seconds']}s")

    return {
        "mode": mode,
        "details": details,
        "experiment_duration_seconds": round(time.time() - started, 3),
    }


def _metrics(details: List[Dict[str, Any]]) -> Dict[str, Any]:
    sql = [d for d in details if d.get("task_type") == "sql_query"]
    # 报告质量只比较三种方案都应生成正式报告的 mixed 任务。分析和预测
    # 在主方案中产出结构化中间结果，若混入会被误记为“空报告 0 分”。
    reports = [d for d in details if d.get("task_type") == "mixed"]
    durations = [float(d.get("duration_seconds", 0)) for d in details]
    sorted_durations = sorted(durations)
    p95_index = max(0, min(len(sorted_durations) - 1, int(0.95 * len(sorted_durations)) - 1)) if sorted_durations else 0
    by_type: Dict[str, Any] = {}
    for task_type in ("sql_query", "analysis", "prediction", "mixed"):
        subset = [d for d in details if d.get("task_type") == task_type]
        by_type[task_type] = {
            "n": len(subset),
            "tcs": round(sum(bool(d.get("success")) for d in subset) / len(subset), 4) if subset else None,
        }
    return {
        "n": len(details),
        "tcs": round(sum(bool(d.get("success")) for d in details) / len(details), 4) if details else 0,
        "sql_accuracy": round(sum(d.get("result_correct") is True for d in sql) / len(sql), 4) if sql else None,
        "report_quality": round(sum(d.get("report_quality_score_v2", 0) for d in reports) / len(reports), 2) if reports else None,
        "report_quality_count": len(reports),
        "avg_duration_seconds": round(sum(durations) / len(durations), 3) if durations else 0,
        "median_duration_seconds": round(statistics.median(durations), 3) if durations else 0,
        "p95_duration_seconds": round(sorted_durations[p95_index], 3) if sorted_durations else 0,
        "by_task_type": by_type,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--skip-judge", action="store_true")
    parser.add_argument("--only", choices=("single_llm", "rag_llm", "full_multi_agent"))
    parser.add_argument("--run-dir", help="复用指定正式运行目录并覆盖所选方案")
    parser.add_argument(
        "--disable-business-semantics",
        action="store_true",
        help="消融实验：关闭业务语义配方与结果形状契约",
    )
    args = parser.parse_args()

    tests = load_json(str(TEST_DATA))
    if args.max_samples:
        tests = tests[:args.max_samples]
    run_dir = Path(args.run_dir).resolve() if args.run_dir else OUTPUT_ROOT / _timestamp()
    (run_dir / "raw").mkdir(parents=True, exist_ok=True)
    if args.disable_business_semantics:
        from config.settings import get_settings
        get_settings().BUSINESS_SEMANTICS_ENABLED = False
    create_manifest(run_dir, tests)

    modes = [args.only] if args.only else ["single_llm", "rag_llm", "full_multi_agent"]
    summary_path = run_dir / "summary.json"
    summary: Dict[str, Any] = (
        load_json(str(summary_path)) if summary_path.exists()
        else {"run_dir": str(run_dir), "schemes": {}}
    )
    for mode in modes:
        result_name = (
            f"{mode}_no_business_semantics"
            if mode == "full_multi_agent" and args.disable_business_semantics
            else mode
        )
        print(f"\n=== {result_name} / {len(tests)} tasks ===")
        if mode == "full_multi_agent":
            result = run_v4_experiment(tests, verbose=True)
            for detail in result["details"]:
                detail["mode"] = mode
                detail["has_db_access"] = True
                if args.disable_business_semantics:
                    detail["ablation"] = "no_business_semantics"
                detail.setdefault("llm_call_count", None)
        else:
            result = run_baseline(tests, mode, verbose=True)

        if not args.skip_judge:
            result["details"] = evaluate_all_reports_v2(result["details"], skip_sql_query=True)
        result["formal_metrics"] = _metrics(result["details"])
        save_json(result, run_dir / "raw" / f"{result_name}.json")
        summary["schemes"][result_name] = result["formal_metrics"]
        save_json(summary, run_dir / "summary.json")
        print(json.dumps(result["formal_metrics"], ensure_ascii=False, indent=2))

    print(f"\nFormal experiment saved to: {run_dir}")


if __name__ == "__main__":
    main()
