"""
V4 优化效果验证实验 —— Schema增强 + 路由优化 + 报告压缩。

基于 V3 实验框架，增加 V4 专属指标:
- SQL Context Quality Score
- Report Agent Latency Analysis
- Full Pipeline Rate (Planner planned vs actual)
- V3 vs V4 综合对比

运行方式:
  # 验证模式（10条代表性任务: 2 SQL + 3 analysis + 3 prediction + 2 mixed）
  python -m evaluation.experiments.run_v4_evaluation --validate

  # 完整模式（50条全部任务）
  python -m evaluation.experiments.run_v4_evaluation --full
"""

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from evaluation.framework.result import save_json, load_json, save_csv
from evaluation.metrics.multi_agent_metrics_v3 import (
    ALL_AGENTS, CORE_AGENTS, AGENT_NAME_CN,
    compute_agent_coverage_rate,
    compute_agent_timing,
    analyze_execution_chain,
    diagnose_sql_accuracy,
    check_report_agent_inputs,
)
from evaluation.metrics.multi_agent_metrics_v4 import (
    compute_sql_context_quality,
    compute_report_latency,
    compute_full_pipeline_rate,
    compute_v4_comparison,
    generate_v4_diagnosis_report,
    save_v4_tables_csv,
    compute_v4_summary,
)


# ============================================================
# Agent Trace Instrumentation (复用 V3)
# ============================================================

_trace_collector: List[Dict[str, Any]] = []


def _make_timed_wrapper(original_func, agent_name: str):
    """创建带 timing 的 wrapper 函数。"""
    def wrapper(state):
        start = time.time()
        error_occurred = False
        try:
            result = original_func(state)
            error_occurred = result.get("error") is not None if isinstance(result, dict) else False
            return result
        except Exception:
            error_occurred = True
            raise
        finally:
            elapsed = time.time() - start
            _trace_collector.append({
                "agent": agent_name,
                "start_ts": start,
                "duration": round(elapsed, 4),
                "error": error_occurred,
            })
    return wrapper


def run_query_with_trace(
    user_query: str, requested_intent: Optional[str] = None
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """执行查询并记录每个 Agent 的执行耗时（V4 增强版）。"""
    global _trace_collector
    _trace_collector = []

    import agents.planner as planner_mod
    import agents.schema_agent as schema_mod
    import agents.sql_agent as sql_mod
    import agents.governance_agent as gov_mod
    import agents.analysis_agent as analysis_mod
    import agents.prediction_agent as pred_mod
    import agents.report_agent as report_mod

    originals = {
        "schema_agent": schema_mod.schema_agent_node,
        "sql_agent": sql_mod.sql_agent_node,
        "governance_agent": gov_mod.governance_agent_node,
        "analysis_agent": analysis_mod.analysis_agent_node,
        "prediction_agent": pred_mod.prediction_agent_node,
        "report_agent": report_mod.report_agent_node,
    }

    planner_mod._graph = None

    try:
        schema_mod.schema_agent_node = _make_timed_wrapper(originals["schema_agent"], "Schema Agent")
        sql_mod.sql_agent_node = _make_timed_wrapper(originals["sql_agent"], "SQL Agent")
        gov_mod.governance_agent_node = _make_timed_wrapper(originals["governance_agent"], "Governance Agent")
        analysis_mod.analysis_agent_node = _make_timed_wrapper(originals["analysis_agent"], "Analysis Agent")
        pred_mod.prediction_agent_node = _make_timed_wrapper(originals["prediction_agent"], "Prediction Agent")
        report_mod.report_agent_node = _make_timed_wrapper(originals["report_agent"], "Report Agent")

        state = planner_mod.run_query(
            user_query, requested_intent=requested_intent
        )

        trace = list(_trace_collector)
        trace.insert(0, {
            "agent": "Planner",
            "start_ts": trace[0]["start_ts"] - 0.001 if trace else time.time(),
            "duration": 0.001,
            "error": False,
        })
    finally:
        schema_mod.schema_agent_node = originals["schema_agent"]
        sql_mod.sql_agent_node = originals["sql_agent"]
        gov_mod.governance_agent_node = originals["governance_agent"]
        analysis_mod.analysis_agent_node = originals["analysis_agent"]
        pred_mod.prediction_agent_node = originals["prediction_agent"]
        report_mod.report_agent_node = originals["report_agent"]
        planner_mod._graph = None

    return state, trace


# ============================================================
# Helpers
# ============================================================

def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _extract_agents_from_state(state: Dict[str, Any]) -> List[str]:
    """从 AgentState 提取实际调用的 Agent 列表。"""
    messages = state.get("messages", [])
    agents = []
    for m in messages:
        if isinstance(m, str) and m.startswith("[") and "]" in m[:40]:
            name = m[1:].split("]")[0].strip()
            if name not in ("Init",) and name not in agents:
                agents.append(name)
    return agents


def _safe_summary(data: Any, max_items: int = 50, max_str: int = 500) -> Any:
    """截断大数据结构用于存储。"""
    if data is None:
        return None
    if isinstance(data, dict):
        return {k: _safe_summary(v, max_items, max_str) for k, v in data.items()}
    if isinstance(data, list) and len(data) > max_items:
        return data[:max_items]
    if isinstance(data, str) and len(data) > max_str:
        return data[:max_str] + "..."
    return data


# ============================================================
# V4 Experiment Runner
# ============================================================

def run_v4_experiment(
    test_queries: List[Dict[str, Any]],
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    运行 V4 实验。

    Returns:
        {details, traces, metrics_v4}
    """
    total = len(test_queries)
    details = []
    all_traces = []

    t0 = time.time()

    for idx, test in enumerate(test_queries):
        qid = test["id"]
        query = test["query"]
        task_type = test.get("task_type", "sql_query")
        category = test.get("category", "unknown")

        if verbose:
            print(f"[{idx + 1}/{total}] Q{qid} [{task_type}] {query[:60]}...")

        detail = {
            "id": qid,
            "query": query,
            "category": category,
            "task_type": task_type,
            "expected_sql": test.get("expected_sql"),
            "expected_insights": test.get("expected_insights", []),
            "success": False,
            "error": None,
            "duration_seconds": 0.0,
            "agent_call_count": 0,
            "agent_trace": [],
            "agent_timing": [],
            "agents_invoked": [],
            "generated_sql": None,
            "execution_success": False,
            "query_result": None,
            "result_correct": None,
            "analysis_result": None,
            "prediction_result": None,
            "governance_result": None,
            "report": None,
            "task_plan": None,
            "planned_agents": [],
            "evidence": None,
        }

        t_start = time.time()
        trace: List[Dict[str, Any]] = []

        try:
            # 固定使用测试集标注的 intent，避免不同方案因意图识别差异
            # 执行了不同任务，破坏对照实验的同口径性。
            state, trace = run_query_with_trace(
                query, requested_intent=task_type
            )

            detail["agent_timing"] = trace
            detail["agent_trace"] = [
                m for m in state.get("messages", [])
                if isinstance(m, str) and m.startswith("[")
            ]
            detail["agents_invoked"] = _extract_agents_from_state(state)
            detail["agent_call_count"] = len(trace)
            detail["generated_sql"] = state.get("sql")
            detail["query_result"] = _safe_summary(state.get("query_result"))
            detail["execution_success"] = state.get("query_result") is not None
            detail["error"] = state.get("error")
            detail["task_plan"] = _safe_summary(state.get("task_plan"))
            detail["planned_agents"] = list(state.get("planned_agents") or [])
            detail["evidence"] = _safe_summary(state.get("evidence"))

            if state.get("analysis_result"):
                detail["analysis_result"] = _safe_summary(state["analysis_result"])
            if state.get("prediction_result"):
                detail["prediction_result"] = _safe_summary(state["prediction_result"])
            if state.get("governance_result"):
                detail["governance_result"] = _safe_summary(state["governance_result"])

            detail["report"] = state.get("report")

            # SQL 结果验证
            expected_sql = detail.get("expected_sql")
            if expected_sql and detail["execution_success"]:
                try:
                    from storage.db_adapter import get_available_adapter
                    from evaluation.metrics.text2sql_metrics import compare_result_sets
                    adapter = get_available_adapter()
                    expected_result = adapter.execute_sql(expected_sql)
                    detail["result_correct"] = compare_result_sets(
                        state["query_result"], expected_result
                    )
                except Exception:
                    detail["result_correct"] = None

            detail["success"] = _judge_success_v4(detail, test)

        except Exception as e:
            detail["error"] = f"{type(e).__name__}: {str(e)[:200]}"
            detail["success"] = False
            if verbose:
                print(f"  [ERR] {e}")

        detail["duration_seconds"] = round(time.time() - t_start, 3)

        if verbose:
            status = "[OK]" if detail["success"] else "[FAIL]"
            agents_str = " → ".join(detail["agents_invoked"]) if detail["agents_invoked"] else "—"
            print(f"  {status} {detail['duration_seconds']}s | {agents_str}")

        details.append(detail)
        all_traces.append({
            "task_id": qid,
            "task_type": task_type,
            "trace": trace,
            "agents_invoked": detail["agents_invoked"],
            "planned_agents": detail["planned_agents"],
        })

    total_time = time.time() - t0

    # ================================================================
    # V4 指标计算
    # ================================================================
    if verbose:
        print(f"\n{'='*60}")
        print("计算 V4 指标...")

    acr_data = compute_agent_coverage_rate(details)
    timing_data = compute_agent_timing(details)
    chain_data = analyze_execution_chain(details)
    sql_diagnosis = diagnose_sql_accuracy(details)
    report_check = check_report_agent_inputs(details)

    # V4 新增指标
    context_quality = compute_sql_context_quality(details)
    report_latency = compute_report_latency(details)
    full_pipeline = compute_full_pipeline_rate(details)
    comparison = compute_v4_comparison(
        v4_acr=acr_data,
        v4_timing=timing_data,
        v4_context_quality=context_quality,
        v4_report_latency=report_latency,
        v4_full_pipeline=full_pipeline,
        v4_sql_diagnosis=sql_diagnosis,
    )

    if verbose:
        print(f"  ACR: {acr_data.get('overall_acr', 0)*100:.1f}%")
        print(f"  Full Chain: {acr_data.get('full_chain_rate', 0)*100:.0f}%")
        print(f"  dtype 覆盖率: {context_quality.get('dtype_coverage', 0)*100:.0f}%")
        print(f"  business_term 覆盖率: {context_quality.get('business_term_coverage', 0)*100:.0f}%")
        print(f"  Report 平均延迟: {report_latency.get('avg', 0):.1f}s")
        print(f"  Full Pipeline: {full_pipeline.get('overall_completion_rate', 0)*100:.0f}%")

    summary = compute_v4_summary(
        acr_data=acr_data,
        timing_data=timing_data,
        chain_data=chain_data,
        sql_diagnosis=sql_diagnosis,
        report_check=report_check,
        context_quality=context_quality,
        report_latency=report_latency,
        full_pipeline=full_pipeline,
        comparison=comparison,
    )

    return {
        "details": details,
        "traces": all_traces,
        "metrics_v4": {
            "experiment_duration_seconds": round(total_time, 1),
            "total_tasks": total,
            "agent_coverage": acr_data,
            "agent_timing": timing_data,
            "execution_chain": chain_data,
            "sql_diagnosis": sql_diagnosis,
            "report_agent_inputs": report_check,
            "context_quality": context_quality,
            "report_latency": report_latency,
            "full_pipeline": full_pipeline,
            "comparison": comparison,
            "summary": summary,
        },
    }


def _judge_success_v4(detail: Dict[str, Any], test: Dict[str, Any]) -> bool:
    """按任务产物判定成功，避免把“执行过”误记为“完成正确”。"""
    if detail.get("error"):
        return False

    task_type = test.get("task_type", "sql_query")

    if task_type == "sql_query":
        # 有标准 SQL 的任务必须返回等价结果；仅执行成功不足以证明正确。
        if test.get("expected_sql"):
            return detail.get("result_correct") is True
        return detail.get("execution_success", False)
    elif task_type == "analysis":
        return detail.get("analysis_result") is not None
    elif task_type == "prediction":
        pred = detail.get("prediction_result", {}) or {}
        churn_result = pred.get("churn", {})
        churn_ok = churn_result.get("auc") is not None
        sales_result = pred.get("sales", {})
        sales_ok = (
            sales_result.get("rmse") is not None
            or sales_result.get("forecast") is not None
        )
        return churn_ok or sales_ok
    elif task_type == "mixed":
        report = detail.get("report") or ""
        return detail.get("execution_success", False) and len(report.strip()) > 100
    else:
        return not detail.get("error")


# ============================================================
# 主入口
# ============================================================

def run_full_v4_evaluation(
    test_data_path: Optional[str] = None,
    max_samples: int = 0,
    validate_only: bool = False,
) -> None:
    """
    运行完整的 V4 评估流程。

    Args:
        test_data_path: 测试数据路径
        max_samples: 最多运行样本数
        validate_only: True=运行10条验证, False=运行全部50条
    """
    output_dir = Path(__file__).resolve().parent.parent / "results" / "multi_agent_v4"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "tables").mkdir(exist_ok=True)
    (output_dir / "raw").mkdir(exist_ok=True)
    (output_dir / "traces").mkdir(exist_ok=True)

    ts = _timestamp()

    # 加载测试数据
    if test_data_path is None:
        test_data_path = str(
            Path(__file__).resolve().parent.parent / "data" / "multi_agent" / "test_queries.json"
        )

    test_queries = load_json(test_data_path)
    print(f"加载测试数据: {len(test_queries)} 条查询")

    if validate_only and max_samples == 0:
        # 选取 10 条代表性任务: 2 sql + 3 analysis + 3 prediction + 2 mixed
        representative = []
        counts = {"sql_query": 0, "analysis": 0, "prediction": 0, "mixed": 0}
        targets = {"sql_query": 2, "analysis": 3, "prediction": 3, "mixed": 2}

        for q in test_queries:
            tt = q.get("task_type", "sql_query")
            if counts.get(tt, 0) < targets.get(tt, 0):
                representative.append(q)
                counts[tt] = counts.get(tt, 0) + 1

        test_queries = representative
        print(f"验证模式: 选取 {len(test_queries)} 条代表性任务")
        for tt, n in counts.items():
            print(f"  {tt}: {n} 条")

    if max_samples > 0:
        test_queries = test_queries[:max_samples]

    # 运行 V4 实验
    print(f"\n{'='*60}")
    print(f"V4 优化效果验证实验")
    print(f"测试样本: {len(test_queries)} 条")
    print(f"优化: Schema增强 + Planner路由 + Report压缩")
    print(f"{'='*60}\n")

    result = run_v4_experiment(test_queries=test_queries, verbose=True)

    details = result["details"]
    traces = result["traces"]
    m = result["metrics_v4"]

    # ---- 保存 ----
    # 1. 完整详情
    details_path = output_dir / "raw" / f"v4_full_details_{ts}.json"
    save_json(details, details_path)
    print(f"\n详情已保存: {details_path}")

    # 2. Agent Trace
    traces_path = output_dir / "traces" / f"v4_agent_traces_{ts}.json"
    save_json(traces, traces_path)
    print(f"Trace 已保存: {traces_path}")

    # 3. 指标摘要
    summary_path = output_dir / "raw" / f"v4_metrics_summary_{ts}.json"
    save_json(m["summary"], summary_path)
    print(f"指标摘要已保存: {summary_path}")

    # 4. CSV 表格
    saved_csv = save_v4_tables_csv(
        m["agent_coverage"],
        m["agent_timing"],
        m["sql_diagnosis"],
        m["report_agent_inputs"],
        m["context_quality"],
        m["report_latency"],
        m["full_pipeline"],
        m["comparison"],
        output_dir,
        ts,
    )
    for name, path in saved_csv.items():
        print(f"CSV [{name}]: {path}")

    # 5. V4 诊断报告
    report_md = generate_v4_diagnosis_report(
        details=details,
        acr_data=m["agent_coverage"],
        timing_data=m["agent_timing"],
        chain_data=m["execution_chain"],
        sql_diagnosis=m["sql_diagnosis"],
        report_check=m["report_agent_inputs"],
        context_quality=m["context_quality"],
        report_latency=m["report_latency"],
        full_pipeline=m["full_pipeline"],
        comparison=m["comparison"],
    )
    report_path = output_dir / "raw" / f"v4_diagnosis_report_{ts}.md"
    report_path.write_text(report_md, encoding="utf-8")
    print(f"诊断报告已保存: {report_path}")

    # 6. 详情 CSV
    rows = []
    for d in details:
        rows.append({
            "id": d.get("id"),
            "query": (d.get("query") or "")[:80],
            "task_type": d.get("task_type"),
            "category": d.get("category"),
            "success": d.get("success"),
            "execution_success": d.get("execution_success"),
            "result_correct": d.get("result_correct"),
            "agents_count": len(d.get("agents_invoked", [])),
            "agents_invoked": " → ".join(d.get("agents_invoked", [])),
            "duration_seconds": d.get("duration_seconds"),
            "error": (d.get("error") or "")[:100],
        })
    details_csv = output_dir / "raw" / f"v4_details_{ts}.csv"
    save_csv(rows, details_csv)
    print(f"详情 CSV: {details_csv}")

    # ---- 结果摘要 ----
    print(f"\n{'='*60}")
    print("V4 实验完成")
    print(f"{'='*60}")
    print(f"ACR: {m['agent_coverage'].get('overall_acr', 0)*100:.1f}%")
    print(f"Full Chain: {m['agent_coverage'].get('full_chain_rate', 0)*100:.0f}%")
    print(f"dtype 覆盖: {m['context_quality'].get('dtype_coverage', 0)*100:.0f}%")
    print(f"biz_term 覆盖: {m['context_quality'].get('business_term_coverage', 0)*100:.0f}%")
    print(f"Report avg: {m['report_latency'].get('avg', 0):.1f}s")
    print(f"SQL acc: {m['sql_diagnosis']['multi_agent'].get('accuracy_v1_method', 0)*100:.1f}%")

    print(f"\n所有结果已保存到: {output_dir}")

    # 历史 V3 仅作背景参考，不是同口径对照实验。
    comp = m["comparison"].get("metrics", {})
    if comp:
        print(f"\n--- 历史 V3 参考 vs 本次 V4（非同口径）---")
        print("  WARNING:", m["comparison"].get("warning"))
        for metric, vals in comp.items():
            print(f"  {metric}: {vals['v3']} → {vals['v4']} ({vals['delta']})")


# ============================================================
# CLI
# ============================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="V4 Multi-Agent 优化效果验证实验")
    parser.add_argument("--test-data", type=str, default=None)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--validate", action="store_true",
                        help="验证模式：运行10条代表性任务 (2 SQL + 3 analysis + 3 prediction + 2 mixed)")
    parser.add_argument("--full", action="store_true",
                        help="完整模式：运行全部50条任务")
    args = parser.parse_args()

    if args.validate:
        run_full_v4_evaluation(
            test_data_path=args.test_data,
            validate_only=True,
        )
    elif args.full:
        run_full_v4_evaluation(
            test_data_path=args.test_data,
            max_samples=args.max_samples,
        )
    else:
        # 默认：验证模式
        print("未指定模式，默认运行 --validate（10条验证）")
        print("使用 --full 运行全部50条任务")
        run_full_v4_evaluation(
            test_data_path=args.test_data,
            validate_only=True,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
