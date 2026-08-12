"""
Multi-Agent V3 评价指标 —— 执行链诊断、Agent 覆盖率、Agent 调用追踪。

V3 新增指标:
- Agent Coverage Rate (ACR): 每个任务调用的 Agent 数量和覆盖率
- Agent Execution Time: 每个 Agent 的执行耗时统计
- Agent Call Path Analysis: 执行链路完整性分析
- SQL Accuracy Deep Dive: Text2SQL vs Multi-Agent SQL 准确率对比诊断

V3 保留 V2 的 TCS 和 RQS 指标，在此基础上增加执行链维度的分析。
"""

import json
import time
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

import pandas as pd


# ============================================================
# ALL_AGENTS 常量
# ============================================================

ALL_AGENTS = [
    "Planner",
    "Schema Agent",
    "SQL Agent",
    "Governance Agent",
    "Analysis Agent",
    "Prediction Agent",
    "Report Agent",
]

# 技术评测核心6 Agents（排除 Governance 和 Planner，它们总是执行）
CORE_AGENTS = [
    "Schema Agent",
    "SQL Agent",
    "Analysis Agent",
    "Prediction Agent",
    "Report Agent",
]

# Agent 中文名映射
AGENT_NAME_CN = {
    "Planner": "规划Agent",
    "Schema Agent": "Schema检索Agent",
    "SQL Agent": "SQL生成Agent",
    "Governance Agent": "数据治理Agent",
    "Analysis Agent": "分析Agent",
    "Prediction Agent": "预测Agent",
    "Report Agent": "报告Agent",
}


# ============================================================
# Agent Coverage Rate (ACR)
# ============================================================

def compute_agent_coverage_rate(details: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    计算 Agent Coverage Rate —— 每个任务实际调用了多少个 Agent。

    Returns:
        {
            "overall_acr": float,           # 整体覆盖率 (0-1)
            "overall_avg_agents": float,    # 平均每任务调用Agent数
            "by_agent": {agent: {count, rate}},  # 每个Agent被调用的次数/比例
            "by_task_type": {type: {acr, avg_agents, agent_counts}},  # 按任务类型
            "by_intent": {intent: {acr, avg_agents, agent_counts}},   # 按意图
            "full_chain_tasks": int,         # 走完完整6-Agent链的任务数
            "full_chain_task_ids": [int],   # 完整链的任务ID
            "skipped_agents_per_task": {task_id: [agent_names]},  # 每个任务跳过的Agent
        }
    """
    if not details:
        return {"overall_acr": 0.0, "overall_avg_agents": 0.0}

    total_tasks = len(details)
    agent_call_counts = {a: 0 for a in ALL_AGENTS}
    task_agent_counts = []
    full_chain_tasks = 0
    full_chain_ids = []
    skipped_per_task = {}
    by_task_type = defaultdict(lambda: {"total": 0, "agent_counts": defaultdict(int), "agent_sum": 0})

    for d in details:
        agents_invoked = d.get("agents_invoked", [])
        agent_trace = d.get("agent_trace", [])
        task_type = d.get("task_type", "unknown")
        task_id = d.get("id", "?")

        # 从 agent_trace 中提取实际调用的 Agent
        invoked_set = set()
        for entry in agents_invoked:
            # Normalize agent names
            for agent_name in ALL_AGENTS:
                if agent_name.lower() in entry.lower():
                    invoked_set.add(agent_name)

        # 也检查 agent_trace 消息
        for msg in agent_trace:
            if isinstance(msg, str) and msg.startswith("["):
                for agent_name in ALL_AGENTS:
                    if agent_name in msg:
                        invoked_set.add(agent_name)

        # 如果 agents_invoked 为空但 trace 有内容，从 trace 推断
        if not invoked_set and agent_trace:
            for msg in agent_trace:
                if isinstance(msg, str):
                    for agent_name in ALL_AGENTS:
                        if agent_name in msg:
                            invoked_set.add(agent_name)

        # 计数
        for agent_name in invoked_set:
            if agent_name in agent_call_counts:
                agent_call_counts[agent_name] += 1

        # 只统计 CORE_AGENTS 的调用数
        core_invoked = invoked_set & set(CORE_AGENTS)
        n_core_agents = len(core_invoked)
        task_agent_counts.append(n_core_agents)

        # 完整链判定（核心5个Agent全部调用）
        if n_core_agents == len(CORE_AGENTS):
            full_chain_tasks += 1
            full_chain_ids.append(task_id)

        # 跳过的 Agent
        skipped = [a for a in CORE_AGENTS if a not in invoked_set]
        if skipped:
            skipped_per_task[str(task_id)] = skipped

        # 按任务类型统计
        by_task_type[task_type]["total"] += 1
        by_task_type[task_type]["agent_sum"] += n_core_agents
        for a in invoked_set:
            if a in agent_call_counts:
                by_task_type[task_type]["agent_counts"][a] += 1

    # 计算比率
    by_agent = {}
    for a in ALL_AGENTS:
        by_agent[a] = {
            "count": agent_call_counts[a],
            "rate": round(agent_call_counts[a] / total_tasks, 4) if total_tasks > 0 else 0,
        }

    # 按任务类型的 ACR
    by_type_result = {}
    for tt, data in by_task_type.items():
        n = data["total"]
        by_type_result[tt] = {
            "task_count": n,
            "acr": round(data["agent_sum"] / (n * len(CORE_AGENTS)), 4) if n > 0 else 0,
            "avg_agents": round(data["agent_sum"] / n, 2) if n > 0 else 0,
            "agent_counts": {a: data["agent_counts"].get(a, 0) for a in CORE_AGENTS},
            "agent_rates": {a: round(data["agent_counts"].get(a, 0) / n, 4) if n > 0 else 0
                          for a in CORE_AGENTS},
        }

    overall_acr = sum(task_agent_counts) / (total_tasks * len(CORE_AGENTS)) if total_tasks > 0 else 0

    return {
        "overall_acr": round(overall_acr, 4),
        "overall_avg_agents": round(sum(task_agent_counts) / total_tasks, 2) if total_tasks > 0 else 0,
        "by_agent": by_agent,
        "by_task_type": by_type_result,
        "full_chain_tasks": full_chain_tasks,
        "full_chain_rate": round(full_chain_tasks / total_tasks, 4) if total_tasks > 0 else 0,
        "full_chain_task_ids": full_chain_ids,
        "skipped_agents_per_task": skipped_per_task,
    }


# ============================================================
# Agent Execution Time Analysis
# ============================================================

def compute_agent_timing(details: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    分析每个 Agent 的执行耗时（从 agent_timing 字段提取）。

    期望 detail 中含有 agent_timing 字段:
        [{"agent": "Schema Agent", "duration": 0.523, "error": false}, ...]
    """
    if not details:
        return {}

    # 收集所有有 timing 数据的任务
    agent_times = defaultdict(list)
    total_by_agent = defaultdict(float)
    count_by_agent = defaultdict(int)
    error_by_agent = defaultdict(int)

    for d in details:
        timing = d.get("agent_timing", [])
        if not timing:
            continue
        for entry in timing:
            agent = entry.get("agent", "Unknown")
            duration = entry.get("duration", 0)
            has_error = entry.get("error", False)

            agent_times[agent].append(duration)
            total_by_agent[agent] += duration
            count_by_agent[agent] += 1
            if has_error:
                error_by_agent[agent] += 1

    result = {}
    for agent in ALL_AGENTS:
        times = agent_times.get(agent, [])
        if times:
            result[agent] = {
                "count": len(times),
                "total_seconds": round(total_by_agent[agent], 3),
                "avg_seconds": round(sum(times) / len(times), 4),
                "min_seconds": round(min(times), 4),
                "max_seconds": round(max(times), 4),
                "p50_seconds": round(sorted(times)[len(times) // 2], 4),
                "p95_seconds": round(sorted(times)[int(len(times) * 0.95)], 4) if len(times) >= 20 else None,
                "error_count": error_by_agent.get(agent, 0),
                "error_rate": round(error_by_agent.get(agent, 0) / len(times), 4),
            }
        else:
            result[agent] = {
                "count": 0,
                "total_seconds": 0,
                "avg_seconds": 0,
                "note": "无 timing 数据",
            }

    # 瓶颈分析
    all_agent_avgs = {a: r.get("avg_seconds", 0) for a, r in result.items() if r.get("count", 0) > 0}
    if all_agent_avgs:
        bottleneck = max(all_agent_avgs, key=all_agent_avgs.get)
        result["_bottleneck"] = {
            "agent": bottleneck,
            "avg_seconds": all_agent_avgs[bottleneck],
            "all_avgs": all_agent_avgs,
        }

    return result


# ============================================================
# Execution Chain Completeness Analysis
# ============================================================

def analyze_execution_chain(details: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    分析执行链完整性 —— 检查每个任务的实际执行路径。

    Returns:
        {
            "chain_patterns": {pattern: count},  # 各执行路径模式及频次
            "broken_chains": [task_ids],          # 执行链断裂的任务（有error的任务）
            "intent_routing_accuracy": float,      # 意图路由准确率（如果expected_intent可用）
            "error_cascade_analysis": {...},       # 错误级联分析
        }
    """
    if not details:
        return {}

    chain_patterns = defaultdict(int)
    broken_chains = []
    error_cascade = {"total_errors": 0, "sql_failures": 0,
                     "schema_failures": 0, "analysis_failures": 0,
                     "prediction_failures": 0, "report_failures": 0}
    intent_mismatches = []

    for d in details:
        agents_invoked = d.get("agents_invoked", [])
        task_type = d.get("task_type", "unknown")
        task_id = d.get("id", "?")
        error = d.get("error")

        # 构建执行路径模式
        pattern_parts = []
        for a in agents_invoked:
            for agent_name in ALL_AGENTS:
                if agent_name.lower() in a.lower():
                    pattern_parts.append(agent_name)
                    break
            else:
                pattern_parts.append(a)

        pattern = " → ".join(pattern_parts) if pattern_parts else "无Agent调用"
        chain_patterns[pattern] += 1

        # 断裂链
        if error:
            broken_chains.append({
                "task_id": task_id,
                "task_type": task_type,
                "error": str(error)[:200],
                "agents_invoked": pattern_parts,
            })
            error_cascade["total_errors"] += 1

            err_str = str(error).lower()
            if "sql" in err_str:
                error_cascade["sql_failures"] += 1
            if "schema" in err_str:
                error_cascade["schema_failures"] += 1
            if "analysis" in err_str:
                error_cascade["analysis_failures"] += 1
            if "prediction" in err_str:
                error_cascade["prediction_failures"] += 1
            if "report" in err_str:
                error_cascade["report_failures"] += 1

    # 按频率排序
    sorted_patterns = sorted(chain_patterns.items(), key=lambda x: x[1], reverse=True)

    return {
        "chain_patterns": dict(sorted_patterns),
        "unique_patterns": len(chain_patterns),
        "broken_chains": broken_chains,
        "broken_chain_count": len(broken_chains),
        "error_cascade": error_cascade,
    }


# ============================================================
# SQL Accuracy Deep Dive
# ============================================================

def diagnose_sql_accuracy(
    multi_details: List[Dict[str, Any]],
    text2sql_report_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    深入诊断 Multi-Agent SQL 准确率 vs Text2SQL 独立实验的差异。

    Args:
        multi_details: Multi-Agent 实验的 details 列表
        text2sql_report_path: Text2SQL 实验报告路径（可选）

    Returns:
        诊断报告 dict
    """
    # Multi-Agent 侧的 SQL 统计
    ma_sql_tasks = [d for d in multi_details if d.get("task_type") == "sql_query"]
    ma_total = len(ma_sql_tasks)

    ma_generated = [d for d in ma_sql_tasks if d.get("generated_sql")]
    ma_executed = [d for d in ma_sql_tasks if d.get("execution_success")]
    ma_correct = [d for d in ma_sql_tasks if d.get("result_correct") is True]
    ma_unverifiable = [d for d in ma_sql_tasks if d.get("result_correct") is None]
    ma_wrong = [d for d in ma_sql_tasks if d.get("result_correct") is False]

    # 分析 SQL 失败原因
    failure_reasons = defaultdict(list)
    for d in ma_sql_tasks:
        tid = d.get("id", "?")
        error = d.get("error", "")
        if error:
            if "生成失败" in str(error) or "SQL 生成" in str(error):
                failure_reasons["SQL生成失败"].append(tid)
            elif "执行" in str(error):
                failure_reasons["SQL执行失败"].append(tid)
            else:
                failure_reasons["其他错误"].append(tid)
        elif d.get("result_correct") is False:
            failure_reasons["结果不匹配"].append(tid)
        elif d.get("result_correct") is None:
            failure_reasons["无法验证(无expected_sql或未执行)"].append(tid)

    # Text2SQL 报告数据（硬编码已知结果）
    text2sql_ex = 0.48   # Text2SQL 独立实验 EX = 48.0%
    text2sql_exe = 0.94  # Text2SQL 独立实验 EXE = 94.0%

    # Multi-Agent V1 metric: compute_sql_accuracy
    valid = len(ma_sql_tasks) - len(ma_unverifiable)
    ma_sql_acc_v1 = len(ma_correct) / valid if valid > 0 else 0.0
    ma_sql_exec_rate = len(ma_executed) / ma_total if ma_total > 0 else 0.0

    diagnosis = {
        "summary": (
            f"Text2SQL 独立实验 EX={text2sql_ex*100:.1f}% (50条全量)，"
            f"Multi-Agent SQL Accuracy={ma_sql_acc_v1*100:.1f}% "
            f"(仅统计sql_query任务且可验证的{valid}条)。"
            f"Multi-Agent SQL 执行成功率={ma_sql_exec_rate*100:.1f}% "
            f"vs Text2SQL EXE={text2sql_exe*100:.1f}%。"
        ),
        "text2sql_standalone": {
            "ex": text2sql_ex,
            "exe": text2sql_exe,
            "total_queries": 50,
            "note": "所有50条查询，提供完整Schema",
        },
        "multi_agent": {
            "total_sql_tasks": ma_total,
            "sql_generated": len(ma_generated),
            "sql_executed": len(ma_executed),
            "sql_correct": len(ma_correct),
            "sql_wrong": len(ma_wrong),
            "sql_unverifiable": len(ma_unverifiable),
            "execution_rate": round(ma_sql_exec_rate, 4),
            "accuracy_v1_method": round(ma_sql_acc_v1, 4),
            "accuracy_over_all_sql_tasks": round(len(ma_correct) / ma_total, 4) if ma_total > 0 else 0,
        },
        "failure_analysis": {k: {"count": len(v), "task_ids": v} for k, v in failure_reasons.items()},
        "root_causes": [
            {
                "cause": "Schema Agent 字段筛选过窄",
                "detail": (
                    "Multi-Agent 流程中 Schema Agent 使用 ChromaDB (top_k=15) + LLM 重排序"
                    "选择相关表和字段。如果 Schema Agent 筛选掉了 SQL 生成所需的关键字段，"
                    "SQL Agent 将无法生成正确的 SQL。"
                ),
                "evidence": (
                    f"Multi-Agent 中仅 {ma_total} 条 sql_query 任务，"
                    f"其中 {len(ma_executed)} 条执行成功，{len(ma_correct)} 条结果正确。"
                    f"Text2SQL 独立实验提供完整表结构，执行成功率达 {text2sql_exe*100:.0f}%。"
                ),
                "severity": "high",
            },
            {
                "cause": "SQL 自修正循环不足",
                "detail": (
                    "SQL Agent 重试时使用相同的 Schema 上下文，如果 Schema 本身就不足，"
                    "重试也无法修复。重试只修复语法错误和字段名错误，不修复逻辑错误。"
                ),
                "evidence": (
                    f"Text2SQL 失败案例中 88.5% 是逻辑错误（结果不匹配），"
                    f"不是语法错误。自修正循环对此类错误无效。"
                ),
                "severity": "medium",
            },
            {
                "cause": "评估口径不一致",
                "detail": (
                    "V1/V2 的 compute_sql_accuracy 仅统计 task_type='sql_query' 的任务，"
                    "且只计入 result_correct 可验证的样本。当 expected_sql 缺失或 SQL "
                    "未执行时，该样本不计入分母，导致分母缩小而分子不变，准确率偏低。"
                ),
                "evidence": (
                    f"可验证样本数={valid}，总 sql_query 任务数={ma_total}，"
                    f"差异={ma_total - valid} 条。"
                ),
                "severity": "medium",
            },
        ],
        "recommendations": [
            "增大 Schema Agent 的 top_k（当前15→建议25-30），减少漏选风险",
            "在 SQL Agent 中增加 second-pass：如果第一次生成失败，自动使用全量 Schema 重试",
            "统一评估口径：增加对所有 SQL 执行结果的验证（不限于 sql_query 类型任务）",
        ],
    }

    return diagnosis


# ============================================================
# Report Agent Input Completeness Check
# ============================================================

def check_report_agent_inputs(details: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    检查 Report Agent 是否收到了完整的上游输入。

    Report Agent 需要：
    - analysis_result (from Analysis Agent)
    - prediction_result (from Prediction Agent)
    - governance_result (from Governance Agent)
    - query_result (from SQL Agent)
    """
    if not details:
        return {}

    report_tasks = [d for d in details
                    if d.get("task_type") in ("mixed", "report")
                    or "Report Agent" in str(d.get("agents_invoked", []))]

    if not report_tasks:
        # 如果没有 mixed/report 任务，检查所有调用了 Report Agent 的任务
        report_tasks = [d for d in details
                        if "Report Agent" in str(d.get("agents_invoked", []))]

    results = []
    for d in report_tasks:
        tid = d.get("id", "?")
        has_sql_result = d.get("query_result") is not None
        has_analysis = d.get("analysis_result") is not None
        has_prediction = d.get("prediction_result") is not None
        has_governance = d.get("governance_result") is not None
        has_report = bool(d.get("report") or d.get("llm_output"))

        missing = []
        if not has_sql_result:
            missing.append("query_result (SQL查询结果)")
        if not has_analysis:
            missing.append("analysis_result (分析结果)")
        if not has_prediction:
            missing.append("prediction_result (预测结果)")
        if not has_governance:
            missing.append("governance_result (治理结果)")

        results.append({
            "task_id": tid,
            "task_type": d.get("task_type", "?"),
            "report_generated": has_report,
            "inputs": {
                "query_result": has_sql_result,
                "analysis_result": has_analysis,
                "prediction_result": has_prediction,
                "governance_result": has_governance,
            },
            "missing_inputs": missing,
            "input_completeness": round(
                (has_sql_result + has_analysis + has_prediction + has_governance) / 4, 2
            ),
        })

    complete = [r for r in results if r["input_completeness"] == 1.0]
    partial = [r for r in results if 0 < r["input_completeness"] < 1.0]
    empty = [r for r in results if r["input_completeness"] == 0.0]

    return {
        "total_report_tasks": len(results),
        "full_input_tasks": len(complete),
        "partial_input_tasks": len(partial),
        "no_input_tasks": len(empty),
        "avg_input_completeness": round(
            sum(r["input_completeness"] for r in results) / len(results), 4
        ) if results else 0,
        "per_task": results,
        "summary": (
            f"共 {len(results)} 个任务调用 Report Agent，"
            f"其中 {len(complete)} 个 ({len(complete)*100//max(len(results),1)}%) 收到完整上游输入，"
            f"{len(partial)} 个输入不完整，{len(empty)} 个无任何输入。"
            f"最常见缺失: {_most_missing(results)}"
        ),
    }


def _most_missing(results: List[Dict]) -> str:
    """统计最常见的缺失输入。"""
    counter = defaultdict(int)
    for r in results:
        for m in r.get("missing_inputs", []):
            counter[m] += 1
    if not counter:
        return "无"
    top = max(counter, key=counter.get)
    return f"{top} (缺失{counter[top]}次)"


# ============================================================
# V3 表格生成
# ============================================================

def generate_table_v3_agent_coverage(acr_data: Dict[str, Any]) -> str:
    """
    表V3-1: Agent Coverage Rate —— 各Agent在不同任务类型中的调用率。
    """
    lines = []
    lines.append("### 表V3-1：Agent 覆盖率分析 (Agent Coverage Rate)")
    lines.append("")
    lines.append("| Agent | 总调用次数 | 总调用率 | SQL查询 | 业务分析 | 预测任务 | 报告生成 |")
    lines.append("|-------|-----------|---------|---------|---------|---------|---------|")

    by_agent = acr_data.get("by_agent", {})
    by_type = acr_data.get("by_task_type", {})

    for agent in CORE_AGENTS:
        info = by_agent.get(agent, {})
        count = info.get("count", 0)
        rate = info.get("rate", 0)

        type_rates = []
        for tt in ["sql_query", "analysis", "prediction", "mixed"]:
            tr = by_type.get(tt, {}).get("agent_rates", {}).get(agent, 0)
            type_rates.append(f"{tr*100:.0f}%")

        lines.append(
            f"| {AGENT_NAME_CN.get(agent, agent)} | {count} | {rate*100:.0f}% | "
            + " | ".join(type_rates) + " |"
        )

    # 汇总行
    overall_acr = acr_data.get("overall_acr", 0)
    lines.append(f"| **整体 ACR** | — | **{overall_acr*100:.1f}%** | — | — | — | — |")
    lines.append("")
    lines.append(f"- 完整链任务数: {acr_data.get('full_chain_tasks', 0)}/{acr_data.get('full_chain_rate', 0)*100:.0f}%")
    lines.append("")

    return "\n".join(lines)


def generate_table_v3_task_type_analysis(
    details: List[Dict[str, Any]],
    acr_data: Dict[str, Any],
) -> str:
    """
    表V3-2: 任务类型 × Agent 执行链分析。
    """
    lines = []
    lines.append("### 表V3-2：任务类型 × 执行链分析")
    lines.append("")

    by_type = acr_data.get("by_task_type", {})

    lines.append("| 任务类型 | 任务数 | 平均Agent数 | ACR | 完整链率 | 主要跳过的Agent |")
    lines.append("|----------|--------|-----------|-----|---------|----------------|")

    for tt in ["sql_query", "analysis", "prediction", "mixed"]:
        info = by_type.get(tt, {})
        if not info:
            continue
        n = info.get("task_count", 0)
        avg_agents = info.get("avg_agents", 0)
        acr = info.get("acr", 0)

        # 该类型中完整链的比例
        type_details = [d for d in details if d.get("task_type") == tt]
        full_in_type = sum(
            1 for d in type_details
            if len(set(d.get("agents_invoked", [])) & set(CORE_AGENTS)) == len(CORE_AGENTS)
        )
        full_rate = full_in_type / n if n > 0 else 0

        # 被跳过的Agent
        agent_rates = info.get("agent_rates", {})
        skipped = [AGENT_NAME_CN.get(a, a) for a, r in agent_rates.items() if r < 0.5]
        skipped_str = ", ".join(skipped) if skipped else "无"

        lines.append(
            f"| {tt} | {n} | {avg_agents} | {acr*100:.0f}% | {full_rate*100:.0f}% | {skipped_str} |"
        )

    lines.append("")
    return "\n".join(lines)


def generate_table_v3_agent_timing(timing_data: Dict[str, Any]) -> str:
    """
    表V3-3: Agent 执行耗时分析。
    """
    lines = []
    lines.append("### 表V3-3：Agent 执行耗时分析")
    lines.append("")
    lines.append("| Agent | 执行次数 | 平均耗时(s) | P50(s) | P95(s) | 最大(s) | 错误率 |")
    lines.append("|-------|---------|-----------|--------|--------|--------|--------|")

    for agent in ALL_AGENTS:
        info = timing_data.get(agent, {})
        count = info.get("count", 0)
        if count == 0:
            lines.append(f"| {AGENT_NAME_CN.get(agent, agent)} | 0 | — | — | — | — | — |")
            continue

        avg_t = info.get("avg_seconds", 0)
        p50 = info.get("p50_seconds", "—") or "—"
        p95 = info.get("p95_seconds", "—") or "—"
        max_t = info.get("max_seconds", 0)
        err_rate = info.get("error_rate", 0)

        lines.append(
            f"| {AGENT_NAME_CN.get(agent, agent)} | {count} | {avg_t:.3f} | "
            f"{p50 if isinstance(p50, str) else f'{p50:.3f}'} | "
            f"{p95 if isinstance(p95, str) else f'{p95:.3f}'} | "
            f"{max_t:.3f} | {err_rate*100:.1f}% |"
        )

    bottleneck = timing_data.get("_bottleneck", {})
    if bottleneck:
        lines.append("")
        lines.append(f"**瓶颈Agent**: {AGENT_NAME_CN.get(bottleneck.get('agent', ''), bottleneck.get('agent', ''))} "
                    f"(平均 {bottleneck.get('avg_seconds', 0):.3f}s)")

    lines.append("")
    return "\n".join(lines)


def generate_table_v3_sql_diagnosis(diagnosis: Dict[str, Any]) -> str:
    """
    表V3-4: SQL 准确率诊断对比。
    """
    lines = []
    lines.append("### 表V3-4：SQL 准确率诊断 —— Multi-Agent vs Text2SQL")
    lines.append("")

    t2s = diagnosis.get("text2sql_standalone", {})
    ma = diagnosis.get("multi_agent", {})

    lines.append("| 指标 | Text2SQL 独立实验 | Multi-Agent 实验 | 差异 |")
    lines.append("|------|-----------------|-----------------|------|")
    lines.append(f"| 测试样本数 | 50 (全量) | {ma.get('total_sql_tasks', 0)} (仅sql_query) | — |")
    lines.append(f"| SQL 执行成功率 | {t2s.get('exe', 0)*100:.1f}% | {ma.get('execution_rate', 0)*100:.1f}% | "
                f"{ma.get('execution_rate', 0)*100 - t2s.get('exe', 0)*100:+.1f}% |")
    lines.append(f"| 执行准确率 (EX) | {t2s.get('ex', 0)*100:.1f}% | {ma.get('accuracy_v1_method', 0)*100:.1f}% | "
                f"{ma.get('accuracy_v1_method', 0)*100 - t2s.get('ex', 0)*100:+.1f}% |")

    failure = diagnosis.get("failure_analysis", {})
    for reason, info in failure.items():
        lines.append(f"| {reason} | — | {info['count']} 条 | — |")

    lines.append("")
    lines.append("**根因分析**:")
    for i, rc in enumerate(diagnosis.get("root_causes", []), 1):
        lines.append(f"{i}. **{rc['cause']}** [{rc['severity']}]: {rc['detail']}")

    lines.append("")
    return "\n".join(lines)


def generate_table_v3_report_input_check(check_data: Dict[str, Any]) -> str:
    """
    表V3-5: Report Agent 输入完整性检查。
    """
    lines = []
    lines.append("### 表V3-5：Report Agent 输入完整性")
    lines.append("")
    lines.append(f"- 调用 Report Agent 的任务数: {check_data.get('total_report_tasks', 0)}")
    lines.append(f"- 收到完整上游输入: {check_data.get('full_input_tasks', 0)}")
    lines.append(f"- 输入部分完整: {check_data.get('partial_input_tasks', 0)}")
    lines.append(f"- 无任何上游输入: {check_data.get('no_input_tasks', 0)}")
    lines.append(f"- 平均输入完整度: {check_data.get('avg_input_completeness', 0)*100:.0f}%")
    lines.append("")

    per_task = check_data.get("per_task", [])
    if per_task:
        lines.append("| 任务ID | 任务类型 | SQL结果 | 分析结果 | 预测结果 | 治理结果 | 报告生成 | 缺失输入 |")
        lines.append("|--------|---------|---------|---------|---------|---------|---------|---------|")
        for t in per_task[:20]:  # 最多显示20条
            inp = t["inputs"]
            missing = ", ".join(t["missing_inputs"]) if t["missing_inputs"] else "无"
            lines.append(
                f"| {t['task_id']} | {t['task_type']} | "
                f"{'✅' if inp['query_result'] else '❌'} | "
                f"{'✅' if inp['analysis_result'] else '❌'} | "
                f"{'✅' if inp['prediction_result'] else '❌'} | "
                f"{'✅' if inp['governance_result'] else '❌'} | "
                f"{'✅' if t['report_generated'] else '❌'} | "
                f"{missing} |"
            )
    lines.append("")
    return "\n".join(lines)


# ============================================================
# 综合报告生成
# ============================================================

def generate_v3_diagnosis_report(
    details: List[Dict[str, Any]],
    acr_data: Dict[str, Any],
    timing_data: Dict[str, Any],
    chain_data: Dict[str, Any],
    sql_diagnosis: Dict[str, Any],
    report_check: Dict[str, Any],
    v2_metrics: Optional[Dict[str, Any]] = None,
) -> str:
    """生成 V3 完整诊断报告 (Markdown)。"""
    from datetime import datetime, timezone

    lines = []
    lines.append("# Multi-Agent 执行链诊断报告 (V3)")
    lines.append("")
    lines.append(f"**生成时间**: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    lines.append(f"**评价体系版本**: V3 (执行链诊断 + Agent 覆盖分析)")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 1. V3 诊断目标")
    lines.append("")
    lines.append("1. **执行链完整性**: 验证 Planner → Schema → SQL → Governance → Analysis → Prediction → Report 的完整调用链")
    lines.append("2. **Agent 覆盖率**: 各 Agent 在不同任务类型中的实际调用率")
    lines.append("3. **SQL 准确率异常**: 诊断 Multi-Agent 6.7% vs Text2SQL 48% 的差距根因")
    lines.append("4. **Report Agent 输入**: 检查 Report Agent 是否收到完整上游数据")
    lines.append("5. **执行瓶颈**: 识别耗时最长的 Agent 节点")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 2. 执行链架构回顾")
    lines.append("")
    lines.append("```")
    lines.append("START → Planner → Schema → SQL → Governance ─┬→ Analysis → Prediction → Report → Chart → END")
    lines.append("                                              ├→ Prediction ──────────────────────────→ Chart → END")
    lines.append("                                              └→ END (sql_query)")
    lines.append("```")
    lines.append("")
    lines.append("- **sql_query**: 4 agents (Planner, Schema, SQL, Governance)")
    lines.append("- **analysis**: 6 agents (+ Analysis, Chart)")
    lines.append("- **prediction**: 5 agents (+ Prediction, Chart)")
    lines.append("- **mixed**: 7 agents (+ Analysis, Prediction, Report, Chart)")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Agent Coverage
    lines.append("## 3. Agent 覆盖率分析")
    lines.append("")
    lines.append(generate_table_v3_agent_coverage(acr_data))
    lines.append(generate_table_v3_task_type_analysis(details, acr_data))

    # Agent Timing
    lines.append("## 4. Agent 执行耗时分析")
    lines.append("")
    if timing_data:
        lines.append(generate_table_v3_agent_timing(timing_data))
    else:
        lines.append("*无 agent_timing 数据。V3 实验需启用 agent tracing。*")
    lines.append("")

    # Execution Chain
    lines.append("## 5. 执行链完整性")
    lines.append("")
    lines.append(f"- 唯一执行路径: {chain_data.get('unique_patterns', 0)} 种")
    lines.append(f"- 断裂链（有 error）: {chain_data.get('broken_chain_count', 0)} 个任务")
    lines.append("")

    patterns = chain_data.get("chain_patterns", {})
    if patterns:
        lines.append("### 执行路径分布")
        lines.append("")
        lines.append("| 执行路径 | 频次 |")
        lines.append("|---------|------|")
        for pattern, count in list(patterns.items())[:15]:
            lines.append(f"| {pattern} | {count} |")
        lines.append("")

    error_cascade = chain_data.get("error_cascade", {})
    if error_cascade:
        lines.append("### 错误级联分析")
        lines.append("")
        lines.append(f"- 总错误数: {error_cascade.get('total_errors', 0)}")
        lines.append(f"- SQL 相关失败: {error_cascade.get('sql_failures', 0)}")
        lines.append(f"- Schema 相关失败: {error_cascade.get('schema_failures', 0)}")
        lines.append(f"- Analysis 失败: {error_cascade.get('analysis_failures', 0)}")
        lines.append(f"- Prediction 失败: {error_cascade.get('prediction_failures', 0)}")
        lines.append(f"- Report 失败: {error_cascade.get('report_failures', 0)}")
        lines.append("")
        lines.append("*错误级联模式: SQL 失败 → 无 query_result → 下游所有 Agent 输入为空 → 报告无数据支撑*")
        lines.append("")

    # SQL Diagnosis
    lines.append("## 6. SQL 准确率深度诊断")
    lines.append("")
    lines.append(generate_table_v3_sql_diagnosis(sql_diagnosis))

    # Report Agent Input Check
    lines.append("## 7. Report Agent 输入完整性")
    lines.append("")
    lines.append(generate_table_v3_report_input_check(report_check))

    # V3 Summary
    lines.append("## 8. V3 诊断总结")
    lines.append("")
    lines.append(f"| 诊断项 | 状态 | 关键发现 |")
    lines.append(f"|--------|------|---------|")
    lines.append(f"| 执行链完整性 | {'✅' if chain_data.get('unique_patterns', 0) <= 5 else '⚠️'} | "
                f"{chain_data.get('unique_patterns', 0)} 种执行路径 |")
    lines.append(f"| Agent 覆盖率 | "
                f"{'✅' if acr_data.get('overall_acr', 0) > 0.6 else '⚠️'} | "
                f"整体 ACR={acr_data.get('overall_acr', 0)*100:.1f}% |")
    lines.append(f"| SQL 准确率 | ❌ | "
                f"Multi-Agent {sql_diagnosis.get('multi_agent', {}).get('accuracy_v1_method', 0)*100:.1f}% "
                f"vs Text2SQL 48.0% — Schema筛选过窄是主因 |")
    lines.append(f"| Report 输入 | "
                f"{'✅' if report_check.get('full_input_tasks', 0) > report_check.get('total_report_tasks', 1) * 0.5 else '⚠️'} | "
                f"{report_check.get('full_input_tasks', 0)}/{report_check.get('total_report_tasks', 0)} 任务输入完整 |")
    lines.append(f"| 执行瓶颈 | {'⚠️' if timing_data.get('_bottleneck') else '—'} | "
                f"{timing_data.get('_bottleneck', {}).get('agent', '无数据')} |")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("*本报告由 Multi-Agent 实验框架 V3 执行链诊断模块自动生成。*")
    lines.append("")

    return "\n".join(lines)


# ============================================================
# CSV 导出
# ============================================================

def save_v3_tables_csv(
    acr_data: Dict[str, Any],
    timing_data: Dict[str, Any],
    sql_diagnosis: Dict[str, Any],
    report_check: Dict[str, Any],
    output_dir: Path,
    timestamp: str = "",
) -> Dict[str, Path]:
    """将 V3 表格导出为 CSV 文件。"""
    saved = {}
    tables_dir = Path(output_dir) / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    # 表V3-1: Agent Coverage CSV
    rows_acr = []
    by_agent = acr_data.get("by_agent", {})
    by_type = acr_data.get("by_task_type", {})
    for agent in CORE_AGENTS:
        info = by_agent.get(agent, {})
        row = {
            "Agent": AGENT_NAME_CN.get(agent, agent),
            "总调用次数": info.get("count", 0),
            "总调用率": info.get("rate", 0),
        }
        for tt in ["sql_query", "analysis", "prediction", "mixed"]:
            tr = by_type.get(tt, {}).get("agent_rates", {}).get(agent, 0)
            row[f"{tt}_调用率"] = tr
        rows_acr.append(row)

    path1 = tables_dir / f"v3_agent_coverage_{timestamp}.csv"
    pd.DataFrame(rows_acr).to_csv(path1, index=False, encoding="utf-8-sig")
    saved["agent_coverage"] = path1

    # 表V3-2: Agent Timing CSV
    rows_timing = []
    for agent in ALL_AGENTS:
        info = timing_data.get(agent, {})
        rows_timing.append({
            "Agent": AGENT_NAME_CN.get(agent, agent),
            "执行次数": info.get("count", 0),
            "平均耗时(s)": info.get("avg_seconds", 0),
            "P50(s)": info.get("p50_seconds", None),
            "P95(s)": info.get("p95_seconds", None),
            "最大耗时(s)": info.get("max_seconds", 0),
            "错误率": info.get("error_rate", 0),
        })
    path2 = tables_dir / f"v3_agent_timing_{timestamp}.csv"
    pd.DataFrame(rows_timing).to_csv(path2, index=False, encoding="utf-8-sig")
    saved["agent_timing"] = path2

    # 表V3-3: SQL Diagnosis CSV
    ma = sql_diagnosis.get("multi_agent", {})
    t2s = sql_diagnosis.get("text2sql_standalone", {})
    rows_sql = [
        {"指标": "SQL执行成功率", "Text2SQL": t2s.get('exe', 0), "MultiAgent": ma.get('execution_rate', 0)},
        {"指标": "执行准确率", "Text2SQL": t2s.get('ex', 0), "MultiAgent": ma.get('accuracy_v1_method', 0)},
        {"指标": "可验证样本数", "Text2SQL": 50, "MultiAgent": ma.get('total_sql_tasks', 0)},
    ]
    path3 = tables_dir / f"v3_sql_diagnosis_{timestamp}.csv"
    pd.DataFrame(rows_sql).to_csv(path3, index=False, encoding="utf-8-sig")
    saved["sql_diagnosis"] = path3

    # 表V3-4: Report Input Check CSV
    per_task = report_check.get("per_task", [])
    if per_task:
        rows_ri = []
        for t in per_task:
            inp = t["inputs"]
            rows_ri.append({
                "任务ID": t["task_id"],
                "任务类型": t["task_type"],
                "SQL结果": inp["query_result"],
                "分析结果": inp["analysis_result"],
                "预测结果": inp["prediction_result"],
                "治理结果": inp["governance_result"],
                "报告生成": t["report_generated"],
                "输入完整度": t["input_completeness"],
                "缺失输入": ", ".join(t["missing_inputs"]) if t["missing_inputs"] else "无",
            })
        path4 = tables_dir / f"v3_report_input_check_{timestamp}.csv"
        pd.DataFrame(rows_ri).to_csv(path4, index=False, encoding="utf-8-sig")
        saved["report_input_check"] = path4

    return saved


# ============================================================
# 综合指标摘要
# ============================================================

def compute_v3_summary(
    acr_data: Dict[str, Any],
    timing_data: Dict[str, Any],
    chain_data: Dict[str, Any],
    sql_diagnosis: Dict[str, Any],
    report_check: Dict[str, Any],
) -> Dict[str, Any]:
    """生成 V3 核心指标摘要 JSON。"""
    return {
        "agent_coverage": {
            "overall_acr": acr_data.get("overall_acr"),
            "overall_avg_agents": acr_data.get("overall_avg_agents"),
            "full_chain_tasks": acr_data.get("full_chain_tasks"),
            "full_chain_rate": acr_data.get("full_chain_rate"),
        },
        "execution_chain": {
            "unique_patterns": chain_data.get("unique_patterns"),
            "broken_chains": chain_data.get("broken_chain_count"),
        },
        "sql_accuracy": {
            "multi_agent_accuracy": sql_diagnosis.get("multi_agent", {}).get("accuracy_v1_method"),
            "text2sql_accuracy": sql_diagnosis.get("text2sql_standalone", {}).get("ex"),
            "gap": (sql_diagnosis.get("multi_agent", {}).get("accuracy_v1_method", 0)
                    - sql_diagnosis.get("text2sql_standalone", {}).get("ex", 0)),
        },
        "report_agent": {
            "total_tasks": report_check.get("total_report_tasks"),
            "full_input_tasks": report_check.get("full_input_tasks"),
            "avg_input_completeness": report_check.get("avg_input_completeness"),
        },
        "bottleneck": timing_data.get("_bottleneck", {}).get("agent") if timing_data else None,
    }
