"""
Multi-Agent V4 评价指标 —— Schema增强 + 路由优化 + 报告压缩 效果评估。

V4 新增指标:
- SQL Context Quality Score: Schema 上下文是否包含 dtype/business_term/table_desc
- Report Latency: Report Agent 单独耗时分析
- Full Pipeline Rate: 完整 Agent 链执行比例（按 planned_agents 对照）
- V3 vs V4 对比表

V4 继承 V3 的所有指标:
- Agent Coverage Rate (ACR)
- Agent Execution Time
- Execution Chain Completeness
- SQL Accuracy Diagnosis
- Report Agent Input Check
"""

import json
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

import pandas as pd

from evaluation.metrics.multi_agent_metrics_v3 import (
    ALL_AGENTS,
    CORE_AGENTS,
    AGENT_NAME_CN,
    compute_agent_coverage_rate,
    compute_agent_timing,
    analyze_execution_chain,
    diagnose_sql_accuracy,
    check_report_agent_inputs,
    compute_v3_summary,
    generate_v3_diagnosis_report as _generate_v3_report_base,
    save_v3_tables_csv,
)


# ============================================================
# V4: SQL Context Quality Score
# ============================================================

def compute_sql_context_quality(details: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    V4: 测量 SQL Agent 收到的 Schema 上下文质量。

    检查 Schema 上下文是否包含:
    - dtype（数据类型）
    - business_term（业务含义）
    - table_context（表级描述）
    - 增强前后的字段对比

    Returns:
        {
            "avg_schema_fields": float,
            "dtype_coverage": float,           # dtype 覆盖率
            "business_term_coverage": float,   # business_term 覆盖率
            "table_context_present_rate": float,  # 表级上下文提供率
            "context_length_stats": {avg, p50, p95, max, min},
            "per_task_summary": [{task_id, fields, has_dtype, has_biz, has_table_ctx, ctx_len}],
        }
    """
    if not details:
        return {"avg_schema_fields": 0, "dtype_coverage": 0, "business_term_coverage": 0}

    total = len(details)
    has_dtype = 0
    has_biz = 0
    has_table_ctx = 0
    field_counts = []
    ctx_lengths = []

    per_task = []

    for d in details:
        task_id = d.get("id", "?")
        # 从 agent_trace 中检查 Schema Agent 日志
        trace = d.get("agent_trace", [])
        dtype_found = False
        biz_found = False
        table_ctx_found = False
        n_fields = 0
        ctx_len = 0

        for msg in trace:
            if not isinstance(msg, str):
                continue
            # Schema Agent 日志: "含 dtype + business_term"
            if "Schema Agent" in msg:
                if "dtype" in msg:
                    dtype_found = True
                if "business_term" in msg:
                    biz_found = True
            # SQL Agent 日志: "dtype=Y" / "business_term=Y"
            if "SQL Agent" in msg and "dtype=" in msg:
                if "dtype=Y" in msg:
                    dtype_found = True
                if "business_term=Y" in msg:
                    biz_found = True
                # 提取字段数和上下文长度
                # 格式: "Schema上下文: N字段, 增强后XXX字符"
                import re
                m = re.search(r'Schema上下文: (\d+)字段.*?增强后(\d+)字符', msg)
                if m:
                    n_fields = int(m.group(1))
                    ctx_len = int(m.group(2))
            # Table context check
            if "table_context" in msg.lower() or "表级" in msg:
                table_ctx_found = True

        if dtype_found:
            has_dtype += 1
        if biz_found:
            has_biz += 1
        if table_ctx_found:
            has_table_ctx += 1
        if n_fields > 0:
            field_counts.append(n_fields)
        if ctx_len > 0:
            ctx_lengths.append(ctx_len)

        per_task.append({
            "task_id": task_id,
            "fields": n_fields,
            "has_dtype": dtype_found,
            "has_business_term": biz_found,
            "has_table_context": table_ctx_found,
            "context_length": ctx_len,
        })

    def _stats(values: List[int]) -> Dict[str, float]:
        if not values:
            return {"avg": 0, "p50": 0, "p95": 0, "max": 0, "min": 0}
        sv = sorted(values)
        n = len(sv)
        return {
            "avg": round(sum(sv) / n, 1),
            "p50": sv[n // 2],
            "p95": sv[int(n * 0.95)] if n >= 20 else sv[-1],
            "max": max(sv),
            "min": min(sv),
        }

    return {
        "avg_schema_fields": round(sum(field_counts) / len(field_counts), 1) if field_counts else 0,
        "dtype_coverage": round(has_dtype / total, 4) if total > 0 else 0,
        "business_term_coverage": round(has_biz / total, 4) if total > 0 else 0,
        "table_context_present_rate": round(has_table_ctx / total, 4) if total > 0 else 0,
        "context_length_stats": _stats(ctx_lengths),
        "per_task_summary": per_task,
    }


# ============================================================
# V4: Report Latency Analysis
# ============================================================

def compute_report_latency(details: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    V4: Report Agent 单独耗时分析。

    Returns:
        {avg, p50, p95, max, min, count, target_met}
        target_met: 低于 25s 的任务比例（V4目标）
    """
    report_times = []
    for d in details:
        timing = d.get("agent_timing", [])
        for t in timing:
            if "Report" in t.get("agent", ""):
                duration = t.get("duration", 0)
                if duration > 0:
                    report_times.append(duration)

    if not report_times:
        return {"avg": 0, "p50": 0, "p95": 0, "max": 0, "min": 0, "count": 0, "target_met": 0}

    sv = sorted(report_times)
    n = len(sv)

    target_25s = sum(1 for t in report_times if t < 25)

    return {
        "avg": round(sum(sv) / n, 3),
        "p50": round(sv[n // 2], 3),
        "p95": round(sv[int(n * 0.95)], 3) if n >= 20 else None,
        "max": round(sv[-1], 3),
        "min": round(sv[0], 3),
        "count": n,
        "target_met": round(target_25s / n, 4) if n > 0 else 0,
        "target_met_count": target_25s,
    }


# ============================================================
# V4: Full Pipeline Rate
# ============================================================

def compute_full_pipeline_rate(details: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    V4: 完整 Agent 链执行比例。

    对照 planned_agents（Planner 计划调用的 Agent）和实际调用的 Agent，
    计算计划完成率。同时按 intent 分类统计。

    Returns:
        {
            "overall_completion_rate": float,
            "by_intent": {intent: {planned, invoked, completion_rate}},
            "per_task": [{task_id, intent, planned, invoked, missing, rate}],
        }
    """
    if not details:
        return {"overall_completion_rate": 0, "by_intent": {}, "per_task": []}

    per_task = []
    by_intent = defaultdict(lambda: {"total": 0, "planned_sum": 0, "invoked_sum": 0})

    for d in details:
        task_id = d.get("id", "?")
        intent = d.get("task_type", d.get("intent", "unknown"))
        agents_invoked = d.get("agents_invoked", [])
        recorded_plan = d.get("planned_agents") or []
        recognized_agents = set(ALL_AGENTS) | set(recorded_plan)

        # 从 agents_invoked 中提取实际调用的 Agent 名称
        invoked_set = set()
        for entry in agents_invoked:
            for agent_name in recognized_agents:
                if agent_name.lower() in entry.lower():
                    invoked_set.add(agent_name)

        # 也检查 agent_trace
        trace = d.get("agent_trace", [])
        for msg in trace:
            if isinstance(msg, str):
                for agent_name in recognized_agents:
                    if f"[{agent_name}" in msg or f"{agent_name}]" in msg:
                        invoked_set.add(agent_name)

        # 新链路以 Planner 实际产出的 planned_agents 为准。仅对不含该字段的
        # 历史结果保留 intent 推断，避免把“按需不调用”误判为链路缺失。
        if recorded_plan:
            planned_set = set(recorded_plan)
        else:
            base = {"Planner", "Schema Agent", "SQL Agent", "Governance Agent"}
            if intent == "sql_query":
                planned_set = base
            elif intent == "analysis":
                planned_set = base | {"Analysis Agent", "Chart Renderer"}
            elif intent == "prediction":
                planned_set = base | {"Prediction Agent", "Chart Renderer"}
            elif intent == "mixed":
                planned_set = base | {"Analysis Agent", "Prediction Agent",
                                      "Report Agent", "Chart Renderer"}
            else:
                planned_set = base | {"Analysis Agent", "Prediction Agent",
                                      "Report Agent", "Chart Renderer"}

        n_planned = len(planned_set)
        n_invoked = len(invoked_set & planned_set)
        rate = n_invoked / n_planned if n_planned > 0 else 0
        missing = planned_set - invoked_set

        per_task.append({
            "task_id": task_id,
            "intent": intent,
            "planned": sorted(planned_set),
            "invoked": sorted(invoked_set),
            "missing": sorted(missing),
            "completion_rate": round(rate, 4),
        })

        by_intent[intent]["total"] += 1
        by_intent[intent]["planned_sum"] += n_planned
        by_intent[intent]["invoked_sum"] += n_invoked

    # 计算按 intent 的完成率
    by_intent_result = {}
    for intent, data in by_intent.items():
        n = data["total"]
        by_intent_result[intent] = {
            "task_count": n,
            "completion_rate": round(data["invoked_sum"] / data["planned_sum"], 4)
            if data["planned_sum"] > 0 else 0,
        }

    overall = (
        sum(t["completion_rate"] for t in per_task) / len(per_task)
        if per_task else 0
    )

    return {
        "overall_completion_rate": round(overall, 4),
        "by_intent": by_intent_result,
        "per_task": per_task,
    }


# ============================================================
# V4: V3 vs V4 对比
# ============================================================

def compute_v4_comparison(
    v4_acr: Dict[str, Any],
    v4_timing: Dict[str, Any],
    v4_context_quality: Dict[str, Any],
    v4_report_latency: Dict[str, Any],
    v4_full_pipeline: Dict[str, Any],
    v4_sql_diagnosis: Dict[str, Any],
    v3_acr_r: float = 0.664,
    v3_report_avg: float = 46.64,
    v3_full_chain: float = 0.32,
    v3_sql_accuracy: float = 0.067,
    v3_analysis_coverage: float = 0.44,
    v3_prediction_coverage: float = 0.56,
    v3_report_coverage: float = 0.32,
) -> Dict[str, Any]:
    """
    V4: 生成 V3 baseline vs V4 优化后的对比表。

    Args:
        v4_*: V4 实验数据
        v3_*: V3 baseline 数据（硬编码，来自 V3 诊断报告）
    """
    # V4 指标提取
    v4_acr_val = v4_acr.get("overall_acr", 0)
    v4_full_chain_val = v4_acr.get("full_chain_rate", 0)
    v4_report_avg = v4_report_latency.get("avg", 0)
    v4_sql_acc = v4_sql_diagnosis.get("multi_agent", {}).get("accuracy_v1_method", 0)

    # Agent 覆盖率
    v4_by_agent = v4_acr.get("by_agent", {})
    v4_analysis_cov = v4_by_agent.get("Analysis Agent", {}).get("rate", 0)
    v4_prediction_cov = v4_by_agent.get("Prediction Agent", {}).get("rate", 0)
    v4_report_cov = v4_by_agent.get("Report Agent", {}).get("rate", 0)

    def _delta(v4, v3) -> str:
        diff = v4 - v3
        if diff > 0:
            return f"↑ +{diff*100:.1f}pp"
        elif diff < 0:
            return f"↓ {diff*100:.1f}pp"
        return "—"

    return {
        "comparison_valid": False,
        "warning": (
            "V3 数值来自历史文件，未与本次 V4 使用同一数据快照、依赖版本和评估口径；"
            "只能作背景参考，不得作为技术评测提升幅度结论。"
        ),
        "metrics": {
            "ACR": {
                "v3": round(v3_acr_r, 4),
                "v4": round(v4_acr_val, 4),
                "delta": _delta(v4_acr_val, v3_acr_r),
            },
            "SQL Accuracy": {
                "v3": round(v3_sql_accuracy, 4),
                "v4": round(v4_sql_acc, 4),
                "delta": _delta(v4_sql_acc, v3_sql_accuracy),
            },
            "Full Chain Rate": {
                "v3": round(v3_full_chain, 4),
                "v4": round(v4_full_chain_val, 4),
                "delta": _delta(v4_full_chain_val, v3_full_chain),
            },
            "Report Avg Latency": {
                "v3": round(v3_report_avg, 1),
                "v4": round(v4_report_avg, 1),
                "delta": f"{v4_report_avg - v3_report_avg:+.1f}s",
            },
            "Analysis Coverage": {
                "v3": round(v3_analysis_coverage, 4),
                "v4": round(v4_analysis_cov, 4),
                "delta": _delta(v4_analysis_cov, v3_analysis_coverage),
            },
            "Prediction Coverage": {
                "v3": round(v3_prediction_coverage, 4),
                "v4": round(v4_prediction_cov, 4),
                "delta": _delta(v4_prediction_cov, v3_prediction_coverage),
            },
            "Report Coverage": {
                "v3": round(v3_report_coverage, 4),
                "v4": round(v4_report_cov, 4),
                "delta": _delta(v4_report_cov, v3_report_coverage),
            },
            "Dtype Coverage": {
                "v3": 0.0,
                "v4": round(v4_context_quality.get("dtype_coverage", 0), 4),
                "delta": "V4 新增",
            },
            "Business Term Coverage": {
                "v3": 0.0,
                "v4": round(v4_context_quality.get("business_term_coverage", 0), 4),
                "delta": "V4 新增",
            },
        },
        "summary": "",
    }


# ============================================================
# V4 综合报告生成
# ============================================================

def generate_v4_diagnosis_report(
    details: List[Dict[str, Any]],
    acr_data: Dict[str, Any],
    timing_data: Dict[str, Any],
    chain_data: Dict[str, Any],
    sql_diagnosis: Dict[str, Any],
    report_check: Dict[str, Any],
    context_quality: Dict[str, Any],
    report_latency: Dict[str, Any],
    full_pipeline: Dict[str, Any],
    comparison: Dict[str, Any],
    v2_metrics: Optional[Dict[str, Any]] = None,
) -> str:
    """生成 V4 完整诊断报告 (Markdown)。"""
    from datetime import datetime, timezone

    lines = []
    lines.append("# Multi-Agent 执行链诊断报告 (V4)")
    lines.append("")
    lines.append(f"**生成时间**: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    lines.append(f"**评价体系版本**: V4 (Schema增强 + 路由优化 + 报告压缩)")
    lines.append("")
    lines.append("---")
    lines.append("")

    # V4 优化说明
    lines.append("## 1. V4 优化内容")
    lines.append("")
    lines.append("1. **Schema 增强**: Schema Agent 输出含 dtype + business_term + 表级描述，SQL Agent 上下文更完整")
    lines.append("2. **Planner 路由**: 关键词 + LLM 两阶段意图分类，减少分析/报告任务漏路由")
    lines.append("3. **Report 压缩**: max_tokens 4096→2000, 6段→4段结构，目标报告耗时<25s")
    lines.append("")
    lines.append("---")
    lines.append("")

    # V3 vs V4 对比（最重要）
    lines.append("## 2. V3 vs V4 核心指标对比")
    lines.append("")
    if not comparison.get("comparison_valid", False):
        lines.append(
            f"> **口径警告**：{comparison.get('warning', '该对比不是同口径实验。')}"
        )
        lines.append("")
    comp = comparison.get("metrics", {})
    if comp:
        lines.append("| 指标 | V3 Baseline | V4 优化后 | 变化 |")
        lines.append("|------|-----------|----------|------|")
        for metric_name, vals in comp.items():
            v3_val = vals["v3"]
            v4_val = vals["v4"]
            delta = vals["delta"]
            # 格式化
            if isinstance(v3_val, float) and v3_val < 1:
                v3_str = f"{v3_val*100:.1f}%"
            elif isinstance(v3_val, float):
                v3_str = f"{v3_val:.2f}"
            else:
                v3_str = str(v3_val)

            if isinstance(v4_val, float) and v4_val < 1:
                v4_str = f"{v4_val*100:.1f}%"
            elif isinstance(v4_val, float):
                v4_str = f"{v4_val:.2f}"
            else:
                v4_str = str(v4_val)

            lines.append(f"| {metric_name} | {v3_str} | {v4_str} | {delta} |")
        lines.append("")
    lines.append("---")
    lines.append("")

    # Agent Coverage (V3 table format)
    lines.append("## 3. Agent 覆盖率分析")
    lines.append("")
    lines.append(_generate_v4_agent_coverage_table(acr_data))
    lines.append("")

    # Agent Timing
    lines.append("## 4. Agent 执行耗时分析")
    lines.append("")
    if timing_data:
        lines.append(_generate_v4_timing_table(timing_data))
    else:
        lines.append("*无 agent_timing 数据。*")
    lines.append("")

    # Schema Context Quality (V4 New)
    lines.append("## 5. Schema 上下文质量 (V4 新增)")
    lines.append("")
    lines.append(f"- 平均字段数: {context_quality.get('avg_schema_fields', 0)}")
    lines.append(f"- dtype 覆盖率: {context_quality.get('dtype_coverage', 0)*100:.0f}%")
    lines.append(f"- business_term 覆盖率: {context_quality.get('business_term_coverage', 0)*100:.0f}%")
    lines.append(f"- 表级上下文提供率: {context_quality.get('table_context_present_rate', 0)*100:.0f}%")
    ctx_stats = context_quality.get("context_length_stats", {})
    if ctx_stats:
        lines.append(f"- 上下文长度: avg={ctx_stats.get('avg', 0)}字符, "
                    f"max={ctx_stats.get('max', 0)}字符")
    lines.append("")

    # Full Pipeline Rate (V4 New)
    lines.append("## 6. 完整链路执行率 (V4 新增)")
    lines.append("")
    lines.append(f"- 整体完成率: {full_pipeline.get('overall_completion_rate', 0)*100:.1f}%")
    by_intent = full_pipeline.get("by_intent", {})
    if by_intent:
        lines.append("")
        lines.append("| 任务类型 | 任务数 | 完成率 |")
        lines.append("|----------|--------|--------|")
        for intent, info in by_intent.items():
            lines.append(f"| {intent} | {info.get('task_count', 0)} | "
                        f"{info.get('completion_rate', 0)*100:.0f}% |")
    lines.append("")

    # Report Latency (V4 New)
    lines.append("## 7. Report Agent 延迟分析 (V4 新增)")
    lines.append("")
    rl = report_latency
    lines.append(f"- 调用次数: {rl.get('count', 0)}")
    lines.append(f"- 平均耗时: {rl.get('avg', 0):.1f}s")
    lines.append(f"- P50: {rl.get('p50', 0):.1f}s")
    if rl.get('p95'):
        lines.append(f"- P95: {rl['p95']:.1f}s")
    lines.append(f"- 最大耗时: {rl.get('max', 0):.1f}s")
    lines.append(f"- <25s 目标达成率: {rl.get('target_met', 0)*100:.0f}% "
                f"({rl.get('target_met_count', 0)}/{rl.get('count', 0)})")
    lines.append("")

    # Execution Chain
    lines.append("## 8. 执行链完整性")
    lines.append("")
    lines.append(f"- 唯一执行路径: {chain_data.get('unique_patterns', 0)} 种")
    lines.append(f"- 断裂链（有 error）: {chain_data.get('broken_chain_count', 0)} 个任务")
    patterns = chain_data.get("chain_patterns", {})
    if patterns:
        lines.append("")
        lines.append("| 执行路径 | 频次 |")
        lines.append("|---------|------|")
        for pattern, count in list(patterns.items())[:15]:
            lines.append(f"| {pattern} | {count} |")
    lines.append("")

    # SQL Accuracy
    lines.append("## 9. SQL 准确率")
    lines.append("")
    ma = sql_diagnosis.get("multi_agent", {})
    t2s = sql_diagnosis.get("text2sql_standalone", {})
    lines.append(f"| 指标 | Text2SQL | Multi-Agent (V4) |")
    lines.append(f"|------|----------|-----------------|")
    lines.append(f"| 执行成功率 | {t2s.get('exe', 0)*100:.1f}% | {ma.get('execution_rate', 0)*100:.1f}% |")
    lines.append(f"| 执行准确率 | {t2s.get('ex', 0)*100:.1f}% | {ma.get('accuracy_v1_method', 0)*100:.1f}% |")
    lines.append("")

    # V4 总结
    lines.append("## 10. V4 诊断总结")
    lines.append("")
    lines.append("| 维度 | V3 | V4 | 评价 |")
    lines.append("|------|----|----|------|")
    lines.append(f"| ACR | 66.4% | {acr_data.get('overall_acr', 0)*100:.1f}% | "
                f"{'✅ 改善' if acr_data.get('overall_acr', 0) > 0.664 else '⚠️ 待优化'} |")
    lines.append(f"| SQL Accuracy | 6.7% | {ma.get('accuracy_v1_method', 0)*100:.1f}% | "
                f"{'✅ 改善' if ma.get('accuracy_v1_method', 0) > 0.067 else '⚠️ 待优化'} |")
    lines.append(f"| Report 延迟 | 46.6s | {rl.get('avg', 0):.1f}s | "
                f"{'✅ 改善' if rl.get('avg', 0) < 46.6 else '⚠️ 待优化'} |")
    lines.append(f"| Full Chain Rate | 32% | {acr_data.get('full_chain_rate', 0)*100:.0f}% | "
                f"{'✅ 改善' if acr_data.get('full_chain_rate', 0) > 0.32 else '⚠️ 待优化'} |")
    lines.append(f"| dtype 覆盖 | 0% | {context_quality.get('dtype_coverage', 0)*100:.0f}% | "
                f"{'✅ 新增' if context_quality.get('dtype_coverage', 0) > 0.5 else '⚠️ 待排查'} |")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("*本报告由 Multi-Agent 实验框架 V4 自动生成。*")
    lines.append("")

    return "\n".join(lines)


def _generate_v4_agent_coverage_table(acr_data: Dict[str, Any]) -> str:
    """生成 Agent 覆盖率表格。"""
    lines = []
    lines.append("### Agent 覆盖率分析")
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

    overall_acr = acr_data.get("overall_acr", 0)
    lines.append(f"| **整体 ACR** | — | **{overall_acr*100:.1f}%** | — | — | — | — |")
    lines.append("")
    return "\n".join(lines)


def _generate_v4_timing_table(timing_data: Dict[str, Any]) -> str:
    """生成 Agent 耗时表格。"""
    lines = []
    lines.append("### Agent 执行耗时")
    lines.append("")
    lines.append("| Agent | 执行次数 | 平均耗时(s) | P50(s) | P95(s) | 最大(s) | 错误率 |")
    lines.append("|-------|---------|-----------|--------|--------|--------|--------|")

    for agent in ALL_AGENTS:
        info = timing_data.get(agent, {})
        count = info.get("count", 0)
        if count == 0:
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
        lines.append(f"**瓶颈Agent**: "
                    f"{AGENT_NAME_CN.get(bottleneck.get('agent', ''), bottleneck.get('agent', ''))} "
                    f"(平均 {bottleneck.get('avg_seconds', 0):.3f}s)")
    lines.append("")
    return "\n".join(lines)


# ============================================================
# V4 综合指标摘要
# ============================================================

def compute_v4_summary(
    acr_data: Dict[str, Any],
    timing_data: Dict[str, Any],
    chain_data: Dict[str, Any],
    sql_diagnosis: Dict[str, Any],
    report_check: Dict[str, Any],
    context_quality: Dict[str, Any],
    report_latency: Dict[str, Any],
    full_pipeline: Dict[str, Any],
    comparison: Dict[str, Any],
) -> Dict[str, Any]:
    """生成 V4 核心指标摘要 JSON。"""
    return {
        "agent_coverage": {
            "overall_acr": acr_data.get("overall_acr"),
            "full_chain_rate": acr_data.get("full_chain_rate"),
            "analysis_coverage": acr_data.get("by_agent", {}).get("Analysis Agent", {}).get("rate", 0),
            "prediction_coverage": acr_data.get("by_agent", {}).get("Prediction Agent", {}).get("rate", 0),
            "report_coverage": acr_data.get("by_agent", {}).get("Report Agent", {}).get("rate", 0),
        },
        "schema_context_quality": {
            "dtype_coverage": context_quality.get("dtype_coverage"),
            "business_term_coverage": context_quality.get("business_term_coverage"),
            "table_context_rate": context_quality.get("table_context_present_rate"),
            "avg_fields": context_quality.get("avg_schema_fields"),
        },
        "report_latency": {
            "avg_s": report_latency.get("avg"),
            "p50_s": report_latency.get("p50"),
            "p95_s": report_latency.get("p95"),
            "target_met_rate": report_latency.get("target_met"),
        },
        "full_pipeline": {
            "overall_completion": full_pipeline.get("overall_completion_rate"),
            "by_intent": full_pipeline.get("by_intent"),
        },
        "sql_accuracy": {
            "v4_accuracy": sql_diagnosis.get("multi_agent", {}).get("accuracy_v1_method"),
            "text2sql_baseline": 0.48,
            "v4_execution_rate": sql_diagnosis.get("multi_agent", {}).get("execution_rate"),
        },
        "v3_vs_v4_comparison": {
            "valid": comparison.get("comparison_valid", False),
            "warning": comparison.get("warning"),
            "metrics": comparison.get("metrics"),
        },
        "bottleneck": timing_data.get("_bottleneck", {}).get("agent") if timing_data else None,
    }


# ============================================================
# V4 CSV 导出
# ============================================================

def save_v4_tables_csv(
    acr_data: Dict[str, Any],
    timing_data: Dict[str, Any],
    sql_diagnosis: Dict[str, Any],
    report_check: Dict[str, Any],
    context_quality: Dict[str, Any],
    report_latency: Dict[str, Any],
    full_pipeline: Dict[str, Any],
    comparison: Dict[str, Any],
    output_dir: Path,
    timestamp: str = "",
) -> Dict[str, Path]:
    """将 V4 表格导出为 CSV 文件。"""
    saved = {}
    tables_dir = Path(output_dir) / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    # 继承 V3 的所有表格导出
    v3_saved = save_v3_tables_csv(acr_data, timing_data, sql_diagnosis,
                                  report_check, output_dir, timestamp)
    for k, v in v3_saved.items():
        saved[f"v4_{k}"] = v

    # V3 vs V4 对比表
    comp_metrics = comparison.get("metrics", {})
    if comp_metrics:
        rows = []
        for name, vals in comp_metrics.items():
            rows.append({
                "指标": name,
                "V3_Baseline": vals.get("v3"),
                "V4_优化后": vals.get("v4"),
                "变化": vals.get("delta"),
                "对比有效性": "仅历史参考，非同口径实验",
            })
        path = tables_dir / f"v4_comparison_{timestamp}.csv"
        pd.DataFrame(rows).to_csv(path, index=False, encoding="utf-8-sig")
        saved["comparison"] = path

    # Schema Context Quality 详情
    per_task_ctx = context_quality.get("per_task_summary", [])
    if per_task_ctx:
        path = tables_dir / f"v4_schema_context_quality_{timestamp}.csv"
        pd.DataFrame(per_task_ctx).to_csv(path, index=False, encoding="utf-8-sig")
        saved["schema_context_quality"] = path

    # Report Latency 摘要
    rl_rows = [{
        "指标": k, "值": v
    } for k, v in report_latency.items() if k != "per_task"]
    if rl_rows:
        path = tables_dir / f"v4_report_latency_{timestamp}.csv"
        pd.DataFrame(rl_rows).to_csv(path, index=False, encoding="utf-8-sig")
        saved["report_latency"] = path

    # Full Pipeline Rate 详情
    fp_per_task = full_pipeline.get("per_task", [])
    if fp_per_task:
        path = tables_dir / f"v4_full_pipeline_{timestamp}.csv"
        pd.DataFrame(fp_per_task).to_csv(path, index=False, encoding="utf-8-sig")
        saved["full_pipeline"] = path

    return saved
