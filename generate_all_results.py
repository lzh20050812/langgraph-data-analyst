"""
Complete end-to-end: Run LLM-as-Judge → Generate paper tables → Generate Markdown report.

Usage: python generate_all_results.py
"""

import json
import sys
import time
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent))

from evaluation.metrics.multi_agent_metrics import (
    evaluate_all_reports,
    compute_all_multi_agent_metrics,
    generate_table1_comparison,
    generate_table2_task_types,
    generate_table3_resources,
    generate_table4_ablation,
    generate_failure_analysis,
    generate_full_experiment_report,
)

# ============================================================
# File paths for the LATEST comparison and ablation results
# ============================================================
RAW = Path("evaluation/results/raw")
TABLES = Path("evaluation/results/tables")
TABLES.mkdir(parents=True, exist_ok=True)

COMPARISON_FILES = {
    "single_llm": RAW / "multi_agent_single_llm_20260811_030000.json",
    "rag_llm": RAW / "multi_agent_rag_llm_20260811_031200.json",
    "full_multi_agent": RAW / "multi_agent_full_multi_agent_20260811_033619.json",
}

ABLATION_FILES = {
    "full_model": RAW / "multi_agent_full_multi_agent_20260811_040312.json",
    "no_rag": RAW / "multi_agent_ablation_no_rag_20260811_041231.json",
    "no_multi_agent": RAW / "multi_agent_ablation_no_multi_agent_20260811_042433.json",
    "no_self_correction": RAW / "multi_agent_ablation_no_self_correction_20260811_043258.json",
}


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    start_time = time.time()

    # ============================================================
    # Step 1: Run LLM-as-Judge on ALL comparison results
    # ============================================================
    print("=" * 60)
    print("Step 1: LLM-as-Judge on Comparison (3 schemes)")
    print("=" * 60)

    comparison_metrics = {}
    for name, path in COMPARISON_FILES.items():
        print(f"\n--- {name} ---")
        data = load_json(path)
        details = data["details"]

        # Count reports before
        non_sql = [d for d in details if d.get("task_type") != "sql_query"]
        has_content = sum(1 for d in non_sql if (
            d.get("llm_output", "") or d.get("report", "")
        ) and len((d.get("llm_output") or d.get("report") or "").strip()) >= 30)
        print(f"  Non-SQL tasks: {len(non_sql)}, with content: {has_content}")

        # Run LLM-as-Judge
        evaluate_all_reports(details, skip_sql_query=True)

        # Count scores
        scored = [d for d in details if d.get("report_quality_score") is not None]
        if scored:
            avg = sum(d["report_quality_score"] for d in scored) / len(scored)
            print(f"  Scored: {len(scored)}, avg score: {avg:.1f}")

        # Save updated results with judge scores
        save_path = RAW / f"{path.stem}_judged.json"
        save_json(data, save_path)

        # Compute metrics
        metrics = compute_all_multi_agent_metrics(details)
        comparison_metrics[name] = metrics
        print(f"  TSR={metrics['task_success_rate']:.1%}, RPT={metrics['report_score']}, SQL_ACC={metrics['sql_accuracy']:.1%}")

    # ============================================================
    # Step 2: Run LLM-as-Judge on ALL ablation results
    # ============================================================
    print("\n" + "=" * 60)
    print("Step 2: LLM-as-Judge on Ablation (4 variants)")
    print("=" * 60)

    ablation_metrics = {}
    for name, path in ABLATION_FILES.items():
        print(f"\n--- {name} ---")
        data = load_json(path)
        details = data["details"]

        non_sql = [d for d in details if d.get("task_type") != "sql_query"]
        has_content = sum(1 for d in non_sql if (
            d.get("llm_output", "") or d.get("report", "")
        ) and len((d.get("llm_output") or d.get("report") or "").strip()) >= 30)
        print(f"  Non-SQL tasks: {len(non_sql)}, with content: {has_content}")

        # Run LLM-as-Judge
        evaluate_all_reports(details, skip_sql_query=True)

        scored = [d for d in details if d.get("report_quality_score") is not None]
        if scored:
            avg = sum(d["report_quality_score"] for d in scored) / len(scored)
            print(f"  Scored: {len(scored)}, avg score: {avg:.1f}")

        # Save
        save_path = RAW / f"{path.stem}_judged.json"
        save_json(data, save_path)

        # Compute metrics
        metrics = compute_all_multi_agent_metrics(details)
        ablation_metrics[name] = metrics
        print(f"  TSR={metrics['task_success_rate']:.1%}, RPT={metrics['report_score']}, SQL_ACC={metrics['sql_accuracy']:.1%}")

    # ============================================================
    # Step 3: Generate Paper Tables
    # ============================================================
    print("\n" + "=" * 60)
    print("Step 3: Generate Paper Tables")
    print("=" * 60)

    s = comparison_metrics["single_llm"]
    r = comparison_metrics["rag_llm"]
    m = comparison_metrics["full_multi_agent"]

    # Table 1
    t1 = generate_table1_comparison(s, r, m)
    (TABLES / "table1_comparison.md").write_text(t1, encoding="utf-8")
    print("  table1_comparison.md")

    # Table 2
    t2 = generate_table2_task_types(s, r, m)
    (TABLES / "table2_by_task_type.md").write_text(t2, encoding="utf-8")
    print("  table2_by_task_type.md")

    # Table 3
    t3 = generate_table3_resources(s, r, m)
    (TABLES / "table3_resources.md").write_text(t3, encoding="utf-8")
    print("  table3_resources.md")

    # Table 4
    t4 = generate_table4_ablation(ablation_metrics)
    (TABLES / "table4_ablation.md").write_text(t4, encoding="utf-8")
    print("  table4_ablation.md")

    # Save metrics as CSV for paper use
    import csv

    # Table 1 CSV
    with open(TABLES / "table1_comparison.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["方案", "任务成功率", "Report Score", "SQL准确率", "平均耗时(s)", "Agent调用", "LLM调用"])
        for label, key in [("Single LLM", "single_llm"), ("LLM+RAG", "rag_llm"), ("Multi-Agent", "full_multi_agent")]:
            m_data = comparison_metrics.get(key, {})
            w.writerow([
                label,
                f"{m_data.get('task_success_rate', 0)*100:.1f}%",
                m_data.get('report_score', 'N/A'),
                f"{m_data.get('sql_accuracy', 0)*100:.1f}%",
                m_data.get('avg_response_time_seconds', 'N/A'),
                m_data.get('avg_agent_calls', 'N/A'),
                m_data.get('avg_llm_calls', 'N/A'),
            ])
    print("  table1_comparison.csv")

    # Table 4 CSV
    with open(TABLES / "table4_ablation.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["模型版本", "任务成功率", "Report Score", "SQL准确率", "平均响应时间(s)", "Agent调用", "LLM调用"])
        for label, key in [
            ("完整模型", "full_model"),
            ("去除RAG", "no_rag"),
            ("去除Multi-Agent", "no_multi_agent"),
            ("去除SQL自修正", "no_self_correction"),
        ]:
            a_data = ablation_metrics.get(key, {})
            w.writerow([
                label,
                f"{a_data.get('task_success_rate', 0)*100:.1f}%",
                a_data.get('report_score', 'N/A'),
                f"{a_data.get('sql_accuracy', 0)*100:.1f}%",
                a_data.get('avg_response_time_seconds', 'N/A'),
                a_data.get('avg_agent_calls', 'N/A'),
                a_data.get('avg_llm_calls', 'N/A'),
            ])
    print("  table4_ablation.csv")

    # Task type breakdown CSV
    with open(TABLES / "table2_task_types.csv", "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["任务类型", "样本数", "SingleLLM_TSR", "SingleLLM_Report", "RAG_TSR", "RAG_Report", "MultiAgent_TSR", "MultiAgent_Report"])
        for tt, label in [("sql_query", "SQL查询"), ("analysis", "业务分析"), ("prediction", "预测任务"), ("mixed", "报告生成")]:
            s_tt = s.get("by_task_type", {}).get(tt, {})
            r_tt = r.get("by_task_type", {}).get(tt, {})
            m_tt = m.get("by_task_type", {}).get(tt, {})
            cnt = s_tt.get("count", 0)
            w.writerow([
                label, cnt,
                f"{s_tt.get('task_success_rate', 0)*100:.1f}%",
                s_tt.get('report_quality_score', s_tt.get('avg_response_time', 'N/A')),
                f"{r_tt.get('task_success_rate', 0)*100:.1f}%",
                r_tt.get('report_quality_score', r_tt.get('avg_response_time', 'N/A')),
                f"{m_tt.get('task_success_rate', 0)*100:.1f}%",
                m_tt.get('report_quality_score', m_tt.get('avg_response_time', 'N/A')),
            ])
    print("  table2_task_types.csv")

    # ============================================================
    # Step 4: Generate Full Markdown Report
    # ============================================================
    print("\n" + "=" * 60)
    print("Step 4: Generate Full Markdown Report")
    print("=" * 60)

    # Load full_multi_agent details for failure analysis
    ma_data = load_json(COMPARISON_FILES["full_multi_agent"])
    ma_details = ma_data["details"]

    duration = time.time() - start_time
    report = generate_full_experiment_report(
        single_metrics=s,
        rag_metrics=r,
        multi_metrics=m,
        ablation_results=ablation_metrics,
        multi_details=ma_details,
        duration_seconds=duration,
    )

    report_path = Path("evaluation/results/experiment_report.md")
    report_path.write_text(report, encoding="utf-8")
    print(f"  Report: {report_path} ({len(report)} chars)")

    # Also save metrics summary JSON for reference
    metrics_summary = {
        "comparison": {k: {
            "task_success_rate": v["task_success_rate"],
            "report_score": v["report_score"],
            "sql_accuracy": v["sql_accuracy"],
            "avg_response_time_seconds": v["avg_response_time_seconds"],
            "avg_agent_calls": v["avg_agent_calls"],
            "avg_llm_calls": v["avg_llm_calls"],
            "by_task_type": v["by_task_type"],
        } for k, v in comparison_metrics.items()},
        "ablation": {k: {
            "task_success_rate": v["task_success_rate"],
            "report_score": v["report_score"],
            "sql_accuracy": v["sql_accuracy"],
            "avg_response_time_seconds": v["avg_response_time_seconds"],
            "avg_agent_calls": v["avg_agent_calls"],
            "avg_llm_calls": v["avg_llm_calls"],
            "by_task_type": v["by_task_type"],
        } for k, v in ablation_metrics.items()},
    }
    save_json(metrics_summary, RAW / "metrics_summary.json")
    print("  metrics_summary.json")

    # ============================================================
    # Print Final Summary
    # ============================================================
    print("\n" + "=" * 60)
    print("FINAL RESULTS SUMMARY")
    print("=" * 60)

    print("\n--- Comparison (3-scheme) ---")
    print(f"{'Scheme':<25} {'TSR':>8} {'RPT':>8} {'SQL_ACC':>8} {'Time(s)':>8} {'Agents':>8} {'LLM':>8}")
    print("-" * 73)
    for label, key in [("Single LLM", "single_llm"), ("LLM+RAG", "rag_llm"), ("Multi-Agent", "full_multi_agent")]:
        m_data = comparison_metrics.get(key, {})
        print(f"{label:<25} {m_data.get('task_success_rate', 0):>7.1%} {m_data.get('report_score', 0):>8} "
              f"{m_data.get('sql_accuracy', 0):>7.1%} {m_data.get('avg_response_time_seconds', 0):>8.2f} "
              f"{m_data.get('avg_agent_calls', 0):>8} {m_data.get('avg_llm_calls', 0):>8}")

    print("\n--- Ablation (4 variants) ---")
    print(f"{'Variant':<25} {'TSR':>8} {'RPT':>8} {'SQL_ACC':>8} {'Time(s)':>8} {'Agents':>8} {'LLM':>8}")
    print("-" * 73)
    for label, key in [
        ("Full Model", "full_model"),
        ("-RAG", "no_rag"),
        ("-Multi-Agent", "no_multi_agent"),
        ("-SQL Self-Correction", "no_self_correction"),
    ]:
        a_data = ablation_metrics.get(key, {})
        print(f"{label:<25} {a_data.get('task_success_rate', 0):>7.1%} {a_data.get('report_score', 0):>8} "
              f"{a_data.get('sql_accuracy', 0):>7.1%} {a_data.get('avg_response_time_seconds', 0):>8.2f} "
              f"{a_data.get('avg_agent_calls', 0):>8} {a_data.get('avg_llm_calls', 0):>8}")

    print(f"\nTotal duration: {duration:.1f}s")
    print("Done!")


if __name__ == "__main__":
    main()
