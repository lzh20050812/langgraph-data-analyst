"""
V2 Fair Evaluation Runner — 重新运行实验并使用新评价体系。

用法:
  # 完整运行（三方案对比 + 消融 + V2评价 + 报告）
  python -m evaluation.experiments.run_v2_evaluation

  # 仅三方案对比
  python -m evaluation.experiments.run_v2_evaluation --compare-only

  # 仅消融实验
  python -m evaluation.experiments.run_v2_evaluation --ablation-only

  # 快速模式：复用已有实验详情，仅重新评价（跳过实验重跑）
  python -m evaluation.experiments.run_v2_evaluation --fast

所有结果保存至: evaluation/results/multi_agent_v2/
"""

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# Add project root to path
_project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_project_root))

# Existing experiment infrastructure (unchanged)
from evaluation.experiments.multi_agent_experiment import (
    MultiAgentExperiment,
    ABLATION_VARIANTS,
    _fmt,
    _timestamp,
)

# V2 metric functions
from evaluation.metrics.multi_agent_metrics_v2 import (
    compute_all_metrics_v2,
    evaluate_all_reports_v2,
    generate_table1_comparison_v2,
    generate_table2_task_types_v2,
    generate_table3_resources_v2,
    generate_table4_ablation_v2,
    generate_full_experiment_report_v2,
    generate_tcs_comparison_table_v2,
    save_all_tables_csv_v2,
)


# ============================================================
# Paths
# ============================================================
V2_RESULTS_DIR = _project_root / "evaluation" / "results" / "multi_agent_v2"
V2_RAW_DIR = V2_RESULTS_DIR / "raw"
V2_TABLES_DIR = V2_RESULTS_DIR / "tables"

# Old results paths for --fast mode
OLD_RAW_DIR = _project_root / "evaluation" / "results" / "raw"
OLD_COMPARISON_FILES = {
    "single_llm": OLD_RAW_DIR / "multi_agent_single_llm_20260811_030000.json",
    "rag_llm": OLD_RAW_DIR / "multi_agent_rag_llm_20260811_031200.json",
    "full_multi_agent": OLD_RAW_DIR / "multi_agent_full_multi_agent_20260811_033619.json",
}
OLD_ABLATION_FILES = {
    "full_model": OLD_RAW_DIR / "multi_agent_full_multi_agent_20260811_040312.json",
    "no_rag": OLD_RAW_DIR / "multi_agent_ablation_no_rag_20260811_041231.json",
    "no_multi_agent": OLD_RAW_DIR / "multi_agent_ablation_no_multi_agent_20260811_042433.json",
    "no_self_correction": OLD_RAW_DIR / "multi_agent_ablation_no_self_correction_20260811_043258.json",
}


# ============================================================
# Core: Run and evaluate a single experiment configuration
# ============================================================

def run_single_v2(
    mode: str = "full_multi_agent",
    ablation: Optional[str] = None,
    test_data_path: Optional[str] = None,
    max_samples: int = 0,
) -> Dict[str, Any]:
    """运行单个实验配置并使用 V2 指标评价。

    返回 {"details": [...], "metrics": {...}, "name": str}
    """
    config = {
        "experiment": {"name": f"multi_agent_v2_{ablation or mode}", "version": "2.0"},
        "limits": {"max_samples": max_samples},
        "mode": mode,
        "ablation": ablation,
    }
    if test_data_path:
        config["data"] = {"test_data_path": test_data_path}

    label = ablation or mode
    print(f"\n{'=' * 60}")
    print(f"V2 Experiment: {label}")
    print(f"{'=' * 60}")

    exp = MultiAgentExperiment(config)
    result = exp.execute()

    # Apply V2 metrics to details
    details = result.details
    metrics_v2 = compute_all_metrics_v2(details)

    # Print summary
    m = metrics_v2
    print(f"[{label}] TCS={m['task_completion_score']:.1f}, "
          f"RQS={m['report_quality_score']:.1f}, "
          f"SQL_ACC={_fmt(m.get('sql_accuracy'))}, "
          f"Time={m['avg_response_time_seconds']:.2f}s, "
          f"Agents={m['avg_agent_calls']}, LLM={m['avg_llm_calls']}")

    return {
        "details": details,
        "metrics": metrics_v2,
        "name": exp.name,
        "duration": result.duration_seconds,
    }


def load_and_evaluate_v2(file_path: Path) -> Dict[str, Any]:
    """加载已有实验详情，使用 V2 指标重新评价。"""
    print(f"\n--- Loading & re-evaluating: {file_path.name} ---")
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    details = data.get("details", [])
    if not details:
        print(f"  WARNING: No details found in {file_path.name}")
        return {"details": [], "metrics": {}, "name": file_path.stem, "duration": 0}

    # Ensure details have mode/ablation info from file metadata
    for d in details:
        if "mode" not in d or not d["mode"]:
            d["mode"] = data.get("config", {}).get("mode", "full_multi_agent")

    metrics_v2 = compute_all_metrics_v2(details)

    m = metrics_v2
    print(f"  TCS={m['task_completion_score']:.1f}, RQS={m['report_quality_score']:.1f}, "
          f"SQL_ACC={_fmt(m.get('sql_accuracy'))}")

    return {
        "details": details,
        "metrics": metrics_v2,
        "name": file_path.stem,
        "duration": 0,
    }


# ============================================================
# Full V2 Evaluation Pipeline
# ============================================================

def run_full_v2_evaluation(
    fast_mode: bool = False,
    test_data_path: Optional[str] = None,
    max_samples: int = 0,
) -> None:
    """完整的 V2 评价流程：运行实验 → V2 评价 → LLM-as-Judge → 生成报告。"""
    total_start = time.time()

    # Create output directories
    V2_RAW_DIR.mkdir(parents=True, exist_ok=True)
    V2_TABLES_DIR.mkdir(parents=True, exist_ok=True)

    # ============================================================
    # Phase 1: Run/Load experiments
    # ============================================================
    print("\n" + "=" * 70)
    print("PHASE 1: 获取实验数据")
    print("=" * 70)

    comparison_data = {}  # {mode: {"details": [...], "metrics": {...}}}
    ablation_data = {}    # {variant: {"details": [...], "metrics": {...}}}

    if fast_mode:
        print("\n[Fast Mode] 复用已有实验详情，仅重新评价...")
        for mode, path in OLD_COMPARISON_FILES.items():
            if path.exists():
                comparison_data[mode] = load_and_evaluate_v2(path)
            else:
                print(f"  WARNING: {path} not found, skipping {mode}")

        for variant, path in OLD_ABLATION_FILES.items():
            if path.exists():
                ablation_data[variant] = load_and_evaluate_v2(path)
            else:
                print(f"  WARNING: {path} not found, skipping {variant}")
    else:
        print("\n[Full Mode] 重新运行所有实验...")
        # Phase 1a: Comparison (3 schemes)
        print("\n" + "-" * 60)
        print("1a: 三方案对比实验 (50 queries × 3)")
        print("-" * 60)
        for mode in ("single_llm", "rag_llm", "full_multi_agent"):
            data = run_single_v2(
                mode=mode,
                test_data_path=test_data_path,
                max_samples=max_samples,
            )
            comparison_data[mode] = data

        # Phase 1b: Ablation (4 variants)
        print("\n" + "-" * 60)
        print("1b: 消融实验 (4 variants)")
        print("-" * 60)
        for label, ablation in ABLATION_VARIANTS:
            data = run_single_v2(
                mode="full_multi_agent",
                ablation=ablation,
                test_data_path=test_data_path,
                max_samples=max_samples,
            )
            ablation_data[label] = data

    # ============================================================
    # Phase 2: V2 LLM-as-Judge on all reports
    # ============================================================
    print("\n" + "=" * 70)
    print("PHASE 2: V2 LLM-as-Judge 报告质量评估")
    print("=" * 70)

    all_data_sets = {**comparison_data, **ablation_data}
    total_scored = 0

    for name, data in all_data_sets.items():
        details = data.get("details", [])
        if not details:
            continue

        non_sql = [d for d in details if d.get("task_type") != "sql_query"]
        has_content = sum(1 for d in non_sql if
            (_get_output_text_v2(d) or "").strip() and
            len((_get_output_text_v2(d) or "").strip()) >= 30)

        if has_content == 0:
            print(f"  {name}: No reports to score ({len(non_sql)} non-SQL, 0 with content)")
            continue

        print(f"  {name}: {len(non_sql)} non-SQL, {has_content} with content...")
        evaluate_all_reports_v2(details, skip_sql_query=True)

        scored = [d for d in details if d.get("report_quality_score_v2") is not None]
        if scored:
            avg = sum(d["report_quality_score_v2"] for d in scored) / len(scored)
            print(f"    Scored: {len(scored)}, avg RQS: {avg:.1f}")
            total_scored += len(scored)

        # Recompute metrics with judge scores
        data["metrics"] = compute_all_metrics_v2(details)

    print(f"\n  Total reports scored: {total_scored}")

    # ============================================================
    # Phase 3: Save results
    # ============================================================
    print("\n" + "=" * 70)
    print("PHASE 3: 保存结果")
    print("=" * 70)

    ts = _timestamp()

    # Save comparison results
    for mode, data in comparison_data.items():
        save_experiment_result_v2(data, mode, ts)

    # Save ablation results
    for variant, data in ablation_data.items():
        save_experiment_result_v2(data, variant, ts)

    # Save metrics summary
    metrics_summary = {
        "evaluation_version": "V2",
        "timestamp": ts,
        "comparison": {
            k: {kk: vv for kk, vv in v["metrics"].items() if kk != "by_task_type"}
            for k, v in comparison_data.items()
        },
        "ablation": {
            k: {kk: vv for kk, vv in v["metrics"].items() if kk != "by_task_type"}
            for k, v in ablation_data.items()
        },
    }
    summary_path = V2_RAW_DIR / f"metrics_summary_v2_{ts}.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, ensure_ascii=False, indent=2)
    print(f"  metrics_summary: {summary_path.name}")

    # ============================================================
    # Phase 4: Generate paper tables + report
    # ============================================================
    print("\n" + "=" * 70)
    print("PHASE 4: 生成技术评测表格和实验报告")
    print("=" * 70)

    s = comparison_data.get("single_llm", {}).get("metrics", {})
    r = comparison_data.get("rag_llm", {}).get("metrics", {})
    m = comparison_data.get("full_multi_agent", {}).get("metrics", {})
    a = {k: v.get("metrics", {}) for k, v in ablation_data.items()}

    # Save MD tables
    (V2_TABLES_DIR / "table1_comparison_v2.md").write_text(
        generate_table1_comparison_v2(s, r, m), encoding="utf-8")
    (V2_TABLES_DIR / "table2_task_types_v2.md").write_text(
        generate_table2_task_types_v2(s, r, m), encoding="utf-8")
    (V2_TABLES_DIR / "table3_resources_v2.md").write_text(
        generate_table3_resources_v2(s, r, m), encoding="utf-8")
    (V2_TABLES_DIR / "table4_ablation_v2.md").write_text(
        generate_table4_ablation_v2(a), encoding="utf-8")

    # Save CSV tables
    csv_paths = save_all_tables_csv_v2(str(V2_RESULTS_DIR), s, r, m, a)
    for k, v in csv_paths.items():
        print(f"  {k}: {Path(v).name}")

    # Generate old vs new comparison
    multi_details = comparison_data.get("full_multi_agent", {}).get("details", [])
    single_details = comparison_data.get("single_llm", {}).get("details", [])
    rag_details = comparison_data.get("rag_llm", {}).get("details", [])
    old_vs_new = generate_tcs_comparison_table_v2(
        single_details, rag_details, multi_details)
    (V2_TABLES_DIR / "old_vs_new_comparison.md").write_text(old_vs_new, encoding="utf-8")

    # Generate full report
    total_duration = time.time() - total_start
    report = generate_full_experiment_report_v2(
        single_metrics=s,
        rag_metrics=r,
        multi_metrics=m,
        ablation_metrics=a,
        multi_details=multi_details,
        single_details=single_details,
        rag_details=rag_details,
        duration_seconds=total_duration,
        old_vs_new_comparison=old_vs_new,
    )

    report_path = V2_RESULTS_DIR / "experiment_report_v2.md"
    report_path.write_text(report, encoding="utf-8")
    print(f"  experiment_report_v2.md ({len(report)} chars)")

    # ============================================================
    # Final summary
    # ============================================================
    print("\n" + "=" * 70)
    print("V2 FAIR EVALUATION — FINAL RESULTS")
    print("=" * 70)

    print("\n--- 三方案对比 (V2统一TCS) ---")
    print(f"{'Scheme':<25} {'TCS':>8} {'RQS':>8} {'SQL_ACC':>8} {'Time(s)':>8} {'LLM':>8}")
    print("-" * 70)
    for label, key in [("Single LLM", "single_llm"), ("LLM+RAG", "rag_llm"),
                        ("Multi-Agent", "full_multi_agent")]:
        metrics = comparison_data.get(key, {}).get("metrics", {})
        print(f"{label:<25} {metrics.get('task_completion_score', 0):>7.1f} "
              f"{metrics.get('report_quality_score', 0):>8.1f} "
              f"{_fmt(metrics.get('sql_accuracy')):>8} "
              f"{metrics.get('avg_response_time_seconds', 0):>8.2f} "
              f"{metrics.get('avg_llm_calls', 0):>7.1f}")

    print("\n--- 消融实验 (V2统一TCS) ---")
    print(f"{'Variant':<25} {'TCS':>8} {'RQS':>8} {'SQL_ACC':>8} {'Time(s)':>8} {'LLM':>8}")
    print("-" * 70)
    for label, key in [
        ("Full Model", "full_model"),
        ("-RAG", "no_rag"),
        ("-Multi-Agent", "no_multi_agent"),
        ("-SQL Self-Correction", "no_self_correction"),
    ]:
        metrics = ablation_data.get(key, {}).get("metrics", {})
        print(f"{label:<25} {metrics.get('task_completion_score', 0):>7.1f} "
              f"{metrics.get('report_quality_score', 0):>8.1f} "
              f"{_fmt(metrics.get('sql_accuracy')):>8} "
              f"{metrics.get('avg_response_time_seconds', 0):>8.2f} "
              f"{metrics.get('avg_llm_calls', 0):>7.1f}")

    print(f"\nAll results saved to: {V2_RESULTS_DIR}")
    print(f"Total duration: {total_duration:.1f}s")
    print("=" * 70)


# ============================================================
# Helpers
# ============================================================

def _get_output_text_v2(detail: Dict[str, Any]) -> str:
    """从 detail 中提取输出文本（同 metrics_v2 中的逻辑）。"""
    mode = detail.get("mode", "full_multi_agent")
    ablation = detail.get("ablation") or ""
    if mode in ("single_llm", "rag_llm") or ablation == "no_multi_agent":
        return detail.get("llm_output", "") or ""
    else:
        return (detail.get("report") or detail.get("llm_output") or "")


def save_experiment_result_v2(
    data: Dict[str, Any], label: str, ts: str
) -> None:
    """保存单个实验结果到 V2 目录。"""
    details = data.get("details", [])
    metrics = data.get("metrics", {})

    # JSON: Full details + metrics
    json_path = V2_RAW_DIR / f"v2_{label}_{ts}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "label": label,
            "evaluation_version": "V2",
            "timestamp": ts,
            "metrics": metrics,
            "details": details,
        }, f, ensure_ascii=False, indent=2, default=str)

    # CSV: Per-task TCS
    import csv
    csv_path = V2_RAW_DIR / f"v2_{label}_tcs_details_{ts}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["id", "task_type", "query", "TCS", "D1_Understanding",
                     "D2_DataGrounding", "D3_AnalysisDepth", "D4_Completeness",
                     "RQS_V2", "duration_seconds", "error"])
        # Import the scoring functions to get per-dimension scores
        from evaluation.metrics.multi_agent_metrics_v2 import (
            _score_task_understanding,
            _score_data_grounding,
            _score_analysis_depth,
            _score_output_completeness,
        )
        for d in details:
            mode = d.get("mode", "")
            ablation = d.get("ablation") or ""
            task_type = d.get("task_type", "")
            w.writerow([
                d.get("id"),
                task_type,
                (d.get("query", "") or "")[:60],
                d.get("tcs", 0),
                _score_task_understanding(d, None),
                _score_data_grounding(d, task_type, mode, ablation),
                _score_analysis_depth(d, task_type, mode, ablation),
                _score_output_completeness(d, task_type, mode, ablation),
                d.get("report_quality_score_v2", "N/A"),
                d.get("duration_seconds", 0),
                (d.get("error") or "")[:80],
            ])


# ============================================================
# CLI
# ============================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(
        description="V2 Fair Evaluation — Multi-Agent 实验公平评价体系")
    parser.add_argument("--test-data", type=str, default=None)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--fast", action="store_true",
                        help="快速模式：复用已有实验详情，仅重新评价（跳过实验重跑）")
    parser.add_argument("--compare-only", action="store_true",
                        help="仅运行三方案对比")
    parser.add_argument("--ablation-only", action="store_true",
                        help="仅运行消融实验")
    args = parser.parse_args()

    if args.compare_only:
        print("仅运行三方案对比 + V2评价...")
        _run_comparison_only(args.test_data, args.max_samples, args.fast)
    elif args.ablation_only:
        print("仅运行消融实验 + V2评价...")
        _run_ablation_only(args.test_data, args.max_samples, args.fast)
    else:
        run_full_v2_evaluation(
            fast_mode=args.fast,
            test_data_path=args.test_data,
            max_samples=args.max_samples,
        )
    return 0


def _run_comparison_only(test_data_path, max_samples, fast):
    """仅运行三方案对比。"""
    V2_RAW_DIR.mkdir(parents=True, exist_ok=True)
    V2_TABLES_DIR.mkdir(parents=True, exist_ok=True)

    comparison_data = {}
    if fast:
        for mode, path in OLD_COMPARISON_FILES.items():
            if path.exists():
                comparison_data[mode] = load_and_evaluate_v2(path)
    else:
        for mode in ("single_llm", "rag_llm", "full_multi_agent"):
            comparison_data[mode] = run_single_v2(
                mode=mode, test_data_path=test_data_path, max_samples=max_samples)

    # Judge + Save + Report
    for data in comparison_data.values():
        evaluate_all_reports_v2(data["details"])
        data["metrics"] = compute_all_metrics_v2(data["details"])

    ts = _timestamp()
    for label, data in comparison_data.items():
        save_experiment_result_v2(data, label, ts)

    s = comparison_data["single_llm"]["metrics"]
    r = comparison_data["rag_llm"]["metrics"]
    m = comparison_data["full_multi_agent"]["metrics"]

    save_all_tables_csv_v2(str(V2_RESULTS_DIR), s, r, m, {})

    print("\n--- 三方案对比 V2 ---")
    for label, key in [("Single LLM", "single_llm"), ("LLM+RAG", "rag_llm"),
                        ("Multi-Agent", "full_multi_agent")]:
        met = comparison_data[key]["metrics"]
        print(f"{label}: TCS={met['task_completion_score']:.1f}, "
              f"RQS={met['report_quality_score']:.1f}, "
              f"SQL_ACC={_fmt(met.get('sql_accuracy'))}")


def _run_ablation_only(test_data_path, max_samples, fast):
    """仅运行消融实验。"""
    V2_RAW_DIR.mkdir(parents=True, exist_ok=True)
    V2_TABLES_DIR.mkdir(parents=True, exist_ok=True)

    ablation_data = {}
    if fast:
        for variant, path in OLD_ABLATION_FILES.items():
            if path.exists():
                ablation_data[variant] = load_and_evaluate_v2(path)
    else:
        for label, ablation in ABLATION_VARIANTS:
            ablation_data[label] = run_single_v2(
                mode="full_multi_agent", ablation=ablation,
                test_data_path=test_data_path, max_samples=max_samples)

    for data in ablation_data.values():
        evaluate_all_reports_v2(data["details"])
        data["metrics"] = compute_all_metrics_v2(data["details"])

    ts = _timestamp()
    for label, data in ablation_data.items():
        save_experiment_result_v2(data, label, ts)

    a = {k: v["metrics"] for k, v in ablation_data.items()}
    save_all_tables_csv_v2(str(V2_RESULTS_DIR), {}, {}, {}, a)

    print("\n--- 消融实验 V2 ---")
    for label, key in [
        ("Full Model", "full_model"), ("-RAG", "no_rag"),
        ("-Multi-Agent", "no_multi_agent"), ("-SQL SC", "no_self_correction"),
    ]:
        met = ablation_data[key]["metrics"]
        print(f"{label}: TCS={met['task_completion_score']:.1f}, "
              f"RQS={met['report_quality_score']:.1f}")


if __name__ == "__main__":
    sys.exit(main())
