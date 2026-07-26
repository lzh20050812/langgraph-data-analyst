"""
质量报告生成器 —— 对比治理前后的数据质量指标。

核心逻辑：
- 读取 injection_report.json（治理前的"理想"注入统计）
- 对比 ETL 清洗后的实际结果
- 输出量化提升数字，直接可用于论文"数据治理效果"章节

为什么放在 ETL 层而不是 Agent 层：
- 这是批处理阶段的质量对比，是确定性统计，不涉及 LLM 语义理解
- 数据治理 Agent（governance_agent.py）负责的是运行时查询结果的质量评分
"""

import json
import pandas as pd
from pathlib import Path
from typing import Dict


def load_injection_report(report_path: Path) -> dict:
    """加载注入报告。"""
    with open(report_path, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_quality_metrics(df: pd.DataFrame, source_name: str) -> dict:
    """
    计算 DataFrame 的实际质量指标。

    Returns:
        {
            "total_rows": int,
            "total_columns": int,
            "missing": {col: {"missing_count": int, "missing_rate": float}},
            "duplicate_rows": int,
        }
    """
    n_rows = len(df)

    # 缺失值统计
    missing = {}
    for col in df.columns:
        n_miss = int(df[col].isna().sum())
        if n_miss > 0:
            missing[col] = {
                "missing_count": n_miss,
                "missing_rate": round(n_miss / n_rows, 4),
            }

    # 重复行统计
    n_dup = int(df.duplicated().sum())

    return {
        "total_rows": n_rows,
        "total_columns": len(df.columns),
        "missing": missing,
        "duplicate_rows": n_dup,
    }


def compare_quality(
    injection_report: dict,
    app_cleaned: pd.DataFrame,
    web_cleaned: pd.DataFrame,
    merged_cleaned: pd.DataFrame,
) -> dict:
    """
    对比注入报告（治理前）和实际清洗结果（治理后），生成质量提升报告。

    注意：治理前的指标来自 injection_report.json（已知注入量），
    治理后的指标来自清洗后 DataFrame 的实际统计。
    """
    sources_injection = {s["source"]: s for s in injection_report["sources"]}

    comparison = {
        "original_customers": injection_report["original_rows"],
        "merged_final_rows": len(merged_cleaned),
        "sources_comparison": [],
    }

    # App端对比
    app_inj = sources_injection.get("customers_app_raw.csv", {})
    app_metrics = compute_quality_metrics(app_cleaned, "app")
    app_comparison = {
        "source": "customers_app_raw.csv",
        "before_cleaning": {
            "total_rows": app_inj.get("total_rows_after", "?"),
            "missing_injections": app_inj.get("injections", {}).get("missing", {}),
            "duplicate_injections": app_inj.get("injections", {}).get("duplicates", {}),
        },
        "after_cleaning": {
            "total_rows": app_metrics["total_rows"],
            "missing_remaining": app_metrics["missing"],
            "duplicate_rows_remaining": app_metrics["duplicate_rows"],
        },
        "improvement": {
            "missing_resolved": _count_resolved(
                app_inj.get("injections", {}).get("missing", {}),
                app_metrics["missing"],
            ),
            "duplicates_removed": _get_dup_removed(
                app_inj.get("injections", {}).get("duplicates", {}),
                app_metrics["duplicate_rows"],
            ),
        },
    }
    comparison["sources_comparison"].append(app_comparison)

    # Web端对比
    web_inj = sources_injection.get("customers_web_raw.csv", {})
    web_metrics = compute_quality_metrics(web_cleaned, "web")
    web_comparison = {
        "source": "customers_web_raw.csv",
        "before_cleaning": {
            "total_rows": web_inj.get("total_rows_after", "?"),
            "missing_injections": web_inj.get("injections", {}).get("missing", {}),
            "duplicate_injections": web_inj.get("injections", {}).get("duplicates", {}),
        },
        "after_cleaning": {
            "total_rows": web_metrics["total_rows"],
            "missing_remaining": web_metrics["missing"],
            "duplicate_rows_remaining": web_metrics["duplicate_rows"],
        },
        "improvement": {
            "missing_resolved": _count_resolved(
                web_inj.get("injections", {}).get("missing", {}),
                web_metrics["missing"],
            ),
            "duplicates_removed": _get_dup_removed(
                web_inj.get("injections", {}).get("duplicates", {}),
                web_metrics["duplicate_rows"],
            ),
        },
    }
    comparison["sources_comparison"].append(web_comparison)

    # 汇总
    comparison["summary"] = {
        "total_missing_injected": _sum_injected_missing(injection_report),
        "total_missing_remaining": (
            sum(len(v) for v in [app_metrics["missing"], web_metrics["missing"]])
        ),
        "total_duplicates_injected": _sum_injected_dupes(injection_report),
        "total_duplicates_remaining": (
            app_metrics["duplicate_rows"] + web_metrics["duplicate_rows"]
        ),
        "final_merged_rows": len(merged_cleaned),
    }

    return comparison


def _count_resolved(injected: dict, remaining: dict) -> dict:
    """计算各列缺失值治理前后的对比。"""
    resolved = {}
    for col, info in injected.items():
        remaining_count = remaining.get(col, {}).get("missing_count", 0)
        injected_count = info.get("missing_count", 0)
        resolved[col] = {
            "injected": injected_count,
            "remaining_after_cleaning": remaining_count,
            "resolution_rate": (
                round((injected_count - remaining_count) / injected_count, 4)
                if injected_count > 0 else 1.0
            ),
        }
    return resolved


def _get_dup_removed(injected: dict, remaining_count: int) -> dict:
    """计算重复行治理前后对比。"""
    injected_count = injected.get("duplicate_rows_added", 0) if injected else 0
    return {
        "injected": injected_count,
        "remaining_after_cleaning": remaining_count,
        "resolution_rate": (
            round((injected_count - remaining_count) / injected_count, 4)
            if injected_count > 0 else 1.0
        ),
    }


def _sum_injected_missing(report: dict) -> int:
    """汇总注入的缺失值总数。"""
    total = 0
    for source in report.get("sources", []):
        for col_info in source.get("injections", {}).get("missing", {}).values():
            total += col_info.get("missing_count", 0)
    return total


def _sum_injected_dupes(report: dict) -> int:
    """汇总注入的重复行总数。"""
    total = 0
    for source in report.get("sources", []):
        dup_info = source.get("injections", {}).get("duplicates", {})
        total += dup_info.get("duplicate_rows_added", 0)
    return total


def print_quality_summary(comparison: dict) -> None:
    """打印可读的质量对比摘要。"""
    print("\n" + "=" * 60)
    print("数据治理效果 —— 前后对比")
    print("=" * 60)

    s = comparison["summary"]
    print(f"  原始 customers 行数: {comparison['original_customers']}")
    print(f"  合并去重后最终行数: {s['final_merged_rows']}")
    print(f"  注入缺失值总数: {s['total_missing_injected']} → 治理后剩余: {s['total_missing_remaining']}")
    print(f"  注入重复行总数: {s['total_duplicates_injected']} → 治理后剩余: {s['total_duplicates_remaining']}")

    for src in comparison["sources_comparison"]:
        print(f"\n--- {src['source']} ---")
        for col, info in src["improvement"]["missing_resolved"].items():
            print(f"  {col}: {info['injected']}缺失 → {info['remaining_after_cleaning']}剩余 "
                  f"(修复率 {info['resolution_rate']*100:.1f}%)")
        dup = src["improvement"]["duplicates_removed"]
        print(f"  重复行: {dup['injected']}注入 → {dup['remaining_after_cleaning']}剩余 "
              f"(修复率 {dup['resolution_rate']*100:.1f}%)")

    print("=" * 60)
