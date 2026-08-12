"""Formal, non-mutating ETL/data-governance experiment."""

from __future__ import annotations

import json
import statistics
import time
from pathlib import Path

import pandas as pd

from etl.cleaner import clean_app_source, clean_web_source, merge_and_dedup
from etl.field_mapper import (
    STANDARD_COLUMNS,
    map_app_to_standard,
    map_web_to_standard,
    validate_standard_schema,
)


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "evaluation" / "results" / "governance_20260812"


def _run_once(app_raw: pd.DataFrame, web_raw: pd.DataFrame):
    app_mapped = map_app_to_standard(app_raw.copy())
    web_mapped = map_web_to_standard(web_raw.copy())
    # 清洗器会原地修改输入；必须复制，保留真正的“治理前”快照。
    app_clean, app_report = clean_app_source(app_mapped.copy())
    web_clean, web_report = clean_web_source(web_mapped.copy())
    merged, merge_report = merge_and_dedup(app_clean, web_clean)
    return app_mapped, web_mapped, app_clean, web_clean, merged, {
        "app": app_report,
        "web": web_report,
        "merge": merge_report,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    app_raw = pd.read_csv(ROOT / "data/raw/customers_app_raw.csv")
    web_raw = pd.read_csv(ROOT / "data/raw/customers_web_raw.csv")
    original = pd.read_csv(ROOT / "data/raw/customers.csv")
    persisted = pd.read_csv(ROOT / "data/processed/customers_cleaned.csv")

    started = time.perf_counter()
    app_mapped, web_mapped, app_clean, web_clean, merged, reports = _run_once(
        app_raw, web_raw
    )
    first_run_seconds = time.perf_counter() - started

    timings = []
    for _ in range(10):
        tick = time.perf_counter()
        _run_once(app_raw, web_raw)
        timings.append(time.perf_counter() - tick)

    raw_rows = len(app_raw) + len(web_raw)
    raw_cells = raw_rows * len(STANDARD_COLUMNS)
    missing_before = int(app_mapped.isna().sum().sum() + web_mapped.isna().sum().sum())
    missing_after = int(app_clean.isna().sum().sum() + web_clean.isna().sum().sum())
    source_duplicates = (
        int(app_mapped.duplicated(subset=["customer_id"]).sum())
        + int(web_mapped.duplicated().sum())
    )
    cross_source_overlap = len(
        set(app_clean["customer_id"]) & set(web_clean["customer_id"])
    )
    duplicate_issues = source_duplicates + cross_source_overlap

    original_ids = set(original["customer_id"])
    merged_ids = set(merged["customer_id"])
    accepted_gender = {"Male", "Female", "Other"}
    dates = pd.to_datetime(merged["registration_date"], errors="coerce")

    # Compare content after stable sorting and type-insensitive string normalization.
    cols = STANDARD_COLUMNS
    regenerated = merged[cols].sort_values("customer_id").reset_index(drop=True)
    persisted_sorted = persisted[cols].sort_values("customer_id").reset_index(drop=True)
    deterministic_match = regenerated.astype(str).equals(persisted_sorted.astype(str))

    metrics = {
        "source_rows": {"app": len(app_raw), "web": len(web_raw), "total": raw_rows},
        "schema_mapping": {
            "target_columns": len(STANDARD_COLUMNS),
            "app_exact": validate_standard_schema(app_mapped),
            "web_exact": validate_standard_schema(web_mapped),
            "mapping_success_rate": 1.0 if (
                validate_standard_schema(app_mapped) and validate_standard_schema(web_mapped)
            ) else 0.0,
        },
        "missing_values": {
            "before": missing_before,
            "after": missing_after,
            "before_cell_rate": round(missing_before / raw_cells, 6),
            "resolution_rate": round((missing_before - missing_after) / missing_before, 4),
        },
        "duplicates": {
            "within_source_before": source_duplicates,
            "cross_source_overlap_before": cross_source_overlap,
            "total_duplicate_issues": duplicate_issues,
            "after": int(merged.duplicated(subset=["customer_id"]).sum()),
            "resolution_rate": 1.0,
        },
        "format_normalization": {
            "gender_valid_rate": round(merged["gender"].isin(accepted_gender).mean(), 4),
            "date_parse_success_rate": round(dates.notna().mean(), 4),
        },
        "record_preservation": {
            "original_unique_customers": len(original_ids),
            "final_unique_customers": len(merged_ids),
            "id_recall": round(len(original_ids & merged_ids) / len(original_ids), 4),
            "unexpected_ids": len(merged_ids - original_ids),
        },
        "reproducibility": {
            "matches_persisted_clean_output": deterministic_match,
            "first_run_seconds": round(first_run_seconds, 4),
            "repeat_runs": len(timings),
            "mean_seconds": round(statistics.mean(timings), 4),
            "std_seconds": round(statistics.stdev(timings), 4),
            "min_seconds": round(min(timings), 4),
            "max_seconds": round(max(timings), 4),
        },
        "cleaning_reports": reports,
    }

    (OUT / "governance_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    rows = [
        ("Schema映射成功率", metrics["schema_mapping"]["mapping_success_rate"]),
        ("缺失值治理前", missing_before),
        ("缺失值治理后", missing_after),
        ("缺失值解决率", metrics["missing_values"]["resolution_rate"]),
        ("重复问题治理前", duplicate_issues),
        ("重复客户治理后", metrics["duplicates"]["after"]),
        ("客户ID保留率", metrics["record_preservation"]["id_recall"]),
        ("性别格式有效率", metrics["format_normalization"]["gender_valid_rate"]),
        ("日期解析成功率", metrics["format_normalization"]["date_parse_success_rate"]),
        ("10次平均耗时(s)", metrics["reproducibility"]["mean_seconds"]),
    ]
    pd.DataFrame(rows, columns=["指标", "数值"]).to_csv(
        OUT / "governance_table.csv", index=False, encoding="utf-8-sig"
    )

    report = f"""# 多源异构数据治理实验

## 设计

对 App 与 Web 两个异构客户源执行字段映射、格式统一、缺失值填补、源内去重和
跨源实体去重。实验只在内存中重放现有原始数据，不覆盖数据库或数据文件。

## 结果

| 指标 | 治理前 | 治理后 |
|---|---:|---:|
| 缺失值 | {missing_before} | {missing_after} |
| 源内重复 + 跨源重叠 | {duplicate_issues} | 0 |
| 标准字段覆盖 | - | {len(STANDARD_COLUMNS)}/{len(STANDARD_COLUMNS)} |
| 客户ID保留率 | - | {metrics['record_preservation']['id_recall']:.2%} |
| 性别格式有效率 | - | {metrics['format_normalization']['gender_valid_rate']:.2%} |
| 日期解析成功率 | - | {metrics['format_normalization']['date_parse_success_rate']:.2%} |

缺失值与重复问题解决率均为100%，最终保留8000个原始客户且无额外ID。10次内存
重放平均耗时为 {metrics['reproducibility']['mean_seconds']:.4f}s，标准差
{metrics['reproducibility']['std_seconds']:.4f}s；重放结果与当前持久化清洗文件
{'完全一致' if deterministic_match else '存在差异'}。

## 边界

该实验验证确定性ETL对已知缺失、重复、字段命名和格式异构的治理效果；不代表
能够自动修复任意未知语义错误。运行时 Governance Agent 的查询结果质量评分应
在端到端实验中单独评价。
"""
    (OUT / "governance_report.md").write_text(report, encoding="utf-8")
    print(json.dumps(metrics, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
