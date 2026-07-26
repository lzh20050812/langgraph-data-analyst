"""
ETL 端到端验证脚本（无需 MySQL）。

运行：python -m etl.run_etl
"""

import sys
import json
from pathlib import Path

import pandas as pd

# 添加项目根目录到 path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from etl.field_mapper import map_app_to_standard, map_web_to_standard, validate_standard_schema
from etl.cleaner import clean_app_source, clean_web_source, merge_and_dedup
from etl.quality_report import load_injection_report, compare_quality, print_quality_summary


DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def main():
    raw_dir = DATA_DIR / "raw"

    # 1. 加载脏数据
    print("=" * 60)
    print("Step 1: 加载异构数据源")
    print("=" * 60)
    app_raw = pd.read_csv(raw_dir / "customers_app_raw.csv")
    web_raw = pd.read_csv(raw_dir / "customers_web_raw.csv")
    print(f"  App来源: {len(app_raw)}行 x {len(app_raw.columns)}列")
    print(f"  Web来源: {len(web_raw)}行 x {len(web_raw.columns)}列")

    # 2. 字段映射
    print("\n" + "=" * 60)
    print("Step 2: 字段名映射 → 标准schema")
    print("=" * 60)
    app_mapped = map_app_to_standard(app_raw)
    web_mapped = map_web_to_standard(web_raw)
    assert validate_standard_schema(app_mapped), "App端映射后列名不匹配！"
    assert validate_standard_schema(web_mapped), "Web端映射后列名不匹配！"
    print(f"  [OK] App端映射完成: {list(app_mapped.columns)}")
    print(f"  [OK] Web端映射完成: {list(web_mapped.columns)}")

    # 3. 清洗（去重 + 缺失值处理 + 格式统一）
    print("\n" + "=" * 60)
    print("Step 3: 数据清洗")
    print("=" * 60)
    app_cleaned, app_clean_report = clean_app_source(app_mapped)
    web_cleaned, web_clean_report = clean_web_source(web_mapped)

    print(f"  App: {app_clean_report['rows_before']}→{app_clean_report['rows_after']} "
          f"(-{app_clean_report['duplicates_removed']}重复)")
    for col, info in app_clean_report["filled_missing"].items():
        print(f"    {col}: {info['filled_count']}个缺失 → {info['strategy']}={info['fill_value']}")

    print(f"  Web: {web_clean_report['rows_before']}→{web_clean_report['rows_after']} "
          f"(-{web_clean_report['duplicates_removed']}重复)")
    for col, info in web_clean_report["filled_missing"].items():
        print(f"    {col}: {info['filled_count']}个缺失 → {info['strategy']}={info['fill_value']}")

    # 验证清洗后无缺失值
    assert app_cleaned.isna().sum().sum() == 0, "App端清洗后仍有缺失值！"
    assert web_cleaned.isna().sum().sum() == 0, "Web端清洗后仍有缺失值！"
    print("  [OK] 清洗后两端均无缺失值")

    # 4. 合并去重
    print("\n" + "=" * 60)
    print("Step 4: 合并App+Web → 去重重叠客户")
    print("=" * 60)
    merged, merge_report = merge_and_dedup(app_cleaned, web_cleaned)
    print(f"  合并前: App={merge_report['app_rows']} + Web={merge_report['web_rows']} "
          f"= {merge_report['rows_before_dedup']}")
    print(f"  移除重叠: {merge_report['overlap_duplicates_removed']}")
    print(f"  最终: {merge_report['rows_after_merge']} 个唯一客户")
    print(f"  [OK] 合并完成，最终 customers 表: {len(merged)}行 x {len(merged.columns)}列")

    # 5. 质量对比
    print("\n" + "=" * 60)
    print("Step 5: 治理前后质量对比")
    print("=" * 60)
    injection_report = load_injection_report(raw_dir / "injection_report.json")
    comparison = compare_quality(injection_report, app_cleaned, web_cleaned, merged)
    print_quality_summary(comparison)

    # 保存质量对比报告
    output_path = DATA_DIR / "processed"
    output_path.mkdir(exist_ok=True)
    with open(output_path / "quality_comparison.json", "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n[OK] 质量对比报告已保存: {output_path / 'quality_comparison.json'}")

    # 保存清洗后的 merged customers 作为 CSV 留档
    merged_path = output_path / "customers_cleaned.csv"
    merged.to_csv(merged_path, index=False)
    print(f"[OK] 清洗后 customers 已保存: {merged_path}")

    print("\n" + "=" * 60)
    print("ETL 全流程验证通过 ✓")
    print("=" * 60)


if __name__ == "__main__":
    main()
