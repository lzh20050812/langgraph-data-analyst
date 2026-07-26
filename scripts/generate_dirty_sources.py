"""
脏数据生成脚本 —— 从干净的 customers.csv 生成两个模拟异构来源。

设计目标（对应论文"数据质量问题描述"章节）：
1. customers_app_raw.csv  —— 模拟App端导出（驼峰/缩写命名，约60%客户）
2. customers_web_raw.csv  —— 模拟Web端导出（snake_case不统一，约40%客户）
3. injection_report.json  —— 每列缺失/重复注入统计，供ETL治理效果对比

可重复运行：random_seed=42，所有随机操作可复现。
"""

import pandas as pd
import numpy as np
import json
import os
from pathlib import Path

# ============================================================
# 配置
# ============================================================
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = BASE_DIR / "data" / "raw"
INPUT_CSV = RAW_DIR / "customers.csv"
OUTPUT_APP = RAW_DIR / "customers_app_raw.csv"
OUTPUT_WEB = RAW_DIR / "customers_web_raw.csv"
OUTPUT_REPORT = RAW_DIR / "injection_report.json"

# App端约60%客户，Web端约40%客户，交集约5%
APP_FRAC = 0.60
WEB_FRAC = 0.40
OVERLAP_FRAC = 0.05  # 占全体客户的5%同时在两端出现

# App端注入参数
APP_MISSING_COLS = ["age", "avg_review_score"]
APP_MISSING_RATE = 0.06   # 各列6%缺失
APP_DUP_RATE = 0.02       # 2%行重复

# Web端注入参数
WEB_MISSING_COLS = ["country", "preferred_device"]
WEB_MISSING_RATE = 0.08   # 各列8%缺失
WEB_DUP_RATE = 0.03       # 3%行重复


# ============================================================
# 辅助函数
# ============================================================

def load_clean_data(path: Path) -> pd.DataFrame:
    """加载干净的 customers.csv"""
    df = pd.read_csv(path)
    print(f"[OK] 加载 {len(df)} 行, {len(df.columns)} 列 from {path.name}")
    return df


def split_customers(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, set]:
    """
    拆分 customers 为 App 和 Web 两个子集。
    返回 (app_df, web_df, overlap_ids)
    """
    all_ids = df["customer_id"].to_numpy().copy()
    np.random.shuffle(all_ids)
    n_total = len(all_ids)
    n_overlap = int(n_total * OVERLAP_FRAC)

    # 重叠ID（约5%）
    overlap_ids = set(all_ids[:n_overlap])

    # 剩余ID分配给App和Web（App占60%，Web占40%）
    remaining = all_ids[n_overlap:]
    n_app = int(n_total * APP_FRAC) - n_overlap
    n_web = int(n_total * WEB_FRAC) - n_overlap

    app_ids = set(remaining[:n_app]) | overlap_ids
    web_ids = set(remaining[n_app:n_app + n_web]) | overlap_ids

    app_df = df[df["customer_id"].isin(app_ids)].copy()
    web_df = df[df["customer_id"].isin(web_ids)].copy()

    actual_overlap = len(set(app_df["customer_id"]) & set(web_df["customer_id"]))
    print(f"[OK] 拆分: App={len(app_df)}行, Web={len(web_df)}行, 重叠={actual_overlap}行")
    return app_df, web_df, overlap_ids


# ============================================================
# App端处理
# ============================================================

def process_app_source(df: pd.DataFrame) -> dict:
    """
    对App端数据进行污染处理，返回 (处理后的DataFrame, 注入统计dict)。
    """
    report = {"source": "customers_app_raw.csv", "total_rows_before": len(df), "injections": {}}

    # 1. 列重命名
    rename_map = {
        "customer_id": "customerId",
        "registration_date": "regDate",
        "preferred_device": "prefDevice",
        "preferred_payment_method": "payMethod",
        "acquisition_channel": "channel",
    }
    df = df.rename(columns=rename_map)
    print(f"  [App] 列重命名: {rename_map}")

    # 2. gender 编码: Male→M, Female→F, Other→Other
    if "gender" in df.columns:
        df["gender"] = df["gender"].map({"Male": "M", "Female": "F", "Other": "Other"})
        print("  [App] gender 编码: Male→M, Female→F, Other→Other")

    # 3. regDate 日期格式: YYYY-MM-DD → MM/DD/YYYY
    if "regDate" in df.columns:
        df["regDate"] = pd.to_datetime(df["regDate"]).dt.strftime("%m/%d/%Y")
        print("  [App] regDate 格式: MM/DD/YYYY")

    # 4. 注入缺失值: age, avg_review_score 各约6%
    missing_report = {}
    for col in APP_MISSING_COLS:
        if col in df.columns:
            n_total = len(df)
            n_missing = int(n_total * APP_MISSING_RATE)
            mask_idx = np.random.choice(df.index, size=n_missing, replace=False)
            df.loc[mask_idx, col] = np.nan
            missing_report[col] = {
                "missing_count": n_missing,
                "missing_rate": round(n_missing / n_total, 4),
            }
            print(f"  [App] {col}: 注入 {n_missing} 个缺失值 ({APP_MISSING_RATE*100:.0f}%)")
    report["injections"]["missing"] = missing_report

    # 5. 注入近似重复行: 约2%行（同一customerId出现两次，total_spend_usd差几分钱）
    n_dup = int(len(df) * APP_DUP_RATE)
    dup_indices = np.random.choice(df.index, size=n_dup, replace=False)
    dup_rows = df.loc[dup_indices].copy()
    # 微调 total_spend_usd（差 ±0.01~0.10）
    if "total_spend_usd" in dup_rows.columns:
        jitter = np.random.uniform(-0.10, 0.10, size=n_dup).round(2)
        # 确保不为0的调整
        jitter[jitter == 0] = 0.01
        dup_rows["total_spend_usd"] = dup_rows["total_spend_usd"] + jitter
    df = pd.concat([df, dup_rows], ignore_index=True)
    report["injections"]["duplicates"] = {
        "duplicate_rows_added": n_dup,
        "duplicate_rate": round(n_dup / len(df), 4),
        "type": "近似重复（同一customerId，total_spend_usd微调）",
    }
    print(f"  [App] 注入 {n_dup} 条近似重复行 ({APP_DUP_RATE*100:.0f}%)")

    report["total_rows_after"] = len(df)
    return df, report


# ============================================================
# Web端处理
# ============================================================

def process_web_source(df: pd.DataFrame) -> dict:
    """
    对Web端数据进行污染处理，返回 (处理后的DataFrame, 注入统计dict)。
    """
    report = {"source": "customers_web_raw.csv", "total_rows_before": len(df), "injections": {}}

    # 1. 列重命名
    rename_map = {
        "customer_id": "cust_id",
        "registration_date": "signup_date",
        "preferred_category": "fav_category",
        "newsletter_subscribed": "subscribed_flag",
    }
    df = df.rename(columns=rename_map)
    print(f"  [Web] 列重命名: {rename_map}")

    # 2. gender 大小写不统一
    if "gender" in df.columns:
        def random_case_gender(val):
            if val == "Male":
                return np.random.choice(["male", "Male", "MALE"])
            elif val == "Female":
                return np.random.choice(["female", "Female", "FEMALE"])
            return val
        df["gender"] = df["gender"].apply(random_case_gender)
        print("  [Web] gender 大小写随机化: male/Male/MALE, female/Female/FEMALE")

    # 3. signup_date 日期格式混用
    if "signup_date" in df.columns:
        def random_date_format(val):
            dt = pd.to_datetime(val)
            if np.random.random() < 0.5:
                return dt.strftime("%Y年%m月%d日")
            else:
                return dt.strftime("%Y-%m-%d")
        df["signup_date"] = df["signup_date"].apply(random_date_format)
        print("  [Web] signup_date 格式混用: YYYY年MM月DD日 / YYYY-MM-DD")

    # 4. 注入缺失值: country, preferred_device 各约8%
    missing_report = {}
    for col in WEB_MISSING_COLS:
        if col in df.columns:
            n_total = len(df)
            n_missing = int(n_total * WEB_MISSING_RATE)
            mask_idx = np.random.choice(df.index, size=n_missing, replace=False)
            df.loc[mask_idx, col] = np.nan
            missing_report[col] = {
                "missing_count": n_missing,
                "missing_rate": round(n_missing / n_total, 4),
            }
            print(f"  [Web] {col}: 注入 {n_missing} 个缺失值 ({WEB_MISSING_RATE*100:.0f}%)")
    report["injections"]["missing"] = missing_report

    # 5. 注入完全重复行: 约3%
    n_dup = int(len(df) * WEB_DUP_RATE)
    dup_indices = np.random.choice(df.index, size=n_dup, replace=False)
    dup_rows = df.loc[dup_indices].copy()
    df = pd.concat([df, dup_rows], ignore_index=True)
    report["injections"]["duplicates"] = {
        "duplicate_rows_added": n_dup,
        "duplicate_rate": round(n_dup / len(df), 4),
        "type": "完全重复（所有字段值相同）",
    }
    print(f"  [Web] 注入 {n_dup} 条完全重复行 ({WEB_DUP_RATE*100:.0f}%)")

    report["total_rows_after"] = len(df)
    return df, report


# ============================================================
# 主流程
# ============================================================

def main():
    print("=" * 60)
    print("脏数据生成脚本 —— 模拟异构数据源")
    print(f"随机种子: {RANDOM_SEED}")
    print("=" * 60)

    # 加载
    df_clean = load_clean_data(INPUT_CSV)

    # 拆分
    app_df, web_df, overlap_ids = split_customers(df_clean)

    # App端处理
    print("\n--- App端处理 ---")
    app_df, app_report = process_app_source(app_df)

    # Web端处理
    print("\n--- Web端处理 ---")
    web_df, web_report = process_web_source(web_df)

    # 保存
    app_df.to_csv(OUTPUT_APP, index=False)
    print(f"\n[OK] 保存 App端: {OUTPUT_APP} ({len(app_df)}行 x {len(app_df.columns)}列)")

    web_df.to_csv(OUTPUT_WEB, index=False)
    print(f"[OK] 保存 Web端: {OUTPUT_WEB} ({len(web_df)}行 x {len(web_df.columns)}列)")

    # 汇总报告
    full_report = {
        "random_seed": RANDOM_SEED,
        "original_file": str(INPUT_CSV.name),
        "original_rows": len(df_clean),
        "original_columns": len(df_clean.columns),
        "overlap_customer_count": len(overlap_ids),
        "overlap_ratio": round(len(overlap_ids) / len(df_clean), 4),
        "sources": [app_report, web_report],
        "summary": {
            "app": {
                "file": "customers_app_raw.csv",
                "rows": len(app_df),
                "columns": len(app_df.columns),
                "missing_cols": APP_MISSING_COLS,
                "missing_rate": APP_MISSING_RATE,
                "dup_rate": APP_DUP_RATE,
            },
            "web": {
                "file": "customers_web_raw.csv",
                "rows": len(web_df),
                "columns": len(web_df.columns),
                "missing_cols": WEB_MISSING_COLS,
                "missing_rate": WEB_MISSING_RATE,
                "dup_rate": WEB_DUP_RATE,
            },
        },
    }

    with open(OUTPUT_REPORT, "w", encoding="utf-8") as f:
        json.dump(full_report, f, indent=2, ensure_ascii=False)
    print(f"[OK] 注入报告: {OUTPUT_REPORT}")

    print("\n" + "=" * 60)
    print("完成！生成文件：")
    print(f"  1. {OUTPUT_APP}")
    print(f"  2. {OUTPUT_WEB}")
    print(f"  3. {OUTPUT_REPORT}")
    print("=" * 60)


if __name__ == "__main__":
    main()
