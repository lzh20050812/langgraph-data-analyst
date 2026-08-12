"""
清洗器 —— 统一格式、处理缺失值、去重（纯Python函数，非Agent）。

清洗策略（支持治理前后量化对比）：
1. 格式统一：gender 编码 / 日期格式
2. 缺失值处理：数值列中位数、类别列众数（仅 customers 表）
3. 去重：App端按 customer_id 保留首条，Web端 drop_duplicates()
4. 合并去重：两个来源合并后，按 customer_id 去重（Web端重复的ID优先保留App端数据）
"""

import pandas as pd
import numpy as np
from typing import Tuple


# ============================================================
# 格式统一
# ============================================================

def normalize_gender(df: pd.DataFrame) -> pd.DataFrame:
    """
    将各种 gender 表示统一为 Male/Female。
    - App端: M/F → Male/Female
    - Web端: male/Male/MALE → Male; female/Female/FEMALE → Female
    """
    if "gender" not in df.columns:
        return df

    gender_map = {
        # 统一所有变体到 Male
        "M": "Male", "m": "Male", "male": "Male", "MALE": "Male",
        # 统一所有变体到 Female
        "F": "Female", "f": "Female", "female": "Female", "FEMALE": "Female",
        # Other 保持不变
        "Other": "Other", "other": "Other", "OTHER": "Other",
    }
    df["gender"] = df["gender"].astype(str).str.strip().map(gender_map).fillna(df["gender"])
    unmapped = df[~df["gender"].isin(["Male", "Female", "Other"])]["gender"].unique()
    if len(unmapped) > 0:
        print(f"  [WARN] gender 仍有未识别的值: {unmapped}")
    return df


def normalize_date(df: pd.DataFrame, col: str = "registration_date") -> pd.DataFrame:
    """
    统一日期格式为 YYYY-MM-DD。
    处理三种输入格式：MM/DD/YYYY（App端）、YYYY年MM月DD日 / YYYY-MM-DD（Web端）。
    """
    if col not in df.columns:
        return df

    def parse_date(val):
        if pd.isna(val):
            return np.nan
        val_str = str(val).strip()

        # 尝试多种格式
        formats = [
            "%Y-%m-%d",       # 标准 ISO
            "%m/%d/%Y",       # App端 MM/DD/YYYY
            "%Y年%m月%d日",    # Web端中文格式
            "%Y/%m/%d",       # 备用
            "%m-%d-%Y",       # 备用
        ]
        for fmt in formats:
            try:
                return pd.to_datetime(val_str, format=fmt).strftime("%Y-%m-%d")
            except (ValueError, TypeError):
                continue
        # 如果所有格式都失败，让 pandas 自动推断
        try:
            return pd.to_datetime(val_str).strftime("%Y-%m-%d")
        except Exception:
            print(f"  [WARN] 无法解析日期: {val_str}")
            return np.nan

    df[col] = df[col].apply(parse_date)
    return df


# ============================================================
# 缺失值处理（仅 customers 表）
# ============================================================

def fill_missing_values(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """
    填充缺失值并返回填充统计。
    - 数值列：填中位数
    - 类别列：填众数
    - 返回 (df, fill_report)
    """
    fill_report = {}

    for col in df.columns:
        if col == "customer_id":
            continue

        n_missing = df[col].isna().sum()
        if n_missing == 0:
            continue

        if df[col].dtype in [np.float64, np.int64, "float64", "int64"]:
            # 数值列 → 中位数
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            fill_report[col] = {
                "strategy": "median",
                "fill_value": float(median_val),
                "filled_count": int(n_missing),
            }
        else:
            # 类别列 → 众数
            mode_val = df[col].mode()
            fill_val = mode_val.iloc[0] if len(mode_val) > 0 else "Unknown"
            df[col] = df[col].fillna(fill_val)
            fill_report[col] = {
                "strategy": "mode",
                "fill_value": str(fill_val),
                "filled_count": int(n_missing),
            }

    return df, fill_report


# ============================================================
# 去重
# ============================================================

def dedup_app(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """
    App端去重：同一 customer_id 保留第一条（模拟"近似重复"的去重）。
    返回 (去重后DataFrame, 移除行数)。
    """
    n_before = len(df)
    df = df.drop_duplicates(subset=["customer_id"], keep="first")
    n_removed = n_before - len(df)
    return df, n_removed


def dedup_web(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """
    Web端去重：完全重复的行直接删除。
    返回 (去重后DataFrame, 移除行数)。
    """
    n_before = len(df)
    df = df.drop_duplicates()
    n_removed = n_before - len(df)
    return df, n_removed


def dedup_merged(df: pd.DataFrame) -> Tuple[pd.DataFrame, int]:
    """
    合并后去重：两个来源的5%交集，按 customer_id 保留第一条。
    """
    n_before = len(df)
    df = df.drop_duplicates(subset=["customer_id"], keep="first")
    n_removed = n_before - len(df)
    return df, n_removed


# ============================================================
# 主清洗流程
# ============================================================

def clean_app_source(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """清洗 App 端数据，返回 (cleaned_df, cleaning_report)。"""
    report = {}
    n_before = len(df)

    # 1. 格式统一
    df = normalize_gender(df)
    df = normalize_date(df, col="registration_date")

    # 2. 缺失值填充
    df, fill_report = fill_missing_values(df)
    report["filled_missing"] = fill_report

    # 3. 去重
    df, n_dup = dedup_app(df)
    report["duplicates_removed"] = n_dup

    report["rows_before"] = n_before
    report["rows_after"] = len(df)
    return df, report


def clean_web_source(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """清洗 Web 端数据，返回 (cleaned_df, cleaning_report)。"""
    report = {}
    n_before = len(df)

    # 1. 格式统一
    df = normalize_gender(df)
    df = normalize_date(df, col="registration_date")

    # 2. 缺失值填充
    df, fill_report = fill_missing_values(df)
    report["filled_missing"] = fill_report

    # 3. 去重（完全重复行）
    df, n_dup = dedup_web(df)
    report["duplicates_removed"] = n_dup

    report["rows_before"] = n_before
    report["rows_after"] = len(df)
    return df, report


def merge_and_dedup(app_df: pd.DataFrame, web_df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """
    合并 App 和 Web 两个来源，处理重叠客户去重。
    App 端数据优先（keep='first'，App在前）。
    """
    report = {
        "app_rows": len(app_df),
        "web_rows": len(web_df),
    }
    merged = pd.concat([app_df, web_df], ignore_index=True)
    report["rows_before_dedup"] = len(merged)

    merged, n_dup = dedup_merged(merged)
    report["overlap_duplicates_removed"] = n_dup
    report["rows_after_merge"] = len(merged)

    return merged, report
