"""
字段映射器 —— 将异构来源的列名统一映射回标准 schema（对齐原始 customers.csv）。

设计原则（多源异构数据集成）：
- App端和Web端各有自己的命名约定（驼峰/缩写 vs snake_case不统一）
- 映射逻辑是确定性的，不涉及LLM，写成普通字典驱动的函数
- 两套映射独立维护，方便增加新来源时扩展
"""

import pandas as pd
from typing import Dict

# ============================================================
# 标准 schema（对齐原始 customers.csv 列名）
# ============================================================
STANDARD_COLUMNS = [
    "customer_id", "country", "age", "gender", "membership_tier",
    "registration_date", "total_orders", "total_spend_usd",
    "avg_order_value_usd", "days_since_last_purchase",
    "preferred_category", "preferred_device",
    "preferred_payment_method", "acquisition_channel",
    "reviews_given", "avg_review_score", "returns_made",
    "wishlist_items", "newsletter_subscribed", "churned",
]

# ============================================================
# 来源 → 标准 schema 映射表
# ============================================================

# App端：驼峰/缩写命名 → 标准名
APP_RENAME_MAP: Dict[str, str] = {
    "customerId": "customer_id",
    "regDate": "registration_date",
    "prefDevice": "preferred_device",
    "payMethod": "preferred_payment_method",
    "channel": "acquisition_channel",
    # 其余列名保持不变
}

# Web端：snake_case不统一 → 标准名
WEB_RENAME_MAP: Dict[str, str] = {
    "cust_id": "customer_id",
    "signup_date": "registration_date",
    "fav_category": "preferred_category",
    "subscribed_flag": "newsletter_subscribed",
    # 其余列名保持不变
}


def map_app_to_standard(df: pd.DataFrame) -> pd.DataFrame:
    """将 App 端 DataFrame 列名映射到标准 schema。"""
    # 验证必需列存在
    missing_cols = [c for c in APP_RENAME_MAP if c not in df.columns]
    if missing_cols:
        raise KeyError(f"App端缺少预期列: {missing_cols}")

    df = df.rename(columns=APP_RENAME_MAP)
    # 只保留标准列
    return df[STANDARD_COLUMNS]


def map_web_to_standard(df: pd.DataFrame) -> pd.DataFrame:
    """将 Web 端 DataFrame 列名映射到标准 schema。"""
    missing_cols = [c for c in WEB_RENAME_MAP if c not in df.columns]
    if missing_cols:
        raise KeyError(f"Web端缺少预期列: {missing_cols}")

    df = df.rename(columns=WEB_RENAME_MAP)
    return df[STANDARD_COLUMNS]


def validate_standard_schema(df: pd.DataFrame) -> bool:
    """验证 DataFrame 的列是否与标准 schema 完全一致。"""
    return list(df.columns) == STANDARD_COLUMNS
