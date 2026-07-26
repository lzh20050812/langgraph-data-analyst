"""
数据加载器 —— 将清洗后的数据写入 MySQL。

ETL 加载分工（对应架构文档 3.2 节）：
- customers: ETL 清洗后写入（App + Web 合并去重后的干净表）
- orders: 原样导入（customer_rating 保留 NULL，不做清洗）
- monthly_revenue: 原样导入
- product_summary: 原样导入
"""

import pandas as pd
from sqlalchemy import create_engine, text
from typing import Optional


def load_csv_to_mysql(
    csv_path: str,
    table_name: str,
    engine,
    if_exists: str = "replace",
    dtype: Optional[dict] = None,
) -> int:
    """
    将 CSV 文件直接导入 MySQL 表。

    Args:
        csv_path: CSV 文件路径
        table_name: 目标表名
        engine: SQLAlchemy engine
        if_exists: 'replace' | 'append' | 'fail'
        dtype: SQL 列类型映射（可选）

    Returns:
        导入的行数
    """
    df = pd.read_csv(csv_path)
    df.to_sql(table_name, engine, if_exists=if_exists, index=False, dtype=dtype)
    return len(df)


def load_dataframe_to_mysql(
    df: pd.DataFrame,
    table_name: str,
    engine,
    if_exists: str = "replace",
    dtype: Optional[dict] = None,
) -> int:
    """将 DataFrame 写入 MySQL 表。"""
    df.to_sql(table_name, engine, if_exists=if_exists, index=False, dtype=dtype)
    return len(df)


def create_daily_orders_view(engine) -> None:
    """
    在 orders 表所在库创建日粒度聚合视图（供 Prediction Agent 直接查询）。
    避免每次都对 25000 行订单做全表扫描。
    """
    ddl = """
    CREATE OR REPLACE VIEW daily_orders_agg AS
    SELECT
        order_date,
        year,
        month,
        COUNT(*)                          AS order_count,
        SUM(subtotal_usd)                 AS revenue_usd,
        AVG(subtotal_usd)                 AS avg_order_value,
        AVG(discount_pct)                 AS avg_discount_pct,
        COUNT(DISTINCT customer_id)       AS unique_customers,
        SUM(CASE WHEN is_repeat_customer = 1 THEN 1 ELSE 0 END) AS repeat_orders,
        AVG(delivery_days)                AS avg_delivery_days,
        AVG(session_duration_minutes)     AS avg_session_minutes,
        AVG(pages_viewed_before_purchase) AS avg_pages_viewed,
        AVG(customer_rating)              AS avg_customer_rating,
        COUNT(CASE WHEN returned = 1 THEN 1 END) AS return_count
    FROM orders
    GROUP BY order_date, year, month
    ORDER BY order_date
    """
    with engine.connect() as conn:
        conn.execute(text(ddl))
        conn.commit()


def create_monthly_orders_view(engine) -> None:
    """月粒度聚合视图（供 Prophet 预测训练使用）。"""
    ddl = """
    CREATE OR REPLACE VIEW monthly_orders_agg AS
    SELECT
        year,
        month,
        COUNT(*)                          AS order_count,
        SUM(subtotal_usd)                 AS revenue_usd,
        AVG(subtotal_usd)                 AS avg_order_value,
        AVG(discount_pct)                 AS avg_discount_pct,
        COUNT(DISTINCT customer_id)       AS unique_customers,
        SUM(CASE WHEN is_repeat_customer = 1 THEN 1 ELSE 0 END) AS repeat_orders,
        AVG(customer_rating)              AS avg_customer_rating,
        COUNT(CASE WHEN returned = 1 THEN 1 END) AS return_count
    FROM orders
    GROUP BY year, month
    ORDER BY year, month
    """
    with engine.connect() as conn:
        conn.execute(text(ddl))
        conn.commit()
