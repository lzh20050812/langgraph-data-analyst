"""
存储层初始化 —— 一键执行：建表 + 导入CSV + 构建聚合视图。

用法：
    python -m storage.init_storage          # 完整初始化（含ETL清洗）
    python -m storage.init_storage --skip-etl  # 跳过ETL，直接从processed/导入
"""

import sys
import argparse
from pathlib import Path

import pandas as pd
from sqlalchemy import text

# 添加项目根目录
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import get_settings
from storage.mysql.client import get_engine, get_connection, check_connection
from etl.field_mapper import map_app_to_standard, map_web_to_standard, validate_standard_schema
from etl.cleaner import clean_app_source, clean_web_source, merge_and_dedup
from etl.loader import (
    load_csv_to_mysql,
    load_dataframe_to_mysql,
    create_daily_orders_view,
    create_monthly_orders_view,
)


def init_customers(settings, skip_etl: bool = False) -> int:
    """初始化 customers 表。如果 skip_etl，直接读取已清洗的 CSV。"""
    engine = get_engine()

    if skip_etl:
        cleaned_path = settings.PROCESSED_DIR / "customers_cleaned.csv"
        if not cleaned_path.exists():
            print(f"[ERROR] 找不到已清洗文件: {cleaned_path}，请先运行 etl.run_etl")
            return 0
        df = pd.read_csv(cleaned_path)
        rows = load_dataframe_to_mysql(df, "customers", engine)
        print(f"  [OK] customers 表从 {cleaned_path.name} 导入: {rows} 行")
        return rows

    # 完整 ETL 流程
    raw_dir = settings.RAW_DIR
    app_raw = pd.read_csv(raw_dir / "customers_app_raw.csv")
    web_raw = pd.read_csv(raw_dir / "customers_web_raw.csv")

    app_mapped = map_app_to_standard(app_raw)
    web_mapped = map_web_to_standard(web_raw)
    assert validate_standard_schema(app_mapped)
    assert validate_standard_schema(web_mapped)

    app_cleaned, _ = clean_app_source(app_mapped)
    web_cleaned, _ = clean_web_source(web_mapped)
    merged, _ = merge_and_dedup(app_cleaned, web_cleaned)

    rows = load_dataframe_to_mysql(merged, "customers", engine)
    print(f"  [OK] customers 表（ETL清洗后）: {rows} 行")
    return rows


def init_orders(settings) -> int:
    """导入 orders 表（原样，保留 customer_rating NULL）。"""
    engine = get_engine()
    csv_path = settings.RAW_DIR / "orders.csv"
    rows = load_csv_to_mysql(str(csv_path), "orders", engine)
    print(f"  [OK] orders 表: {rows} 行")
    return rows


def init_monthly_revenue(settings) -> int:
    """导入 monthly_revenue 表。"""
    engine = get_engine()
    csv_path = settings.RAW_DIR / "monthly_revenue.csv"
    rows = load_csv_to_mysql(str(csv_path), "monthly_revenue", engine)
    print(f"  [OK] monthly_revenue 表: {rows} 行")
    return rows


def init_product_summary(settings) -> int:
    """导入 product_summary 表。"""
    engine = get_engine()
    csv_path = settings.RAW_DIR / "product_summary.csv"
    rows = load_csv_to_mysql(str(csv_path), "product_summary", engine)
    print(f"  [OK] product_summary 表: {rows} 行")
    return rows


def create_views() -> None:
    """创建聚合视图。"""
    engine = get_engine()
    create_daily_orders_view(engine)
    print("  [OK] daily_orders_agg 视图")
    create_monthly_orders_view(engine)
    print("  [OK] monthly_orders_agg 视图")


def print_table_stats() -> None:
    """打印各表的统计信息。"""
    engine = get_engine()
    tables = ["customers", "orders", "monthly_revenue", "product_summary"]
    print("\n" + "=" * 50)
    print("数据库表统计")
    print("=" * 50)
    with get_connection() as conn:
        for t in tables:
            result = conn.execute(text(f"SELECT COUNT(*) AS cnt FROM {t}"))
            cnt = result.fetchone()[0]
            print(f"  {t}: {cnt} 行")
        # 检查 orders 中 customer_rating 缺失比例
        result = conn.execute(text(
            "SELECT COUNT(*) AS total, "
            "SUM(CASE WHEN customer_rating IS NULL THEN 1 ELSE 0 END) AS null_count "
            "FROM orders"
        ))
        row = result.fetchone()
        total, null_count = row[0], row[1]
        print(f"  orders.customer_rating 缺失: {null_count}/{total} "
              f"({null_count/total*100:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="初始化存储层")
    parser.add_argument("--skip-etl", action="store_true", help="跳过ETL清洗，从processed导入")
    args = parser.parse_args()

    settings = get_settings()
    print("=" * 60)
    print("存储层初始化")
    print(f"  MySQL: {settings.MYSQL_HOST}:{settings.MYSQL_PORT}/{settings.MYSQL_DATABASE}")
    print("=" * 60)

    # 检查连接
    if not check_connection():
        print("\n[FATAL] 无法连接数据库，请确认 MySQL 已启动。")
        print("  启动方式: docker-compose -f docker/docker-compose.yml up -d")
        sys.exit(1)

    print("\n[1/5] 导入 customers 表...")
    init_customers(settings, skip_etl=args.skip_etl)

    print("\n[2/5] 导入 orders 表...")
    init_orders(settings)

    print("\n[3/5] 导入 monthly_revenue 表...")
    init_monthly_revenue(settings)

    print("\n[4/5] 导入 product_summary 表...")
    init_product_summary(settings)

    print("\n[5/5] 创建聚合视图...")
    create_views()

    print_table_stats()
    print("\n[OK] 存储层初始化完成！")


if __name__ == "__main__":
    main()
