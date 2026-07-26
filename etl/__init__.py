# ETL工程层
# 纯Python函数，不涉及LLM。
# 职责：将 customers_app_raw.csv + customers_web_raw.csv → 统一干净的 customers 表
# 注意：ETL 只清洗 customers 表，orders/monthly_revenue/product_summary 原样导入。
