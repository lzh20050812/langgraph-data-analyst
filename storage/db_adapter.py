"""
数据库适配器 —— 抽象 MySQL / DuckDB 两种后端，提供统一的 execute_sql 接口。

设计目的：
- MySQL: 生产环境，Docker 容器化
- DuckDB: 测试环境，零配置，内存运行，标准 SQL 兼容
- 测试时自动降级到 DuckDB（无需启动 Docker），生产时切回 MySQL
"""

from typing import List, Dict, Any, Optional
from contextlib import contextmanager


class DBAdapter:
    """统一数据库适配器接口。"""

    def execute_sql(self, sql: str) -> List[Dict[str, Any]]:
        raise NotImplementedError

    def check_connection(self) -> bool:
        raise NotImplementedError

    def load_dataframe(self, df, table_name: str) -> int:
        raise NotImplementedError

    def close(self):
        pass


# ============================================================
# DuckDB 实现（测试用）
# ============================================================

class DuckDBAdapter(DBAdapter):
    """DuckDB 内存数据库适配器 —— 零配置，用于测试和开发。"""

    def __init__(self):
        import duckdb
        self._conn = duckdb.connect(":memory:")
        self._conn.execute("SET threads=2")

    def execute_sql(self, sql: str) -> List[Dict[str, Any]]:
        import duckdb
        try:
            result = self._conn.execute(sql)
            columns = [desc[0] for desc in result.description]
            rows = result.fetchall()
            return [dict(zip(columns, row)) for row in rows]
        except duckdb.Error as e:
            raise RuntimeError(str(e))

    def check_connection(self) -> bool:
        try:
            self._conn.execute("SELECT 1")
            return True
        except Exception:
            return False

    def load_dataframe(self, df, table_name: str) -> int:
        """将 pandas DataFrame 注册为 DuckDB 表。"""
        self._conn.register(table_name, df)
        return len(df)

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None


# ============================================================
# MySQL 实现（生产用）
# ============================================================

class MySQLAdapter(DBAdapter):
    """MySQL 数据库适配器 —— 封装 storage.mysql.client。"""

    def execute_sql(self, sql: str) -> List[Dict[str, Any]]:
        from storage.mysql.client import execute_sql
        return execute_sql(sql)

    def check_connection(self) -> bool:
        from storage.mysql.client import check_connection
        return check_connection()

    def load_dataframe(self, df, table_name: str) -> int:
        from storage.mysql.client import get_engine
        engine = get_engine()
        df.to_sql(table_name, engine, if_exists="replace", index=False)
        return len(df)


# ============================================================
# 自动选择
# ============================================================

def get_available_adapter() -> DBAdapter:
    """
    自动检测可用后端：优先 MySQL，不可用时降级到 DuckDB。
    """
    mysql = MySQLAdapter()
    if mysql.check_connection():
        print("[DBAdapter] 使用 MySQL 后端")
        return mysql

    print("[DBAdapter] MySQL 不可用，降级为 DuckDB 内存数据库")
    return DuckDBAdapter()
