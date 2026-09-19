"""
MySQL 连接封装 —— 提供上下文管理器，自动处理事务提交/回滚。

用法:
    from storage.mysql.client import get_connection
    with get_connection() as conn:
        result = conn.execute(text("SELECT ..."))
"""

from contextlib import contextmanager
from typing import Any, Mapping
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine, Connection
from config.settings import get_settings
from agents.sql_safety import enforce_row_limit, validate_read_only_sql

_engine: Engine | None = None


def get_engine() -> Engine:
    """获取 SQLAlchemy Engine 单例。"""
    global _engine
    if _engine is None:
        settings = get_settings()
        _engine = create_engine(
            settings.mysql_url,
            pool_size=5,
            pool_pre_ping=True,
            pool_timeout=settings.MYSQL_POOL_TIMEOUT_SECONDS,
            pool_recycle=3600,
            echo=False,  # 生产环境关掉 SQL 日志
            connect_args={
                "connect_timeout": settings.MYSQL_CONNECT_TIMEOUT_SECONDS
            },
        )
    return _engine


@contextmanager
def get_connection():
    """
    获取数据库连接（上下文管理器）。
    退出时自动 commit 或 rollback。
    """
    settings = get_settings()
    engine = get_engine()
    conn = engine.connect()
    try:
        # 防止 LLM 生成的笛卡尔积或低效聚合无限占用数据库。
        # MySQL 的 MAX_EXECUTION_TIME 只约束只读 SELECT（毫秒）。
        conn.execute(text(
            f"SET SESSION MAX_EXECUTION_TIME={settings.SQL_EXECUTION_TIMEOUT_MS}"
        ))
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def execute_sql(
    sql: str, parameters: Mapping[str, Any] | None = None
) -> list[dict]:
    """
    执行一条只读 SELECT（含以 WITH 开头的 CTE）并返回 dict 列表。
    写操作或多语句会抛出 ValueError。

    Args:
        sql: SQL 语句。动态值应使用 ``:name`` 绑定参数，不应拼接到 SQL。
        parameters: 可选的 SQLAlchemy 绑定参数。

    Returns:
        [{col: val, ...}, ...]

    Raises:
        ValueError: 如果不是单条只读查询
    """
    settings = get_settings()
    sql_stripped = sql.strip()
    is_safe, reason = validate_read_only_sql(
        sql_stripped, allowed_tables=settings.SQL_ALLOWED_TABLES
    )
    if not is_safe:
        raise ValueError(f"仅允许单条只读 SELECT/CTE：{reason}")
    bounded_sql = enforce_row_limit(sql_stripped, settings.SQL_MAX_RESULT_ROWS)

    with get_connection() as conn:
        result = conn.execute(text(bounded_sql), dict(parameters or {}))
        rows = result.fetchmany(settings.SQL_MAX_RESULT_ROWS + 1)
        rows = rows[:settings.SQL_MAX_RESULT_ROWS]
        if not rows:
            return []
        columns = list(result.keys())
        return [dict(zip(columns, row)) for row in rows]


def check_connection() -> bool:
    """测试数据库连接是否正常。"""
    try:
        with get_connection() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception as e:
        print(f"[ERROR] 数据库连接失败: {e}")
        return False
