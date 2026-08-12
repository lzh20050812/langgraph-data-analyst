"""
MySQL 连接封装 —— 提供上下文管理器，自动处理事务提交/回滚。

用法:
    from storage.mysql.client import get_connection
    with get_connection() as conn:
        result = conn.execute(text("SELECT ..."))
"""

from contextlib import contextmanager
import re
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine, Connection
from config.settings import get_settings

_engine: Engine | None = None


def get_engine() -> Engine:
    """获取 SQLAlchemy Engine 单例。"""
    global _engine
    if _engine is None:
        settings = get_settings()
        _engine = create_engine(
            settings.mysql_url,
            pool_size=5,
            pool_recycle=3600,
            echo=False,  # 生产环境关掉 SQL 日志
            connect_args={"connect_timeout": 5},  # 连接超时5秒
        )
    return _engine


@contextmanager
def get_connection():
    """
    获取数据库连接（上下文管理器）。
    退出时自动 commit 或 rollback。
    """
    engine = get_engine()
    conn = engine.connect()
    try:
        # 防止 LLM 生成的笛卡尔积或低效聚合无限占用数据库。
        # MySQL 的 MAX_EXECUTION_TIME 只约束只读 SELECT（毫秒）。
        conn.execute(text("SET SESSION MAX_EXECUTION_TIME=60000"))
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def execute_sql(sql: str) -> list[dict]:
    """
    执行一条只读 SELECT（含以 WITH 开头的 CTE）并返回 dict 列表。
    写操作或多语句会抛出 ValueError。

    Args:
        sql: SQL 语句

    Returns:
        [{col: val, ...}, ...]

    Raises:
        ValueError: 如果不是单条只读查询
    """
    sql_stripped = sql.strip()
    normalized = sql_stripped.rstrip(";").strip()
    # CTE 同样是只读查询的常见写法；同时拒绝写操作和多语句，避免把
    # “允许 WITH”扩大成允许 WITH ... UPDATE/DELETE。
    starts_read_only = bool(re.match(r"^(SELECT|WITH)\b", normalized, re.IGNORECASE))
    has_write_keyword = bool(re.search(
        r"\b(INSERT|UPDATE|DELETE|DROP|ALTER|TRUNCATE|CREATE|REPLACE|GRANT|REVOKE|CALL|LOAD)\b",
        normalized,
        re.IGNORECASE,
    ))
    has_multiple_statements = ";" in normalized
    if not starts_read_only or has_write_keyword or has_multiple_statements:
        raise ValueError(f"仅允许单条只读 SELECT/CTE，收到: {sql_stripped[:50]}...")

    with get_connection() as conn:
        result = conn.execute(text(sql_stripped))
        rows = result.fetchall()
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
