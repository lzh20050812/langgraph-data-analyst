"""
MySQL 连接封装 —— 提供上下文管理器，自动处理事务提交/回滚。

用法:
    from storage.mysql.client import get_connection
    with get_connection() as conn:
        result = conn.execute(text("SELECT ..."))
"""

from contextlib import contextmanager
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
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def execute_sql(sql: str) -> list[dict]:
    """
    执行一条 SELECT 语句，返回 dict 列表。
    非 SELECT 语句会抛出 ValueError。

    Args:
        sql: SQL 语句

    Returns:
        [{col: val, ...}, ...]

    Raises:
        ValueError: 如果不是 SELECT 语句
    """
    sql_stripped = sql.strip()
    if not sql_stripped.upper().startswith("SELECT"):
        raise ValueError(f"仅允许 SELECT 语句，收到: {sql_stripped[:50]}...")

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
