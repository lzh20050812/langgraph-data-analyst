"""
全局配置 —— 从环境变量读取，提供统一的配置入口。

所有模块通过 `from config.settings import get_settings()` 获取配置，
方便后续做"不同LLM下SQL生成准确率对比"时只需要改一处。
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# 项目根目录
BASE_DIR = Path(__file__).resolve().parent.parent

# 加载 .env 文件
load_dotenv(BASE_DIR / ".env")


class Settings:
    """全局配置单例。"""

    # ---- 项目路径 ----
    BASE_DIR: Path = BASE_DIR
    DATA_DIR: Path = BASE_DIR / "data"
    RAW_DIR: Path = DATA_DIR / "raw"
    PROCESSED_DIR: Path = DATA_DIR / "processed"

    # ---- MySQL ----
    MYSQL_HOST: str = os.getenv("MYSQL_HOST", "127.0.0.1")
    MYSQL_PORT: int = int(os.getenv("MYSQL_PORT", "3306"))
    MYSQL_USER: str = os.getenv("MYSQL_USER", "root")
    MYSQL_PASSWORD: str = os.getenv("MYSQL_PASSWORD", "analytics_dev_password")
    MYSQL_DATABASE: str = os.getenv("MYSQL_DATABASE", "ai_analytics")
    MYSQL_CONNECT_TIMEOUT_SECONDS: int = int(
        os.getenv("MYSQL_CONNECT_TIMEOUT_SECONDS", "5")
    )
    MYSQL_POOL_TIMEOUT_SECONDS: int = int(
        os.getenv("MYSQL_POOL_TIMEOUT_SECONDS", "10")
    )
    SQL_EXECUTION_TIMEOUT_MS: int = int(
        os.getenv("SQL_EXECUTION_TIMEOUT_MS", "60000")
    )
    SQL_MAX_RESULT_ROWS: int = int(os.getenv("SQL_MAX_RESULT_ROWS", "1000"))
    SQL_ALLOWED_TABLES: tuple[str, ...] = tuple(
        name.strip()
        for name in os.getenv(
            "SQL_ALLOWED_TABLES",
            "customers,orders,monthly_revenue,product_summary,support_agents,support_tickets",
        ).split(",")
        if name.strip()
    )

    @property
    def mysql_url(self) -> str:
        return (
            f"mysql+pymysql://{self.MYSQL_USER}:{self.MYSQL_PASSWORD}"
            f"@{self.MYSQL_HOST}:{self.MYSQL_PORT}/{self.MYSQL_DATABASE}"
            f"?charset=utf8mb4"
        )

    # ---- LLM (DeepSeek-V3) ----
    LLM_API_KEY: str = os.getenv("LLM_API_KEY", "")
    LLM_API_BASE: str = os.getenv("LLM_API_BASE", "https://api.deepseek.com/v1")
    LLM_MODEL: str = os.getenv("LLM_MODEL", "deepseek-chat")  # DeepSeek-V3
    LLM_FALLBACK_MODEL: str = os.getenv("LLM_FALLBACK_MODEL", "")
    LLM_TEMPERATURE: float = 0.0  # SQL生成用0温度，确保确定性
    LLM_TIMEOUT_SECONDS: float = float(os.getenv("LLM_TIMEOUT_SECONDS", "30"))
    LLM_MAX_RETRIES: int = int(os.getenv("LLM_MAX_RETRIES", "2"))
    TASK_MAX_LLM_CALLS: int = int(os.getenv("TASK_MAX_LLM_CALLS", "16"))
    TASK_MAX_CONTEXT_CHARS: int = int(os.getenv("TASK_MAX_CONTEXT_CHARS", "200000"))
    TASK_MAX_TOTAL_TOKENS: int = int(os.getenv("TASK_MAX_TOTAL_TOKENS", "100000"))
    TASK_MAX_DURATION_SECONDS: float = float(os.getenv("TASK_MAX_DURATION_SECONDS", "300"))
    TASK_MAX_ESTIMATED_COST_USD: float = float(os.getenv("TASK_MAX_ESTIMATED_COST_USD", "1"))
    LLM_INPUT_COST_PER_MILLION: float = float(os.getenv("LLM_INPUT_COST_PER_MILLION", "0"))
    LLM_OUTPUT_COST_PER_MILLION: float = float(os.getenv("LLM_OUTPUT_COST_PER_MILLION", "0"))

    # ---- API resource protection ----
    MAX_CONCURRENT_QUERIES: int = int(os.getenv("MAX_CONCURRENT_QUERIES", "4"))
    MAX_QUEUED_TASKS: int = int(os.getenv("MAX_QUEUED_TASKS", "20"))
    API_RATE_LIMIT_REQUESTS: int = int(
        os.getenv("API_RATE_LIMIT_REQUESTS", "120")
    )
    API_RATE_LIMIT_WINDOW_SECONDS: int = int(
        os.getenv("API_RATE_LIMIT_WINDOW_SECONDS", "60")
    )
    API_AUTH_ENABLED: bool = os.getenv(
        "API_AUTH_ENABLED", "0"
    ).strip().lower() in {"1", "true", "yes", "on"}
    LOCAL_ACCESS_ENABLED: bool = os.getenv(
        "LOCAL_ACCESS_ENABLED", "0"
    ).strip().lower() in {"1", "true", "yes", "on"}
    API_KEYS: tuple[str, ...] = tuple(
        key.strip() for key in os.getenv("API_KEYS", "").split(",") if key.strip()
    )
    API_PRINCIPALS_JSON: str = os.getenv("API_PRINCIPALS_JSON", "")
    WEB_LOGIN_ENABLED: bool = os.getenv(
        "WEB_LOGIN_ENABLED", "1"
    ).strip().lower() not in {"0", "false", "no", "off"}
    SESSION_COOKIE_SECURE: bool = os.getenv(
        "SESSION_COOKIE_SECURE", "0"
    ).strip().lower() in {"1", "true", "yes", "on"}
    SESSION_TTL_SECONDS: int = int(os.getenv("SESSION_TTL_SECONDS", "28800"))
    INITIAL_ADMIN_USERNAME: str = os.getenv("INITIAL_ADMIN_USERNAME", "admin")
    INITIAL_ADMIN_PASSWORD: str = os.getenv("INITIAL_ADMIN_PASSWORD", "Admin123!")
    TASK_DB_PATH: Path = Path(
        os.getenv("TASK_DB_PATH", str(BASE_DIR / "data" / "runtime" / "tasks.db"))
    )
    TASK_EVENT_POLL_SECONDS: float = float(
        os.getenv("TASK_EVENT_POLL_SECONDS", "0.25")
    )
    TASK_STALE_SECONDS: float = float(os.getenv("TASK_STALE_SECONDS", "300"))
    CLARIFICATION_TTL_SECONDS: float = float(
        os.getenv("CLARIFICATION_TTL_SECONDS", "1800")
    )
    TASK_RETENTION_SECONDS: float = float(
        os.getenv("TASK_RETENTION_SECONDS", "604800")
    )
    TASK_MAX_RECORDS: int = int(os.getenv("TASK_MAX_RECORDS", "1000"))
    AUDIT_RETENTION_SECONDS: float = float(
        os.getenv("AUDIT_RETENTION_SECONDS", "2592000")
    )
    AUDIT_MAX_RECORDS: int = int(os.getenv("AUDIT_MAX_RECORDS", "10000"))
    TASK_EXECUTION_MODE: str = os.getenv("TASK_EXECUTION_MODE", "embedded")
    TASK_WORKER_POLL_SECONDS: float = float(
        os.getenv("TASK_WORKER_POLL_SECONDS", "0.5")
    )
    TASK_WORKER_LEASE_SECONDS: float = float(
        os.getenv("TASK_WORKER_LEASE_SECONDS", "60")
    )
    TASK_WORKER_MAX_ATTEMPTS: int = int(
        os.getenv("TASK_WORKER_MAX_ATTEMPTS", "2")
    )

    # ---- Embedding (BAAI/bge-small-zh) ----
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-zh")
    CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", str(BASE_DIR / "data" / "chromadb"))

    # ---- ChromaDB ----
    CHROMA_COLLECTION: str = os.getenv("CHROMA_COLLECTION", "schema_metadata")

    # ---- SQL Agent ----
    SQL_MAX_RETRIES: int = 3  # 自修正最多重试次数
    BUSINESS_SEMANTICS_ENABLED: bool = os.getenv(
        "BUSINESS_SEMANTICS_ENABLED", "1"
    ).strip().lower() not in {"0", "false", "no", "off"}
    ANALYSIS_MEMORY_ENABLED: bool = os.getenv(
        "ANALYSIS_MEMORY_ENABLED", "1"
    ).strip().lower() not in {"0", "false", "no", "off"}
    DEMO_MODE_ENABLED: bool = os.getenv(
        "DEMO_MODE_ENABLED", "1"
    ).strip().lower() not in {"0", "false", "no", "off"}


_settings: Settings | None = None


def get_settings() -> Settings:
    """获取全局配置单例。"""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
