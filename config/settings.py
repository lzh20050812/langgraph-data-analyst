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
    LLM_TEMPERATURE: float = 0.0  # SQL生成用0温度，确保确定性

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


_settings: Settings | None = None


def get_settings() -> Settings:
    """获取全局配置单例。"""
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
