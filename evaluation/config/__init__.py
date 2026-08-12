"""
实验配置目录 —— 存放技术评测实验的配置文件。

后续可为每个实验类别创建独立配置文件：
- config/text2sql.yaml       # Text2SQL 实验配置
- config/rag_retrieval.yaml   # RAG 检索实验配置
- config/multi_agent.yaml    # 多智能体协同实验配置
- config/prediction.yaml     # 预测模型实验配置
"""

from pathlib import Path

CONFIG_DIR = Path(__file__).resolve().parent


def get_config_path(name: str) -> Path:
    """获取指定配置文件的路径。

    Args:
        name: 配置文件名（如 "text2sql" 或 "text2sql.yaml"）

    Returns:
        配置文件的绝对路径
    """
    if not name.endswith((".yaml", ".yml", ".json")):
        name = f"{name}.yaml"
    return CONFIG_DIR / name
