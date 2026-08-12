"""
实验数据模块 —— 存放技术评测实验的测试数据集。

目录结构规划：
- data/text2sql/      # Text2SQL 实验测试集（自然语言查询 + 期望SQL + 期望结果）
- data/schema/         # Schema 检索实验测试集
- data/prediction/     # 预测模型实验数据
- data/report/         # 报告质量评估测试集

数据格式规范请参阅 data/README.md。

工具函数：
- get_data_path(name): 获取指定测试数据文件的路径
- list_datasets(): 列出所有可用的测试数据集
"""

from pathlib import Path
from typing import List, Optional

_DATA_DIR = Path(__file__).resolve().parent


def get_data_path(name: str) -> Path:
    """获取指定测试数据文件的路径。

    Args:
        name: 数据文件名（如 "text2sql/test_easy.json"）

    Returns:
        文件的绝对路径
    """
    return _DATA_DIR / name


def list_datasets(category: Optional[str] = None) -> List[str]:
    """列出所有可用的测试数据集。

    Args:
        category: 按子目录过滤（如 "text2sql"），None 表示全部

    Returns:
        相对路径列表
    """
    datasets = []
    search_dir = _DATA_DIR / category if category else _DATA_DIR
    if search_dir.exists():
        for f in sorted(search_dir.rglob("*.json")):
            datasets.append(str(f.relative_to(_DATA_DIR)))
        for f in sorted(search_dir.rglob("*.csv")):
            datasets.append(str(f.relative_to(_DATA_DIR)))
    return datasets
