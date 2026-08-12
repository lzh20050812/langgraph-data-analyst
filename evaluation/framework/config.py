"""
实验配置类 —— 管理技术评测实验的通用配置项。

所有实验均可通过 ExperimentConfig 统一管理配置，支持：
- 从字典加载
- 从 YAML/JSON 文件加载（可选）
- 默认值覆盖
- 属性风格访问（config.llm_model）

设计约束：
- 不依赖任何已有业务配置（config/settings.py）
- 实验配置独立于系统配置，避免冲突
"""

import json
from pathlib import Path
from typing import Any, Dict, Optional


class ExperimentConfig:
    """技术评测实验通用配置。

    配置项按类别分组：
    - experiment: 实验元信息
    - llm: LLM 相关配置
    - data: 测试数据路径
    - output: 输出路径
    - limits: 运行限制（超时、重试次数等）
    - extended: 扩展配置（自由键值，供特定实验使用）

    使用示例:
        config = ExperimentConfig({
            "experiment": {"name": "text2sql", "version": "1.0"},
            "llm": {"model": "deepseek-chat", "temperature": 0.0},
            "limits": {"max_retries": 3, "timeout_seconds": 300},
        })
        print(config.llm_model)  # "deepseek-chat"
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Args:
            config: 配置字典，支持嵌套结构
        """
        self._data: Dict[str, Any] = {}

        # 设置默认值
        self._data.update(self._defaults())

        # 合并用户配置
        if config:
            self._merge(config)

    @staticmethod
    def _defaults() -> Dict[str, Any]:
        """默认配置值。"""
        base_dir = Path(__file__).resolve().parent.parent
        return {
            "experiment": {
                "name": "",
                "description": "",
                "version": "1.0",
                "random_seed": 42,
            },
            "llm": {
                "model": "deepseek-chat",
                "temperature": 0.0,
                "max_tokens": 2048,
            },
            "data": {
                "test_data_dir": str(base_dir / "data"),
                "schema_test_file": "",
                "sql_test_file": "",
            },
            "output": {
                "results_dir": str(base_dir / "results" / "raw"),
                "tables_dir": str(base_dir / "results" / "tables"),
                "logs_dir": str(base_dir / "logs"),
            },
            "limits": {
                "max_retries": 3,
                "timeout_seconds": 600,
                "max_samples": 0,  # 0 表示不限制
            },
            "extended": {},
        }

    def _merge(self, config: Dict[str, Any]) -> None:
        """深度合并配置（仅覆盖已有 key，不删除未提供的 key）。"""
        for key, value in config.items():
            if key in self._data and isinstance(self._data[key], dict) and isinstance(value, dict):
                self._data[key].update(value)
            else:
                self._data[key] = value

    # ---- 便捷属性访问 ----

    @property
    def experiment_name(self) -> str:
        return self._data["experiment"]["name"]

    @property
    def experiment_description(self) -> str:
        return self._data["experiment"]["description"]

    @property
    def random_seed(self) -> int:
        return self._data["experiment"]["random_seed"]

    @property
    def llm_model(self) -> str:
        return self._data["llm"]["model"]

    @property
    def llm_temperature(self) -> float:
        return self._data["llm"]["temperature"]

    @property
    def max_retries(self) -> int:
        return self._data["limits"]["max_retries"]

    @property
    def timeout_seconds(self) -> int:
        return self._data["limits"]["timeout_seconds"]

    # ---- 字典风格访问 ----

    def get(self, key: str, default: Any = None) -> Any:
        """获取顶级配置项。支持点号分隔的嵌套 key（如 "llm.model"）。"""
        keys = key.split(".")
        value = self._data
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
        return value if value is not None else default

    def set(self, key: str, value: Any) -> None:
        """设置顶级或嵌套配置项。"""
        keys = key.split(".")
        target = self._data
        for k in keys[:-1]:
            if k not in target:
                target[k] = {}
            target = target[k]
        target[keys[-1]] = value

    def to_dict(self) -> Dict[str, Any]:
        """返回完整的配置字典。"""
        return dict(self._data)

    # ---- 文件加载 ----

    @classmethod
    def from_json(cls, path: Path) -> "ExperimentConfig":
        """从 JSON 文件加载配置。"""
        with open(path, "r", encoding="utf-8") as f:
            return cls(json.load(f))

    def to_json(self, path: Path) -> Path:
        """将当前配置保存为 JSON 文件。"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self._data, f, indent=2, ensure_ascii=False)
        return path

    # ---- 特殊方法 ----

    def __repr__(self) -> str:
        return f"ExperimentConfig({self._data['experiment']['name']!r})"

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __contains__(self, key: str) -> bool:
        return key in self._data
