"""
Evaluation Framework —— 技术评测实验基础框架。

本模块提供统一的实验基础设施，后续所有技术评测实验（Text2SQL 准确率、
RAG 检索效果、多智能体协同对比、Memory 机制、预测模型、系统性能测试等）
均应基于此框架开发。

公共 API：
- ExperimentBase    —— 实验基类（抽象接口）
- ExperimentResult  —— 实验结果数据结构
- ExperimentLogger  —— 实验日志记录器
- ResultManager     —— 实验结果管理器
- ExperimentConfig  —— 实验配置类
- MetricRegistry    —— 指标注册表

设计原则：
- 与现有业务代码（agents/、storage/、api/）完全解耦
- 不修改任何已有 Agent 调用方式
- 不引入额外框架依赖（仅标准库 + 项目已有依赖）
"""

from .base import ExperimentBase, ExperimentResult
from .logger import ExperimentLogger
from .result import ResultManager, save_json, save_csv, load_json, load_csv
from .config import ExperimentConfig

__all__ = [
    "ExperimentBase",
    "ExperimentResult",
    "ExperimentLogger",
    "ResultManager",
    "ExperimentConfig",
    "save_json",
    "save_csv",
    "load_json",
    "load_csv",
]
