"""
Experiments —— 技术评测实验脚本目录。

每个实验脚本继承 evaluation.framework.ExperimentBase，
遵循统一流程：加载数据 → 执行实验 → 计算指标 → 保存结果。

已实现实验：
- text2sql_experiment.py      : Text2SQL 生成准确率与执行准确率实验
- rag_schema_experiment.py     : RAG Schema 语义检索 Recall@K + RAG vs No-RAG 对比实验
- multi_agent_experiment.py    : Multi-Agent 协同 vs Single LLM vs LLM+RAG 三方案对比 + 消融实验
"""

from pathlib import Path

EXPERIMENTS_DIR = Path(__file__).resolve().parent
