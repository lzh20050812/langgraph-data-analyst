"""
评估指标模块 —— 技术评测实验所需的评价指标计算。

本模块提供：
- MetricRegistry: 指标注册表，统一管理所有指标函数
- 后续在此目录下按指标类别组织文件：
  - metrics/classification.py   (accuracy, precision, recall, f1, auc)
  - metrics/regression.py       (mse, rmse, mae, mape)
  - metrics/retrieval.py        (top_k_hit_rate, precision_at_k, recall_at_k, mrr)
  - metrics/text_quality.py     (exact_match, rouge_l, bleu)
  - metrics/system.py           (avg_latency, p95_latency, success_rate)
"""

from .registry import MetricRegistry, get_metric_registry

__all__ = ["MetricRegistry", "get_metric_registry"]
