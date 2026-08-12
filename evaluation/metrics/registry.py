"""
指标注册表 —— 集中管理技术评测实验所需的所有评价指标。

设计思路：
- 所有指标函数统一签名为 (y_true, y_pred, **kwargs) -> float
- 通过 MetricRegistry 按名称注册和调用
- 后续实验脚本通过 registry.compute("accuracy", y_true, y_pred) 调用

支持的指标类别：
- classification: accuracy, precision, recall, f1, auc
- regression: mse, rmse, mae, mape
- retrieval: top_k_hit_rate, precision_at_k, recall_at_k, mrr
- text: exact_match, rouge_l, bleu（预留）
- system: avg_latency, p95_latency, success_rate
"""

from typing import Any, Callable, Dict, List, Optional


# ============================================================
# 指标函数类型
# ============================================================

# 通用指标函数：接收任意位置参数和关键字参数，返回数值
MetricFn = Callable[..., float]


# ============================================================
# 指标注册表
# ============================================================

class MetricRegistry:
    """指标注册表。

    使用示例:
        registry = MetricRegistry()

        @registry.register("accuracy")
        def accuracy(y_true, y_pred):
            return sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)

        score = registry.compute("accuracy", y_true, y_pred)
    """

    def __init__(self):
        self._metrics: Dict[str, MetricFn] = {}

    def register(self, name: str, category: str = "general") -> Callable:
        """装饰器：注册一个指标函数。

        Args:
            name: 指标名称（如 "accuracy"）
            category: 指标类别（如 "classification"）
        """

        def decorator(fn: MetricFn) -> MetricFn:
            self._metrics[name] = fn
            # 附加元信息
            fn._metric_name = name
            fn._metric_category = category
            return fn

        return decorator

    def compute(self, name: str, *args, **kwargs) -> float:
        """计算指定指标。

        Args:
            name: 指标名称
            *args, **kwargs: 传递给指标函数的参数

        Returns:
            指标值

        Raises:
            KeyError: 指标未注册
        """
        if name not in self._metrics:
            raise KeyError(
                f"指标 '{name}' 未注册。可用指标: {list(self._metrics.keys())}"
            )
        return self._metrics[name](*args, **kwargs)

    def compute_all(
        self,
        metric_names: List[str],
        *args,
        **kwargs,
    ) -> Dict[str, float]:
        """批量计算多个指标。

        Args:
            metric_names: 指标名称列表
            *args, **kwargs: 共享的参数

        Returns:
            {metric_name: value} 字典
        """
        return {name: self.compute(name, *args, **kwargs) for name in metric_names}

    def list_metrics(self, category: Optional[str] = None) -> List[str]:
        """列出已注册的指标。

        Args:
            category: 按类别过滤（为 None 则不过滤）
        """
        if category is None:
            return list(self._metrics.keys())
        return [
            name
            for name, fn in self._metrics.items()
            if getattr(fn, "_metric_category", "") == category
        ]

    def __contains__(self, name: str) -> bool:
        return name in self._metrics


# ============================================================
# 全局单例
# ============================================================

_registry: Optional[MetricRegistry] = None


def get_metric_registry() -> MetricRegistry:
    """获取全局指标注册表单例。"""
    global _registry
    if _registry is None:
        _registry = MetricRegistry()
    return _registry
