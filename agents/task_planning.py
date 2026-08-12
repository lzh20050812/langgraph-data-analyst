"""Dependency-light task planning helpers.

The LangGraph planner uses these helpers to turn a coarse intent into an
explicit execution contract.  Keeping this module free of LangGraph and LLM
dependencies makes the planning rules easy to test and audit.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List


def _contains_any(text: str, keywords: tuple[str, ...]) -> bool:
    return any(keyword in text for keyword in keywords)


def _append_unique(items: List[str], value: str) -> None:
    if value not in items:
        items.append(value)


def build_task_plan(user_query: str, intent: str) -> Dict[str, Any]:
    """Build a deterministic, inspectable execution plan for one request."""
    query = user_query.lower().strip()
    metrics: List[str] = []
    dimensions: List[str] = []
    analysis_tools: List[str] = []
    prediction_tools: List[str] = []
    filters: Dict[str, Any] = {}
    prediction_horizon_months = None

    metric_keywords = {
        "revenue": ("营收", "销售额", "收入", "gmv", "revenue"),
        "return_rate": ("退货率", "退货", "return"),
        "customer_count": ("客户数", "用户数", "人数", "数量"),
        "avg_order_value": ("客单价", "平均订单", "aov"),
        "repeat_purchase_rate": ("复购", "repeat"),
        "churn_rate": ("流失率", "流失客户", "churn"),
    }
    for metric, keywords in metric_keywords.items():
        if _contains_any(query, keywords):
            metrics.append(metric)

    dimension_keywords = {
        "category": ("品类", "类别", "商品分类"),
        "country": ("国家", "地区", "区域", "中国区"),
        "membership_tier": ("会员等级", "会员层级"),
        "acquisition_channel": ("获客渠道", "渠道"),
        "time": ("年度", "季度", "月份", "每月", "趋势"),
        "device": ("设备", "终端"),
        "customer": ("客户", "用户"),
    }
    for dimension, keywords in dimension_keywords.items():
        if _contains_any(query, keywords):
            dimensions.append(dimension)

    years = sorted(set(re.findall(r"(?<!\d)(20\d{2})(?!\d)", query)))
    if years:
        filters["years"] = [int(year) for year in years]

    month_horizon = re.search(r"(?:未来|预测)?\s*(\d+)\s*个?月", query)
    if month_horizon:
        prediction_horizon_months = int(month_horizon.group(1))
    elif _contains_any(query, ("下一年", "未来一年", "未来1年")):
        prediction_horizon_months = 12
    elif _contains_any(query, ("下个季度", "下一季度", "未来一个季度")):
        prediction_horizon_months = 3

    wants_rfm = _contains_any(query, ("rfm", "客户价值", "客户分层"))
    wants_kmeans = _contains_any(query, ("k-means", "kmeans", "聚类", "分群"))
    wants_operational_metrics = _contains_any(
        query, ("运营指标", "综合经营", "核心指标", "经营概览")
    )

    if wants_rfm:
        analysis_tools.append("rfm")
    if wants_kmeans:
        analysis_tools.append("kmeans")
    if wants_operational_metrics:
        analysis_tools.append("operational_metrics")

    need_report = intent == "mixed" or _contains_any(
        query, ("报告", "report", "经营分析", "策略建议", "方案")
    )
    if intent in ("analysis", "mixed") or need_report:
        _append_unique(analysis_tools, "query_summary")

    prediction_requested = intent in ("prediction", "mixed") and _contains_any(
        query, ("预测", "未来", "forecast", "prophet", "xgboost", "流失")
    )
    if prediction_requested and _contains_any(query, ("流失", "churn", "xgboost")):
        prediction_tools.append("churn_prediction")
    if prediction_requested and _contains_any(
        query, ("销售", "营收", "收入", "趋势", "未来", "prophet", "forecast")
    ):
        prediction_tools.append("sales_forecast")
    if intent == "prediction" and not prediction_tools:
        prediction_tools.append("sales_forecast")

    return {
        "intent": intent,
        "metrics": metrics,
        "dimensions": dimensions,
        "filters": filters,
        "analysis_tools": analysis_tools,
        "prediction_tools": prediction_tools,
        "prediction_horizon_months": prediction_horizon_months,
        "need_report": need_report,
    }


def planned_nodes(task_plan: Dict[str, Any]) -> List[str]:
    """Return the expected execution nodes for trace completeness checks."""
    nodes = ["Planner", "Schema Agent", "SQL Agent", "Governance Agent"]
    if task_plan.get("analysis_tools"):
        nodes.append("Analysis Agent")
    if task_plan.get("prediction_tools"):
        nodes.append("Prediction Agent")
    if task_plan.get("need_report"):
        nodes.append("Report Agent")
    if task_plan.get("analysis_tools") or task_plan.get("prediction_tools"):
        nodes.append("Chart Renderer")
    return nodes


def deterministic_evidence_sql(task_plan: Dict[str, Any]) -> str | None:
    """Return a reproducible evidence query for fixed-input specialist tools."""
    analysis_tools = set(task_plan.get("analysis_tools") or [])
    prediction_tools = set(task_plan.get("prediction_tools") or [])

    if analysis_tools.intersection({"rfm", "kmeans"}) or "churn_prediction" in prediction_tools:
        return (
            "SELECT customer_id, country, age, gender, membership_tier, "
            "total_orders, total_spend_usd, avg_order_value_usd, "
            "days_since_last_purchase, preferred_category, acquisition_channel, "
            "avg_review_score, returns_made, wishlist_items, churned "
            "FROM customers"
        )

    if "sales_forecast" in prediction_tools:
        return (
            "SELECT year, month, quarter, orders, revenue_usd, "
            "avg_order_value, avg_discount_pct, return_rate, "
            "unique_customers, new_customers "
            "FROM monthly_revenue ORDER BY year, month"
        )

    return None
