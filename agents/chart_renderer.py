"""
图表渲染器 —— 将查询、Analysis / Prediction Agent 的输出转为 ECharts 配置。

非 Agent，纯渲染逻辑。接收 analysis_result 或 prediction_result，
输出 ECharts option JSON 字典列表。

支持图表：
- RFM 分段饼图
- K-Means 聚类散点图（PCA 2D）
- 特征重要度横向柱状图
- ROC 曲线 / 流失概率分布
- 销售预测趋势图（历史+预测+置信区间）
- 运营指标仪表盘
- SQL 查询结果自动折线图 / 柱状图
"""

from numbers import Number
import re
from typing import List, Dict, Any, Optional
from agents.state import AgentState


_TIME_FIELD_PATTERN = re.compile(
    r"(?:date|time|month|year|quarter|日期|时间|月份|年度|季度)",
    re.IGNORECASE,
)


def _is_number(value: object) -> bool:
    return isinstance(value, Number) and not isinstance(value, bool)


def _is_identifier(column: str) -> bool:
    lowered = column.lower()
    return lowered == "id" or lowered.endswith("_id") or lowered.endswith("编号")


def _usable_numeric_columns(rows: list[dict], columns: list[str]) -> list[str]:
    numeric = []
    for column in columns:
        if _is_identifier(column):
            continue
        values = [row.get(column) for row in rows if row.get(column) is not None]
        if values and all(_is_number(value) for value in values):
            numeric.append(column)
    return numeric


def _time_labels(rows: list[dict], columns: list[str]) -> tuple[list[str], set[str]] | None:
    lowered = {column.lower(): column for column in columns}
    if "year" in lowered and "month" in lowered:
        year_column = lowered["year"]
        month_column = lowered["month"]
        try:
            labels = [
                f"{int(row[year_column])}-{int(row[month_column]):02d}"
                for row in rows
            ]
            return labels, {year_column, month_column}
        except (KeyError, TypeError, ValueError):
            pass

    for column in columns:
        if not _TIME_FIELD_PATTERN.search(column):
            continue
        values = [row.get(column) for row in rows]
        if all(value is not None for value in values):
            return [str(value) for value in values], {column}
    return None


def build_query_result_chart(rows: list[dict], max_points: int = 30) -> dict | None:
    """Build one deterministic chart when specialist agents produced none."""
    rows = [row for row in (rows or []) if isinstance(row, dict)][:max_points]
    if not rows:
        return None

    columns = list(dict.fromkeys(
        column for row in rows for column in row.keys()
    ))
    numeric_columns = _usable_numeric_columns(rows, columns)
    if not numeric_columns:
        return None

    # A one-row aggregate is best represented as a compact metric comparison.
    if len(rows) == 1:
        metrics = numeric_columns[:10]
        return {
            "id": "query_result_auto",
            "title": "查询结果概览",
            "type": "bar",
            "generated_by": "query_result_auto",
            "option": {
                "xAxis": {
                    "type": "category",
                    "data": metrics,
                    "axisLabel": {"rotate": 25},
                },
                "yAxis": {"type": "value"},
                "series": [{
                    "name": "数值",
                    "type": "bar",
                    "data": [rows[0].get(column) for column in metrics],
                }],
            },
        }

    time_axis = _time_labels(rows, columns)
    if time_axis:
        labels, time_columns = time_axis
        measures = [
            column for column in numeric_columns if column not in time_columns
        ][:4]
        if measures:
            return {
                "id": "query_result_auto",
                "title": "查询结果趋势",
                "type": "line",
                "generated_by": "query_result_auto",
                "option": {
                    "xAxis": {"type": "category", "data": labels},
                    "yAxis": {"type": "value"},
                    "series": [
                        {
                            "name": column,
                            "type": "line",
                            "smooth": True,
                            "data": [row.get(column) for row in rows],
                        }
                        for column in measures
                    ],
                },
            }

    category_columns = [
        column
        for column in columns
        if column not in numeric_columns
        and any(row.get(column) is not None for row in rows)
    ]
    category = category_columns[0] if category_columns else None
    labels = (
        [str(row.get(category, "")) for row in rows]
        if category
        else [str(index + 1) for index in range(len(rows))]
    )
    measures = numeric_columns[:4]
    return {
        "id": "query_result_auto",
        "title": "查询结果对比" if category else "查询结果趋势",
        "type": "bar" if category else "line",
        "generated_by": "query_result_auto",
        "option": {
            "xAxis": {
                "type": "category",
                "data": labels,
                "axisLabel": {"rotate": 25 if category else 0},
            },
            "yAxis": {"type": "value"},
            "series": [
                {
                    "name": column,
                    "type": "bar" if category else "line",
                    "data": [row.get(column) for row in rows],
                }
                for column in measures
            ],
        },
    }


def render_charts(state: AgentState) -> AgentState:
    """
    将 analysis_result 和 prediction_result 转为 ECharts 配置，
    写入 state["charts"]。
    """
    charts = []

    # ---- 来自 Analysis Agent ----
    analysis = state.get("analysis_result") or {}

    # RFM 饼图
    rfm_segments = analysis.get("rfm", {}).get("segments", {})
    if rfm_segments:
        charts.append({
            "id": "rfm_pie",
            "title": "RFM 客户价值分布",
            "type": "pie",
            "option": {
                "tooltip": {"trigger": "item"},
                "legend": {"orient": "vertical", "left": "left"},
                "series": [{
                    "name": "客户数",
                    "type": "pie",
                    "radius": "60%",
                    "data": [{"name": k, "value": v} for k, v in rfm_segments.items()],
                    "emphasis": {"itemStyle": {"shadowBlur": 10, "shadowColor": "rgba(0,0,0,0.3)"}},
                }],
            },
        })

    # K-Means 散点图
    kmeans = analysis.get("kmeans", {})
    coords = kmeans.get("pca_coords_2d") or kmeans.get("coords_2d")
    if coords and kmeans.get("labels"):
        labels = kmeans["labels"]
        clusters = sorted(set(labels))
        scatter_data = [
            [coords[i][0], coords[i][1], int(labels[i])]
            for i in range(len(coords))
        ]
        charts.append({
            "id": "kmeans_scatter",
            "title": f"K-Means 客户聚类 (轮廓系数: {kmeans.get('silhouette_score', 'N/A')})",
            "type": "scatter",
            "option": {
                "tooltip": {"trigger": "item", "formatter": "{@[2]}"},
                "xAxis": {"name": "PC1"},
                "yAxis": {"name": "PC2"},
                "series": [{
                    "type": "scatter",
                    "data": scatter_data,
                    "symbolSize": 6,
                }],
            },
        })

    # 运营指标仪表盘
    metrics = analysis.get("metrics", {})
    if metrics:
        charts.append({
            "id": "kpi_dashboard",
            "title": "核心运营指标",
            "type": "dashboard",
            "option": {
                "gmv_usd": metrics.get("gmv_usd", 0),
                "avg_order_value_usd": metrics.get("avg_order_value_usd", 0),
                "repeat_purchase_rate": metrics.get("repeat_purchase_rate", 0),
                "churn_rate": metrics.get("churn_rate", 0),
            },
        })

        # 品类收入柱状图
        cat_rev = metrics.get("category_revenue", {})
        if cat_rev:
            charts.append({
                "id": "category_bar",
                "title": "品类收入分布",
                "type": "bar",
                "option": {
                    "xAxis": {"type": "category", "data": list(cat_rev.keys()),
                              "axisLabel": {"rotate": 30}},
                    "yAxis": {"type": "value", "name": "营收 (USD)"},
                    "series": [{"type": "bar", "data": list(cat_rev.values())}],
                },
            })

        # 月度营收趋势
        trend = metrics.get("monthly_revenue_trend", [])
        if trend:
            charts.append({
                "id": "revenue_trend",
                "title": "月度营收趋势",
                "type": "line",
                "option": {
                    "xAxis": {
                        "type": "category",
                        "data": [f"{t['year']}-{t['month']:02d}" for t in trend],
                    },
                    "yAxis": {"type": "value", "name": "营收 (USD)"},
                    "series": [{
                        "type": "line",
                        "data": [t["revenue_usd"] for t in trend],
                        "smooth": True,
                    }],
                },
            })

        # 会员等级分布
        member_dist = metrics.get("membership_distribution", {})
        if member_dist:
            charts.append({
                "id": "membership_pie",
                "title": "会员等级分布",
                "type": "pie",
                "option": {
                    "series": [{
                        "type": "pie",
                        "radius": "55%",
                        "data": [{"name": k, "value": v} for k, v in member_dist.items()],
                    }],
                },
            })

    # ---- 来自 Prediction Agent ----
    prediction = state.get("prediction_result") or {}

    # 流失预测 - 特征重要度
    churn = prediction.get("churn", {})
    importance = churn.get("feature_importance", [])
    if importance:
        # Top 10
        top10 = importance[:10]
        charts.append({
            "id": "churn_feature_importance",
            "title": (
                f"流失预测特征重要度 [{churn.get('model', 'N/A')}] "
                f"(AUC={churn.get('auc', 'N/A')})"
            ),
            "type": "bar",
            "option": {
                "xAxis": {"type": "value"},
                "yAxis": {
                    "type": "category",
                    "data": [f["feature"] for f in reversed(top10)],
                },
                "series": [{
                    "type": "bar",
                    "data": [f["importance"] for f in reversed(top10)],
                }],
            },
        })

    # 销售预测趋势（历史+预测）
    sales = prediction.get("sales", {})
    hist_data = sales.get("historical", [])
    forecast_data = sales.get("forecast", [])
    if hist_data and forecast_data:
        charts.append({
            "id": "sales_forecast",
            "title": (
                f"销售预测 [{sales.get('model', 'N/A')}] "
                f"(RMSE={sales.get('rmse', 'N/A')}, "
                f"MAPE={sales.get('mape_pct', 'N/A')}%)"
            ),
            "type": "line",
            "option": {
                "xAxis": {
                    "type": "category",
                    "data": (
                        [h["ds"][:7] for h in hist_data] +
                        [f["ds"][:7] for f in forecast_data]
                    ),
                },
                "yAxis": {"type": "value", "name": "营收 (USD)"},
                "series": [
                    {
                        "name": "历史营收",
                        "type": "line",
                        "data": [h["y"] for h in hist_data] + [None] * len(forecast_data),
                    },
                    {
                        "name": "预测营收",
                        "type": "line",
                        "data": [None] * len(hist_data) + [f["yhat"] for f in forecast_data],
                        "lineStyle": {"type": "dashed"},
                    },
                ],
            },
        })

    if not charts:
        automatic_chart = build_query_result_chart(state.get("query_result") or [])
        if automatic_chart:
            charts.append(automatic_chart)

    state["charts"] = charts
    state["messages"].append(f"[Chart Renderer] 生成 {len(charts)} 个图表配置")
    return state
