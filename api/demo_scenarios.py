"""Deterministic, explicitly-labelled scenarios for offline defense demos."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


SOURCE_NOTICE = (
    "这是答辩缓存演示结果：数值取自项目内固定电商数据集的只读聚合快照，"
    "本次展示不调用 LLM，也不代表实时生产数据。"
)


def _bar_chart(title: str, rows: list[dict[str, Any]], category: str, measures: list[str]) -> dict:
    return {
        "id": f"demo_{category}_bar",
        "title": title,
        "type": "bar",
        "option": {
            "xAxis": {"type": "category", "data": [str(row[category]) for row in rows], "axisLabel": {"rotate": 25}},
            "yAxis": {"type": "value"},
            "series": [
                {"name": measure, "type": "bar", "data": [row[measure] for row in rows]}
                for measure in measures
            ],
        },
    }


_CATEGORY_ROWS = [
    {"category": "Electronics", "revenue_usd": 1148937.00, "orders": 4526, "return_rate_pct": 8.04},
    {"category": "Clothing & Apparel", "revenue_usd": 449288.68, "orders": 3981, "return_rate_pct": 7.28},
    {"category": "Home & Kitchen", "revenue_usd": 434212.35, "orders": 3068, "return_rate_pct": 8.80},
    {"category": "Sports & Outdoors", "revenue_usd": 159620.11, "orders": 1761, "return_rate_pct": 7.78},
    {"category": "Jewelry & Accessories", "revenue_usd": 145627.20, "orders": 973, "return_rate_pct": 7.81},
    {"category": "Office Supplies", "revenue_usd": 133472.15, "orders": 770, "return_rate_pct": 8.05},
]

_TIER_ROWS = [
    {"membership_tier": "Free", "customers": 4443, "avg_spend_usd": 1393.21},
    {"membership_tier": "Silver", "customers": 1736, "avg_spend_usd": 1388.75},
    {"membership_tier": "Gold", "customers": 1177, "avg_spend_usd": 1863.59},
    {"membership_tier": "Platinum", "customers": 644, "avg_spend_usd": 2600.64},
]

_MONTHLY_ROWS = [
    {"month": "2024-10", "revenue_usd": 31867.25, "orders": 285},
    {"month": "2024-11", "revenue_usd": 34159.40, "orders": 260},
    {"month": "2024-12", "revenue_usd": 31629.30, "orders": 263},
    {"month": "2025-01", "revenue_usd": 32012.75, "orders": 255},
    {"month": "2025-02", "revenue_usd": 29197.88, "orders": 255},
    {"month": "2025-03", "revenue_usd": 38073.94, "orders": 302},
    {"month": "2025-04", "revenue_usd": 31374.44, "orders": 264},
    {"month": "2025-05", "revenue_usd": 36208.38, "orders": 287},
    {"month": "2025-06", "revenue_usd": 40777.79, "orders": 306},
    {"month": "2025-07", "revenue_usd": 41236.16, "orders": 269},
    {"month": "2025-08", "revenue_usd": 34388.76, "orders": 290},
    {"month": "2025-09", "revenue_usd": 34675.45, "orders": 264},
    {"month": "2025-10", "revenue_usd": 39940.71, "orders": 287},
    {"month": "2025-11", "revenue_usd": 28149.83, "orders": 240},
    {"month": "2025-12", "revenue_usd": 39321.84, "orders": 304},
    {"month": "2026-01", "revenue_usd": 44793.18, "orders": 316},
    {"month": "2026-02", "revenue_usd": 38488.48, "orders": 254},
    {"month": "2026-03", "revenue_usd": 32872.17, "orders": 266},
]


SCENARIOS: dict[str, dict[str, Any]] = {
    "category-performance": {
        "title": "品类经营表现",
        "description": "展示品类营收、订单量、退货率与可追溯 SQL。",
        "kind": "correct_analysis",
        "steps": ["检查结构化条件", "查看只读 SQL 与结果", "展开事实证据与限制"],
        "query": "分析各品类营收、订单量与退货率，给出经营建议",
        "intent": "analysis",
        "result": {
            "success": True,
            "intent": "analysis",
            "sql": "SELECT category, ROUND(SUM(total_amount_usd), 2) AS revenue_usd, COUNT(*) AS orders, ROUND(AVG(returned) * 100, 2) AS return_rate_pct FROM orders GROUP BY category ORDER BY revenue_usd DESC;",
            "query_result": _CATEGORY_ROWS,
            "query_result_total_rows": len(_CATEGORY_ROWS),
            "query_result_truncated": False,
            "report": "# 核心结论\n\n- Electronics 营收 1,148,937 美元，在展示品类中明显领先。\n- Clothing & Apparel 与 Home & Kitchen 构成第二梯队。\n\n# 风险提示\n\n- Home & Kitchen 退货率为 8.80%，高于服装品类的 7.28%。\n- 当前为固定数据集快照，只能解释样本期内表现。\n\n# 建议\n\n1. 优先拆解 Electronics 的流量、转化与客单价来源。\n2. 对 Home & Kitchen 开展退货原因和 SKU 级质量排查。",
            "charts": [_bar_chart("品类营收对比", _CATEGORY_ROWS, "category", ["revenue_usd"])],
            "governance_result": {"quality_score": 1.0, "source_tables": ["orders"], "limitations": ["固定数据集聚合快照"]},
            "demo_steps": [
                {"title": "条件", "detail": "指标、维度与排序来自受控分析请求。", "status": "verified"},
                {"title": "执行", "detail": "只读聚合 SQL 与固定快照结果逐行对应。", "status": "verified"},
                {"title": "证据", "detail": "结论展示来源表、限制与生成 SQL。", "status": "verified"},
            ],
        },
    },
    "customer-value": {
        "title": "客户价值概览",
        "description": "展示客户 KPI、会员结构和价值分层差异。",
        "query": "分析客户价值、会员层级与流失风险概况",
        "intent": "analysis",
        "result": {
            "success": True,
            "intent": "analysis",
            "sql": "SELECT membership_tier, COUNT(*) AS customers, ROUND(AVG(total_spend_usd), 2) AS avg_spend_usd FROM customers GROUP BY membership_tier ORDER BY customers DESC;",
            "query_result": _TIER_ROWS,
            "query_result_total_rows": len(_TIER_ROWS),
            "query_result_truncated": False,
            "report": "# 核心结论\n\n- 固定样本包含 8,000 名客户，总消费额为 12,469,138.80 美元。\n- Platinum 客户平均消费 2,600.64 美元，是 Free 客户的约 1.87 倍。\n- 样本流失率为 8.94%。\n\n# 经营建议\n\n1. 针对 Gold 和 Platinum 客户设计高价值留存权益。\n2. 对 Free 客户建立升级路径，并结合流失模型筛选重点触达人群。\n\n# 限制\n\n- 该结果描述历史样本，不等同于实时客户状态或因果结论。",
            "charts": [
                {"id": "demo_customer_kpi", "title": "客户核心指标", "type": "dashboard", "option": {"gmv_usd": 12469138.80, "avg_order_value_usd": 94.85, "repeat_purchase_rate": None, "churn_rate": 0.0894}},
                _bar_chart("会员层级平均消费", _TIER_ROWS, "membership_tier", ["avg_spend_usd"]),
            ],
            "governance_result": {"quality_score": 1.0, "source_tables": ["customers"], "limitations": ["未在此缓存场景中计算复购率"]},
        },
    },
    "monthly-trend": {
        "title": "月度营收趋势",
        "description": "展示最近 18 个月营收和订单趋势，不生成虚构预测。",
        "query": "分析最近 18 个月销售趋势与波动",
        "intent": "sql_query",
        "result": {
            "success": True,
            "intent": "sql_query",
            "sql": "SELECT year, month, revenue_usd, orders FROM monthly_revenue ORDER BY year DESC, month DESC LIMIT 18;",
            "query_result": _MONTHLY_ROWS,
            "query_result_total_rows": len(_MONTHLY_ROWS),
            "query_result_truncated": False,
            "report": "# 趋势观察\n\n- 2026 年 1 月营收达到 44,793.18 美元，是展示区间内高点。\n- 月度营收存在明显波动，2025 年 11 月降至 28,149.83 美元。\n\n# 建议\n\n1. 将活动、渠道和品类数据与异常月份做关联分析。\n2. 正式预测应运行完整预测链路并展示验证集、测试集和置信区间。\n\n# 说明\n\n- 本场景只展示历史真实聚合，不将趋势外推伪装成预测。",
            "charts": [{"id": "demo_monthly_line", "title": "月度营收与订单趋势", "type": "line", "option": {"xAxis": {"type": "category", "data": [row["month"] for row in _MONTHLY_ROWS]}, "yAxis": [{"type": "value", "name": "营收"}, {"type": "value", "name": "订单", "splitLine": {"show": False}}], "series": [{"name": "营收（USD）", "type": "line", "smooth": True, "data": [row["revenue_usd"] for row in _MONTHLY_ROWS]}, {"name": "订单量", "type": "bar", "yAxisIndex": 1, "data": [row["orders"] for row in _MONTHLY_ROWS]}]}}],
            "governance_result": {"quality_score": 1.0, "source_tables": ["monthly_revenue"], "limitations": ["历史聚合，不包含未来预测"]},
        },
    },
    "multiturn-clarification": {
        "title": "多轮澄清与条件继承",
        "description": "演示指标澄清、地区继承、维度替换与条件清除。",
        "kind": "multiturn_clarification",
        "steps": ["统计去年各地区实付金额", "只看美国", "按月拆分", "改成净销售额并触发缺失字段说明"],
        "query": "统计去年各地区实付金额 → 只看美国 → 按月拆分 → 改成净销售额",
        "intent": "analysis",
        "result": {
            "success": True,
            "intent": "analysis",
            "sql": "SELECT YEAR(order_date) AS year, MONTH(order_date) AS month, ROUND(SUM(total_amount_usd), 2) AS paid_amount_usd FROM orders WHERE country = :country AND YEAR(order_date) = :year GROUP BY year, month ORDER BY year, month;",
            "query_result": _MONTHLY_ROWS[-12:],
            "query_result_total_rows": 12,
            "query_result_truncated": False,
            "report": "# 多轮结果\n\n- 年份和美国筛选在后续轮次中保留。\n- ‘按月拆分’只替换分组维度，不清除指标与筛选。\n- ‘净销售额’需要退款金额字段；当前 Schema 缺少该字段，因此系统要求澄清并拒绝编造。\n\n# 限制\n\n本演示使用固定状态迁移快照，不调用模型。",
            "charts": [{"id": "demo_multiturn_line", "title": "继承条件后的月度实付金额", "type": "line", "option": {"xAxis": {"type": "category", "data": [row["month"] for row in _MONTHLY_ROWS[-12:]]}, "yAxis": {"type": "value"}, "series": [{"name": "实付金额（USD）", "type": "line", "data": [row["revenue_usd"] for row in _MONTHLY_ROWS[-12:]]}]}}],
            "governance_result": {"quality_score": 1.0, "source_tables": ["orders"], "limitations": ["净销售额因缺少 refund_amount 未执行"]},
            "analysis_request": {"metrics": ["paid_amount"], "dimensions": ["month"], "time_scope": {"years": [2025]}, "filters": {"country": ["United States"]}, "metric_catalog_version": "2026.09.1"},
            "context_changes": [
                {"turn": 1, "change": "创建指标 paid_amount、年份 2025、地区维度"},
                {"turn": 2, "change": "继承指标与年份，新增 country=United States"},
                {"turn": 3, "change": "维度替换为 month，其余条件保持"},
                {"turn": 4, "change": "请求 net_sales；缺少 refund_amount，进入澄清/拒绝"},
            ],
            "demo_steps": [
                {"title": "继承", "detail": "年份和国家筛选跨轮保留。", "status": "verified"},
                {"title": "替换", "detail": "地区分组替换为月份，不污染其他条件。", "status": "verified"},
                {"title": "澄清", "detail": "缺少退款字段时停止执行，不虚构净销售额。", "status": "unverifiable"},
            ],
        },
    },
    "worker-recovery": {
        "title": "Worker 中断与租约接管",
        "description": "展示旧执行者失效、任务重排队、新 Worker 接管和唯一终态。",
        "kind": "failure_recovery",
        "steps": ["Worker A 领取", "租约过期并重排队", "Worker B 使用新令牌接管", "旧令牌拒写且 B 完成"],
        "runtime_trace": True,
        "query": "演示 Worker 中断后的安全恢复",
        "intent": "analysis",
        "result": {
            "success": True,
            "intent": "analysis",
            "sql": None,
            "query_result": [],
            "query_result_total_rows": 0,
            "query_result_truncated": False,
            "report": "# 恢复结果\n\n- Worker A 的租约失效后，任务从 running 合法迁移回 queued。\n- Worker B 获得新的执行归属并提交唯一完成结果。\n- A 的旧令牌不能写入事件、检查点或终态。\n\n# 说明\n\n这是本地确定性故障演示，不执行 SQL 或调用模型；真实故障行为由阶段六自动化测试覆盖。",
            "charts": [],
            "governance_result": {"quality_score": 1.0, "source_tables": [], "limitations": ["确定性故障演示，不代表真实模型吞吐"]},
            "recovery_timeline": [
                {"state": "queued", "owner": None, "detail": "任务持久化"},
                {"state": "running", "owner": "worker-a", "detail": "第一次领取"},
                {"state": "queued", "owner": None, "detail": "租约过期，旧令牌失效"},
                {"state": "running", "owner": "worker-b", "detail": "新令牌接管"},
                {"state": "completed", "owner": None, "detail": "唯一终态提交"},
            ],
            "demo_steps": [
                {"title": "中断", "detail": "过期租约触发有界重排队。", "status": "verified"},
                {"title": "接管", "detail": "新 Worker 获得不同归属令牌。", "status": "verified"},
                {"title": "防迟到写", "detail": "旧令牌事件、检查点和结果写入均被拒绝。", "status": "verified"},
            ],
        },
    },
}


def list_demo_scenarios() -> list[dict[str, Any]]:
    return [
        {
            "id": scenario_id,
            "title": item["title"],
            "description": item["description"],
            "query": item["query"],
            "kind": item.get("kind", "cached_analysis"),
            "steps": item.get("steps", []),
            "mode": "cached_demo",
        }
        for scenario_id, item in SCENARIOS.items()
    ]


def build_demo_result(scenario_id: str, request_id: str) -> tuple[dict[str, Any], dict[str, Any]]:
    scenario = SCENARIOS.get(scenario_id)
    if scenario is None:
        raise KeyError(scenario_id)
    result = deepcopy(scenario["result"])
    result.update({
        "request_id": request_id,
        "demo_mode": True,
        "result_mode": "cached_demo",
        "demo_kind": scenario.get("kind", "cached_analysis"),
        "source_notice": SOURCE_NOTICE,
        "evidence": {"kind": "fixed_local_dataset_snapshot", "source_tables": result["governance_result"]["source_tables"]},
        "execution_trace": [{"node": "demo_cache", "status": "completed", "duration_ms": 0}],
    })
    return deepcopy(scenario), result
