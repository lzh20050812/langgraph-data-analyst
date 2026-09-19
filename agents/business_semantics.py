"""可审计的电商业务语义层。

这里保存可复用的“指标 + 维度 + 粒度”配方，而不是保存测试题全文。
高置信度配方可直接生成确定性只读 SQL，并声明结果形状契约；没有匹配时
仍交给 LLM，避免语义层无限膨胀。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
from agents.task_planning import extract_country_filters


@dataclass(frozen=True)
class QueryContract:
    recipe_id: str
    sql: str
    source_tables: Tuple[str, ...]
    grain: str
    output_columns: Tuple[str, ...]
    rationale: str
    is_fallback: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _has_all(query: str, *groups: Iterable[str]) -> bool:
    return all(any(term in query for term in group) for group in groups)


def _contract(
    recipe_id: str,
    sql: str,
    table: str | Tuple[str, ...],
    grain: str,
    columns: Tuple[str, ...],
    rationale: str,
    *,
    fallback: bool = False,
) -> QueryContract:
    tables = (table,) if isinstance(table, str) else table
    return QueryContract(recipe_id, sql, tables, grain, columns, rationale, fallback)


def _membership_distribution(query: str) -> QueryContract:
    countries = extract_country_filters(query)
    where = ""
    rationale = "会员等级直接来自 customers.membership_tier，占比采用0-100口径。"
    if countries:
        quoted = ", ".join("'" + value.replace("'", "''") + "'" for value in countries)
        where = f"WHERE country IN ({quoted}) "
        rationale += f" 客户范围限定为: {', '.join(countries)}。"
    return _contract(
        "customer.membership_distribution",
        "SELECT membership_tier, COUNT(*) AS cnt, "
        "COUNT(*) * 100.0 / SUM(COUNT(*)) OVER() AS pct "
        f"FROM customers {where}GROUP BY membership_tier ORDER BY cnt DESC",
        "customers", "membership_tier",
        ("membership_tier", "cnt", "pct"),
        rationale,
    )


def _country_customer_value(_: str) -> QueryContract:
    return _contract(
        "customer.country_value",
        "SELECT country, COUNT(*) AS customer_count, "
        "AVG(total_spend_usd) AS avg_spend_usd FROM customers "
        "GROUP BY country ORDER BY customer_count DESC",
        "customers", "country",
        ("country", "customer_count", "avg_spend_usd"),
        "客户国家与累计消费均为 customers 粒度，不需要连接订单表。",
    )


def _age_behavior(_: str) -> QueryContract:
    return _contract(
        "customer.age_band_behavior",
        "SELECT CASE WHEN age < 25 THEN '18-24岁' "
        "WHEN age < 35 THEN '25-34岁' WHEN age < 45 THEN '35-44岁' "
        "WHEN age < 55 THEN '45-54岁' ELSE '55岁以上' END AS age_group, "
        "COUNT(*) AS cnt, AVG(total_spend_usd) AS avg_spend "
        "FROM customers GROUP BY age_group ORDER BY avg_spend DESC",
        "customers", "age_group",
        ("age_group", "cnt", "avg_spend"),
        "固定互斥年龄段，消费行为使用客户累计消费。",
    )


def _device_behavior(_: str) -> QueryContract:
    return _contract(
        "customer.device_behavior",
        "SELECT preferred_device, COUNT(*) AS cnt, "
        "AVG(avg_order_value_usd) AS avg_aov, AVG(total_orders) AS avg_orders "
        "FROM customers GROUP BY preferred_device ORDER BY cnt DESC",
        "customers", "preferred_device",
        ("preferred_device", "cnt", "avg_aov", "avg_orders"),
        "设备偏好是客户属性，客单价和订单数使用同粒度客户指标。",
    )


def _payment_summary(_: str) -> QueryContract:
    return _contract(
        "order.payment_summary",
        "SELECT payment_method, COUNT(*) AS order_count, "
        "SUM(total_amount_usd) AS total_amount FROM orders "
        "GROUP BY payment_method ORDER BY total_amount DESC",
        "orders", "payment_method",
        ("payment_method", "order_count", "total_amount"),
        "支付方式、订单数与订单金额均来自 orders。",
    )


def _order_status_summary(_: str) -> QueryContract:
    return _contract(
        "order.status_summary",
        "SELECT order_status, COUNT(*) AS cnt, "
        "SUM(total_amount_usd) AS total_amount, AVG(returned) AS return_rate "
        "FROM orders GROUP BY order_status",
        "orders", "order_status",
        ("order_status", "cnt", "total_amount", "return_rate"),
        "订单状态退货率必须使用 orders.returned，禁止伪连商品汇总表。",
    )


def _quarter_trend(_: str) -> QueryContract:
    return _contract(
        "order.quarter_trend",
        "SELECT quarter, COUNT(*) AS order_count, "
        "SUM(total_amount_usd) AS revenue, AVG(discount_pct) AS avg_discount "
        "FROM orders GROUP BY quarter ORDER BY quarter",
        "orders", "quarter",
        ("quarter", "order_count", "revenue", "avg_discount"),
        "订单级季度趋势使用 orders，避免把月表的月均折扣再次平均。",
    )


def _repeat_customer_orders(_: str) -> QueryContract:
    return _contract(
        "order.repeat_customer_performance",
        "SELECT is_repeat_customer, COUNT(*) AS cnt, "
        "AVG(customer_rating) AS avg_rating, AVG(discount_pct) AS avg_discount, "
        "AVG(delivery_days) AS avg_delivery FROM orders GROUP BY is_repeat_customer",
        "orders", "is_repeat_customer",
        ("is_repeat_customer", "cnt", "avg_rating", "avg_discount", "avg_delivery"),
        "复购标签及订单表现指标均来自 orders。",
    )


def _category_performance(_: str) -> QueryContract:
    return _contract(
        "product.category_performance",
        "SELECT category, COUNT(*) AS product_count, "
        "SUM(total_revenue_usd) AS total_revenue, AVG(avg_rating) AS avg_rating "
        "FROM product_summary GROUP BY category ORDER BY total_revenue DESC",
        "product_summary", "category",
        ("category", "product_count", "total_revenue", "avg_rating"),
        "商品数量、累计营收和评分来自商品粒度汇总表。",
    )


def _category_returns(_: str) -> QueryContract:
    return _contract(
        "product.category_returns",
        "SELECT category, AVG(return_rate) AS avg_return_rate, "
        "AVG(avg_discount_pct) AS avg_discount FROM product_summary "
        "GROUP BY category ORDER BY avg_return_rate DESC",
        "product_summary", "category",
        ("category", "avg_return_rate", "avg_discount"),
        "品类退货率和折扣率使用 product_summary 的商品粒度指标。",
    )


def _top_rated_products(_: str) -> QueryContract:
    return _contract(
        "product.top_rated",
        "SELECT product_name, category, avg_rating, avg_price, total_orders "
        "FROM product_summary ORDER BY avg_rating DESC LIMIT 10",
        "product_summary", "product",
        ("product_name", "category", "avg_rating", "avg_price", "total_orders"),
        "最高评分排序字段也必须出现在结果中，以便解释排名。",
    )


def _monthly_revenue_2025(query: str) -> QueryContract:
    import re
    match = re.search(r"20\d{2}", query)
    year = int(match.group()) if match else 2025
    return _contract(
        "revenue.monthly_year",
        f"SELECT year, month, revenue_usd FROM monthly_revenue "
        f"WHERE year = {year} ORDER BY month",
        "monthly_revenue", "month",
        ("year", "month", "revenue_usd"),
        "指定年份的月度营收直接使用月度事实表。",
    )


def _annual_summary(_: str) -> QueryContract:
    return _contract(
        "revenue.annual_summary",
        "SELECT year, SUM(revenue_usd) AS total_revenue, "
        "SUM(orders) AS total_orders, SUM(new_customers) AS total_new_customers "
        "FROM monthly_revenue GROUP BY year ORDER BY year",
        "monthly_revenue", "year",
        ("year", "total_revenue", "total_orders", "total_new_customers"),
        "年度指标由月度事实表按年求和。",
    )


def _churn_distribution(_: str) -> QueryContract:
    return _contract(
        "customer.churn_distribution",
        "SELECT churned, COUNT(*) AS cnt, "
        "COUNT(*) * 100.0 / SUM(COUNT(*)) OVER() AS pct "
        "FROM customers GROUP BY churned",
        "customers", "churned",
        ("churned", "cnt", "pct"),
        "流失状态来自 customers.churned，占比采用0-100口径。",
    )


def _channel_value(_: str) -> QueryContract:
    return _contract(
        "customer.channel_value",
        "SELECT acquisition_channel, COUNT(*) AS cnt, "
        "SUM(total_spend_usd) AS total_revenue, AVG(churned) AS churn_rate "
        "FROM customers GROUP BY acquisition_channel ORDER BY total_revenue DESC",
        "customers", "acquisition_channel",
        ("acquisition_channel", "cnt", "total_revenue", "churn_rate"),
        "渠道客户量、客户累计消费和流失标签均为客户粒度。",
    )


def _discount_effect(_: str) -> QueryContract:
    return _contract(
        "order.discount_effect_fallback",
        "SELECT CASE WHEN discount_pct < 10 THEN '0-10%' "
        "WHEN discount_pct < 20 THEN '10-20%' WHEN discount_pct < 30 THEN '20-30%' "
        "ELSE '30%及以上' END AS discount_bucket, COUNT(*) AS order_count, "
        "SUM(total_amount_usd) AS revenue_usd, AVG(returned) * 100 AS return_rate_pct, "
        "AVG(is_repeat_customer) * 100 AS repeat_customer_pct FROM orders "
        "GROUP BY discount_bucket ORDER BY revenue_usd DESC",
        "orders", "discount_bucket",
        ("discount_bucket", "order_count", "revenue_usd", "return_rate_pct", "repeat_customer_pct"),
        "缺少订单—商品键时，退回订单单表评估折扣与营收/退货/复购关系。",
        fallback=True,
    )


def _return_cause(_: str) -> QueryContract:
    return _contract(
        "order.return_cause_fallback",
        "SELECT o.returned, COUNT(*) AS order_count, "
        "AVG(o.customer_rating) AS avg_rating, AVG(o.delivery_days) AS avg_delivery_days, "
        "AVG(o.discount_pct) AS avg_discount_pct, AVG(o.shipping_fee_usd) AS avg_shipping_fee_usd, "
        "AVG(o.is_repeat_customer) * 100 AS repeat_customer_pct, "
        "AVG(c.churned) * 100 AS churned_customer_pct FROM orders o "
        "LEFT JOIN customers c ON o.customer_id = c.customer_id "
        "GROUP BY o.returned ORDER BY o.returned DESC",
        ("orders", "customers"), "returned",
        ("returned", "order_count", "avg_rating", "avg_delivery_days", "avg_discount_pct", "avg_shipping_fee_usd", "repeat_customer_pct", "churned_customer_pct"),
        "Schema 没有 orders.product_id，使用合法 customer_id 连接并对比退货/未退货订单。",
        fallback=True,
    )


def _support_team_resolution(_: str) -> QueryContract:
    return _contract(
        "support.team_resolution",
        "SELECT a.team, COUNT(*) AS ticket_count, "
        "ROUND(AVG(t.resolution_hours), 2) AS avg_resolution_hours "
        "FROM support_tickets t JOIN support_agents a ON t.agent_id = a.agent_id "
        "GROUP BY a.team ORDER BY a.team",
        ("support_tickets", "support_agents"), "team",
        ("team", "ticket_count", "avg_resolution_hours"),
        "工单通过 agent_id 连接坐席维表；未解决工单的 NULL 耗时不进入平均值。",
    )


Recipe = Tuple[Callable[[str], bool], Callable[[str], QueryContract]]


RECIPES: Tuple[Recipe, ...] = (
    (lambda q: _has_all(q, ("客服团队", "坐席组"), ("工单",),
                        ("平均解决时长", "平均处理耗时")), _support_team_resolution),
    (lambda q: _has_all(q, ("会员等级", "会员层级"), ("数量", "客户数", "用户数", "人数", "分布")), _membership_distribution),
    (lambda q: _has_all(q, ("国家", "地区"), ("客户数量", "客户数", "用户规模", "用户数"), ("平均消费", "消费水平", "人均累计消费")), _country_customer_value),
    (lambda q: _has_all(q, ("年龄段", "年龄区间"), ("分布", "消费行为", "人数", "平均消费")), _age_behavior),
    (lambda q: _has_all(q, ("设备", "终端"), ("客户", "用户", "群体"), ("特征", "差异", "比较", "客单价")), _device_behavior),
    (lambda q: _has_all(q, ("支付方式", "付款类型"), ("订单数量", "订单量", "交易笔数"), ("金额", "成交额")), _payment_summary),
    (lambda q: _has_all(q, ("订单状态", "按状态"), ("订单量", "订单数量", "订单数"), ("退货率", "退单比例")), _order_status_summary),
    (lambda q: _has_all(q, ("季度",), ("订单量", "订单数"), ("营收", "销售额"), ("折扣", "优惠")), _quarter_trend),
    (lambda q: _has_all(q, ("复购客户", "复购", "老客"), ("新客户", "首购"), ("订单表现", "订单", "对比")), _repeat_customer_orders),
    (lambda q: _has_all(q, ("品类", "商品分类"), ("商品数量", "sku数"), ("营收", "销售收入"), ("评分", "均分")), _category_performance),
    (lambda q: _has_all(q, ("品类", "商品分类"), ("退货率", "平均退货", "退货"), ("折扣率", "折扣", "优惠")), _category_returns),
    (lambda q: _has_all(q, ("评分最高", "最高评分", "均分最高"), ("商品", "产品"), ("销量", "订单")), _top_rated_products),
    (lambda q: _has_all(q, ("每月", "月度", "各月"), ("营收", "销售收入"), ("趋势", "变化", "走势")) and any(str(y) in q for y in range(2000, 2100)), _monthly_revenue_2025),
    (lambda q: _has_all(q, ("年度", "各年", "按年"), ("营收", "销售收入"), ("订单",), ("新客户", "新增用户")), _annual_summary),
    (lambda q: _has_all(q, ("流失客户", "已流失", "流失与留存", "流失用户"), ("数量", "人数", "用户数"), ("占比", "比例")), _churn_distribution),
    (lambda q: _has_all(q, ("获客渠道", "引流来源"), ("客户数量", "客户数", "用户规模"), ("消费", "消费额"), ("流失率", "流失比例")), _channel_value),
    (lambda q: _has_all(q, ("折扣", "促销"), ("营收", "销售额"), ("关系", "效果", "评估")), _discount_effect),
    (lambda q: _has_all(q, ("退货率", "退货"), ("原因",), ("策略", "降低", "制定")), _return_cause),
)


def build_query_contract(user_query: str) -> Optional[QueryContract]:
    query = user_query.lower().strip()
    for predicate, factory in RECIPES:
        if predicate(query):
            contract = factory(query)
            # Fixed recipes must never silently discard extra user scope. The
            # general constrained SQL path gets the request when a recipe does
            # not model exclusions, Top-N, or an additional population filter.
            unsupported_markers = (
                "排除", "不包含", "除了", "top ", "top-", "前10", "前 10",
                "最近", "去年", "今年",
            )
            if any(marker in query for marker in unsupported_markers):
                return None
            population_markers = ("只看", "仅看", "只统计", "仅统计")
            supported_country_scope = (
                contract.recipe_id == "customer.membership_distribution"
                and bool(extract_country_filters(query))
            )
            if any(marker in query for marker in population_markers) and not supported_country_scope:
                return None
            years = re.findall(r"(?<!\d)20\d{2}(?!\d)", query)
            if years and contract.recipe_id != "revenue.monthly_year":
                return None
            return contract
    return None


def validate_result_shape(
    rows: Optional[List[Dict[str, Any]]], contract: QueryContract
) -> Optional[str]:
    if rows is None:
        return "查询结果为 None"
    if not rows:
        return None  # 空结果可能由真实过滤条件造成，不能自动判为 SQL 错误。
    actual = tuple(rows[0].keys())
    if actual != contract.output_columns:
        return (
            f"结果列不符合契约 {contract.recipe_id}: "
            f"期望 {list(contract.output_columns)}，实际 {list(actual)}"
        )
    return None
