"""Versioned business metrics and deterministic unified analysis requests."""

from __future__ import annotations

from copy import deepcopy
from datetime import date
import json
import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


METRIC_CATALOG_VERSION = "2026.09.1"
CONFIRMED_REQUEST_PREFIX = "已确认的结构化分析条件（不得静默忽略）："


class MetricDefinition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    metric_id: str
    version: str
    name: str
    aliases: list[str]
    formula: str
    source_table: str
    required_fields: list[str]
    time_field: str | None = None
    supported_dimensions: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)


METRIC_CATALOG: dict[str, MetricDefinition] = {
    "revenue": MetricDefinition(
        metric_id="revenue", version="1.0.0", name="营收",
        aliases=["营收", "销售额", "销售", "收入", "gmv", "revenue"],
        formula="SUM(monthly_revenue.revenue_usd)",
        source_table="monthly_revenue",
        required_fields=["year", "month", "revenue_usd"],
        time_field="year,month",
        supported_dimensions=["time"],
        limitations=["月度营收表不含国家或客户维度"],
    ),
    "customer_count": MetricDefinition(
        metric_id="customer_count", version="1.0.0", name="客户数",
        aliases=["客户数", "用户数", "人数", "customer count"],
        formula="COUNT(*)",
        source_table="customers",
        required_fields=["customer_id"],
        supported_dimensions=["country", "membership_tier", "acquisition_channel", "device"],
    ),
    "avg_order_value": MetricDefinition(
        metric_id="avg_order_value", version="1.0.0", name="平均客单价",
        aliases=["客单价", "平均订单", "aov"],
        formula="AVG(orders.total_amount_usd)", source_table="orders",
        required_fields=["total_amount_usd"],
        supported_dimensions=["time", "payment_method", "order_status"],
    ),
    "paid_amount": MetricDefinition(
        metric_id="paid_amount", version="1.0.0", name="实付金额",
        aliases=["实付金额", "成交金额", "paid amount"],
        formula="SUM(orders.total_amount_usd)", source_table="orders",
        required_fields=["total_amount_usd", "order_date", "customer_id"],
        time_field="orders.order_date",
        supported_dimensions=["time", "country", "payment_method", "order_status"],
        limitations=["当前 total_amount_usd 按订单口径，不等同于扣除退款后的净销售额"],
    ),
    "return_rate": MetricDefinition(
        metric_id="return_rate", version="1.0.0", name="退货率",
        aliases=["退货率", "退货", "return rate"],
        formula="AVG(orders.returned)", source_table="orders",
        required_fields=["returned"],
        supported_dimensions=["time", "payment_method", "order_status"],
    ),
    "churn_rate": MetricDefinition(
        metric_id="churn_rate", version="1.0.0", name="当前流失率",
        aliases=["流失率", "流失客户", "churn rate"],
        formula="AVG(customers.churned)", source_table="customers",
        required_fields=["churned"],
        supported_dimensions=["country", "membership_tier", "acquisition_channel", "device"],
        limitations=["churned 为当前静态标签，不支持按历史月份回溯"],
    ),
    "net_revenue": MetricDefinition(
        metric_id="net_revenue", version="1.0.0", name="净销售额",
        aliases=["净销售额", "净营收", "net revenue"],
        formula="revenue - refunds", source_table="unavailable",
        required_fields=["revenue", "refund_amount"],
        supported_dimensions=["time", "country"],
        limitations=["当前 Schema 缺少 refund_amount，无法可靠计算"],
    ),
    "customer_value": MetricDefinition(
        metric_id="customer_value", version="1.0.0", name="RFM 客户价值分层",
        aliases=["rfm", "客户价值", "客户分层"],
        formula="RFM(recency, frequency, monetary)", source_table="customers",
        required_fields=["days_since_last_purchase", "total_orders", "total_spend_usd"],
        supported_dimensions=["country", "membership_tier", "acquisition_channel", "device"],
        limitations=["客户特征为当前快照，不支持按历史日期回溯"],
    ),
    "ticket_count": MetricDefinition(
        metric_id="ticket_count", version="1.0.0", name="工单数",
        aliases=["工单数", "工单量", "ticket count"],
        formula="COUNT(*)", source_table="support_tickets",
        required_fields=["ticket_id"], time_field="support_tickets.opened_at",
        supported_dimensions=["time", "channel", "priority", "status", "team"],
    ),
    "avg_resolution_hours": MetricDefinition(
        metric_id="avg_resolution_hours", version="1.0.0", name="平均解决时长",
        aliases=["平均解决时长", "平均处理耗时", "resolution time"],
        formula="AVG(support_tickets.resolution_hours)", source_table="support_tickets",
        required_fields=["resolution_hours"], time_field="support_tickets.opened_at",
        supported_dimensions=["time", "channel", "priority", "status", "team"],
    ),
    "satisfaction_score": MetricDefinition(
        metric_id="satisfaction_score", version="1.0.0", name="客服满意度",
        aliases=["客服满意度", "满意度评分", "csat"],
        formula="AVG(support_tickets.satisfaction_score)", source_table="support_tickets",
        required_fields=["satisfaction_score"], time_field="support_tickets.opened_at",
        supported_dimensions=["time", "channel", "priority", "status", "team"],
    ),
}


COUNTRY_ALIASES: dict[str, str] = {
    "美国": "United States", "united states": "United States", "usa": "United States",
    "中国": "China", "china": "China", "德国": "Germany", "germany": "Germany",
    "法国": "France", "france": "France", "英国": "United Kingdom",
    "united kingdom": "United Kingdom", "uk": "United Kingdom", "日本": "Japan",
    "japan": "Japan", "韩国": "South Korea", "south korea": "South Korea",
    "印度": "India", "india": "India", "加拿大": "Canada", "canada": "Canada",
    "澳大利亚": "Australia", "australia": "Australia", "新加坡": "Singapore",
    "singapore": "Singapore", "巴西": "Brazil", "brazil": "Brazil",
    "墨西哥": "Mexico", "mexico": "Mexico",
}


def extract_country_filters(query: str) -> list[str]:
    text = query.lower()
    countries: list[str] = []
    for alias, canonical in sorted(
        COUNTRY_ALIASES.items(), key=lambda item: len(item[0]), reverse=True
    ):
        if alias in text and canonical not in countries:
            countries.append(canonical)
    return countries


class TimeScope(BaseModel):
    model_config = ConfigDict(extra="forbid")

    field: str | None = None
    years: list[int] = Field(default_factory=list)
    start: str | None = None
    end: str | None = None


class SortRule(BaseModel):
    model_config = ConfigDict(extra="forbid")

    field: str
    direction: Literal["asc", "desc"] = "desc"


class AnalysisRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    schema_version: str = "1.0"
    metric_catalog_version: str = METRIC_CATALOG_VERSION
    metrics: list[str] = Field(default_factory=list)
    dimensions: list[str] = Field(default_factory=list)
    time_scope: TimeScope = Field(default_factory=TimeScope)
    filters: dict[str, list[str]] = Field(default_factory=dict)
    sort: list[SortRule] = Field(default_factory=list)
    limit: int | None = Field(default=None, ge=1, le=1000)
    data_source: str = "ai_analytics"
    unresolved_ambiguities: list[str] = Field(default_factory=list)
    unsupported_conditions: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_metric_ids(self) -> "AnalysisRequest":
        unknown = [item for item in self.metrics if item not in METRIC_CATALOG]
        if unknown:
            raise ValueError(f"unknown metric ids: {unknown}")
        return self


def metric_catalog_payload() -> dict[str, Any]:
    return {
        "version": METRIC_CATALOG_VERSION,
        "metrics": [item.model_dump(mode="python") for item in METRIC_CATALOG.values()],
    }


def analysis_request_semantic_issues(request: dict[str, Any]) -> list[str]:
    """Validate supported sources, dimensions, filters and historical scope."""
    metrics = request.get("metrics") or []
    dimensions = request.get("dimensions") or []
    filters = request.get("filters") or {}
    years = (request.get("time_scope") or {}).get("years") or []
    issues: list[str] = []
    sources = set()
    for metric_id in metrics:
        definition = METRIC_CATALOG[metric_id]
        source = (
            "support_ops"
            if definition.source_table.startswith("support_")
            else "ai_analytics"
        )
        if definition.source_table != "unavailable":
            sources.add(source)
        if definition.source_table == "unavailable":
            issues.extend(definition.limitations)
        unsupported_dimensions = [
            dimension for dimension in dimensions
            if dimension not in definition.supported_dimensions
        ]
        if unsupported_dimensions:
            issues.append(
                f"{definition.name} 不支持维度: {', '.join(unsupported_dimensions)}"
            )
        if "country" in filters and "country" not in definition.supported_dimensions:
            issues.append(f"{definition.name} 不支持国家筛选")
        if years and not definition.time_field:
            issues.append(f"{definition.name} 没有可回溯的时间字段")
    if len(sources) > 1:
        issues.append("单次分析暂不支持跨数据源组合指标")
    return list(dict.fromkeys(issues))


def extract_confirmed_analysis_request(query: str) -> dict[str, Any] | None:
    """Read the API-injected request contract without re-interpreting it."""
    if not query.startswith(CONFIRMED_REQUEST_PREFIX):
        return None
    payload = query[len(CONFIRMED_REQUEST_PREFIX):].split("\n当前用户表达：", 1)[0]
    try:
        return AnalysisRequest.model_validate(json.loads(payload)).model_dump(
            mode="python"
        )
    except (json.JSONDecodeError, ValueError, TypeError):
        return None


def build_analysis_request(query: str) -> dict[str, Any]:
    """Parse the deterministic subset; preserve uncertainty explicitly."""
    text = query.lower().strip()
    metrics = [
        metric_id for metric_id, definition in METRIC_CATALOG.items()
        if any(alias.lower() in text for alias in definition.aliases)
    ]
    # Net revenue is more specific than generic revenue.
    if "net_revenue" in metrics and "revenue" in metrics:
        metrics.remove("revenue")

    dimension_terms = {
        "country": ("国家", "地区", "区域"),
        "membership_tier": ("会员等级", "会员层级"),
        "acquisition_channel": ("获客渠道", "渠道"),
        "time": ("按月", "月度", "每月", "年度", "趋势"),
        "device": ("设备", "终端"),
        "channel": ("工单渠道", "客服渠道"),
        "priority": ("优先级", "紧急程度"),
        "status": ("工单状态", "处理状态"),
        "team": ("客服团队", "坐席组"),
    }
    dimensions = [
        name for name, terms in dimension_terms.items()
        if any(term in text for term in terms)
    ]
    years = sorted({int(value) for value in re.findall(r"(?<!\d)(20\d{2})(?!\d)", text)})
    if "去年" in text:
        years = [date.today().year - 1]
    elif "今年" in text:
        years = [date.today().year]
    countries = extract_country_filters(text)
    filters = {"country": countries} if countries else {}

    limit_match = re.search(r"(?:top\s*|\u524d\s*)(\d{1,4})", text, re.IGNORECASE)
    limit = min(1000, int(limit_match.group(1))) if limit_match else None
    sort: list[SortRule] = []
    if limit or any(term in text for term in ("最高", "最多", "降序")):
        sort.append(SortRule(field=metrics[0] if metrics else "result", direction="desc"))
    elif any(term in text for term in ("最低", "最少", "升序")):
        sort.append(SortRule(field=metrics[0] if metrics else "result", direction="asc"))

    ambiguities: list[str] = []
    unsupported: list[str] = []
    if any(term in text for term in ("华东", "华南", "华北", "华中", "西南", "西北", "东北")):
        unsupported.append("当前 Schema 只有 country，没有华东等国内区域字段")
    if not metrics:
        ambiguities.append("未确定分析指标")

    time_field = None
    if len(metrics) == 1:
        time_field = METRIC_CATALOG[metrics[0]].time_field
    elif years:
        fields = {METRIC_CATALOG[item].time_field for item in metrics}
        if len(fields) == 1:
            time_field = next(iter(fields))
        else:
            ambiguities.append("多个指标使用不同时间字段")

    data_source = (
        "support_ops"
        if any(METRIC_CATALOG[item].source_table.startswith("support_") for item in metrics)
        or any(term in text for term in ("客服工单", "服务工单", "客服团队", "坐席"))
        else "ai_analytics"
    )
    request = AnalysisRequest(
        metrics=metrics,
        dimensions=dimensions,
        time_scope=TimeScope(field=time_field, years=years),
        filters=filters,
        sort=sort,
        limit=limit,
        data_source=data_source,
        unresolved_ambiguities=list(dict.fromkeys(ambiguities)),
        unsupported_conditions=list(dict.fromkeys(unsupported)),
    )
    result = request.model_dump(mode="python")
    result["unsupported_conditions"] = list(dict.fromkeys(
        result["unsupported_conditions"] + analysis_request_semantic_issues(result)
    ))
    return AnalysisRequest.model_validate(result).model_dump(mode="python")


def inherit_analysis_request(
    previous: dict[str, Any] | None,
    current: dict[str, Any],
    utterance: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Apply explicit inherit/replace/clear rules and return a change log."""
    if not previous or any(
        marker in utterance for marker in ("新分析", "重新开始", "清除所有条件")
    ):
        return deepcopy(current), [{"field": "request", "action": "initialize"}]

    merged = deepcopy(previous)
    changes: list[dict[str, Any]] = []
    for field in ("metrics", "dimensions", "sort", "limit"):
        value = current.get(field)
        if value not in (None, [], {}):
            old = merged.get(field)
            merged[field] = deepcopy(value)
            if old != value:
                changes.append({"field": field, "action": "replace", "from": old, "to": value})

    if current.get("metrics") and current.get("data_source") != merged.get("data_source"):
        old = merged.get("data_source")
        merged["data_source"] = current["data_source"]
        changes.append({
            "field": "data_source", "action": "replace", "from": old,
            "to": current["data_source"],
        })
    if current.get("metrics"):
        metric_time_fields = {
            METRIC_CATALOG[item].time_field for item in current["metrics"]
        }
        if len(metric_time_fields) == 1:
            merged.setdefault("time_scope", {})["field"] = next(iter(metric_time_fields))

    current_time = current.get("time_scope") or {}
    if current_time.get("years") or current_time.get("start") or current_time.get("end"):
        old = merged.get("time_scope")
        merged["time_scope"] = deepcopy(current_time)
        if old != current_time:
            changes.append({"field": "time_scope", "action": "replace", "from": old, "to": current_time})

    merged_filters = deepcopy(merged.get("filters") or {})
    for key, value in (current.get("filters") or {}).items():
        old = merged_filters.get(key)
        merged_filters[key] = deepcopy(value)
        if old != value:
            changes.append({"field": f"filters.{key}", "action": "replace", "from": old, "to": value})
    if any(marker in utterance for marker in ("不限地区", "清除地区", "所有国家")):
        old = merged_filters.pop("country", None)
        if old is not None:
            changes.append({"field": "filters.country", "action": "clear", "from": old, "to": None})
    merged["filters"] = merged_filters

    merged["unresolved_ambiguities"] = list(dict.fromkeys(
        current.get("unresolved_ambiguities") or []
    ))
    merged["unsupported_conditions"] = list(dict.fromkeys(
        (current.get("unsupported_conditions") or [])
        + analysis_request_semantic_issues(merged)
    ))
    if merged.get("metrics"):
        merged["unresolved_ambiguities"] = [
            item for item in merged.get("unresolved_ambiguities", [])
            if item != "未确定分析指标"
        ]
    return AnalysisRequest.model_validate(merged).model_dump(mode="python"), changes
