"""
数据治理 Agent（简化版） —— 运行时数据质量评分与异常标记。

职责（确定性规则，非 LLM Agent）：
1. 对 SQL Agent 返回的查询结果做轻量级质量评估（结果集层面）
2. 追溯源表，查询源表字段的真实缺失率（源表层面）—— 解决"GROUP BY 压缩了缺失率"的问题
3. 合并两层评估，输出质量分数（0-100）+ 告警列表 + 字段级详情

设计定位（工程职责边界）：
- ETL 层的数据清洗是"批处理"，处理的是已知的脏数据（格式不一致、重复等）
- 数据治理Agent 是"运行时"，回答"这批查询结果靠谱吗？"—— 因为 SQL 查出来的数据
  可能依然有问题（如 customer_rating 天然缺失 63%，这不是 ETL 能修的），
  而且聚合查询（GROUP BY）会把真实缺失率压缩到几行里，只看结果集看不出来。

简化范围：只做描述性统计 + 阈值规则判断，不做复杂异常检测算法。
"""

import re
from typing import List, Dict, Any, Optional
from agents.state import AgentState
from agents.evidence import update_evidence


# ============================================================
# 已知列 → 源表映射（基于项目 schema）
# ============================================================

_COLUMN_TABLE_MAP: Dict[str, str] = {
    # ---- customers 表 ----
    "customer_id": "customers",
    "country": "customers",
    "age": "customers",
    "gender": "customers",
    "membership_tier": "customers",
    "registration_date": "customers",
    "total_orders": "customers",
    "total_spend_usd": "customers",
    "avg_order_value_usd": "customers",
    "days_since_last_purchase": "customers",
    "preferred_category": "customers",
    "preferred_device": "customers",
    "preferred_payment_method": "customers",
    "acquisition_channel": "customers",
    "reviews_given": "customers",
    "avg_review_score": "customers",
    "returns_made": "customers",
    "wishlist_items": "customers",
    "newsletter_subscribed": "customers",
    "churned": "customers",
    # ---- orders 表 ----
    "order_id": "orders",
    "order_date": "orders",
    "total_amount_usd": "orders",
    "discount_pct": "orders",
    "delivery_days": "orders",
    "returned": "orders",
    "customer_rating": "orders",
    "is_repeat_customer": "orders",
    "device_used": "orders",
    "session_duration_minutes": "orders",
    "pages_viewed_before_purchase": "orders",
    # ---- monthly_revenue 表 ----
    "year": "monthly_revenue",
    "month": "monthly_revenue",
    "quarter": "monthly_revenue",
    "revenue_usd": "monthly_revenue",
    "order_count": "monthly_revenue",
    "new_customers": "monthly_revenue",
    "unique_customers": "monthly_revenue",
    # 注意: orders 和 monthly_revenue 都有 revenue_usd/order_count 等，以优先匹配为准
    # ---- product_summary 表 ----
    "category": "product_summary",
    "product_name": "product_summary",
    "total_revenue_usd": "product_summary",
    "avg_price": "product_summary",
    "avg_rating": "product_summary",
    "return_rate": "product_summary",
}


# ============================================================
# 结果集层面质量检查
# ============================================================

def _compute_column_quality(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    对查询结果逐列计算质量指标（结果集层面）。

    Returns:
        {column_name: {total, null_count, null_rate, unique_count, ...}, ...}
    """
    if not rows:
        return {}

    columns = list(rows[0].keys())
    total = len(rows)
    col_quality = {}

    for col in columns:
        values = [row.get(col) for row in rows]
        null_count = sum(1 for v in values if v is None)
        null_rate = null_count / total

        non_null = [v for v in values if v is not None]
        unique_count = len(set(str(v) for v in non_null)) if non_null else 0
        unique_rate = unique_count / len(non_null) if non_null else 0

        # 类型推断
        numeric_vals = []
        for v in non_null:
            try:
                numeric_vals.append(float(v))
            except (ValueError, TypeError):
                pass
        dtype = "numeric" if len(numeric_vals) > len(non_null) * 0.8 else "categorical"
        if len(numeric_vals) == 0:
            dtype = "categorical"

        has_negative = any(v < 0 for v in numeric_vals) if numeric_vals else False
        all_same = unique_count <= 1 and len(non_null) > 0

        col_quality[col] = {
            "total": total,
            "null_count": null_count,
            "null_rate": round(null_rate, 4),
            "unique_count": unique_count,
            "unique_rate": round(unique_rate, 4),
            "dtype": dtype,
            "has_negative": has_negative,
            "all_same": all_same,
        }

    return col_quality


# ============================================================
# 源表层面质量检查（解决 GROUP BY 压缩缺失率的问题）
# ============================================================

def _query_source_table_quality(
    adapter, columns: List[str], result_col_quality: Dict[str, Any]
) -> Dict[str, Any]:
    """
    对结果集中涉及的列，回溯源表查询真实缺失率。

    核心逻辑：
    - 对每个字段，通过 _COLUMN_TABLE_MAP 找到它所属的源表
    - 对每个源表，查询其总行数和各字段的 NULL 行数
    - 如果结果集缺失率与源表缺失率差异显著，说明查询中存在过滤/聚合，
      结果集不能反映真实数据质量

    Returns:
        {column_name: {source_table, source_total, source_null_count,
                       source_null_rate, result_null_rate, discrepancy, ...}, ...}
    """
    if not columns or not adapter:
        return {}

    # 找到需要查询的源表及其列
    table_columns: Dict[str, List[str]] = {}
    col_to_table: Dict[str, str] = {}

    for col in columns:
        table = _COLUMN_TABLE_MAP.get(col)
        if table and col not in col_to_table:
            col_to_table[col] = table
            if table not in table_columns:
                table_columns[table] = []
            table_columns[table].append(col)

    if not table_columns:
        return {}

    # 对每个源表查询列级缺失统计
    source_stats: Dict[str, Any] = {}

    for table, cols in table_columns.items():
        try:
            # 查询总行数
            count_rows = adapter.execute_sql(f"SELECT COUNT(*) AS cnt FROM `{table}`")
            if not count_rows:
                continue
            total = int(count_rows[0]["cnt"])

            # 对每列查询 NULL 数
            for col in cols:
                try:
                    null_rows = adapter.execute_sql(
                        f"SELECT COUNT(*) AS cnt FROM `{table}` WHERE `{col}` IS NULL"
                    )
                    null_count = int(null_rows[0]["cnt"]) if null_rows else 0
                except Exception:
                    null_count = 0

                result_null_rate = result_col_quality.get(col, {}).get("null_rate", 0)

                source_stats[col] = {
                    "source_table": table,
                    "source_total": total,
                    "source_null_count": null_count,
                    "source_null_rate": round(null_count / total, 4) if total > 0 else 0,
                    "result_null_rate": round(result_null_rate, 4),
                }

        except Exception:
            continue

    return source_stats


# ============================================================
# 整体评分（合并结果集 + 源表两层信息）
# ============================================================

def _compute_overall_score(
    col_quality: Dict[str, Any],
    row_count: int,
    source_stats: Dict[str, Any],
) -> dict:
    """
    综合结果集和源表两层信息计算质量分数（0-100）。

    评分规则：
    - 以源表缺失率为主要扣分依据（因为结果集可能被聚合压缩）
    - 如果源表数据不可得，降级用结果集缺失率
    """
    if not col_quality:
        return {"score": 0, "level": "无数据", "reasons": ["查询结果为空"]}

    score = 100
    reasons = []
    warnings = []

    # 行数检查（结果集层面）
    if row_count == 0:
        return {"score": 0, "level": "无数据", "reasons": ["查询结果为空（0行）"]}
    if row_count < 5:
        score -= 20
        reasons.append(f"样本量过小（仅 {row_count} 行），统计推断可信度低")

    # 以源表缺失率为准，逐列扣分
    null_penalties = []

    for col, detail in col_quality.items():
        # 优先用源表缺失率
        src = source_stats.get(col, {})
        if src and src.get("source_total", 0) > 0:
            nr = src["source_null_rate"]
            source_total = src["source_total"]
            source_nulls = src["source_null_count"]

            if nr > 0.5:
                null_penalties.append(35)
                warnings.append(
                    f"'{col}' 在源表 {src['source_table']} 中缺失 {nr*100:.1f}% "
                    f"（{source_nulls}/{source_total}），该维度结论可信度低"
                )
            elif nr > 0.2:
                null_penalties.append(15)
                warnings.append(
                    f"'{col}' 在源表 {src['source_table']} 中缺失 {nr*100:.1f}% "
                    f"（{source_nulls}/{source_total}），使用时需注意"
                )

            # 结果集 vs 源表差异标记（聚合压缩场景）
            result_nr = detail.get("null_rate", 0)
            if nr > 0.2 and result_nr < 0.05:
                reasons.append(
                    f"'{col}' 源表缺失率 {nr*100:.1f}% 远高于结果集 {result_nr*100:.1f}%，"
                    f"查询可能使用了聚合或过滤，结果集不能反映真实数据质量"
                )
        else:
            # 没有源表映射，降级用结果集缺失率
            nr = detail["null_rate"]
            if nr > 0.5:
                null_penalties.append(30)
                warnings.append(f"'{col}' 缺失 {nr*100:.1f}%，该维度结论可信度低")
            elif nr > 0.2:
                null_penalties.append(10)
                warnings.append(f"'{col}' 缺失 {nr*100:.1f}%，使用时需注意")

        # 全同列
        if detail.get("all_same"):
            score -= 5
            reasons.append(f"'{col}' 所有值相同，分析价值低")

    # 应用缺失扣分
    if null_penalties:
        null_penalties.sort(reverse=True)
        score -= null_penalties[0]  # 最严重的列
        score -= sum(null_penalties[1:]) * 0.3  # 其余列打折

    # 如果结果集是聚合的（行数 << 源表行数），追加说明
    if source_stats and row_count < 100:
        max_source = max(
            (s.get("source_total", 0) for s in source_stats.values()), default=0
        )
        if max_source > row_count * 10:
            reasons.append(
                f"结果集仅 {row_count} 行（聚合视图），以上评分基于源表统计而非结果集"
            )

    score = max(0, min(100, int(score)))

    if score >= 80:
        level = "良好"
    elif score >= 60:
        level = "一般"
    elif score >= 40:
        level = "较差"
    else:
        level = "不可信"

    return {
        "score": score,
        "level": level,
        "reasons": reasons,
        "warnings": warnings,
    }


# ============================================================
# LangGraph 节点
# ============================================================

def governance_agent_node(state: AgentState) -> AgentState:
    """
    数据治理 Agent 的 LangGraph 节点函数。

    两层评估：
    1. 结果集层面：对 query_result 做逐列质量统计
    2. 源表层面：回溯源表查询真实缺失率（解决 GROUP BY 压缩缺失率问题）

    结果写入 state["governance_result"]。
    """
    state["current_step"] = "governance_agent"
    state["messages"].append("[Governance Agent] 开始数据质量评估...")

    try:
        from storage.db_adapter import get_available_adapter

        query_result = state.get("query_result")

        # 如果 query_result 为空，无法评估
        if not query_result:
            state["governance_result"] = {
                "quality_score": "N/A",
                "overall": {"score": 0, "level": "无数据", "reasons": ["查询结果为空"], "warnings": []},
                "columns": {},
                "source_stats": {},
                "row_count": 0,
                "message": "无可评估的查询结果",
            }
            state["evidence"] = update_evidence(
                state.get("evidence"),
                data_quality=state["governance_result"],
            )
            state["messages"].append("[Governance Agent] 无可评估数据，跳过")
            return state

        # ---- 第 1 层：结果集质量 ----
        col_quality = _compute_column_quality(query_result)
        row_count = len(query_result)

        # ---- 第 2 层：源表质量（追溯真实缺失率） ----
        adapter = get_available_adapter()
        columns = list(col_quality.keys())
        source_stats = _query_source_table_quality(adapter, columns, col_quality)

        if source_stats:
            state["messages"].append(
                f"[Governance Agent] 已追溯 {len(source_stats)} 个字段的源表统计"
            )

        # ---- 综合评分 ----
        overall = _compute_overall_score(col_quality, row_count, source_stats)
        all_warnings = overall.pop("warnings", [])

        # 组装输出
        state["governance_result"] = {
            "quality_score": f"{overall['score']}/100 ({overall['level']})",
            "overall": overall,
            "columns": col_quality,
            "source_stats": source_stats,
            "row_count": row_count,
            "column_count": len(col_quality),
            "warnings": all_warnings,
        }
        state["evidence"] = update_evidence(
            state.get("evidence"),
            data_quality=state["governance_result"],
        )

        state["messages"].append(
            f"[Governance Agent] 质量评分: {overall['score']}/100 ({overall['level']}), "
            f"共 {len(all_warnings)} 条告警"
        )

    except Exception as e:
        state["error"] = f"Governance Agent 失败: {e}"
        state["messages"].append(f"[Governance Agent] ERROR: {e}")

    return state
