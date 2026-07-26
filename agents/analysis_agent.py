"""
Analysis Agent —— RFM 客户价值分析 + K-Means 客户聚类 + 运营指标计算。

职责（纯计算，非 LLM Agent）：
1. RFM 分析：对每位客户计算 Recency / Frequency / Monetary 三项得分，
   按分位数分 4 段，生成 RFM 分段标签（如"高价值客户""流失风险客户"）
2. K-Means 聚类：基于 RFM + 行为特征做客户分群，用轮廓系数验证聚类质量
3. 运营指标：GMV / 客单价 / 复购率 / 品类分布 / 会员等级分布

这是 Phase 2 的核心分析 Agent，所有计算均为确定性规则（不调用 LLM）。
输出写入 state["analysis_result"]，随后由 chart_renderer 转为 ECharts 配置。
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

from agents.state import AgentState
from storage.db_adapter import get_available_adapter


# ============================================================
# RFM 分析
# ============================================================

def compute_rfm(customers_df: pd.DataFrame) -> pd.DataFrame:
    """
    计算每位客户的 RFM 得分。

    R (Recency):   days_since_last_purchase — 越小越好（最近才买过）
    F (Frequency): total_orders — 越大越好
    M (Monetary):  total_spend_usd — 越大越好

    每项按四分位数分为 1-4 分，4 为最优。
    返回带 r_score / f_score / m_score / rfm_segment 列的 DataFrame。
    """
    df = customers_df.copy()

    # ---- R 得分：值越小越好，所以反转分位数 ----
    try:
        df["r_score"] = pd.qcut(df["days_since_last_purchase"], q=4, labels=[4, 3, 2, 1])
    except ValueError:
        # 分位数边界重合时降级为 rank 分段
        df["r_score"] = pd.cut(
            df["days_since_last_purchase"].rank(pct=True), bins=4, labels=[4, 3, 2, 1]
        )
    df["r_score"] = df["r_score"].astype(int)

    # ---- F 得分：值越大越好 ----
    try:
        df["f_score"] = pd.qcut(df["total_orders"], q=4, labels=[1, 2, 3, 4])
    except ValueError:
        df["f_score"] = pd.cut(
            df["total_orders"].rank(pct=True), bins=4, labels=[1, 2, 3, 4]
        )
    df["f_score"] = df["f_score"].astype(int)

    # ---- M 得分：值越大越好 ----
    try:
        df["m_score"] = pd.qcut(df["total_spend_usd"], q=4, labels=[1, 2, 3, 4])
    except ValueError:
        df["m_score"] = pd.cut(
            df["total_spend_usd"].rank(pct=True), bins=4, labels=[1, 2, 3, 4]
        )
    df["m_score"] = df["m_score"].astype(int)

    # ---- RFM 综合标签 ----
    df["rfm_total"] = df["r_score"] + df["f_score"] + df["m_score"]

    def _label_rfm(total: int) -> str:
        if total >= 10:
            return "高价值客户"
        elif total >= 7:
            return "潜力客户"
        elif total >= 4:
            return "一般客户"
        else:
            return "流失风险客户"

    df["rfm_segment"] = df["rfm_total"].apply(_label_rfm)

    return df


# ============================================================
# K-Means 聚类
# ============================================================

def compute_kmeans(customers_df: pd.DataFrame, n_clusters: int = 4) -> dict:
    """
    基于客户特征做 K-Means 聚类。

    特征选取：RFM 三项 + age + avg_review_score + returns_made + wishlist_items
    标准化后降维到 2D 用于可视化，计算轮廓系数评估聚类质量。
    """
    feature_cols = [
        "days_since_last_purchase",
        "total_orders",
        "total_spend_usd",
        "age",
        "avg_review_score",
        "returns_made",
        "wishlist_items",
    ]

    # 过滤完整数据
    df = customers_df.dropna(subset=feature_cols).copy()
    if len(df) < n_clusters * 10:
        return {"error": f"可用于聚类的样本不足（{len(df)} 行）", "clusters": None}

    X = df[feature_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # K-Means
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)

    # 轮廓系数
    sil = silhouette_score(X_scaled, labels)

    # PCA 降维到 2D（用于可视化）
    pca = PCA(n_components=2, random_state=42)
    coords_2d = pca.fit_transform(X_scaled)

    # 按簇汇总特征均值
    df["cluster"] = labels
    cluster_profile = df.groupby("cluster")[feature_cols].mean().round(2)

    # 簇大小
    cluster_sizes = df["cluster"].value_counts().sort_index().to_dict()

    return {
        "n_clusters": n_clusters,
        "silhouette_score": round(sil, 4),
        "pca_variance_ratio": [round(v, 4) for v in pca.explained_variance_ratio_],
        "cluster_sizes": cluster_sizes,
        "cluster_profile": cluster_profile.to_dict(),
        "pca_coords_2d": coords_2d.tolist(),
        "labels": labels.tolist(),
        "sample_count": len(df),
    }


# ============================================================
# 运营指标
# ============================================================

def compute_metrics(adapter) -> dict:
    """
    从数据库计算核心运营指标。

    返回：
    - gmv: 总 GMV
    - avg_order_value: 平均客单价
    - repeat_purchase_rate: 复购率
    - churn_rate: 流失率
    - category_distribution: 品类收入分布
    - membership_distribution: 会员等级分布
    - monthly_revenue_trend: 月度营收趋势
    """
    metrics = {}

    # ---- GMV & 客单价 ----
    rows = adapter.execute_sql(
        "SELECT SUM(total_amount_usd) AS gmv, AVG(total_amount_usd) AS aov FROM orders"
    )
    if rows:
        metrics["gmv_usd"] = round(float(rows[0]["gmv"] or 0), 2)
        metrics["avg_order_value_usd"] = round(float(rows[0]["aov"] or 0), 2)

    # ---- 复购率 ----
    rows = adapter.execute_sql(
        "SELECT AVG(is_repeat_customer) AS repeat_rate FROM orders"
    )
    if rows:
        metrics["repeat_purchase_rate"] = round(float(rows[0]["repeat_rate"] or 0), 4)

    # ---- 流失率 ----
    rows = adapter.execute_sql(
        "SELECT AVG(churned) AS churn_rate FROM customers"
    )
    if rows:
        metrics["churn_rate"] = round(float(rows[0]["churn_rate"] or 0), 4)

    # ---- 品类收入分布 ----
    rows = adapter.execute_sql(
        "SELECT category, SUM(total_revenue_usd) AS revenue "
        "FROM product_summary GROUP BY category ORDER BY revenue DESC"
    )
    metrics["category_revenue"] = {
        r["category"]: round(float(r["revenue"]), 2) for r in rows
    }

    # ---- 会员等级分布 ----
    rows = adapter.execute_sql(
        "SELECT membership_tier, COUNT(*) AS cnt FROM customers GROUP BY membership_tier"
    )
    metrics["membership_distribution"] = {
        r["membership_tier"]: r["cnt"] for r in rows
    }

    # ---- 月度营收趋势 ----
    rows = adapter.execute_sql(
        "SELECT year, month, revenue_usd FROM monthly_revenue ORDER BY year, month"
    )
    metrics["monthly_revenue_trend"] = [
        {"year": r["year"], "month": r["month"], "revenue_usd": float(r["revenue_usd"])}
        for r in rows
    ]

    return metrics


# ============================================================
# LangGraph 节点
# ============================================================

def analysis_agent_node(state: AgentState) -> AgentState:
    """
    Analysis Agent 的 LangGraph 节点函数。

    从数据库读取客户数据，执行 RFM 分析 + K-Means 聚类 + 运营指标计算，
    结果写入 state["analysis_result"]。
    """
    state["current_step"] = "analysis_agent"
    state["messages"].append("[Analysis Agent] 开始分析...")

    try:
        adapter = get_available_adapter()

        # 1. 获取客户数据
        customers_rows = adapter.execute_sql(
            "SELECT customer_id, age, total_orders, total_spend_usd, "
            "avg_order_value_usd, days_since_last_purchase, avg_review_score, "
            "returns_made, wishlist_items, churned, membership_tier, country "
            "FROM customers"
        )
        if not customers_rows:
            state["error"] = "Analysis Agent: customers 表无数据"
            state["messages"].append(f"[Analysis Agent] ERROR: {state['error']}")
            return state

        customers_df = pd.DataFrame(customers_rows)
        state["messages"].append(
            f"[Analysis Agent] 获取 {len(customers_df)} 条客户数据"
        )

        # 2. RFM 分析
        rfm_df = compute_rfm(customers_df)
        rfm_summary = rfm_df["rfm_segment"].value_counts().to_dict()
        state["messages"].append(
            f"[Analysis Agent] RFM 分段: {rfm_summary}"
        )

        # 3. K-Means 聚类
        kmeans_result = compute_kmeans(customers_df)
        state["messages"].append(
            f"[Analysis Agent] K-Means 轮廓系数: {kmeans_result.get('silhouette_score', 'N/A')}"
        )

        # 4. 运营指标
        metrics = compute_metrics(adapter)
        state["messages"].append(
            f"[Analysis Agent] 运营指标: GMV=${metrics.get('gmv_usd', 0):,.0f}, "
            f"客单价=${metrics.get('avg_order_value_usd', 0):,.2f}"
        )

        # 组装输出
        state["analysis_result"] = {
            "rfm": {
                "segments": rfm_summary,
                "sample": rfm_df[
                    ["customer_id", "r_score", "f_score", "m_score", "rfm_total", "rfm_segment"]
                ].head(20).to_dict(orient="records"),
            },
            "kmeans": kmeans_result,
            "metrics": metrics,
        }

        state["messages"].append("[Analysis Agent] 分析完成。")

    except Exception as e:
        state["error"] = f"Analysis Agent 失败: {e}"
        state["messages"].append(f"[Analysis Agent] ERROR: {e}")

    return state
