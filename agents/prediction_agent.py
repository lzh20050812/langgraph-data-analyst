"""
Prediction Agent —— XGBoost 客户流失预测 + Prophet 销售趋势预测。

职责（纯计算，非 LLM Agent）：
1. 流失预测：从 customers + orders 提取特征 → XGBoost 训练 → 评估（AUC/F1/准确率）
   → 输出特征重要度和 Top-N 高风险客户
2. 销售预测：从 monthly_revenue 取历史月度营收 → Prophet 拟合 → 未来6个月预测
   → 评估（RMSE/MAPE/MAE）

这是 Phase 2 的核心预测 Agent，所有计算均为确定性规则。
输出写入 state["prediction_result"]，随后由 chart_renderer 转为 ECharts 配置。
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score, f1_score, accuracy_score, confusion_matrix,
    mean_squared_error, mean_absolute_error,
    precision_score, recall_score,
)
import xgboost as xgb

from agents.state import AgentState
from storage.db_adapter import get_available_adapter


# ============================================================
# XGBoost 流失预测
# ============================================================

def _build_churn_features(adapter) -> pd.DataFrame:
    """
    从数据库构建流失预测特征矩阵。

    特征维度：
    - 客户画像：age, membership_tier(one-hot), country(one-hot), gender
    - 行为特征：total_orders, total_spend_usd, avg_order_value_usd,
      days_since_last_purchase, reviews_given, avg_review_score,
      returns_made, wishlist_items, newsletter_subscribed
    - 订单聚合特征：从 orders 表按 customer_id 聚合
      avg_discount_pct, avg_delivery_days, total_returned,
      avg_session_minutes, avg_pages_viewed
    """
    # 客户主表
    cust_rows = adapter.execute_sql("SELECT * FROM customers")
    if not cust_rows:
        raise ValueError("customers 表无数据")
    df = pd.DataFrame(cust_rows)

    # 订单聚合特征
    order_agg_rows = adapter.execute_sql(
        "SELECT customer_id, "
        "AVG(discount_pct) AS avg_discount_pct, "
        "AVG(delivery_days) AS avg_delivery_days, "
        "SUM(returned) AS total_returned, "
        "AVG(session_duration_minutes) AS avg_session_minutes, "
        "AVG(pages_viewed_before_purchase) AS avg_pages_viewed "
        "FROM orders GROUP BY customer_id"
    )
    order_agg = pd.DataFrame(order_agg_rows)

    # 合并
    df = df.merge(order_agg, on="customer_id", how="left")
    for col in ["avg_discount_pct", "avg_delivery_days", "total_returned",
                "avg_session_minutes", "avg_pages_viewed"]:
        # MySQL nullable 列返回 object dtype，需要先转数值
        df[col] = pd.to_numeric(df[col], errors="coerce")
        df[col] = df[col].fillna(df[col].median() if df[col].notna().any() else 0)

    # 标签
    y = df["churned"].astype(int).values

    # 特征工程
    cat_cols = ["membership_tier", "country", "gender", "preferred_device"]
    num_cols = [
        "age", "total_orders", "total_spend_usd", "avg_order_value_usd",
        "days_since_last_purchase", "reviews_given", "avg_review_score",
        "returns_made", "wishlist_items", "newsletter_subscribed",
        "avg_discount_pct", "avg_delivery_days", "total_returned",
        "avg_session_minutes", "avg_pages_viewed",
    ]

    # One-hot 编码
    X_cat = pd.get_dummies(df[cat_cols], drop_first=True)
    X_num = df[num_cols].fillna(0)
    for col in X_num.columns:
        X_num[col] = pd.to_numeric(X_num[col], errors="coerce").fillna(0)
    X = pd.concat([X_num, X_cat], axis=1)

    return X, y, df["customer_id"].tolist(), X.columns.tolist()


def train_churn_model(adapter) -> dict:
    """
    训练 XGBoost 流失预测模型并评估。

    Returns:
        dict with keys: auc, f1, accuracy, confusion_matrix,
                        feature_importance, train_samples, test_samples,
                        top_risk_customers
    """
    X, y, customer_ids, feature_names = _build_churn_features(adapter)

    # 检查类别平衡
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

    # 三段划分：阈值只能在 validation 上选择，test 仅用于最终评估。
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val,
        y_train_val,
        test_size=0.20,
        random_state=42,
        stratify=y_train_val,
    )

    # 训练
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric="logloss",
    )
    model.fit(X_train, y_train)

    # 预测
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # ---- 默认阈值 (0.5) 评估 ----
    auc = roc_auc_score(y_test, y_prob)
    acc = accuracy_score(y_test, y_pred)
    precision_default = precision_score(y_test, y_pred, zero_division=0)
    recall_default = recall_score(y_test, y_pred, zero_division=0)
    f1_default = f1_score(y_test, y_pred, zero_division=0)
    cm_default = confusion_matrix(y_test, y_pred)

    # ---- 阈值优化（只在 validation 上最大化 F1） ----
    from sklearn.metrics import precision_recall_curve
    val_prob = model.predict_proba(X_val)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_val, val_prob)
    # 计算每个阈值下的 F1
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    validation_best_f1 = f1_scores[best_idx]

    # 用最优阈值重新预测
    y_pred_best = (y_prob >= best_threshold).astype(int)
    cm_best = confusion_matrix(y_test, y_pred_best)
    precision_best = precision_score(y_test, y_pred_best, zero_division=0)
    recall_best = recall_score(y_test, y_pred_best, zero_division=0)
    test_f1_best = f1_score(y_test, y_pred_best, zero_division=0)

    # 评估完成后用全量历史样本重训运营评分模型；该模型不参与上述测试指标。
    final_model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric="logloss",
    )
    final_model.fit(X, y)

    # 特征重要度（Top 15）
    importance = sorted(
        zip(feature_names, final_model.feature_importances_),
        key=lambda x: x[1], reverse=True
    )[:15]

    # Top-N 高风险客户（全量预测）
    all_prob = final_model.predict_proba(X)[:, 1]
    risk_df = pd.DataFrame({
        "customer_id": customer_ids,
        "churn_probability": np.round(all_prob, 4),
    })
    top_risk = risk_df.nlargest(20, "churn_probability").to_dict(orient="records")

    return {
        "model": "XGBoost",
        "positive_rate": round(n_pos / len(y), 4),
        "train_samples": len(X_train),
        "validation_samples": len(X_val),
        "test_samples": len(X_test),
        "auc": round(auc, 4),
        # 默认阈值 (0.5)
        "default_threshold": {
            "threshold": 0.5,
            "accuracy": round(acc, 4),
            "precision": round(precision_default, 4),
            "recall": round(recall_default, 4),
            "f1": round(f1_default, 4),
            "confusion_matrix": cm_default.tolist(),
        },
        # 最优阈值（最大化 F1）
        "optimal_threshold": {
            "threshold": round(float(best_threshold), 4),
            "precision": round(precision_best, 4),
            "recall": round(recall_best, 4),
            "f1": round(float(test_f1_best), 4),
            "validation_f1": round(float(validation_best_f1), 4),
            "selection_set": "validation",
            "confusion_matrix": cm_best.tolist(),
        },
        "feature_importance": [{"feature": f, "importance": round(v, 4)} for f, v in importance],
        "top_risk_customers": top_risk,
    }


# ============================================================
# Prophet 销售预测
# ============================================================

_SALES_MODEL_CACHE = {}


def train_sales_forecast(
    adapter,
    evidence_rows: list[dict] | None = None,
    forecast_months: int = 6,
) -> dict:
    """
    使用 Prophet 对月度营收做时序预测（未来6个月）。

    Returns:
        dict with keys: rmse, mape, mae, forecast (未来6个月),
                        historical (训练数据), model_params
    """
    required = {"year", "month", "revenue_usd"}
    rows = evidence_rows or []
    if not rows or not required.issubset(rows[0].keys()):
        rows = adapter.execute_sql(
            "SELECT year, month, revenue_usd FROM monthly_revenue ORDER BY year, month"
        )
    if not rows:
        return {"error": "monthly_revenue 表无数据"}

    df = pd.DataFrame(rows)
    # 构造日期列
    df["ds"] = pd.to_datetime(df["year"].astype(str) + "-" + df["month"].astype(str) + "-01")
    df["y"] = df["revenue_usd"].astype(float)
    df = df.sort_values("ds")

    # 划分训练/测试（最后6个月作为测试集）
    train_df = df.iloc[:-6].copy()
    test_df = df.iloc[-6:].copy()

    if len(train_df) < 12:
        # 数据太少，减少测试集
        split = max(len(df) * 3 // 4, 6)
        train_df = df.iloc[:split].copy()
        test_df = df.iloc[split:].copy()

    try:
        from prophet import Prophet
    except ImportError:
        return {"error": "prophet 未安装，请执行: pip install prophet"}

    forecast_months = max(1, min(int(forecast_months or 6), 36))
    cache_key = tuple(
        (str(row.ds.date()), round(float(row.y), 6))
        for row in df[["ds", "y"]].itertuples(index=False)
    )
    model = _SALES_MODEL_CACHE.get(cache_key)
    if model is None:
        model = Prophet(
            stan_backend="CMDSTANPY",
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            changepoint_prior_scale=0.05,
        )
        model.fit(train_df[["ds", "y"]])
        _SALES_MODEL_CACHE.clear()
        _SALES_MODEL_CACHE[cache_key] = model

    # 未来预测
    future = model.make_future_dataframe(
        periods=len(test_df) + forecast_months, freq="MS"
    )
    forecast = model.predict(future)

    # 评估（测试集）
    forecast_test = forecast.set_index("ds").loc[test_df["ds"].values]
    y_true = test_df["y"].values
    y_pred = forecast_test["yhat"].values

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100

    # 历史 + 未来6个月预测
    historical = [
        {"ds": str(r["ds"].date()), "y": float(r["y"])}
        for _, r in df.iterrows()
    ]
    future_forecast = [
        {
            "ds": str(r["ds"].date()),
            "yhat": round(float(r["yhat"]), 2),
            "yhat_lower": round(float(r["yhat_lower"]), 2),
            "yhat_upper": round(float(r["yhat_upper"]), 2),
        }
        for _, r in forecast.tail(len(test_df) + forecast_months).iterrows()
    ]

    return {
        "model": "Prophet",
        "rmse": round(rmse, 2),
        "mape_pct": round(mape, 2),
        "mae": round(mae, 2),
        "train_periods": len(train_df),
        "test_periods": len(test_df),
        "forecast_periods": len(test_df) + forecast_months,
        "requested_forecast_months": forecast_months,
        "historical": historical,
        "forecast": future_forecast,
    }


# ============================================================
# LangGraph 节点
# ============================================================

def prediction_agent_node(state: AgentState) -> AgentState:
    """
    Prediction Agent 的 LangGraph 节点函数。

    执行流失预测 + 销售预测，结果写入 state["prediction_result"]。
    """
    state["current_step"] = "prediction_agent"
    state["messages"].append("[Prediction Agent] 开始预测...")

    try:
        from agents.evidence import update_evidence

        adapter = get_available_adapter()
        task_plan = state.get("task_plan") or {}
        tools = list(task_plan.get("prediction_tools") or [])
        evidence = state.get("evidence") or {}
        evidence_rows = evidence.get("rows") or state.get("query_result") or []
        result = {
            "source": "task_plan",
            "tools_executed": [],
            "task_plan": task_plan,
        }

        if "churn_prediction" in tools:
            churn_result = train_churn_model(adapter)
            result["churn"] = churn_result
            result["tools_executed"].append("churn_prediction")
            optimal = churn_result.get("optimal_threshold", {})
            state["messages"].append(
                f"[Prediction Agent] 流失预测: AUC={churn_result.get('auc', 'N/A')}, "
                f"F1={optimal.get('f1', 'N/A')}"
            )

        if "sales_forecast" in tools:
            sales_result = train_sales_forecast(
                adapter,
                evidence_rows=evidence_rows,
                forecast_months=task_plan.get("prediction_horizon_months") or 6,
            )
            result["sales"] = sales_result
            result["tools_executed"].append("sales_forecast")
            state["messages"].append(
                f"[Prediction Agent] 销售预测: RMSE={sales_result.get('rmse', 'N/A')}, "
                f"MAPE={sales_result.get('mape_pct', 'N/A')}%"
            )

        state["prediction_result"] = result
        state["evidence"] = update_evidence(
            state.get("evidence"), prediction_results=result
        )

        state["messages"].append("[Prediction Agent] 预测完成。")

    except Exception as e:
        state["error"] = f"Prediction Agent 失败: {e}"
        state["messages"].append(f"[Prediction Agent] ERROR: {e}")

    return state
