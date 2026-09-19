"""
Prediction Agent —— 面向客户流失和销售趋势的候选模型自动选优。

职责（纯计算，非 LLM Agent）：
1. 流失预测：从 customers + orders 提取特征 → XGBoost/逻辑回归比较
   → 按验证集 AUC 选优并评估（AUC/F1/准确率）
   → 输出特征重要度和 Top-N 高风险客户
2. 销售预测：比较 Prophet、Last-value、Seasonal Naive 与 Drift
   → 按留出集 MAPE 选优并预测未来月份

这是 Phase 2 的核心预测 Agent，所有计算均为确定性规则。
输出写入 state["prediction_result"]，随后由 chart_renderer 转为 ECharts 配置。
"""

import numpy as np
import pandas as pd
import statistics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, f1_score, accuracy_score, confusion_matrix,
    mean_squared_error, mean_absolute_error,
    precision_score, recall_score, average_precision_score,
)
import xgboost as xgb

from agents.state import AgentState
from storage.db_adapter import get_available_adapter


# ============================================================
# XGBoost 流失预测
# ============================================================


def _new_xgboost(scale_pos_weight: float):
    return xgb.XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric="logloss",
    )


def _new_logistic_regression():
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=42,
        ),
    )


def _select_churn_candidate(validation_auc: dict[str, float]) -> str:
    """Select by validation AUC; prefer the simpler model on an exact tie."""
    if not validation_auc:
        raise ValueError("no churn candidate metrics were supplied")
    return max(
        validation_auc,
        key=lambda name: (
            validation_auc[name],
            name == "LogisticRegression",
        ),
    )


def _model_feature_importance(model, model_name: str) -> np.ndarray:
    if model_name == "XGBoost":
        return np.asarray(model.feature_importances_, dtype=float)
    classifier = model.named_steps["logisticregression"]
    return np.abs(np.asarray(classifier.coef_[0], dtype=float))


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


def _scope_risk_scores(
    risk_df: pd.DataFrame, scoring_customer_ids: set[str] | None
) -> tuple[pd.DataFrame, dict]:
    """Limit operational scores without changing the model-training cohort."""
    if scoring_customer_ids is None:
        return risk_df, {
            "source": "training_population",
            "requested_count": len(risk_df),
            "matched_count": len(risk_df),
        }
    scoped = risk_df[
        risk_df["customer_id"].astype(str).isin(scoring_customer_ids)
    ].copy()
    if scoped.empty:
        raise ValueError("当前请求范围与可评分客户没有交集")
    return scoped, {
        "source": "current_sql_evidence",
        "requested_count": len(scoring_customer_ids),
        "matched_count": len(scoped),
    }


def train_churn_model(
    adapter, *, scoring_customer_ids: set[str] | None = None
) -> dict:
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

    # 在 validation 集上比较复杂模型和可解释基线，test 只用于最终评估。
    candidates = {
        "XGBoost": _new_xgboost(scale_pos_weight),
        "LogisticRegression": _new_logistic_regression(),
    }
    validation_probabilities = {}
    validation_auc = {}
    validation_pr_auc = {}
    for name, candidate in candidates.items():
        candidate.fit(X_train, y_train)
        probabilities = candidate.predict_proba(X_val)[:, 1]
        validation_probabilities[name] = probabilities
        validation_auc[name] = float(roc_auc_score(y_val, probabilities))
        validation_pr_auc[name] = float(
            average_precision_score(y_val, probabilities)
        )

    selected_name = _select_churn_candidate(validation_auc)
    model = candidates[selected_name]

    # 预测
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    # ---- 默认阈值 (0.5) 评估 ----
    auc = roc_auc_score(y_test, y_prob)
    pr_auc = average_precision_score(y_test, y_prob)
    acc = accuracy_score(y_test, y_pred)
    precision_default = precision_score(y_test, y_pred, zero_division=0)
    recall_default = recall_score(y_test, y_pred, zero_division=0)
    f1_default = f1_score(y_test, y_pred, zero_division=0)
    cm_default = confusion_matrix(y_test, y_pred)

    # ---- 阈值优化（只在 validation 上最大化 F1） ----
    from sklearn.metrics import precision_recall_curve
    val_prob = validation_probabilities[selected_name]
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
    top_k_count = max(1, int(np.ceil(len(y_test) * 0.10)))
    top_k_indices = np.argsort(-y_prob)[:top_k_count]
    recall_at_top_10_pct = (
        float(y_test[top_k_indices].sum() / y_test.sum())
        if y_test.sum() else 0.0
    )

    # 评估完成后用全量历史样本重训运营评分模型；该模型不参与上述测试指标。
    final_model = (
        _new_xgboost(scale_pos_weight)
        if selected_name == "XGBoost"
        else _new_logistic_regression()
    )
    final_model.fit(X, y)

    # 特征重要度（Top 15）
    importance = sorted(
        zip(feature_names, _model_feature_importance(final_model, selected_name)),
        key=lambda x: x[1], reverse=True
    )[:15]

    # Top-N 高风险客户（全量预测）
    all_prob = final_model.predict_proba(X)[:, 1]
    risk_df = pd.DataFrame({
        "customer_id": customer_ids,
        "churn_probability": np.round(all_prob, 4),
    })
    scoped_risk_df, scoring_scope = _scope_risk_scores(
        risk_df, scoring_customer_ids
    )
    top_risk = scoped_risk_df.nlargest(
        20, "churn_probability"
    ).to_dict(orient="records")

    return {
        "model": selected_name,
        "selected_model": selected_name,
        "selection_metric": "validation_auc",
        "selection_set": "validation",
        "candidate_metrics": {
            name: {
                "validation_auc": round(value, 4),
                "validation_pr_auc": round(validation_pr_auc[name], 4),
            }
            for name, value in validation_auc.items()
        },
        "positive_rate": round(n_pos / len(y), 4),
        "train_samples": len(X_train),
        "validation_samples": len(X_val),
        "test_samples": len(X_test),
        "training_scope": {
            "source": "all_accessible_customers",
            "sample_count": len(X),
            "purpose": "model_training_and_evaluation",
        },
        "scoring_scope": scoring_scope,
        "auc": round(auc, 4),
        "pr_auc": round(float(pr_auc), 4),
        "recall_at_top_10_pct": round(recall_at_top_10_pct, 4),
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


def _forecast_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    nonzero = np.abs(y_true) > 1e-12
    if nonzero.any():
        relative_error = np.abs(
            (y_true[nonzero] - y_pred[nonzero]) / y_true[nonzero]
        )
        mape = float(np.mean(relative_error) * 100)
    else:
        mape = float("inf")
    return {"rmse": rmse, "mape_pct": mape, "mae": mae}


def _baseline_forecast_candidates(
    train_values: np.ndarray, periods: int
) -> dict[str, np.ndarray]:
    values = np.asarray(train_values, dtype=float)
    if not len(values):
        raise ValueError("forecast training data is empty")

    last_value = np.repeat(values[-1], periods)
    slope = (values[-1] - values[0]) / max(1, len(values) - 1)
    drift = values[-1] + slope * np.arange(1, periods + 1)

    seasonal_history = values.tolist()
    seasonal = []
    for _ in range(periods):
        prediction = (
            seasonal_history[-12]
            if len(seasonal_history) >= 12
            else seasonal_history[-1]
        )
        seasonal.append(prediction)
        seasonal_history.append(prediction)

    return {
        "LastValue": np.asarray(last_value, dtype=float),
        "SeasonalNaive": np.asarray(seasonal, dtype=float),
        "Drift": np.asarray(drift, dtype=float),
    }


def _prophet_forecast_candidate(train_df: pd.DataFrame, periods: int) -> dict | None:
    try:
        from prophet import Prophet
    except ImportError:
        return None

    try:
        cache_key = tuple(
            (str(row.ds.date()), round(float(row.y), 6))
            for row in train_df[["ds", "y"]].itertuples(index=False)
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
            while len(_SALES_MODEL_CACHE) >= 8:
                _SALES_MODEL_CACHE.pop(next(iter(_SALES_MODEL_CACHE)))
            _SALES_MODEL_CACHE[cache_key] = model

        future = model.make_future_dataframe(periods=periods, freq="MS")
        prediction = model.predict(future).tail(periods)
    except Exception:
        return None
    return {
        "values": prediction["yhat"].to_numpy(dtype=float),
        "lower": prediction["yhat_lower"].to_numpy(dtype=float),
        "upper": prediction["yhat_upper"].to_numpy(dtype=float),
    }


def _select_sales_candidate(candidate_metrics: dict[str, dict[str, float]]) -> str:
    """Choose the lowest validation MAPE, preferring simpler models on ties."""
    if not candidate_metrics:
        raise ValueError("no sales forecast candidate metrics were supplied")
    simplicity = {
        "LastValue": 0,
        "SeasonalNaive": 1,
        "Drift": 2,
        "Prophet": 3,
    }
    return min(
        candidate_metrics,
        key=lambda name: (
            candidate_metrics[name]["mape_pct"],
            simplicity.get(name, 99),
        ),
    )


def _forecast_candidates(
    train_df: pd.DataFrame, periods: int
) -> tuple[dict[str, np.ndarray], dict[str, tuple[np.ndarray, np.ndarray]]]:
    predictions = _baseline_forecast_candidates(
        train_df["y"].to_numpy(dtype=float), periods
    )
    bounds = {}
    prophet = _prophet_forecast_candidate(train_df, periods)
    if prophet is not None:
        predictions["Prophet"] = prophet["values"]
        bounds["Prophet"] = (prophet["lower"], prophet["upper"])
    return predictions, bounds


def _rolling_validation_metrics(
    selection_df: pd.DataFrame,
    validation_periods: int = 6,
    max_folds: int = 3,
) -> tuple[dict[str, dict[str, float]], list[dict]]:
    """Evaluate candidates on expanding-window rolling-origin folds."""
    validation_periods = max(1, min(validation_periods, len(selection_df) - 1))
    folds = []
    for offset in range(max_folds, 0, -1):
        validation_start = len(selection_df) - offset * validation_periods
        validation_end = validation_start + validation_periods
        if validation_start < 2 or validation_end > len(selection_df):
            continue
        folds.append((validation_start, validation_end))
    if not folds:
        validation_start = max(1, len(selection_df) // 2)
        folds = [(validation_start, len(selection_df))]

    per_model: dict[str, list[dict[str, float]]] = {}
    fold_details = []
    for fold_number, (validation_start, validation_end) in enumerate(folds, start=1):
        train_df = selection_df.iloc[:validation_start].copy()
        validation_df = selection_df.iloc[validation_start:validation_end].copy()
        predictions, _ = _forecast_candidates(train_df, len(validation_df))
        metrics = {
            name: _forecast_metrics(
                validation_df["y"].to_numpy(dtype=float), values
            )
            for name, values in predictions.items()
        }
        for name, values in metrics.items():
            per_model.setdefault(name, []).append(values)
        fold_details.append({
            "fold": fold_number,
            "train_periods": len(train_df),
            "validation_periods": len(validation_df),
            "validation_start": str(validation_df["ds"].iloc[0].date()),
            "validation_end": str(validation_df["ds"].iloc[-1].date()),
            "metrics": metrics,
        })

    complete_models = {
        name: values
        for name, values in per_model.items()
        if len(values) == len(fold_details)
    }
    aggregated = {}
    for name, values in complete_models.items():
        mapes = [item["mape_pct"] for item in values]
        aggregated[name] = {
            "mape_pct": float(statistics.mean(mapes)),
            "validation_mape_mean": float(statistics.mean(mapes)),
            "validation_mape_std": (
                float(statistics.pstdev(mapes)) if len(mapes) > 1 else 0.0
            ),
            "validation_rmse_mean": float(
                statistics.mean(item["rmse"] for item in values)
            ),
            "validation_mae_mean": float(
                statistics.mean(item["mae"] for item in values)
            ),
            "validation_folds": len(values),
        }
    return aggregated, fold_details


def train_sales_forecast(
    adapter,
    evidence_rows: list[dict] | None = None,
    forecast_months: int = 6,
) -> dict:
    """
    使用滚动验证比较 Prophet 与简单时序基线，再在独立最终测试集评估。

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

    forecast_months = max(1, min(int(forecast_months or 6), 36))
    if len(df) < 3:
        return {"error": "月度营收数据不足，至少需要3个周期"}

    # 最后最多6个月只用于最终测试，不参与模型选择。
    test_periods = 6 if len(df) >= 18 else max(1, len(df) // 4)
    train_df = df.iloc[:-test_periods].copy()
    test_df = df.iloc[-test_periods:].copy()
    total_periods = len(test_df) + forecast_months
    y_true = test_df["y"].values

    validation_periods = min(6, max(1, len(train_df) // 4))
    candidate_metrics, validation_folds = _rolling_validation_metrics(
        train_df,
        validation_periods=validation_periods,
        max_folds=3,
    )
    selected_name = _select_sales_candidate(candidate_metrics)
    candidate_predictions, candidate_bounds = _forecast_candidates(
        train_df, len(test_df)
    )
    test_candidate_metrics = {
        name: _forecast_metrics(y_true, predictions[:len(test_df)])
        for name, predictions in candidate_predictions.items()
    }
    if selected_name not in test_candidate_metrics:
        selected_name = _select_sales_candidate({
            name: candidate_metrics[name]
            for name in test_candidate_metrics
            if name in candidate_metrics
        })
    selected_metrics = test_candidate_metrics[selected_name]
    selected_test_predictions = candidate_predictions[selected_name]
    selected_test_lower, selected_test_upper = candidate_bounds.get(
        selected_name, (selected_test_predictions, selected_test_predictions)
    )

    # After honest final-holdout evaluation, refit only the validation-selected
    # model on all observed history for the actual future forecast.
    future_predictions, future_bounds = _forecast_candidates(df, forecast_months)
    selected_future_predictions = future_predictions[selected_name]
    selected_future_lower, selected_future_upper = future_bounds.get(
        selected_name,
        (selected_future_predictions, selected_future_predictions),
    )
    selected_predictions = np.concatenate([
        selected_test_predictions, selected_future_predictions
    ])
    selected_lower = np.concatenate([
        selected_test_lower, selected_future_lower
    ])
    selected_upper = np.concatenate([
        selected_test_upper, selected_future_upper
    ])
    forecast_dates = list(test_df["ds"]) + list(pd.date_range(
        start=df["ds"].iloc[-1] + pd.offsets.MonthBegin(1),
        periods=forecast_months,
        freq="MS",
    ))
    forecast_phases = ["backtest"] * len(test_df) + [
        "future"
    ] * forecast_months

    # 历史 + 未来6个月预测
    historical = [
        {"ds": str(r["ds"].date()), "y": float(r["y"])}
        for _, r in df.iterrows()
    ]
    future_forecast = [
        {
            "ds": str(date.date()),
            "yhat": round(float(value), 2),
            "yhat_lower": round(float(lower), 2),
            "yhat_upper": round(float(upper), 2),
            "phase": phase,
        }
        for date, value, lower, upper, phase in zip(
            forecast_dates,
            selected_predictions,
            selected_lower,
            selected_upper,
            forecast_phases,
        )
    ]

    return {
        "model": selected_name,
        "selected_model": selected_name,
        "selection_metric": "mape_pct",
        "selection_set": "rolling_validation",
        "test_set": "final_holdout",
        "candidate_metrics": {
            name: {
                metric: round(value, 2)
                for metric, value in metrics.items()
            }
            for name, metrics in candidate_metrics.items()
        },
        "test_candidate_metrics": {
            name: {
                metric: round(value, 2)
                for metric, value in metrics.items()
            }
            for name, metrics in test_candidate_metrics.items()
        },
        "validation_folds": validation_folds,
        "rmse": round(selected_metrics["rmse"], 2),
        "mape_pct": round(selected_metrics["mape_pct"], 2),
        "mae": round(selected_metrics["mae"], 2),
        "train_periods": len(train_df),
        "test_periods": len(test_df),
        "forecast_periods": total_periods,
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
            scoring_rows = [
                row for row in evidence_rows if row.get("customer_id") is not None
            ]
            if not scoring_rows:
                state["error"] = (
                    "Prediction Agent: 当前 SQL 证据不包含流失评分对象"
                )
                state["messages"].append(f"[Prediction Agent] ERROR: {state['error']}")
                return state
            from config.settings import get_settings
            if len(scoring_rows) >= get_settings().SQL_MAX_RESULT_ROWS:
                state["error"] = (
                    "Prediction Agent: 评分对象证据已达行数上限，"
                    "拒绝输出可能不完整的高风险客户名单"
                )
                state["messages"].append(f"[Prediction Agent] ERROR: {state['error']}")
                return state
            scoring_customer_ids = {
                str(row["customer_id"]) for row in scoring_rows
            }
            churn_result = train_churn_model(
                adapter, scoring_customer_ids=scoring_customer_ids
            )
            result["churn"] = churn_result
            result["tools_executed"].append("churn_prediction")
            optimal = churn_result.get("optimal_threshold", {})
            state["messages"].append(
                f"[Prediction Agent] 流失预测选用 "
                f"{churn_result.get('model', 'N/A')}: "
                f"AUC={churn_result.get('auc', 'N/A')}, "
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
                f"[Prediction Agent] 销售预测选用 "
                f"{sales_result.get('model', 'N/A')}: "
                f"RMSE={sales_result.get('rmse', 'N/A')}, "
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
