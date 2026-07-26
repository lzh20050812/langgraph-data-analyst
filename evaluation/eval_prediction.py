"""
Prediction Agent 模型评估脚本。

评估指标：
- XGBoost 流失预测：AUC / F1 / 准确率 / 混淆矩阵
- Prophet 销售预测：RMSE / MAPE / MAE

遵循 eval_schema / eval_sql 的评估报告格式：
  {"summary": {...}, "details": {...}}

数据来源：优先 MySQL，不可用时降级为 CSV 文件。
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import get_settings


def evaluate_churn_prediction() -> dict:
    """评估 XGBoost 流失预测模型。"""
    settings = get_settings()

    # 尝试从 MySQL 获取数据
    try:
        from storage.mysql.client import check_connection
        if check_connection():
            from storage.db_adapter import get_available_adapter
            adapter = get_available_adapter()
            from agents.prediction_agent import train_churn_model
            result = train_churn_model(adapter)
            return result
    except Exception as e:
        print(f"[WARN] MySQL 不可用，使用 CSV 数据评估流失预测: {e}")

    # Fallback: 从 CSV 加载
    customers_path = settings.RAW_DIR / "customers.csv"
    orders_path = settings.RAW_DIR / "orders.csv"
    if not customers_path.exists():
        return {"error": "customers.csv 不存在，无法评估"}

    from sklearn.model_selection import train_test_split
    from sklearn.metrics import (
    roc_auc_score, f1_score, accuracy_score, confusion_matrix,
    precision_score, recall_score,
)
    import xgboost as xgb

    customers = pd.read_csv(customers_path)
    orders = pd.read_csv(orders_path)

    # 订单聚合
    order_agg = orders.groupby("customer_id").agg(
        avg_discount_pct=("discount_pct", "mean"),
        avg_delivery_days=("delivery_days", "mean"),
        total_returned=("returned", "sum"),
        avg_session_minutes=("session_duration_minutes", "mean"),
        avg_pages_viewed=("pages_viewed_before_purchase", "mean"),
    ).reset_index()

    df = customers.merge(order_agg, on="customer_id", how="left")
    for col in ["avg_discount_pct", "avg_delivery_days", "total_returned",
                "avg_session_minutes", "avg_pages_viewed"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    y = df["churned"].values
    cat_cols = ["membership_tier", "country", "gender", "preferred_device"]
    num_cols = [
        "age", "total_orders", "total_spend_usd", "avg_order_value_usd",
        "days_since_last_purchase", "reviews_given", "avg_review_score",
        "returns_made", "wishlist_items", "newsletter_subscribed",
        "avg_discount_pct", "avg_delivery_days", "total_returned",
        "avg_session_minutes", "avg_pages_viewed",
    ]
    X_cat = pd.get_dummies(df[cat_cols], drop_first=True)
    X_num = df[num_cols].fillna(0)
    X = pd.concat([X_num, X_cat], axis=1)

    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    model = xgb.XGBClassifier(
        n_estimators=100, max_depth=5, learning_rate=0.1,
        scale_pos_weight=scale_pos_weight, random_state=42, eval_metric="logloss",
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # 默认阈值
    acc = accuracy_score(y_test, y_pred)
    precision_default = precision_score(y_test, y_pred)
    recall_default = recall_score(y_test, y_pred)
    f1_default = f1_score(y_test, y_pred)
    cm_default = confusion_matrix(y_test, y_pred)

    # 阈值优化
    from sklearn.metrics import precision_recall_curve
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5

    y_pred_best = (y_prob >= best_threshold).astype(int)
    cm_best = confusion_matrix(y_test, y_pred_best)
    precision_best = precision_score(y_test, y_pred_best)
    recall_best = recall_score(y_test, y_pred_best)
    best_f1 = f1_scores[best_idx]

    importance = sorted(
        zip(X.columns.tolist(), model.feature_importances_),
        key=lambda x: x[1], reverse=True
    )[:15]

    return {
        "model": "XGBoost",
        "data_source": "CSV (fallback)",
        "positive_rate": round(n_pos / len(y), 4),
        "train_samples": len(X_train),
        "test_samples": len(X_test),
        "auc": round(roc_auc_score(y_test, y_prob), 4),
        "default_threshold": {
            "threshold": 0.5,
            "accuracy": round(acc, 4),
            "precision": round(precision_default, 4),
            "recall": round(recall_default, 4),
            "f1": round(f1_default, 4),
            "confusion_matrix": cm_default.tolist(),
        },
        "optimal_threshold": {
            "threshold": round(float(best_threshold), 4),
            "precision": round(precision_best, 4),
            "recall": round(recall_best, 4),
            "f1": round(float(best_f1), 4),
            "confusion_matrix": cm_best.tolist(),
        },
        "feature_importance": [
            {"feature": f, "importance": round(v, 4)} for f, v in importance
        ],
    }


def evaluate_sales_forecast() -> dict:
    """评估 Prophet 销售预测模型。"""
    settings = get_settings()

    # 尝试从 MySQL
    try:
        from storage.mysql.client import check_connection
        if check_connection():
            from storage.db_adapter import get_available_adapter
            adapter = get_available_adapter()
            from agents.prediction_agent import train_sales_forecast
            result = train_sales_forecast(adapter)
            return result
    except Exception as e:
        print(f"[WARN] MySQL 不可用，使用 CSV 数据评估销售预测: {e}")

    # Fallback: CSV
    mr_path = settings.RAW_DIR / "monthly_revenue.csv"
    if not mr_path.exists():
        return {"error": "monthly_revenue.csv 不存在，无法评估"}

    df = pd.read_csv(mr_path)
    df["ds"] = pd.to_datetime(df["year"].astype(str) + "-" + df["month"].astype(str) + "-01")
    df["y"] = df["revenue_usd"].astype(float)
    df = df.sort_values("ds")

    split = max(len(df) * 3 // 4, 6)
    train_df = df.iloc[:split].copy()
    test_df = df.iloc[split:].copy()

    try:
        from prophet import Prophet
    except ImportError:
        return {"error": "prophet 未安装，请执行: pip install prophet"}

    model = Prophet(
        yearly_seasonality=True, weekly_seasonality=False,
        daily_seasonality=False, changepoint_prior_scale=0.05,
    )
    model.fit(train_df[["ds", "y"]])

    future = model.make_future_dataframe(periods=len(test_df) + 6, freq="MS")
    forecast = model.predict(future)

    forecast_test = forecast.set_index("ds").loc[test_df["ds"].values]
    y_true = test_df["y"].values
    y_pred = forecast_test["yhat"].values

    from sklearn.metrics import mean_squared_error, mean_absolute_error

    return {
        "model": "Prophet",
        "data_source": "CSV (fallback)",
        "rmse": round(np.sqrt(mean_squared_error(y_true, y_pred)), 2),
        "mape_pct": round(np.mean(np.abs((y_true - y_pred) / y_true)) * 100, 2),
        "mae": round(mean_absolute_error(y_true, y_pred), 2),
        "train_periods": len(train_df),
        "test_periods": len(test_df),
        "forecast_periods": len(test_df) + 6,
    }


def print_summary(report: dict) -> None:
    """打印可读的模型评估摘要（含阈值优化对比）。"""
    churn = report.get("churn", {})
    sales = report.get("sales", {})

    print("\n" + "=" * 60)
    print("Prediction Agent 模型评估")
    print("=" * 60)

    if churn and "error" not in churn:
        print(f"\n[XGBoost 流失预测] (数据源: {churn.get('data_source', 'MySQL')})")
        print(f"  训练样本: {churn.get('train_samples', 'N/A')}")
        print(f"  测试样本: {churn.get('test_samples', 'N/A')}")
        print(f"  正样本率: {churn.get('positive_rate', 'N/A'):.1%}")
        print(f"  AUC: {churn.get('auc', 'N/A')}")

        default = churn.get("default_threshold", {})
        optimal = churn.get("optimal_threshold", {})
        print(f"\n  默认阈值 0.5:")
        print(f"    准确率={default.get('accuracy', 'N/A')}  "
              f"精确率={default.get('precision', 'N/A')}  "
              f"召回率={default.get('recall', 'N/A')}  "
              f"F1={default.get('f1', 'N/A')}")
        print(f"    混淆矩阵: {default.get('confusion_matrix', 'N/A')}")

        print(f"\n  最优阈值 {optimal.get('threshold', 'N/A')} (最大化F1):")
        print(f"    精确率={optimal.get('precision', 'N/A')}  "
              f"召回率={optimal.get('recall', 'N/A')}  "
              f"F1={optimal.get('f1', 'N/A')}")
        print(f"    混淆矩阵: {optimal.get('confusion_matrix', 'N/A')}")

        print(f"\n  Top-5 特征重要度:")
        for f in churn.get("feature_importance", [])[:5]:
            print(f"    {f['feature']}: {f['importance']:.4f}")

    if sales and "error" not in sales:
        print(f"\n[Prophet 销售预测] (数据源: {sales.get('data_source', 'MySQL')})")
        print(f"  划分方式: 时间序列按先后顺序（前{sales.get('train_periods', '?')}月训练, "
              f"后{sales.get('test_periods', '?')}月测试）")
        print(f"  RMSE:  {sales.get('rmse', 'N/A')}")
        print(f"  MAPE:  {sales.get('mape_pct', 'N/A')}%")
        print(f"  MAE:   {sales.get('mae', 'N/A')}")


def main():
    print("=" * 60)
    print("Prediction Agent 模型评估")
    print("=" * 60)

    # 环境检查
    try:
        from storage.mysql.client import check_connection
        db_ok = check_connection()
        print(f"  MySQL: {'可用' if db_ok else '不可用，降级为 CSV'}")
    except Exception:
        db_ok = False
        print("  MySQL: 不可用，降级为 CSV")

    # 评估
    print("\n[1/2] 训练 XGBoost 流失预测模型...")
    churn = evaluate_churn_prediction()

    print("\n[2/2] 训练 Prophet 销售预测模型...")
    sales = evaluate_sales_forecast()

    report = {"churn": churn, "sales": sales}
    print_summary(report)

    # 保存报告
    output_dir = Path(__file__).resolve().parent.parent / "data" / "processed"
    output_dir.mkdir(exist_ok=True)
    report_path = output_dir / "eval_prediction_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n[OK] 评估报告已保存: {report_path}")


if __name__ == "__main__":
    main()
