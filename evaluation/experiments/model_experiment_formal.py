"""Formal RFM, clustering, churn and revenue-forecast experiments."""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    adjusted_rand_score, calinski_harabasz_score, davies_bouldin_score,
    f1_score, mean_absolute_error, mean_squared_error, precision_score,
    recall_score, roc_auc_score, silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

from agents.analysis_agent import compute_rfm
from agents.prediction_agent import _build_churn_features
from storage.db_adapter import get_available_adapter


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "evaluation/results/models_20260812"
SEED = 42


def metric_set(y_true, prob, threshold):
    pred = (prob >= threshold).astype(int)
    return {
        "auc": round(float(roc_auc_score(y_true, prob)), 6),
        "f1": round(float(f1_score(y_true, pred, zero_division=0)), 6),
        "precision": round(float(precision_score(y_true, pred, zero_division=0)), 6),
        "recall": round(float(recall_score(y_true, pred, zero_division=0)), 6),
        "threshold": round(float(threshold), 6),
    }


def choose_threshold(y_true, prob):
    candidates = np.unique(np.r_[0.0, prob, 1.0])
    scores = [f1_score(y_true, prob >= t, zero_division=0) for t in candidates]
    return float(candidates[int(np.argmax(scores))])


def churn_experiment(adapter):
    X, y, _, names = _build_churn_features(adapter)
    X = X.astype(float)
    indexes = np.arange(len(y))
    train_val, test = train_test_split(indexes, test_size=.25, random_state=SEED, stratify=y)
    train, val = train_test_split(train_val, test_size=.20, random_state=SEED, stratify=y[train_val])
    scaler = StandardScaler().fit(X.iloc[train])
    scaled = scaler.transform(X)
    ratio = (len(y[train]) - int(y[train].sum())) / max(int(y[train].sum()), 1)
    models = {
        "xgboost": xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=.1,
            scale_pos_weight=ratio, random_state=SEED, eval_metric="logloss"),
        "logistic": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=SEED),
        "dummy": DummyClassifier(strategy="prior", random_state=SEED),
    }
    details = {}
    test_probs = {}
    for name, model in models.items():
        model.fit(scaled[train], y[train])
        val_prob = model.predict_proba(scaled[val])[:, 1]
        test_prob = model.predict_proba(scaled[test])[:, 1]
        threshold = choose_threshold(y[val], val_prob)
        details[name] = metric_set(y[test], test_prob, threshold)
        details[name]["validation_f1"] = round(float(f1_score(y[val], val_prob >= threshold)), 6)
        test_probs[name] = test_prob
    rng = np.random.default_rng(SEED)
    diffs = []
    for _ in range(10000):
        sample = rng.integers(0, len(test), len(test))
        ys = y[test][sample]
        if len(np.unique(ys)) < 2:
            continue
        diffs.append(roc_auc_score(ys, test_probs["xgboost"][sample]) -
                     roc_auc_score(ys, test_probs["logistic"][sample]))
    details["paired_xgboost_vs_logistic"] = {
        "auc_difference": round(details["xgboost"]["auc"] - details["logistic"]["auc"], 6),
        "bootstrap_95_ci": [round(float(x), 6) for x in np.quantile(diffs, [.025, .975])],
    }
    details["protocol"] = {"train": len(train), "validation": len(val), "test": len(test),
        "positive_rate": round(float(np.mean(y)), 6), "feature_count": len(names)}
    return details


def rfm_cluster_experiment(adapter):
    rows = adapter.execute_sql("SELECT * FROM customers")
    customers = pd.DataFrame(rows)
    numeric = ["days_since_last_purchase", "total_orders", "total_spend_usd", "age",
               "avg_review_score", "returns_made", "wishlist_items"]
    for col in numeric:
        customers[col] = pd.to_numeric(customers[col], errors="coerce")
    rfm = compute_rfm(customers)
    grouped = rfm.groupby("rfm_segment", observed=True).agg(
        customers=("customer_id", "count"), avg_recency=("days_since_last_purchase", "mean"),
        avg_frequency=("total_orders", "mean"), avg_monetary=("total_spend_usd", "mean"),
        churn_rate=("churned", "mean")).round(4).reset_index()
    correlations = {col: round(float(rfm["rfm_total"].corr(rfm[col], method="spearman")), 6)
                    for col in ["total_orders", "total_spend_usd"]}
    correlations["days_since_last_purchase"] = round(float(
        rfm["rfm_total"].corr(rfm["days_since_last_purchase"], method="spearman")), 6)

    complete = customers.dropna(subset=numeric).copy()
    scaled = StandardScaler().fit_transform(complete[numeric])
    k_rows = []
    fitted = {}
    for k in range(2, 9):
        labels = KMeans(n_clusters=k, random_state=SEED, n_init=20).fit_predict(scaled)
        fitted[k] = labels
        k_rows.append({"k": k, "silhouette": round(float(silhouette_score(scaled, labels)), 6),
            "davies_bouldin": round(float(davies_bouldin_score(scaled, labels)), 6),
            "calinski_harabasz": round(float(calinski_harabasz_score(scaled, labels)), 6)})
    best_k = max(k_rows, key=lambda row: row["silhouette"])["k"]
    reference = fitted[best_k]
    aris = []
    for seed in range(10):
        labels = KMeans(n_clusters=best_k, random_state=seed, n_init=20).fit_predict(scaled)
        aris.append(adjusted_rand_score(reference, labels))
    complete["cluster"] = reference
    profiles = complete.groupby("cluster")[numeric].mean().round(3).reset_index()
    return {
        "sample_count": len(complete), "rfm_segments": grouped.to_dict("records"),
        "rfm_spearman": correlations, "k_selection": k_rows, "selected_k": best_k,
        "stability_ari_mean": round(float(np.mean(aris)), 6),
        "stability_ari_min": round(float(np.min(aris)), 6),
        "cluster_profiles": profiles.to_dict("records"),
    }


def forecast_metrics(actual, predicted):
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    return {"rmse": round(float(np.sqrt(mean_squared_error(actual, predicted))), 3),
            "mae": round(float(mean_absolute_error(actual, predicted)), 3),
            "mape_pct": round(float(np.mean(np.abs((actual-predicted)/actual))*100), 4)}


def forecast_experiment(adapter):
    from prophet import Prophet
    rows = adapter.execute_sql("SELECT year, month, revenue_usd FROM monthly_revenue ORDER BY year, month")
    df = pd.DataFrame(rows)
    df["ds"] = pd.to_datetime(df.year.astype(str) + "-" + df.month.astype(str) + "-01")
    df["y"] = pd.to_numeric(df.revenue_usd)
    train, test = df.iloc[:-6].copy(), df.iloc[-6:].copy()
    tick = time.perf_counter()
    model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False,
                    changepoint_prior_scale=.05)
    model.fit(train[["ds", "y"]])
    future = model.make_future_dataframe(periods=6, freq="MS")
    prophet_pred = model.predict(future).tail(6)["yhat"].to_numpy()
    elapsed = time.perf_counter() - tick
    seasonal = train["y"].iloc[-12:-6].to_numpy()
    last_value = np.repeat(train["y"].iloc[-1], 6)
    drift_step = (train["y"].iloc[-1] - train["y"].iloc[0]) / (len(train)-1)
    drift = train["y"].iloc[-1] + drift_step * np.arange(1, 7)
    actual = test["y"].to_numpy()
    return {"protocol": {"train_periods": len(train), "test_periods": 6},
        "prophet": {**forecast_metrics(actual, prophet_pred), "fit_predict_seconds": round(elapsed, 4)},
        "seasonal_naive": forecast_metrics(actual, seasonal),
        "last_value": forecast_metrics(actual, last_value), "drift": forecast_metrics(actual, drift),
        "holdout": [{"date": str(d.date()), "actual": round(float(a), 2),
                     "prophet": round(float(p), 2), "seasonal_naive": round(float(s), 2)}
                    for d, a, p, s in zip(test.ds, actual, prophet_pred, seasonal)]}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    adapter = get_available_adapter()
    result = {"seed": SEED, "rfm_clustering": rfm_cluster_experiment(adapter),
              "churn": churn_experiment(adapter), "revenue_forecast": forecast_experiment(adapter)}
    (OUT / "model_metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(result["rfm_clustering"]["k_selection"]).to_csv(OUT/"k_selection.csv", index=False)
    pd.DataFrame(result["rfm_clustering"]["rfm_segments"]).to_csv(OUT/"rfm_segments.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(result["revenue_forecast"]["holdout"]).to_csv(OUT/"forecast_holdout.csv", index=False)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
