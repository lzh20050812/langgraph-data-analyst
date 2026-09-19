from datetime import date

import pandas as pd

from agents import prediction_agent


def test_churn_selection_prefers_best_validation_auc_and_simpler_tie():
    assert prediction_agent._select_churn_candidate({
        "XGBoost": 0.72,
        "LogisticRegression": 0.69,
    }) == "XGBoost"
    assert prediction_agent._select_churn_candidate({
        "XGBoost": 0.72,
        "LogisticRegression": 0.72,
    }) == "LogisticRegression"


def test_sales_selection_uses_lowest_holdout_mape_and_simpler_tie():
    assert prediction_agent._select_sales_candidate({
        "Prophet": {"mape_pct": 12.0},
        "LastValue": {"mape_pct": 8.0},
    }) == "LastValue"
    assert prediction_agent._select_sales_candidate({
        "Prophet": {"mape_pct": 8.0},
        "LastValue": {"mape_pct": 8.0},
    }) == "LastValue"


def test_sales_forecast_falls_back_to_and_reports_baseline_selection(monkeypatch):
    monkeypatch.setattr(
        prediction_agent,
        "_prophet_forecast_candidate",
        lambda train_df, periods: None,
    )
    months = pd.date_range(date(2024, 1, 1), periods=24, freq="MS")
    rows = [
        {"year": value.year, "month": value.month, "revenue_usd": 100.0}
        for value in months
    ]

    result = prediction_agent.train_sales_forecast(
        adapter=object(), evidence_rows=rows, forecast_months=3
    )

    assert result["model"] == "LastValue"
    assert result["selection_set"] == "rolling_validation"
    assert result["test_set"] == "final_holdout"
    assert set(result["candidate_metrics"]) == {
        "LastValue", "SeasonalNaive", "Drift"
    }
    assert result["candidate_metrics"]["LastValue"]["validation_folds"] >= 1
    assert set(result["test_candidate_metrics"]) == {
        "LastValue", "SeasonalNaive", "Drift"
    }
    assert len(result["forecast"]) == result["test_periods"] + 3
    assert all(
        row["phase"] == "backtest"
        for row in result["forecast"][:result["test_periods"]]
    )
    assert all(
        row["phase"] == "future"
        for row in result["forecast"][-3:]
    )
