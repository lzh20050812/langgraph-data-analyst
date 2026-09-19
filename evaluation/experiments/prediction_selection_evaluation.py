"""Generate post-optimization artifacts for rolling forecast model selection."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(_PROJECT_ROOT / ".cache/matplotlib"))
os.environ.setdefault("MPLBACKEND", "Agg")

import pandas as pd

from agents.prediction_agent import train_sales_forecast


ROOT = _PROJECT_ROOT
DATA = ROOT / "data/raw/monthly_revenue.csv"
OUT = ROOT / "evaluation/optimization_results/prediction_selection_20260825"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _save_figure(result: dict) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
    os.environ.setdefault("MPLBACKEND", "Agg")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    candidates = list(result["candidate_metrics"])
    validation = [
        result["candidate_metrics"][name]["validation_mape_mean"]
        for name in candidates
    ]
    test = [
        result["test_candidate_metrics"][name]["mape_pct"]
        for name in candidates
    ]
    x = np.arange(len(candidates))
    width = 0.36
    fig, ax = plt.subplots(figsize=(10, 5.4))
    validation_bars = ax.bar(
        x - width / 2, validation, width, label="Rolling validation", color="#4F6EF7"
    )
    test_bars = ax.bar(
        x + width / 2, test, width, label="Final holdout", color="#FF9F43"
    )
    ax.bar_label(validation_bars, labels=[f"{value:.2f}%" for value in validation])
    ax.bar_label(test_bars, labels=[f"{value:.2f}%" for value in test])
    ax.set_title(
        f"Forecast Model Selection (selected: {result['selected_model']})"
    )
    ax.set_ylabel("MAPE (lower is better)")
    ax.set_xticks(x, candidates)
    ax.grid(axis="y", alpha=0.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "forecast_model_selection.png", dpi=220)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = pd.read_csv(DATA).to_dict(orient="records")
    result = train_sales_forecast(
        adapter=object(), evidence_rows=rows, forecast_months=6
    )
    artifact = {
        "protocol": "rolling-origin-selection-independent-holdout-v1",
        "created_at": "2026-08-25",
        "dataset_sha256": _sha256(DATA),
        "selected_model": result["selected_model"],
        "selection_set": result["selection_set"],
        "test_set": result["test_set"],
        "candidate_metrics": result["candidate_metrics"],
        "test_candidate_metrics": result["test_candidate_metrics"],
        "validation_folds": result["validation_folds"],
        "selected_test_metrics": {
            "rmse": result["rmse"],
            "mape_pct": result["mape_pct"],
            "mae": result["mae"],
        },
    }
    (OUT / "metrics.json").write_text(
        json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    comparison = []
    for model, validation in result["candidate_metrics"].items():
        comparison.append({
            "model": model,
            "selected": model == result["selected_model"],
            **validation,
            **{
                f"test_{key}": value
                for key, value in result["test_candidate_metrics"][model].items()
            },
        })
    pd.DataFrame(comparison).to_csv(
        OUT / "model_comparison.csv", index=False, encoding="utf-8-sig"
    )
    _save_figure(result)

    selected = result["selected_model"]
    selected_validation = result["candidate_metrics"][selected]
    selected_test = result["test_candidate_metrics"][selected]
    best_test = min(
        result["test_candidate_metrics"],
        key=lambda name: result["test_candidate_metrics"][name]["mape_pct"],
    )
    report = f"""# 销售预测滚动选模与独立测试

系统使用 {selected_validation['validation_folds']} 个扩展窗口做滚动验证，按平均 MAPE
选择 `{selected}`，验证 MAPE 为 {selected_validation['validation_mape_mean']:.2f}%
（标准差 {selected_validation['validation_mape_std']:.2f}）。

最终 6 个月完全不参与选模。`{selected}` 在该独立测试集上的 MAPE 为
{selected_test['mape_pct']:.2f}%；测试集事后最优模型是 `{best_test}`，MAPE 为
{result['test_candidate_metrics'][best_test]['mape_pct']:.2f}%。该差异用于如实展示
模型选择的不确定性，不能反向用测试集重选模型。

完成独立测试后，运行时会把验证阶段选定的模型在全部已观测历史上重新拟合，再生成
真正的未来预测。
"""
    (OUT / "REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps({
        "selected_model": selected,
        "selected_validation": selected_validation,
        "selected_test": selected_test,
        "best_test_model_for_audit_only": best_test,
        "output": str(OUT),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
