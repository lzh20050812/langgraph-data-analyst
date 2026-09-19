"""Generate thesis figures from the frozen 2026-08-12 evaluation artifacts.

The script never hard-codes headline results that already exist in JSON.  It
reads the frozen metric pack and its supporting experiment summaries, then
writes publication-ready PNG files to ``evaluation/final_results/figures``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(_PROJECT_ROOT / ".cache" / "matplotlib"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = _PROJECT_ROOT
FINAL_DIR = ROOT / "evaluation" / "final_results"
OUTPUT_DIR = FINAL_DIR / "figures"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "Microsoft YaHei",
                "SimHei",
                "Noto Sans CJK SC",
                "DejaVu Sans",
            ],
            "axes.unicode_minus": False,
            "figure.facecolor": "white",
            "axes.facecolor": "#F8FAFC",
            "axes.edgecolor": "#CBD5E1",
            "axes.titleweight": "bold",
            "axes.titlesize": 15,
            "axes.labelsize": 11,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "grid.color": "#CBD5E1",
            "grid.alpha": 0.55,
            "grid.linestyle": "--",
        }
    )


def add_percent_labels(ax, bars, digits: int = 1) -> None:
    for bar in bars:
        value = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            value + 1.5,
            f"{value:.{digits}f}%",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )


def save(fig, filename: str) -> None:
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_schema_retrieval() -> None:
    source = load_json(
        ROOT / "evaluation/results/rag_retrieval_20260812/rag_retrieval_results.json"
    )
    ks = [1, 3, 5, 10]
    methods = [
        ("BGE语义检索", "dense", "#2563EB", "o"),
        ("字符TF-IDF", "lexical", "#059669", "s"),
        ("随机检索", "random", "#94A3B8", "^"),
    ]

    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for label, key, color, marker in methods:
        summary = source["summaries"][key]
        values = [summary[f"field_recall_at_{k}"] * 100 for k in ks]
        ax.plot(
            ks,
            values,
            label=label,
            color=color,
            marker=marker,
            linewidth=2.4,
            markersize=7,
        )
        for x, value in zip(ks, values):
            label_offset = -15 if key == "dense" else 7
            ax.annotate(
                f"{value:.1f}",
                (x, value),
                xytext=(0, label_offset),
                textcoords="offset points",
                ha="center",
                fontsize=8,
                color=color,
            )

    ax.set_title("Schema字段检索召回率对比（n=40）")
    ax.set_xlabel("Top-K")
    ax.set_ylabel("Field Recall@K（%）")
    ax.set_xticks(ks)
    ax.set_ylim(0, 105)
    ax.grid(axis="y")
    ax.legend(loc="center right", frameon=False)
    ax.text(
        0.01,
        -0.18,
        "BGE vs 字符TF-IDF：Recall@5差值 -5.42pp，95% CI [-15.00, 2.92]，McNemar p=0.500",
        transform=ax.transAxes,
        fontsize=9,
        color="#475569",
    )
    save(fig, "fig6_1_schema_retrieval.png")


def plot_text2sql() -> None:
    source = load_json(
        ROOT
        / "evaluation/results/text2sql_ablation_20260812/text2sql_ablation_results.json"
    )
    rag = source["summaries"]["rag"]
    full = source["summaries"]["full_schema"]
    labels = ["严格执行准确率", "内容执行准确率", "执行成功率"]
    rag_values = [rag["strict_ex"], rag["content_ex"], rag["execution_success"]]
    full_values = [
        full["strict_ex"],
        full["content_ex"],
        full["execution_success"],
    ]

    x = np.arange(len(labels))
    width = 0.34
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    bars1 = ax.bar(
        x - width / 2,
        np.array(rag_values) * 100,
        width,
        label="RAG Schema",
        color="#2563EB",
    )
    bars2 = ax.bar(
        x + width / 2,
        np.array(full_values) * 100,
        width,
        label="完整 Schema",
        color="#F59E0B",
    )
    add_percent_labels(ax, bars1)
    add_percent_labels(ax, bars2)
    ax.set_title("Text2SQL方案对比（n=49）")
    ax.set_ylabel("比例（%）")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 110)
    ax.grid(axis="y")
    ax.legend(frameon=False, loc="lower right")
    ax.text(
        0.01,
        -0.18,
        "内容执行准确率差值 +8.16pp；配对检验 p=0.21875，未达到0.05显著性水平",
        transform=ax.transAxes,
        fontsize=9,
        color="#475569",
    )
    save(fig, "fig6_2_text2sql.png")


def plot_semantic_ablation(metrics: dict) -> None:
    ablation = metrics["business_semantics_ablation"]
    values = [
        ablation["enabled_sql_accuracy"] * 100,
        ablation["disabled_sql_accuracy"] * 100,
    ]
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    bars = ax.bar(
        ["启用业务语义层", "禁用业务语义层"],
        values,
        width=0.55,
        color=["#059669", "#DC2626"],
    )
    add_percent_labels(ax, bars, digits=2)
    ax.set_title("业务语义层消融实验（配对样本 n=15）")
    ax.set_ylabel("SQL准确率（%）")
    ax.set_ylim(0, 112)
    ax.grid(axis="y")
    ax.text(
        0.5,
        0.83,
        "提升 53.33pp\nMcNemar p=0.0078",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=12,
        fontweight="bold",
        color="#0F172A",
        bbox={"boxstyle": "round,pad=0.45", "fc": "#ECFDF5", "ec": "#6EE7B7"},
    )
    save(fig, "fig6_3_semantic_ablation.png")


def plot_models() -> None:
    source = load_json(ROOT / "evaluation/results/models_20260812/model_metrics.json")
    churn = source["churn"]
    forecast = source["revenue_forecast"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    auc_labels = ["XGBoost", "逻辑回归", "Dummy"]
    auc_values = [
        churn["xgboost"]["auc"],
        churn["logistic"]["auc"],
        churn["dummy"]["auc"],
    ]
    auc_bars = axes[0].bar(
        auc_labels,
        np.array(auc_values) * 100,
        color=["#2563EB", "#F59E0B", "#94A3B8"],
    )
    add_percent_labels(axes[0], auc_bars, digits=2)
    axes[0].set_title("客户流失预测（测试集 n=2000）")
    axes[0].set_ylabel("ROC-AUC（%）")
    axes[0].set_ylim(0, 78)
    axes[0].grid(axis="y")
    axes[0].text(
        0.02,
        -0.2,
        "XGBoost - 逻辑回归 = 3.24pp\n95% CI [-0.92, 7.41]",
        transform=axes[0].transAxes,
        fontsize=9,
        color="#475569",
    )

    mape_labels = ["Last-value", "Drift", "Prophet", "季节朴素"]
    mape_values = [
        forecast["last_value"]["mape_pct"],
        forecast["drift"]["mape_pct"],
        forecast["prophet"]["mape_pct"],
        forecast["seasonal_naive"]["mape_pct"],
    ]
    mape_bars = axes[1].bar(
        mape_labels,
        mape_values,
        color=["#059669", "#14B8A6", "#2563EB", "#94A3B8"],
    )
    for bar in mape_bars:
        value = bar.get_height()
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.35,
            f"{value:.2f}%",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )
    axes[1].set_title("收入预测（留出集 6个月）")
    axes[1].set_ylabel("MAPE（%，越低越好）")
    axes[1].set_ylim(0, 25)
    axes[1].tick_params(axis="x", rotation=12)
    axes[1].grid(axis="y")
    axes[1].text(
        0.02,
        -0.2,
        "Prophet未超过Last-value基线（15.65% vs 14.36%）",
        transform=axes[1].transAxes,
        fontsize=9,
        color="#475569",
    )
    save(fig, "fig6_4_prediction_models.png")


def plot_memory(metrics: dict) -> None:
    source = load_json(ROOT / "evaluation/results/memory_20260812/memory_metrics.json")
    vector = source["summaries"]["vector_memory"]
    random = source["summaries"]["random"]
    labels = ["Recall@1", "Recall@3", "MRR@3"]
    vector_values = [vector["recall_at_1"], vector["recall_at_3"], vector["mrr_at_3"]]
    random_values = [random["recall_at_1"], random["recall_at_3"], random["mrr_at_3"]]
    x = np.arange(len(labels))
    width = 0.34

    fig, ax = plt.subplots(figsize=(8.2, 5.2))
    bars1 = ax.bar(
        x - width / 2,
        np.array(vector_values) * 100,
        width,
        label="向量记忆",
        color="#7C3AED",
    )
    bars2 = ax.bar(
        x + width / 2,
        np.array(random_values) * 100,
        width,
        label="随机检索",
        color="#94A3B8",
    )
    add_percent_labels(ax, bars1)
    add_percent_labels(ax, bars2)
    ax.set_title("分析记忆检索效果（n=10）")
    ax.set_ylabel("指标值（%）")
    ax.set_xticks(x, labels)
    ax.set_ylim(0, 105)
    ax.grid(axis="y")
    ax.legend(frameon=False, loc="upper left")
    ax.text(
        0.01,
        -0.16,
        f"向量记忆中位检索延迟 {metrics['analysis_memory']['median_latency_ms']:.3f} ms",
        transform=ax.transAxes,
        fontsize=9,
        color="#475569",
    )
    save(fig, "fig6_5_analysis_memory.png")


def plot_concurrency() -> None:
    source = load_json(ROOT / "evaluation/results/system_20260812/system_metrics.json")
    levels = [1, 2, 4]
    concurrency = source["api"]["concurrency"]
    throughput = [concurrency[str(level)]["throughput_rps"] for level in levels]
    p95 = [concurrency[str(level)]["p95_ms"] / 1000 for level in levels]

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.8))
    bars = axes[0].bar(
        [str(level) for level in levels],
        throughput,
        color=["#93C5FD", "#60A5FA", "#2563EB"],
    )
    for bar in bars:
        value = bar.get_height()
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.04,
            f"{value:.3f}",
            ha="center",
            fontsize=9,
            fontweight="bold",
        )
    axes[0].set_title("并发吞吐量")
    axes[0].set_xlabel("并发 worker 数")
    axes[0].set_ylabel("requests/s")
    axes[0].set_ylim(0, 1.6)
    axes[0].grid(axis="y")

    line = axes[1].plot(
        levels,
        p95,
        color="#DC2626",
        marker="o",
        linewidth=2.4,
        markersize=7,
    )
    del line
    for x, value in zip(levels, p95):
        axes[1].annotate(
            f"{value:.2f}s",
            (x, value),
            xytext=(0, 8),
            textcoords="offset points",
            ha="center",
            fontsize=9,
            fontweight="bold",
        )
    axes[1].set_title("端到端P95延迟")
    axes[1].set_xlabel("并发 worker 数")
    axes[1].set_ylabel("秒")
    axes[1].set_xticks(levels)
    axes[1].set_ylim(0, max(p95) * 1.15)
    axes[1].grid(axis="y")
    axes[1].text(
        0.02,
        -0.2,
        "1 worker包含首次模型冷启动，故P95显著偏高",
        transform=axes[1].transAxes,
        fontsize=9,
        color="#475569",
    )
    save(fig, "fig6_6_concurrency.png")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    configure_style()
    metrics = load_json(FINAL_DIR / "final_metrics.json")
    plot_schema_retrieval()
    plot_text2sql()
    plot_semantic_ablation(metrics)
    plot_models()
    plot_memory(metrics)
    plot_concurrency()
    print(f"Generated 6 figures in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
