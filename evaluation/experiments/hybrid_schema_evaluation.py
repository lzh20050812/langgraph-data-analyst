"""Held-out evaluation for the runtime hybrid Schema retriever.

The validation split is used only to select RRF weights. The held-out test
split is evaluated once with the selected parameters, so the reported test
metrics are not the same samples used for tuning.
"""

from __future__ import annotations

import hashlib
import json
import os
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from agents.schema_retrieval import (
    fuse_schema_candidates,
    lexical_schema_search,
)


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "evaluation/data/rag_schema/test_queries.json"
OUT = ROOT / "evaluation/optimization_results/schema_hybrid_20260825"
KS = (1, 3, 5)
VECTOR_WEIGHTS = tuple(round(value, 1) for value in np.arange(0.1, 1.0, 0.1))
RRF_K_VALUES = (5, 10, 20, 40, 60)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _field_name(row: dict[str, Any]) -> str:
    return f"{row['table_name']}.{row['column_name']}"


def deterministic_category_split(
    queries: list[dict], validation_stride: int = 4
) -> tuple[list[dict], list[dict]]:
    """Place every fourth sample per category into validation deterministically."""
    by_category: dict[str, list[dict]] = defaultdict(list)
    for query in sorted(queries, key=lambda item: int(item["id"])):
        by_category[str(query.get("category") or "uncategorized")].append(query)

    validation_ids = set()
    for items in by_category.values():
        for index, item in enumerate(items):
            if index % validation_stride == 0:
                validation_ids.add(item["id"])

    validation = [item for item in queries if item["id"] in validation_ids]
    test = [item for item in queries if item["id"] not in validation_ids]
    if not validation or not test:
        raise ValueError("validation/test split must contain samples in both sets")
    return validation, test


def _sample_metrics(expected_fields: list[str], ranked_fields: list[str]) -> dict:
    expected = set(expected_fields)
    metrics = {}
    for k in KS:
        retrieved = set(ranked_fields[:k])
        metrics[f"field_recall_at_{k}"] = (
            len(expected & retrieved) / len(expected) if expected else 1.0
        )
        metrics[f"field_precision_at_{k}"] = (
            len(expected & retrieved) / k if k else 0.0
        )
    first_rank = next(
        (
            rank
            for rank, field in enumerate(ranked_fields, start=1)
            if field in expected
        ),
        None,
    )
    metrics["reciprocal_rank_at_10"] = 1 / first_rank if first_rank else 0.0
    return metrics


def _aggregate(rows: list[dict], method: str) -> dict:
    samples = [row["methods"][method] for row in rows]
    result = {"n": len(samples)}
    for k in KS:
        for metric in ("field_recall", "field_precision"):
            name = f"{metric}_at_{k}"
            result[name] = round(
                statistics.mean(sample[name] for sample in samples), 6
            )
    result["mrr_at_10"] = round(
        statistics.mean(sample["reciprocal_rank_at_10"] for sample in samples),
        6,
    )
    latencies = [sample["latency_seconds"] for sample in samples]
    result["latency_median_ms"] = round(statistics.median(latencies) * 1000, 3)
    result["latency_p95_ms"] = round(float(np.percentile(latencies, 95)) * 1000, 3)
    return result


def _rank_all(embedder, queries: list[dict]) -> list[dict]:
    ranked = []
    for query in queries:
        dense_started = time.perf_counter()
        dense = embedder.search(query["query"], top_k=20)
        dense_latency = time.perf_counter() - dense_started

        lexical_started = time.perf_counter()
        lexical = lexical_schema_search(query["query"], top_k=20)
        lexical_latency = time.perf_counter() - lexical_started
        ranked.append({
            "query": query,
            "dense": dense,
            "lexical": lexical,
            "dense_latency_seconds": dense_latency,
            "lexical_latency_seconds": lexical_latency,
        })
    return ranked


def _evaluate_params(
    ranked: list[dict], vector_weight: float, rrf_k: int
) -> tuple[float, float]:
    recalls = []
    reciprocal_ranks = []
    for row in ranked:
        fused = fuse_schema_candidates(
            row["dense"],
            row["lexical"],
            top_k=10,
            vector_weight=vector_weight,
            lexical_weight=1 - vector_weight,
            rrf_k=rrf_k,
        )
        metrics = _sample_metrics(
            row["query"].get("expected_fields", []),
            [_field_name(item) for item in fused],
        )
        recalls.append(metrics["field_recall_at_5"])
        reciprocal_ranks.append(metrics["reciprocal_rank_at_10"])
    return statistics.mean(recalls), statistics.mean(reciprocal_ranks)


def select_hybrid_parameters(ranked_validation: list[dict]) -> dict:
    """Grid-search only the validation partition."""
    trials = []
    for vector_weight in VECTOR_WEIGHTS:
        for rrf_k in RRF_K_VALUES:
            recall, mrr = _evaluate_params(
                ranked_validation, vector_weight, rrf_k
            )
            trials.append({
                "vector_weight": vector_weight,
                "lexical_weight": round(1 - vector_weight, 1),
                "rrf_k": rrf_k,
                "validation_field_recall_at_5": round(recall, 6),
                "validation_mrr_at_10": round(mrr, 6),
            })
    trials.sort(
        key=lambda item: (
            -item["validation_field_recall_at_5"],
            -item["validation_mrr_at_10"],
            abs(item["vector_weight"] - 0.5),
            item["rrf_k"],
        )
    )
    return {"selected": trials[0], "trials": trials}


def _build_details(
    ranked: list[dict], vector_weight: float, rrf_k: int
) -> list[dict]:
    details = []
    for row in ranked:
        fusion_started = time.perf_counter()
        hybrid = fuse_schema_candidates(
            row["dense"],
            row["lexical"],
            top_k=10,
            vector_weight=vector_weight,
            lexical_weight=1 - vector_weight,
            rrf_k=rrf_k,
        )
        fusion_latency = time.perf_counter() - fusion_started
        method_rows = {
            "dense": (
                row["dense"], row["dense_latency_seconds"]
            ),
            "lexical": (
                row["lexical"], row["lexical_latency_seconds"]
            ),
            "hybrid": (
                hybrid,
                row["dense_latency_seconds"]
                + row["lexical_latency_seconds"]
                + fusion_latency,
            ),
        }
        methods = {}
        for name, (candidate_rows, latency) in method_rows.items():
            fields = [_field_name(item) for item in candidate_rows[:10]]
            methods[name] = {
                "ranked_fields": fields,
                "latency_seconds": latency,
                **_sample_metrics(
                    row["query"].get("expected_fields", []), fields
                ),
            }
        details.append({
            "id": row["query"]["id"],
            "query": row["query"]["query"],
            "category": row["query"].get("category"),
            "expected_fields": row["query"].get("expected_fields", []),
            "methods": methods,
        })
    return details


def _bootstrap_difference(
    details: list[dict], reference_method: str
) -> dict:
    differences = np.asarray([
        row["methods"]["hybrid"]["field_recall_at_5"]
        - row["methods"][reference_method]["field_recall_at_5"]
        for row in details
    ])
    rng = np.random.default_rng(20260825)
    bootstrap = np.asarray([
        rng.choice(differences, size=len(differences), replace=True).mean()
        for _ in range(10000)
    ])
    return {
        "reference_method": reference_method,
        "mean_difference": round(float(differences.mean()), 6),
        "bootstrap_95_ci": [
            round(float(np.percentile(bootstrap, 2.5)), 6),
            round(float(np.percentile(bootstrap, 97.5)), 6),
        ],
    }


def _save_figure(test_summaries: dict[str, dict]) -> None:
    os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".cache/matplotlib"))
    os.environ.setdefault("MPLBACKEND", "Agg")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    methods = ("dense", "lexical", "hybrid")
    labels = ("Dense", "Lexical", "Hybrid")
    colors = ("#4F6EF7", "#36CFC9", "#FF9F43")
    x = np.arange(len(KS))
    width = 0.24
    fig, ax = plt.subplots(figsize=(9, 5.2))
    for index, (method, label, color) in enumerate(zip(methods, labels, colors)):
        values = [
            test_summaries[method][f"field_recall_at_{k}"] for k in KS
        ]
        bars = ax.bar(
            x + (index - 1) * width,
            values,
            width,
            label=label,
            color=color,
        )
        ax.bar_label(bars, labels=[f"{value:.1%}" for value in values], fontsize=8)
    ax.set_title("Held-out Schema Retrieval Performance")
    ax.set_ylabel("Macro Field Recall")
    ax.set_xticks(x, [f"Recall@{k}" for k in KS])
    ax.set_ylim(0, 1.08)
    ax.grid(axis="y", alpha=0.2)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(OUT / "schema_hybrid_heldout.png", dpi=220)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    queries = json.loads(DATA.read_text(encoding="utf-8"))
    validation_queries, test_queries = deterministic_category_split(queries)

    from storage.chromadb.embedder import get_embedder

    embedder = get_embedder()
    if not embedder.index_is_current():
        embedder.build()

    validation_ranked = _rank_all(embedder, validation_queries)
    test_ranked = _rank_all(embedder, test_queries)
    selection = select_hybrid_parameters(validation_ranked)
    selected = selection["selected"]

    validation_details = _build_details(
        validation_ranked, selected["vector_weight"], selected["rrf_k"]
    )
    test_details = _build_details(
        test_ranked, selected["vector_weight"], selected["rrf_k"]
    )
    methods = ("dense", "lexical", "hybrid")
    validation_summaries = {
        method: _aggregate(validation_details, method) for method in methods
    }
    test_summaries = {
        method: _aggregate(test_details, method) for method in methods
    }
    reference_method = max(
        ("dense", "lexical"),
        key=lambda method: validation_summaries[method]["field_recall_at_5"],
    )
    paired = _bootstrap_difference(test_details, reference_method)

    result = {
        "protocol": "held-out-hybrid-schema-retrieval-v1",
        "created_at": "2026-08-25",
        "revalidated_at": "2026-09-18",
        "dataset_sha256": _sha256(DATA),
        "runtime_retriever_sha256": _sha256(
            ROOT / "agents/schema_retrieval.py"
        ),
        "split": {
            "strategy": "category-stratified deterministic 1-in-4 validation",
            "validation_ids": [row["id"] for row in validation_queries],
            "test_ids": [row["id"] for row in test_queries],
        },
        "parameter_selection": selection,
        "validation_summaries": validation_summaries,
        "test_summaries": test_summaries,
        "paired_hybrid_vs_validation_selected_baseline": paired,
        "validation_details": validation_details,
        "test_details": test_details,
    }
    (OUT / "metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    pd.DataFrame([
        {"method": method, **metrics}
        for method, metrics in test_summaries.items()
    ]).to_csv(OUT / "heldout_comparison.csv", index=False, encoding="utf-8-sig")
    _save_figure(test_summaries)

    dense = test_summaries["dense"]
    lexical = test_summaries["lexical"]
    hybrid = test_summaries["hybrid"]
    report = f"""# 混合 Schema 检索独立测试报告

本实验将固定 40 题按业务类别确定性拆分为 {len(validation_queries)} 条验证题和
{len(test_queries)} 条独立测试题。RRF 参数只在验证题上选择，测试题不参与调参。

选定参数：向量权重 {selected['vector_weight']:.1f}、词法权重
{selected['lexical_weight']:.1f}、RRF k={selected['rrf_k']}。

| 方法 | Field R@1 | Field R@3 | Field R@5 | MRR@10 | P50延迟 |
|---|---:|---:|---:|---:|---:|
| Dense | {dense['field_recall_at_1']:.2%} | {dense['field_recall_at_3']:.2%} | {dense['field_recall_at_5']:.2%} | {dense['mrr_at_10']:.4f} | {dense['latency_median_ms']:.2f} ms |
| Lexical | {lexical['field_recall_at_1']:.2%} | {lexical['field_recall_at_3']:.2%} | {lexical['field_recall_at_5']:.2%} | {lexical['mrr_at_10']:.4f} | {lexical['latency_median_ms']:.2f} ms |
| Hybrid | {hybrid['field_recall_at_1']:.2%} | {hybrid['field_recall_at_3']:.2%} | {hybrid['field_recall_at_5']:.2%} | {hybrid['mrr_at_10']:.4f} | {hybrid['latency_median_ms']:.2f} ms |

Hybrid 相对验证集选出的较强基线 `{reference_method}`，测试集 Field R@5
平均差值为 {paired['mean_difference']:+.2%}，Bootstrap 95% CI 为
[{paired['bootstrap_95_ci'][0]:+.2%}, {paired['bootstrap_95_ci'][1]:+.2%}]。

该结果是优化后的独立评测，不覆盖 2026-08-12 冻结实验包。
"""
    (OUT / "REPORT.md").write_text(report, encoding="utf-8")
    print(json.dumps({
        "selected": selected,
        "test_summaries": test_summaries,
        "paired": paired,
        "output": str(OUT),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
