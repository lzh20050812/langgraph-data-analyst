"""Formal Schema retrieval experiment: dense RAG vs lexical vs random."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from evaluation.framework.result import load_json
from storage.chromadb.schema_metadata import SCHEMA_FIELDS


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "evaluation/data/rag_schema/test_queries.json"
OUT = ROOT / "evaluation/results/rag_retrieval_20260812"
KS = (1, 3, 5, 10)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _field_name(row: Dict[str, Any]) -> str:
    return f"{row['table_name']}.{row['column_name']}"


def _schema_document(field: Dict[str, Any]) -> str:
    aliases = field.get("aliases") or []
    if isinstance(aliases, str):
        aliases = [aliases]
    return " ".join([
        field["table_name"], field["column_name"],
        field.get("business_term", ""), *aliases,
    ]).lower()


def _rank_dense(embedder, query: str) -> tuple[List[str], float]:
    tick = time.perf_counter()
    rows = embedder.search(query, top_k=10)
    elapsed = time.perf_counter() - tick
    return [_field_name(row) for row in rows], elapsed


def _rank_lexical(vectorizer, matrix, query: str) -> tuple[List[str], float]:
    tick = time.perf_counter()
    query_vector = vectorizer.transform([query])
    scores = (matrix @ query_vector.T).toarray().ravel()
    ranking = np.argsort(-scores, kind="stable")[:10]
    elapsed = time.perf_counter() - tick
    fields = [_field_name(SCHEMA_FIELDS[index]) for index in ranking]
    return fields, elapsed


def _rank_random(query_id: int) -> tuple[List[str], float]:
    rng = np.random.default_rng(42 + int(query_id))
    ranking = rng.choice(len(SCHEMA_FIELDS), size=10, replace=False)
    return [_field_name(SCHEMA_FIELDS[index]) for index in ranking], 0.0


def _sample_metrics(expected_tables, expected_fields, ranked_fields, latency):
    expected_tables = set(expected_tables)
    expected_fields = set(expected_fields)
    result: Dict[str, Any] = {"latency_seconds": latency}
    for k in KS:
        fields = ranked_fields[:k]
        tables = {field.split(".", 1)[0] for field in fields}
        field_set = set(fields)
        result[f"table_recall_at_{k}"] = (
            len(expected_tables & tables) / len(expected_tables) if expected_tables else 1.0
        )
        result[f"field_recall_at_{k}"] = (
            len(expected_fields & field_set) / len(expected_fields) if expected_fields else 1.0
        )
        result[f"field_precision_at_{k}"] = len(expected_fields & field_set) / len(fields)
        result[f"field_hit_at_{k}"] = bool(expected_fields & field_set)
    first_rank = next(
        (rank for rank, field in enumerate(ranked_fields, 1) if field in expected_fields),
        None,
    )
    result["reciprocal_rank_at_10"] = 1.0 / first_rank if first_rank else 0.0
    return result


def _aggregate(details: List[Dict[str, Any]], method: str) -> Dict[str, Any]:
    items = [detail["methods"][method] for detail in details]
    metrics: Dict[str, Any] = {"n": len(items)}
    for k in KS:
        for name in ("table_recall", "field_recall", "field_precision"):
            key = f"{name}_at_{k}"
            metrics[key] = round(statistics.mean(item[key] for item in items), 4)
        key = f"field_hit_at_{k}"
        metrics[key] = round(statistics.mean(bool(item[key]) for item in items), 4)
    metrics["mrr_at_10"] = round(
        statistics.mean(item["reciprocal_rank_at_10"] for item in items), 4
    )
    latencies = [item["latency_seconds"] for item in items]
    metrics["latency_mean_ms"] = round(statistics.mean(latencies) * 1000, 3)
    metrics["latency_median_ms"] = round(statistics.median(latencies) * 1000, 3)
    metrics["latency_p95_ms"] = round(float(np.percentile(latencies, 95)) * 1000, 3)
    return metrics


def _paired_analysis(details: List[Dict[str, Any]]) -> Dict[str, Any]:
    dense = np.array([
        detail["methods"]["dense"]["field_recall_at_5"] for detail in details
    ])
    lexical = np.array([
        detail["methods"]["lexical"]["field_recall_at_5"] for detail in details
    ])
    differences = dense - lexical
    rng = np.random.default_rng(42)
    boot = np.array([
        rng.choice(differences, size=len(differences), replace=True).mean()
        for _ in range(10000)
    ])

    dense_hit = np.array([
        detail["methods"]["dense"]["field_hit_at_5"] for detail in details
    ], dtype=bool)
    lexical_hit = np.array([
        detail["methods"]["lexical"]["field_hit_at_5"] for detail in details
    ], dtype=bool)
    dense_only = int(np.sum(dense_hit & ~lexical_hit))
    lexical_only = int(np.sum(~dense_hit & lexical_hit))
    discordant = dense_only + lexical_only
    if discordant:
        smaller = min(dense_only, lexical_only)
        tail = sum(math.comb(discordant, i) for i in range(smaller + 1)) / (2 ** discordant)
        mcnemar_p = min(1.0, 2 * tail)
    else:
        mcnemar_p = 1.0

    return {
        "field_recall_at_5_difference_dense_minus_lexical": round(float(differences.mean()), 4),
        "bootstrap_95_ci": [round(float(np.percentile(boot, 2.5)), 4), round(float(np.percentile(boot, 97.5)), 4)],
        "dense_only_field_hits_at_5": dense_only,
        "lexical_only_field_hits_at_5": lexical_only,
        "exact_mcnemar_p_two_sided": round(mcnemar_p, 6),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    tests = load_json(str(DATA))

    from storage.chromadb.embedder import get_embedder
    embedder = get_embedder()

    documents = [_schema_document(field) for field in SCHEMA_FIELDS]
    vectorizer = TfidfVectorizer(analyzer="char", ngram_range=(1, 3), lowercase=True)
    lexical_matrix = vectorizer.fit_transform(documents)

    details = []
    for test in tests:
        dense_fields, dense_time = _rank_dense(embedder, test["query"])
        lexical_fields, lexical_time = _rank_lexical(
            vectorizer, lexical_matrix, test["query"]
        )
        random_fields, random_time = _rank_random(test["id"])
        methods = {}
        for name, fields, latency in (
            ("dense", dense_fields, dense_time),
            ("lexical", lexical_fields, lexical_time),
            ("random", random_fields, random_time),
        ):
            methods[name] = {
                "ranked_fields": fields,
                **_sample_metrics(
                    test.get("expected_tables", []),
                    test.get("expected_fields", []),
                    fields,
                    latency,
                ),
            }
        details.append({
            "id": test["id"], "query": test["query"],
            "category": test.get("category"),
            "expected_tables": test.get("expected_tables", []),
            "expected_fields": test.get("expected_fields", []),
            "methods": methods,
        })

    summaries = {name: _aggregate(details, name) for name in ("dense", "lexical", "random")}
    paired = _paired_analysis(details)
    result = {
        "protocol": "strict-macro-recall-v1",
        "dataset_sha256": _sha256(DATA),
        "schema_field_count": len(SCHEMA_FIELDS),
        "sample_count": len(tests),
        "summaries": summaries,
        "paired_dense_vs_lexical": paired,
        "details": details,
    }
    (OUT / "rag_retrieval_results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    table_rows = []
    for method, metrics in summaries.items():
        table_rows.append({"method": method, **metrics})
    pd.DataFrame(table_rows).to_csv(
        OUT / "rag_retrieval_comparison.csv", index=False, encoding="utf-8-sig"
    )

    dense = summaries["dense"]
    lexical = summaries["lexical"]
    random = summaries["random"]
    report = f"""# RAG Schema检索正式实验

## 设计

40条业务问题、67个Schema字段。比较BGE稠密向量检索、字符n-gram TF-IDF
词法检索和固定随机基线。Recall按“检出的标准字段数/全部标准字段数”宏平均，
不再把命中任一字段误称为完整Recall。模型加载不计入单查询延迟。

## 结果

| 方法 | Field R@1 | Field R@3 | Field R@5 | Field R@10 | MRR@10 | P50延迟 |
|---|---:|---:|---:|---:|---:|---:|
| BGE Dense RAG | {dense['field_recall_at_1']:.2%} | {dense['field_recall_at_3']:.2%} | {dense['field_recall_at_5']:.2%} | {dense['field_recall_at_10']:.2%} | {dense['mrr_at_10']:.4f} | {dense['latency_median_ms']:.2f}ms |
| TF-IDF Lexical | {lexical['field_recall_at_1']:.2%} | {lexical['field_recall_at_3']:.2%} | {lexical['field_recall_at_5']:.2%} | {lexical['field_recall_at_10']:.2%} | {lexical['mrr_at_10']:.4f} | {lexical['latency_median_ms']:.2f}ms |
| Random | {random['field_recall_at_1']:.2%} | {random['field_recall_at_3']:.2%} | {random['field_recall_at_5']:.2%} | {random['field_recall_at_10']:.2%} | {random['mrr_at_10']:.4f} | - |

Dense相对Lexical的Field R@5差值为
{paired['field_recall_at_5_difference_dense_minus_lexical']:+.2%}，Bootstrap 95% CI
为[{paired['bootstrap_95_ci'][0]:+.2%}, {paired['bootstrap_95_ci'][1]:+.2%}]；
Hit@5配对McNemar p={paired['exact_mcnemar_p_two_sided']:.4f}。

## 边界

该实验只评价Schema候选检索，不等同于最终SQL准确率。RAG对Text2SQL的下游贡献
需在关闭业务语义层后，以相同LLM、执行器和自修正条件单独消融。
"""
    (OUT / "rag_retrieval_report.md").write_text(report, encoding="utf-8")
    print(json.dumps({"summaries": summaries, "paired": paired}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
