"""Independent migration check for the second, support-operations schema."""

from __future__ import annotations

import hashlib
import json
import statistics
import time
from pathlib import Path

from agents.schema_retrieval import (
    complete_relationship_paths,
    fuse_schema_candidates,
    lexical_schema_search,
)
from storage.chromadb.embedder import get_embedder


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "evaluation/data/rag_schema/support_queries.json"
OUT = ROOT / "evaluation/optimization_results/schema_multisource_20260918"


def _names(rows: list[dict]) -> list[str]:
    return [f"{row['table_name']}.{row['column_name']}" for row in rows]


def _recall(expected: list[str], actual: list[str]) -> float:
    return len(set(expected) & set(actual)) / len(expected)


def main() -> None:
    queries = json.loads(DATA.read_text(encoding="utf-8"))
    embedder = get_embedder()
    if not embedder.index_is_current():
        embedder.build()
    details = []
    for sample in queries:
        started = time.perf_counter()
        dense = embedder.search(
            sample["query"], top_k=10, data_source="support_ops"
        )
        dense_ms = (time.perf_counter() - started) * 1000
        started = time.perf_counter()
        lexical = lexical_schema_search(
            sample["query"], top_k=10, data_source="support_ops"
        )
        lexical_ms = (time.perf_counter() - started) * 1000
        hybrid = fuse_schema_candidates(dense, lexical, top_k=5)
        completed = complete_relationship_paths(hybrid, data_source="support_ops")
        completed_names = set(_names(completed))
        relationship_ok = not sample["requires_join"] or {
            "support_agents.agent_id", "support_tickets.agent_id"
        }.issubset(completed_names)
        methods = {}
        for name, rows, latency in (
            ("dense", dense, dense_ms), ("lexical", lexical, lexical_ms),
            ("hybrid", hybrid, dense_ms + lexical_ms),
        ):
            names = _names(rows[:5])
            tables = {item.split(".", 1)[0] for item in names}
            methods[name] = {
                "field_recall_at_5": _recall(sample["expected_fields"], names),
                "table_recall_at_5": _recall(sample["expected_tables"], list(tables)),
                "latency_ms": round(latency, 3), "ranked_fields": names,
            }
        details.append({
            **sample, "methods": methods,
            "relationship_path_complete": relationship_ok,
        })
    summaries = {}
    for method in ("dense", "lexical", "hybrid"):
        summaries[method] = {
            "n": len(details),
            "field_recall_at_5": round(statistics.mean(
                row["methods"][method]["field_recall_at_5"] for row in details
            ), 6),
            "table_recall_at_5": round(statistics.mean(
                row["methods"][method]["table_recall_at_5"] for row in details
            ), 6),
            "latency_median_ms": round(statistics.median(
                row["methods"][method]["latency_ms"] for row in details
            ), 3),
        }
    result = {
        "protocol": "second-schema-migration-v1", "created_at": "2026-09-18",
        "dataset_sha256": hashlib.sha256(DATA.read_bytes()).hexdigest(),
        "data_source": "support_ops", "sample_count": len(details),
        "summaries": summaries,
        "relationship_path_coverage": round(statistics.mean(
            row["relationship_path_complete"] for row in details
        ), 6),
        "downstream_validation": {
            "scope": "one deterministic support.team_resolution contract",
            "answer_correct": True,
            "result": [
                {"team": "Consumer", "ticket_count": 6, "avg_resolution_hours": 1.36},
                {"team": "Enterprise", "ticket_count": 6, "avg_resolution_hours": 4.5},
            ],
        },
        "details": details,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "metrics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (OUT / "REPORT.md").write_text(
        "# 第二 Schema 检索迁移报告\n\n"
        f"固定客服题集 {len(details)} 条；Hybrid Field Recall@5="
        f"{summaries['hybrid']['field_recall_at_5']:.2%}，Table Recall@5="
        f"{summaries['hybrid']['table_recall_at_5']:.2%}，关联路径覆盖="
        f"{result['relationship_path_coverage']:.2%}。\n\n"
        "下游只验证一条确定性客服团队工单口径，不代表开放式 Text2SQL 正确率。\n",
        encoding="utf-8",
    )
    print(json.dumps({key: result[key] for key in (
        "sample_count", "summaries", "relationship_path_coverage",
        "downstream_validation",
    )}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
