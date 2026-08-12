"""Paired Text2SQL experiment for RAG and execution self-correction."""

from __future__ import annotations

import hashlib
import itertools
import json
import os
import time
from collections import Counter
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path

import numpy as np

from agents.schema_agent import schema_agent_node
from agents.sql_agent import sql_agent_node
from agents.state import create_initial_state
from config.settings import get_settings
from evaluation.metrics.text2sql_metrics import compare_result_sets
from storage.db_adapter import get_available_adapter


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "evaluation/data/text2sql/test_queries.json"
OUT = ROOT / "evaluation/results/text2sql_ablation_20260812"

AUDIT_CHANGES = {
    28: ("SELECT order_status, COUNT(*) AS cnt, COUNT(*) * 100.0 / SUM(COUNT(*)) OVER() AS pct FROM orders GROUP BY order_status",
         "题目明确要求数量和占比，原标准答案遗漏占比"),
    31: ("SELECT quarter, SUM(total_amount_usd) AS total_revenue, COUNT(*) AS total_orders FROM orders WHERE year=2025 GROUP BY quarter ORDER BY quarter",
         "统一以订单明细为事实源，避免预聚合表与明细口径并存"),
    35: ("SELECT customer_id, total_orders AS order_count, total_spend_usd AS total_spend FROM customers",
         "客户表已提供全客户累计口径，且能保留无订单客户的零值"),
    40: ("SELECT category, AVG(customer_rating) AS avg_rating FROM orders WHERE customer_rating IS NOT NULL GROUP BY category ORDER BY avg_rating DESC",
         "题目未要求客户数，原标准答案多出无关度量"),
    45: ("SELECT year, month, SUM(total_amount_usd) AS revenue, COUNT(*) AS orders, AVG(total_amount_usd) AS avg_order_value FROM orders WHERE year=2025 GROUP BY year,month ORDER BY revenue DESC LIMIT 1",
         "统一以订单明细计算营收、订单量和客单价"),
}
UNSCORABLE = {47: "customers 只有静态 churned 标签而没有 churn_date，无法按月归因流失"}


def norm(value):
    if value is None:
        return None
    if isinstance(value, (Decimal, float, np.floating)):
        return round(float(value), 5)
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return value


def tuples(rows):
    return [tuple(norm(v) for v in row.values()) for row in (rows or [])]


def content_equivalent(actual, expected):
    """Alias-insensitive equality allowing a non-conflicting extra projection."""
    if compare_result_sets(actual, expected):
        return True
    a, e = tuples(actual), tuples(expected)
    if len(a) != len(e):
        return False
    if not a and not e:
        return True
    aw, ew = len(a[0]), len(e[0])
    if aw == ew:
        return Counter(a) == Counter(e)
    small, large = (a, e) if aw < ew else (e, a)
    sw, lw = min(aw, ew), max(aw, ew)
    if lw > 12:
        return False
    target = Counter(small)
    for indexes in itertools.combinations(range(lw), sw):
        projected = Counter(tuple(row[i] for i in indexes) for row in large)
        if projected == target:
            return True
    return False


def first_attempt_sql(log):
    attempts = [entry for entry in log if entry.get("attempt") == 1]
    for entry in reversed(attempts):
        if entry.get("sql"):
            return entry["sql"], entry.get("phase") == "execute" and entry.get("result") == "SUCCESS"
    return None, False


def summarize(rows):
    scored = [r for r in rows if r["scorable"]]
    duration = [r["duration_seconds"] for r in scored]
    return {
        "n": len(scored), "strict_ex": round(sum(r["strict_correct"] for r in scored)/len(scored), 6),
        "content_ex": round(sum(r["content_correct"] for r in scored)/len(scored), 6),
        "execution_success": round(sum(r["execution_success"] for r in scored)/len(scored), 6),
        "initial_execution_success": round(sum(r["initial_execution_success"] for r in scored)/len(scored), 6),
        "initial_content_ex": round(sum(r["initial_content_correct"] for r in scored)/len(scored), 6),
        "mean_seconds": round(float(np.mean(duration)), 4),
        "median_seconds": round(float(np.median(duration)), 4),
        "p95_seconds": round(float(np.quantile(duration, .95)), 4),
        "total_retries": int(sum(r["retry_count"] for r in scored)),
    }


def mcnemar(a, b, key):
    pairs = [(x[key], y[key]) for x, y in zip(a, b) if x["scorable"] and y["scorable"]]
    b_only = sum((not x) and y for x, y in pairs)
    a_only = sum(x and (not y) for x, y in pairs)
    n = a_only + b_only
    if n == 0:
        p = 1.0
    else:
        from math import comb
        p = min(1.0, 2 * sum(comb(n, k) for k in range(min(a_only, b_only)+1)) / (2**n))
    return {"a_only": a_only, "b_only": b_only, "exact_two_sided_p": round(p, 6)}


def run_arm(name, tests, rag_disabled, adapter):
    os.environ["RAG_DISABLED"] = "1" if rag_disabled else "0"
    rows = []
    arm_dir = OUT / name
    arm_dir.mkdir(parents=True, exist_ok=True)
    for index, test in enumerate(tests, 1):
        qid = int(test["id"])
        expected_sql = AUDIT_CHANGES.get(qid, (test["expected_sql"], ""))[0]
        if qid in UNSCORABLE:
            rows.append({"id": qid, "query": test["query"], "scorable": False,
                         "unscorable_reason": UNSCORABLE[qid]})
            continue
        expected = adapter.execute_sql(expected_sql)
        state = create_initial_state(test["query"], requested_intent="sql_query")
        tick = time.perf_counter()
        state = schema_agent_node(state)
        if not state.get("error"):
            state = sql_agent_node(state)
        elapsed = time.perf_counter() - tick
        actual = state.get("query_result")
        log = state.get("sql_retry_log") or []
        initial_sql, initial_ok = first_attempt_sql(log)
        initial_rows = adapter.execute_sql(initial_sql) if initial_sql and initial_ok else None
        row = {"id": qid, "query": test["query"], "difficulty": test["difficulty"],
               "scorable": True, "expected_sql": expected_sql, "generated_sql": state.get("sql"),
               "execution_success": actual is not None, "strict_correct": compare_result_sets(actual, expected) if actual is not None else False,
               "content_correct": content_equivalent(actual, expected) if actual is not None else False,
               "initial_sql": initial_sql, "initial_execution_success": bool(initial_ok),
               "initial_content_correct": content_equivalent(initial_rows, expected) if initial_rows is not None else False,
               "retry_count": state.get("sql_retries", 0), "duration_seconds": round(elapsed, 4),
               "error": state.get("error"), "retry_log": log,
               "actual_sample": (actual or [])[:3], "expected_sample": expected[:3]}
        rows.append(row)
        (arm_dir / "checkpoint.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        print(f"[{name}] {index}/{len(tests)} Q{qid} exec={row['execution_success']} content={row['content_correct']} retries={row['retry_count']}", flush=True)
    return rows


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    tests = json.loads(DATA.read_text(encoding="utf-8"))
    settings = get_settings()
    settings.BUSINESS_SEMANTICS_ENABLED = False
    settings.SQL_MAX_RETRIES = 3
    adapter = get_available_adapter()
    rag = run_arm("rag", tests, False, adapter)
    full = run_arm("full_schema", tests, True, adapter)
    result = {"protocol": {"business_semantics": False, "max_retries": 3,
               "scoring": "strict plus alias/projection-tolerant answer-content equivalence",
               "dataset_sha256": hashlib.sha256(DATA.read_bytes()).hexdigest(),
               "audit_changes": {str(k): {"expected_sql": v[0], "reason": v[1]} for k,v in AUDIT_CHANGES.items()},
               "unscorable": {str(k): v for k,v in UNSCORABLE.items()}},
              "summaries": {"rag": summarize(rag), "full_schema": summarize(full)},
              "paired_rag_vs_full_schema": mcnemar(rag, full, "content_correct"),
              "self_correction_rag": {"initial_content_ex": summarize(rag)["initial_content_ex"],
                 "final_content_ex": summarize(rag)["content_ex"],
                 "recovered_queries": [r["id"] for r in rag if r.get("content_correct") and not r.get("initial_content_correct")]},
              "details": {"rag": rag, "full_schema": full}}
    (OUT / "text2sql_ablation_results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k:v for k,v in result.items() if k != "details"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
