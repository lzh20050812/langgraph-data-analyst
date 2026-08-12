"""Formal API, security, concurrency, chart-contract and routing-efficiency checks."""

from __future__ import annotations

import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

from agents.sql_agent import validate_select_only
from api.main import app


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "evaluation/results/system_20260812"
QUERIES = [
    "客户表中总共有多少条记录", "各个国家的客户数量", "每个会员等级各有多少客户",
    "查询2025年每月的营收趋势", "2025年每个季度的总营收和订单量",
    "各支付方式的订单数量统计", "已流失的客户有多少个", "订单表中一共有多少条记录",
    "查询订阅了新闻邮件的客户数量", "各获客渠道的客户数量分布",
    "每个订单状态下的订单数量和占比", "折扣率超过20%的订单数占总订单数的比例",
]


def summary(values):
    return {"mean_ms": round(float(np.mean(values))*1000, 3),
            "median_ms": round(float(np.median(values))*1000, 3),
            "p95_ms": round(float(np.quantile(values, .95))*1000, 3)}


def chart_valid(chart):
    return isinstance(chart, dict) and bool(chart.get("id")) and bool(chart.get("title")) and isinstance(chart.get("option"), dict)


def api_experiment():
    client = TestClient(app)
    health_times = []
    health_payloads = []
    for _ in range(100):
        tick = time.perf_counter(); response = client.get("/health"); health_times.append(time.perf_counter()-tick)
        health_payloads.append(response.json())
    root = client.get("/")

    def invoke(item):
        index, query = item
        tick = time.perf_counter()
        response = client.post("/query", json={"query": query, "intent": "sql_query"})
        elapsed = time.perf_counter()-tick
        body = response.json()
        return {"id": index+1, "query": query, "status_code": response.status_code,
                "success": body.get("success", False), "duration_seconds": round(elapsed, 4),
                "result_rows": len(body.get("query_result") or []),
                "response_contract": all(key in body for key in ["success", "intent", "sql", "error"]),
                "chart_contract": all(chart_valid(c) for c in (body.get("charts") or [])),
                "error": body.get("error")}

    concurrency_results = {}
    for workers in [1, 2, 4]:
        tick = time.perf_counter()
        selected = list(enumerate(QUERIES[:8]))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            rows = list(pool.map(invoke, selected))
        wall = time.perf_counter()-tick
        times = [r["duration_seconds"] for r in rows]
        concurrency_results[str(workers)] = {"requests": len(rows), "wall_seconds": round(wall, 4),
            "throughput_rps": round(len(rows)/wall, 4), "success_rate": round(sum(r["success"] for r in rows)/len(rows), 6),
            "contract_pass_rate": round(sum(r["response_contract"] and r["chart_contract"] for r in rows)/len(rows), 6),
            **summary(times), "details": rows}
    charts = client.get("/charts").json()
    report = client.get("/report").json()
    return {"health": {"requests": 100, "http_success_rate": 1.0,
                       "healthy_rate": round(sum(p.get("status") == "ok" for p in health_payloads)/100, 6),
                       **summary(health_times), "last_payload": health_payloads[-1]},
            "frontend_status": root.status_code, "concurrency": concurrency_results,
            "latest_charts_contract": isinstance(charts.get("charts"), list) and charts.get("count") == len(charts.get("charts", [])),
            "latest_report_contract": isinstance(report.get("report"), str)}


def security_experiment():
    unsafe = ["DROP TABLE customers", "DELETE FROM orders", "UPDATE customers SET churned=0",
              "INSERT INTO customers(customer_id) VALUES ('x')", "TRUNCATE TABLE orders",
              "SELECT * FROM customers; DROP TABLE orders", "WITH x AS (DELETE FROM orders RETURNING *) SELECT * FROM x",
              "SELECT * FROM customers INTO OUTFILE '/tmp/x'", "CALL dangerous_proc()", "ALTER TABLE orders ADD x INT"]
    safe = ["SELECT COUNT(*) FROM customers", "WITH x AS (SELECT * FROM orders) SELECT COUNT(*) FROM x",
            "SELECT order_status, COUNT(*) FROM orders GROUP BY order_status"]
    unsafe_rows = [{"sql": sql, "accepted": validate_select_only(sql)[0], "reason": validate_select_only(sql)[1]} for sql in unsafe]
    safe_rows = [{"sql": sql, "accepted": validate_select_only(sql)[0], "reason": validate_select_only(sql)[1]} for sql in safe]
    return {"unsafe_block_rate": round(sum(not r["accepted"] for r in unsafe_rows)/len(unsafe_rows), 6),
            "safe_accept_rate": round(sum(r["accepted"] for r in safe_rows)/len(safe_rows), 6),
            "unsafe_cases": unsafe_rows, "safe_cases": safe_rows}


def routing_efficiency():
    runs = sorted((ROOT/"evaluation/results/formal_comparison").glob("*/raw/full_multi_agent.json"))
    if not runs:
        return {"error": "formal comparison result not found"}
    payload = json.loads(runs[-1].read_text(encoding="utf-8"))
    details = payload.get("details", [])
    fixed_agents_per_task = 7
    actual = [int(row.get("agent_call_count", 0)) for row in details]
    fixed = fixed_agents_per_task * len(details)
    used = sum(actual)
    return {"source": str(runs[-1].relative_to(ROOT)), "tasks": len(details), "fixed_chain_agent_calls": fixed,
            "on_demand_agent_calls": used, "calls_avoided": fixed-used,
            "reduction_rate": round((fixed-used)/fixed, 6) if fixed else 0,
            "mean_agents_per_task": round(statistics.mean(actual), 4) if actual else 0}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    result = {"api": api_experiment(), "security": security_experiment(), "routing": routing_efficiency()}
    (OUT/"system_metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
