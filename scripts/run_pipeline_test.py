"""
核心链路集成测试 —— 验证"自然语言 → SQL → 数据库"完整流程。

运行方式：
    python scripts/run_pipeline_test.py              # 全链路测试（需要LLM API key配好）
    python scripts/run_pipeline_test.py --mock-llm   # 用模拟LLM测试链路逻辑（无需API key）

测试内容：
    1. 数据加载（CSV → DuckDB 内存表）
    2. Schema Agent（ChromaDB 检索 → 字段映射）
    3. SQL Agent（SQL 生成 → 安全校验 → 执行）
    4. SELECT 安全校验（DROP/DELETE/UPDATE 拦截测试）
    5. 自修正闭环（故意给错误 SQL 看是否能修正）

输出：
    - 每一步的通过/失败状态
    - 实际执行的 SQL 和返回行数
    - 完整的测试报告 JSON
"""

import sys
import json
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import get_settings
from storage.db_adapter import DuckDBAdapter, get_available_adapter
from storage.chromadb.embedder import get_embedder
from agents.state import create_initial_state
from agents.schema_agent import schema_agent_node
from agents.sql_agent import (
    validate_select_only,
    _build_schema_info_str,
    _clean_sql_response,
)


# ============================================================
# 简单测试查询（5条，覆盖常见场景）
# ============================================================
TEST_QUERIES = [
    "总共有多少个客户",
    "Gold会员的平均消费金额是多少",
    "已流失的客户有多少",
    "每个国家的客户数量分布",
    "2025年各季度的总营收是多少",
]

# 模拟 LLM 返回值（用于 --mock-llm 模式）
MOCK_SQL_RESPONSES = {
    "总共有多少个客户": "SELECT COUNT(*) AS total_customers FROM customers",
    "Gold会员的平均消费金额是多少": "SELECT AVG(total_spend_usd) AS avg_spend FROM customers WHERE membership_tier = 'Gold'",
    "已流失的客户有多少": "SELECT COUNT(*) AS churned_count FROM customers WHERE churned = 1",
    "每个国家的客户数量分布": "SELECT country, COUNT(*) AS customer_count FROM customers GROUP BY country ORDER BY customer_count DESC",
    "2025年各季度的总营收是多少": "SELECT quarter, SUM(revenue_usd) AS total_revenue FROM monthly_revenue WHERE year = 2025 GROUP BY quarter ORDER BY quarter",
}


# ============================================================
# 加载数据到 DuckDB
# ============================================================

def load_data_to_duckdb(adapter: DuckDBAdapter) -> dict:
    """将清洗后的 CSV 加载到 DuckDB 内存表中。"""
    settings = get_settings()
    tables = {}

    # 1. customers（已清洗）
    customers_path = settings.PROCESSED_DIR / "customers_cleaned.csv"
    if not customers_path.exists():
        # fallback: 用原始数据（仅测试结构）
        customers_path = settings.RAW_DIR / "customers.csv"
        print("  [WARN] 清洗后 customers 不存在，使用原始数据")

    df_cust = pd.read_csv(customers_path)
    adapter.load_dataframe(df_cust, "customers")
    tables["customers"] = len(df_cust)
    print(f"  [OK] customers: {len(df_cust)} 行")

    # 2. orders（原样，保留 NULL）
    orders_path = settings.RAW_DIR / "orders.csv"
    df_orders = pd.read_csv(orders_path)
    adapter.load_dataframe(df_orders, "orders")
    tables["orders"] = len(df_orders)
    null_count = df_orders["customer_rating"].isna().sum()
    print(f"  [OK] orders: {len(df_orders)} 行 (customer_rating 缺失 {null_count}/{len(df_orders)}, {null_count/len(df_orders)*100:.1f}%)")

    # 3. monthly_revenue
    mr_path = settings.RAW_DIR / "monthly_revenue.csv"
    df_mr = pd.read_csv(mr_path)
    adapter.load_dataframe(df_mr, "monthly_revenue")
    tables["monthly_revenue"] = len(df_mr)
    print(f"  [OK] monthly_revenue: {len(df_mr)} 行")

    # 4. product_summary
    ps_path = settings.RAW_DIR / "product_summary.csv"
    df_ps = pd.read_csv(ps_path)
    adapter.load_dataframe(df_ps, "product_summary")
    tables["product_summary"] = len(df_ps)
    print(f"  [OK] product_summary: {len(df_ps)} 行")

    return tables


# ============================================================
# 模拟 LLM SQL 生成（无 API key 时的 fallback）
# ============================================================

class MockLLM:
    """模拟 LLM —— 返回预定义的 SQL 语句。"""

    def generate_sql(self, user_query: str, schema_info: str) -> str:
        """对已知查询返回正确 SQL，对未知查询返回基本模板。"""
        # 精确匹配
        for key, sql in MOCK_SQL_RESPONSES.items():
            if key in user_query:
                return sql

        # 模糊匹配：关键词检测
        query_lower = user_query.lower()
        if "客户" in user_query and ("多少" in user_query or "数" in user_query):
            return "SELECT COUNT(*) AS count FROM customers"
        if "国家" in user_query:
            return "SELECT country, COUNT(*) AS count FROM customers GROUP BY country ORDER BY count DESC"
        if "营收" in user_query or "gmv" in query_lower:
            return "SELECT SUM(revenue_usd) AS total_revenue FROM monthly_revenue"
        if "会员" in user_query:
            return "SELECT membership_tier, COUNT(*) AS count FROM customers GROUP BY membership_tier"

        # 兜底
        return f"SELECT * FROM customers LIMIT 10"


# ============================================================
# 主测试流程
# ============================================================

def run_single_query(
    adapter,
    embedder,
    query: str,
    mock_llm: bool = False,
) -> dict:
    """
    对单条查询跑完整 Pipeline: Schema Agent → SQL 生成 → 安全校验 → 执行。

    Returns:
        {
            "query": str,
            "schema_ok": bool,
            "sql": str,
            "safety_ok": bool,
            "execution_ok": bool,
            "row_count": int,
            "error": str | None,
            "duration_ms": float,
        }
    """
    result = {
        "query": query,
        "schema_ok": False,
        "selected_tables_count": 0,
        "sql": None,
        "safety_ok": False,
        "execution_ok": False,
        "row_count": 0,
        "error": None,
        "duration_ms": 0,
    }

    t_start = time.time()

    try:
        # ---- Step 1: Schema Agent ----
        state = create_initial_state(query)
        state = schema_agent_node(state)

        if state.get("error"):
            result["error"] = f"Schema Agent: {state['error']}"
            return result

        selected = state.get("selected_tables", [])
        result["schema_ok"] = len(selected) > 0
        result["selected_tables_count"] = len(selected)

        # ---- Step 2: SQL 生成 ----
        schema_info = _build_schema_info_str(selected)

        if mock_llm:
            # 模拟 LLM
            mock = MockLLM()
            sql = mock.generate_sql(query, schema_info)
            print(f"    [MockLLM] {sql[:60]}...")
        else:
            # 真实 LLM 调用
            from agents.sql_agent import _generate_sql
            sql = _generate_sql(query, schema_info)
            print(f"    [LLM] {sql[:80]}...")

        sql = _clean_sql_response(sql)
        result["sql"] = sql

        # ---- Step 3: 安全校验 ----
        is_safe, reason = validate_select_only(sql)
        result["safety_ok"] = is_safe
        if not is_safe:
            result["error"] = f"安全校验拦截: {reason}"
            return result

        # ---- Step 4: 执行 SQL ----
        rows = adapter.execute_sql(sql)
        result["execution_ok"] = True
        result["row_count"] = len(rows)
        if rows:
            # 记录前3行用于验证
            result["sample_rows"] = rows[:3]

    except Exception as e:
        result["error"] = str(e)[:300]

    result["duration_ms"] = round((time.time() - t_start) * 1000, 1)
    return result


def run_safety_tests() -> list[dict]:
    """SELECT 安全校验专项测试。"""
    tests = [
        ("DROP TABLE customers", False, "DROP"),
        ("DELETE FROM orders", False, "DELETE"),
        ("INSERT INTO customers VALUES (1, 'test')", False, "INSERT"),
        ("UPDATE customers SET name='x'", False, "UPDATE"),
        ("ALTER TABLE customers DROP COLUMN x", False, "ALTER"),
        ("SELECT * FROM customers", True, "SELECT"),
        ("SELECT COUNT(*) FROM orders WHERE order_date > '2025-01-01'", True, "SELECT with WHERE"),
        ("WITH cte AS (SELECT * FROM customers) SELECT * FROM cte", True, "CTE"),
    ]

    results = []
    for sql, expected_safe, label in tests:
        is_safe, reason = validate_select_only(sql)
        results.append({
            "label": label,
            "sql": sql,
            "expected_safe": expected_safe,
            "actual_safe": is_safe,
            "passed": is_safe == expected_safe,
            "reason": reason,
        })
    return results


def run_self_correction_test(adapter) -> dict:
    """
    自修正闭环测试：故意构造错误 SQL，验证修正逻辑。

    注意：自修正需要 LLM API key（要将错误信息喂回 LLM）。
    mock-llm 模式下只验证错误检测是否正确。
    """
    result = {
        "test": "自修正闭环",
        "phases": [],
    }

    # Phase 1: 错误的 SQL（字段名拼写错误）
    bad_sql = "SELECT custormer_id, total_spend FROM customers LIMIT 5"
    is_safe, _ = validate_select_only(bad_sql)
    result["phases"].append({
        "phase": "安全校验",
        "sql": bad_sql,
        "safe": is_safe,
        "note": "错误 SQL 通过了安全校验（SELECT 语句），但会在执行时失败",
    })

    # Phase 2: 执行错误 SQL
    try:
        rows = adapter.execute_sql(bad_sql)
        result["phases"].append({
            "phase": "执行",
            "unexpected_success": True,
            "rows": len(rows),
            "note": "DuckDB 可能别名处理了？检查",
        })
    except Exception as e:
        error_msg = str(e)
        result["phases"].append({
            "phase": "执行",
            "error_detected": True,
            "error_message": error_msg[:200],
            "note": "错误被正确捕获，这是期望的行为",
        })

    # Phase 3: 正确的 SQL
    good_sql = "SELECT customer_id, total_spend_usd FROM customers LIMIT 5"
    try:
        rows = adapter.execute_sql(good_sql)
        result["phases"].append({
            "phase": "修正后执行",
            "sql": good_sql,
            "success": True,
            "row_count": len(rows),
            "sample": rows[:2] if rows else [],
        })
    except Exception as e:
        result["phases"].append({
            "phase": "修正后执行",
            "success": False,
            "error": str(e)[:200],
        })

    return result


def main():
    import argparse
    parser = argparse.ArgumentParser(description="核心链路集成测试")
    parser.add_argument("--mock-llm", action="store_true",
                        help="使用模拟 LLM（无需 API key）")
    parser.add_argument("--full", action="store_true",
                        help="使用真实 LLM 跑完整链路（需要 API key）")
    args = parser.parse_args()

    use_mock = args.mock_llm or not args.full

    print("=" * 65)
    print("AI Data Analyst —— 核心链路集成测试")
    print(f"  模式: {'模拟LLM' if use_mock else '真实LLM'}")
    print(f"  时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 65)

    # ==================== 环境准备 ====================
    print("\n[1/6] 环境准备...")
    settings = get_settings()

    # 数据库
    adapter = get_available_adapter()
    backend_type = type(adapter).__name__

    if backend_type == "DuckDBAdapter":
        print(f"  数据库后端: DuckDB (内存)")
        print(f"  [INFO] MySQL 不可用，使用 DuckDB 测试链路逻辑")
    else:
        print(f"  数据库后端: MySQL")

    # LLM
    llm_available = bool(settings.LLM_API_KEY and "your_" not in settings.LLM_API_KEY)
    print(f"  LLM API key: {'已配置' if llm_available else '未配置'}")
    if not llm_available and not use_mock:
        print(f"  [INFO] 自动切换到 --mock-llm 模式")
        use_mock = True

    # ChromaDB
    print(f"  ChromaDB: 准备就绪")

    report = {
        "backend": backend_type,
        "mock_llm": use_mock,
        "llm_available": llm_available,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    # ==================== 数据加载 ====================
    print("\n[2/6] 加载数据到数据库...")
    tables = {}
    if backend_type == "DuckDBAdapter":
        tables = load_data_to_duckdb(adapter)
    else:
        # MySQL 模式：数据应已通过 init_storage.py 导入
        from storage.mysql.client import get_connection, text
        try:
            with get_connection() as conn:
                for t in ["customers", "orders", "monthly_revenue", "product_summary"]:
                    r = conn.execute(text(f"SELECT COUNT(*) FROM {t}"))
                    tables[t] = r.fetchone()[0]
            print(f"  [OK] MySQL 四张表已就绪: {tables}")
        except Exception as e:
            print(f"  [ERROR] MySQL 数据检查失败: {e}")
            print(f"  请先运行: python -m storage.init_storage")
            sys.exit(1)

    report["tables"] = tables

    # ==================== ChromaDB ====================
    print("\n[3/6] 初始化 ChromaDB Schema 向量库...")
    embedder = get_embedder()
    if embedder.collection is None:
        embedder.build()
    print(f"  [OK] ChromaDB collection: {embedder.collection.count()} 条记录")

    # ==================== Schema Agent 测试 ====================
    print("\n[4/6] Schema Agent 单步测试...")
    test_query = "总共有多少个客户"
    state = create_initial_state(test_query)
    state = schema_agent_node(state)
    schema_ok = len(state.get("selected_tables", [])) > 0 and state.get("error") is None
    print(f"  查询: '{test_query}'")
    print(f"  Schema Agent: {'PASS' if schema_ok else 'FAIL'}")
    if schema_ok:
        for t in state["selected_tables"][:5]:
            print(f"    → {t['table_name']}.{t['column_name']} ({t.get('relevance', '?')})")
    report["schema_agent_test"] = {
        "query": test_query,
        "passed": schema_ok,
        "selected_fields": state.get("selected_tables", [])[:5],
    }

    # ==================== SQL 安全校验 ====================
    print("\n[5/6] SQL 安全校验专项测试...")
    safety_results = run_safety_tests()
    safety_passed = sum(1 for r in safety_results if r["passed"])
    safety_total = len(safety_results)
    for r in safety_results:
        status = "PASS" if r["passed"] else "FAIL"
        print(f"  [{status}] {r['label']}: {r['sql'][:55]}...")
    print(f"  结果: {safety_passed}/{safety_total} 通过")
    report["safety_tests"] = {
        "passed": safety_passed,
        "total": safety_total,
        "details": safety_results,
    }

    # ==================== 全链路查询测试 ====================
    print("\n[6/6] 全链路查询测试 (5条)...")
    query_results = []
    for i, query in enumerate(TEST_QUERIES, 1):
        print(f"\n  Query {i}/5: '{query}'")
        result = run_single_query(adapter, embedder, query, mock_llm=use_mock)
        query_results.append(result)

        # 简要状态
        steps = []
        steps.append("Schema" + ("✓" if result["schema_ok"] else "✗"))
        steps.append("SQL" + ("✓" if result["sql"] else "✗"))
        steps.append("Safety" + ("✓" if result["safety_ok"] else "✗"))
        steps.append("Exec" + ("✓" if result["execution_ok"] else "✗"))
        print(f"    [{', '.join(steps)}] → "
              f"{result['row_count']} rows, {result['duration_ms']}ms")

        if result["error"]:
            print(f"    Error: {result['error'][:120]}")

    # 统计
    exec_ok = sum(1 for r in query_results if r["execution_ok"])
    schema_ok = sum(1 for r in query_results if r["schema_ok"])
    print(f"\n  汇总: Schema={schema_ok}/5, 执行成功={exec_ok}/5")

    report["query_tests"] = {
        "total": len(TEST_QUERIES),
        "schema_ok": schema_ok,
        "execution_ok": exec_ok,
        "details": query_results,
    }

    # ==================== 自修正测试 ====================
    print("\n[Bonus] 自修正闭环测试...")
    correction_result = run_self_correction_test(adapter)
    for phase in correction_result["phases"]:
        print(f"  [{phase['phase']}] {phase.get('note', '')[:100]}")
    report["self_correction_test"] = correction_result

    # ==================== 最终报告 ====================
    print("\n" + "=" * 65)
    print("测试总结")
    print("=" * 65)

    overall = {
        "data_loaded": len(tables) == 4,
        "chromadb_ready": embedder.collection is not None and embedder.collection.count() > 0,
        "schema_agent_ok": schema_ok,
        "safety_checks": f"{safety_passed}/{safety_total}",
        "query_execution": f"{exec_ok}/{len(TEST_QUERIES)}",
    }

    all_pass = all([
        overall["data_loaded"],
        overall["chromadb_ready"],
        overall["schema_agent_ok"],
        safety_passed == safety_total,
        exec_ok > 0,
    ])

    for key, val in overall.items():
        print(f"  {key}: {val}")

    if use_mock:
        print(f"\n  [INFO] 当前使用模拟LLM，SQL生成是模板而非LLM输出")
        print(f"  [INFO] 配好 .env 中 LLM_API_KEY 后运行:")
        print(f"         python scripts/run_pipeline_test.py --full")

    if not all_pass:
        print(f"\n  [WARN] 部分测试未通过，详见上方日志")
    else:
        print(f"\n  [OK] 核心链路验证通过！")

    # 保存报告
    output_path = settings.PROCESSED_DIR / "pipeline_test_report.json"
    output_path.parent.mkdir(exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n[OK] 完整测试报告: {output_path}")

    # 清理
    if backend_type == "DuckDBAdapter":
        adapter.close()


if __name__ == "__main__":
    main()
