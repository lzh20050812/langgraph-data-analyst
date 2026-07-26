"""
SQL Agent 生成准确率与执行成功率评估。

评估指标：
- 首次生成成功率（首次 SQL 即可执行成功）
- 自修正后成功率（含重试的最终成功率）
- 平均重试次数
- 按难度分层统计

测试集：25 条自然语言查询，分 easy / medium / hard 三档。

注意：
- 此脚本需要 MySQL 可用（数据已导入）
- 如果缺少 LLM API key，只评估 Schema Agent 的输出质量（SQL 生成部分跳过）
"""

import json
import sys
import time
from pathlib import Path
from typing import List, Dict, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


# ============================================================
# 测试查询集（25条，分三档难度）
# ============================================================

SQL_TEST_QUERIES: List[Dict] = [
    # ========== Easy（单表单条件，点查询/简单聚合）==========
    {
        "id": 1,
        "difficulty": "easy",
        "query": "查询所有Gold会员的客户ID和消费总额",
        "expected_table": "customers",
        "expected_keywords": ["SELECT", "customer_id", "total_spend_usd", "membership_tier", "Gold"],
    },
    {
        "id": 2,
        "difficulty": "easy",
        "query": "已流失的客户有多少个",
        "expected_table": "customers",
        "expected_keywords": ["SELECT", "COUNT", "churned", "= 1"],
    },
    {
        "id": 3,
        "difficulty": "easy",
        "query": "每个国家的客户数量",
        "expected_table": "customers",
        "expected_keywords": ["SELECT", "country", "COUNT", "GROUP BY"],
    },
    {
        "id": 4,
        "difficulty": "easy",
        "query": "平均客单价最高的前5个会员等级",
        "expected_table": "customers",
        "expected_keywords": ["SELECT", "membership_tier", "AVG", "avg_order_value", "ORDER BY", "LIMIT 5"],
    },
    {
        "id": 5,
        "difficulty": "easy",
        "query": "查询2024年1月的总营收",
        "expected_table": "monthly_revenue",
        "expected_keywords": ["SELECT", "revenue_usd", "year = 2024", "month = 1"],
    },
    {
        "id": 6,
        "difficulty": "easy",
        "query": "各品类的商品总数",
        "expected_table": "product_summary",
        "expected_keywords": ["SELECT", "category", "COUNT", "GROUP BY"],
    },
    {
        "id": 7,
        "difficulty": "easy",
        "query": "评分最高的10个商品名称和评分",
        "expected_table": "product_summary",
        "expected_keywords": ["SELECT", "product_name", "avg_rating", "ORDER BY", "DESC", "LIMIT 10"],
    },
    {
        "id": 8,
        "difficulty": "easy",
        "query": "订单表中一共有多少条记录",
        "expected_table": "orders",
        "expected_keywords": ["SELECT", "COUNT(*)", "orders"],
    },
    # ========== Medium（多条件/聚合/关联）==========
    {
        "id": 9,
        "difficulty": "medium",
        "query": "每个会员等级在2025年的月均消费额",
        "expected_table": "customers, orders",
        "expected_keywords": ["SELECT", "membership_tier", "AVG", "GROUP BY", "year", "2025"],
    },
    {
        "id": 10,
        "difficulty": "medium",
        "query": "退货率最高的5个品类及其退货率",
        "expected_table": "product_summary",
        "expected_keywords": ["SELECT", "category", "return_rate", "ORDER BY", "DESC", "LIMIT 5"],
    },
    {
        "id": 11,
        "difficulty": "medium",
        "query": "各获客渠道的客户平均消费总额",
        "expected_table": "customers",
        "expected_keywords": ["SELECT", "acquisition_channel", "AVG", "total_spend_usd", "GROUP BY"],
    },
    {
        "id": 12,
        "difficulty": "medium",
        "query": "2025年每个季度的订单量和总营收",
        "expected_table": "monthly_revenue",
        "expected_keywords": ["SELECT", "quarter", "SUM", "orders", "revenue", "year = 2025", "GROUP BY"],
    },
    {
        "id": 13,
        "difficulty": "medium",
        "query": "订单评分（customer_rating）不为空的订单中，平均评分是多少",
        "expected_table": "orders",
        "expected_keywords": ["SELECT", "AVG", "customer_rating", "IS NOT NULL"],
    },
    {
        "id": 14,
        "difficulty": "medium",
        "query": "使用Mobile设备下单的客户中流失比例是多少",
        "expected_table": "customers, orders",
        "expected_keywords": ["SELECT", "device_used", "churned", "Mobile"],
    },
    {
        "id": 15,
        "difficulty": "medium",
        "query": "每个月的营收环比增长率（按月度营收表）",
        "expected_table": "monthly_revenue",
        "expected_keywords": ["SELECT", "revenue_usd", "month", "year", "ORDER BY"],
    },
    {
        "id": 16,
        "difficulty": "medium",
        "query": "折扣率超过20%的订单占总订单的比例",
        "expected_table": "orders",
        "expected_keywords": ["SELECT", "COUNT", "discount_pct", "> 0.20"],
    },
    # ========== Hard（多表关联/子查询/窗口函数）==========
    {
        "id": 17,
        "difficulty": "hard",
        "query": "消费总额超过平均消费总额2倍的客户有哪些",
        "expected_table": "customers",
        "expected_keywords": ["SELECT", "total_spend_usd", "AVG", "HAVING", ">"],
    },
    {
        "id": 18,
        "difficulty": "hard",
        "query": "每个品类中，销售额排名前3的商品",
        "expected_table": "product_summary",
        "expected_keywords": ["SELECT", "category", "product_name", "total_revenue_usd", "ORDER BY", "LIMIT"],
    },
    {
        "id": 19,
        "difficulty": "hard",
        "query": "复购客户（is_repeat_customer=1）平均评分与非复购客户平均评分的对比",
        "expected_table": "orders",
        "expected_keywords": ["SELECT", "is_repeat_customer", "AVG", "customer_rating", "GROUP BY"],
    },
    {
        "id": 20,
        "difficulty": "hard",
        "query": "每个国家的会员等级分布（透视/交叉统计）",
        "expected_table": "customers",
        "expected_keywords": ["SELECT", "country", "membership_tier", "COUNT", "GROUP BY"],
    },
    {
        "id": 21,
        "difficulty": "unsolvable",
        "query": "2025年每月的新增客户数和流失客户数对比",
        "expected_table": "customers, monthly_revenue",
        "expected_keywords": ["SELECT", "new_customers", "churned", "month", "2025"],
        "solvable": False,
        "unsolvable_reason": "数据模型限制：customers.churned 是静态0/1标签，无 churn_date 列，"
                            "无法按月份统计流失客户数。monthly_revenue 仅有 new_customers 月度数据，"
                            "无 churned_customers 列。该问题在当前 schema 下不可解。",
    },
    {
        "id": 22,
        "difficulty": "hard",
        "query": "配送天数超过平均值1.5倍的订单对应的客户年龄分布",
        "expected_table": "orders, customers",
        "expected_keywords": ["SELECT", "delivery_days", "age", "AVG", "JOIN"],
    },
    {
        "id": 23,
        "difficulty": "hard",
        "query": "各品类中，退货率高于品类平均退货率的商品",
        "expected_table": "product_summary",
        "expected_keywords": ["SELECT", "category", "product_name", "return_rate", "AVG", "HAVING"],
    },
    {
        "id": 24,
        "difficulty": "hard",
        "query": "订单评分分布（1-5分各有多少订单）",
        "expected_table": "orders",
        "expected_keywords": ["SELECT", "customer_rating", "COUNT", "GROUP BY"],
    },
    {
        "id": 25,
        "difficulty": "hard",
        "query": "过去90天内有退货记录的客户的会员等级分布",
        "expected_table": "orders, customers",
        "expected_keywords": ["SELECT", "membership_tier", "returned", "COUNT", "order_date"],
    },
]


# ============================================================
# 评估逻辑
# ============================================================

def run_sql_evaluation() -> dict:
    """
    运行全部25条测试查询，统计 SQL 生成和执行结果。

    评估两层数据（对应论文"LLM SQL生成与自修正"章节）：
    1. 首次生成成功率（first-try）—— 量化基础生成能力
    2. 自修正后成功率（after-retry）—— 量化自修正闭环的提效效果

    注意：此函数需要 MySQL + LLM API key 才能完整运行。
    如果环境不完备，返回跳过状态。
    """
    from config.settings import get_settings
    from storage.mysql.client import check_connection
    from agents.state import create_initial_state
    from agents.schema_agent import schema_agent_node
    from agents.sql_agent import (
        validate_select_only,
        _build_schema_info_str,
        _generate_sql,
        _retry_sql,
        _execute_sql,
    )

    settings = get_settings()

    # 环境检查
    db_available = check_connection()
    llm_available = bool(settings.LLM_API_KEY and "your_" not in settings.LLM_API_KEY)

    if not db_available:
        print("[WARN] MySQL 不可用，SQL 评估将跳过执行环节")
    if not llm_available:
        print("[WARN] LLM API key 未配置，SQL 生成将跳过")

    results = []
    stats = {
        "total": len(SQL_TEST_QUERIES),
        "unsolvable_count": sum(1 for t in SQL_TEST_QUERIES if not t.get("solvable", True)),
        "solvable_count": sum(1 for t in SQL_TEST_QUERIES if t.get("solvable", True)),
        "by_difficulty": {},
        "sql_generated": 0,
        "sql_first_try_success": 0,
        "sql_after_retry_success": 0,
        "total_retries_used": 0,
        "retries_fixed": 0,
        "safety_rejections": 0,
        # 可解题独立统计
        "solvable_first_try_success": 0,
        "solvable_after_retry_success": 0,
        "solvable_retries_fixed": 0,
    }

    for test in SQL_TEST_QUERIES:
        is_solvable = test.get("solvable", True)
        result = {
            "id": test["id"],
            "query": test["query"],
            "difficulty": test["difficulty"],
            "solvable": is_solvable,
            "sql": None,
            "first_try_success": False,
            "after_retry_success": False,
            "retries_used": 0,
            "error": None,
            "safety_passed": True,
            "row_count": 0,
            "retry_detail": [],
        }

        # Step 1: Schema Agent
        state = create_initial_state(test["query"])
        state = schema_agent_node(state)

        if state.get("error"):
            result["error"] = f"Schema Agent: {state['error']}"
            results.append(result)
            continue

        # Step 2: SQL 生成（需要 LLM）
        if not llm_available:
            result["error"] = "LLM API key 未配置，跳过"
            results.append(result)
            continue

        schema_info = _build_schema_info_str(state["selected_tables"])

        try:
            # ---- 首次尝试 ----
            sql = _generate_sql(test["query"], schema_info)
            result["sql"] = sql
            stats["sql_generated"] += 1

            # 安全检查
            is_safe, reason = validate_select_only(sql)
            if not is_safe:
                result["safety_passed"] = False
                result["error"] = f"安全拦截: {reason}"
                stats["safety_rejections"] += 1
                results.append(result)
                continue

            # 执行首次 SQL
            if db_available:
                rows, exec_error = _execute_sql(sql)
                if exec_error is None:
                    result["first_try_success"] = True
                    result["after_retry_success"] = True
                    result["row_count"] = len(rows) if rows else 0
                    stats["sql_first_try_success"] += 1
                    stats["sql_after_retry_success"] += 1
                    if is_solvable:
                        stats["solvable_first_try_success"] += 1
                        stats["solvable_after_retry_success"] += 1
                else:
                    # ---- 自修正闭环（最多3次重试） ----
                    result["error"] = f"[首次] {exec_error}"
                    error_msg = exec_error
                    retry_fixed = False

                    for retry_n in range(1, settings.SQL_MAX_RETRIES + 1):
                        try:
                            retry_sql = _retry_sql(test["query"], sql, error_msg)
                        except Exception as e:
                            result["retry_detail"].append({
                                "retry": retry_n,
                                "phase": "llm_retry_failed",
                                "error": str(e)[:200],
                            })
                            continue

                        # 安全校验
                        is_safe, reason = validate_select_only(retry_sql)
                        if not is_safe:
                            result["retry_detail"].append({
                                "retry": retry_n,
                                "phase": "safety_blocked",
                                "sql": retry_sql[:200],
                                "reason": reason,
                            })
                            error_msg = f"安全校验失败: {reason}"
                            continue

                        # 执行修正后的 SQL
                        rows, exec_error = _execute_sql(retry_sql)
                        sql = retry_sql  # 更新为最新 SQL

                        if exec_error is None:
                            result["retry_detail"].append({
                                "retry": retry_n,
                                "phase": "success",
                                "sql": retry_sql[:500],
                            })
                            result["after_retry_success"] = True
                            result["row_count"] = len(rows) if rows else 0
                            result["sql"] = retry_sql
                            stats["sql_after_retry_success"] += 1
                            stats["retries_fixed"] += 1
                            if is_solvable:
                                stats["solvable_after_retry_success"] += 1
                                stats["solvable_retries_fixed"] += 1
                            retry_fixed = True
                            result["retries_used"] = retry_n
                            break
                        else:
                            result["retry_detail"].append({
                                "retry": retry_n,
                                "phase": "failed",
                                "sql": retry_sql[:200],
                                "error": exec_error[:200],
                            })
                            error_msg = exec_error

                    result["retries_used"] = retry_n
                    stats["total_retries_used"] += retry_n

                    if not retry_fixed:
                        result["after_retry_success"] = False
                        result["error"] = f"[{retry_n}次重试后仍失败] {error_msg[:150]}"

        except Exception as e:
            result["error"] = str(e)[:200]

        results.append(result)

    # 汇总统计（全部25条，含1条unsolvable）
    completed = [r for r in results if r["sql"] is not None]
    if completed:
        n = len(completed)
        stats["sql_first_try_success_rate"] = round(
            stats["sql_first_try_success"] / n, 3
        ) if n > 0 else 0
        stats["sql_after_retry_success_rate"] = round(
            stats["sql_after_retry_success"] / n, 3
        ) if n > 0 else 0
        stats["retry_improvement"] = round(
            (stats["sql_after_retry_success"] - stats["sql_first_try_success"]) / n, 3
        ) if n > 0 else 0

    # 可解题独立统计（排除unsolvable）
    solvable_completed = [r for r in results if r["sql"] is not None and r.get("solvable", True)]
    if solvable_completed:
        sn = len(solvable_completed)
        stats["solvable_first_try_success_rate"] = round(
            stats["solvable_first_try_success"] / sn, 3
        ) if sn > 0 else 0
        stats["solvable_after_retry_success_rate"] = round(
            stats["solvable_after_retry_success"] / sn, 3
        ) if sn > 0 else 0
        stats["solvable_retry_improvement"] = round(
            (stats["solvable_after_retry_success"] - stats["solvable_first_try_success"]) / sn, 3
        ) if sn > 0 else 0

    # 按难度分层
    for diff in ["easy", "medium", "hard"]:
        subset = [r for r in results if r["difficulty"] == diff]
        executed = [r for r in subset if r["sql"] is not None]
        diff_stats = {
            "count": len(subset),
            "generated": len(executed),
            "first_try_success": sum(1 for r in executed if r["first_try_success"]),
            "after_retry_success": sum(1 for r in executed if r["after_retry_success"]),
        }
        if executed:
            diff_stats["first_try_rate"] = round(
                diff_stats["first_try_success"] / len(executed), 3
            )
            diff_stats["after_retry_rate"] = round(
                diff_stats["after_retry_success"] / len(executed), 3
            )
        stats["by_difficulty"][diff] = diff_stats

    return {"summary": stats, "details": results}


def print_summary(report: dict) -> None:
    """打印可读的评估摘要（含自修正前后对比，区分可解/不可解题）。"""
    s = report["summary"]
    print("\n" + "=" * 60)
    print("SQL Agent 评估结果（含自修正闭环）")
    print("=" * 60)
    print(f"  总查询数: {s['total']}")
    print(f"  其中不可解题（数据模型限制）: {s['unsolvable_count']}")
    print(f"  SQL 生成数: {s['sql_generated']}")
    print(f"  安全拦截数: {s['safety_rejections']}")
    print()

    # ---- 全部25条（含不可解题） ----
    print(f"--- 全部 {s['total']} 条（含 {s['unsolvable_count']} 条不可解） ---")
    print(f"  首次生成成功率: {s.get('sql_first_try_success_rate', 'N/A'):.1%}")
    print(f"  自修正后成功率: {s.get('sql_after_retry_success_rate', 'N/A'):.1%}")
    print(f"  重试修复数: {s.get('retries_fixed', 'N/A')}")
    print(f"  总重试次数: {s.get('total_retries_used', 'N/A')}")
    if "retry_improvement" in s:
        print(f"  自修正提效: {s['retry_improvement']:+.1%}")

    # ---- 可解24条 ----
    if s.get('solvable_count', 0) > 0:
        print(f"\n--- 可解题 {s['solvable_count']} 条（排除数据模型限制） ---")
        print(f"  首次生成成功率: {s.get('solvable_first_try_success_rate', 'N/A'):.1%}")
        print(f"  自修正后成功率: {s.get('solvable_after_retry_success_rate', 'N/A'):.1%}")
        print(f"  重试修复数: {s.get('solvable_retries_fixed', 'N/A')}")
        if "solvable_retry_improvement" in s:
            print(f"  自修正提效: {s['solvable_retry_improvement']:+.1%}")

    # 按难度分层
    print("\n按难度分层（可解题目，自修正前后对比）:")
    for diff in ["easy", "medium", "hard"]:
        stats = s.get("by_difficulty", {}).get(diff, {})
        ft = stats.get("first_try_rate", "N/A")
        ar = stats.get("after_retry_rate", "N/A")
        print(f"  {diff} ({stats.get('count', '?')}条): "
              f"首次={ft}, 修正后={ar}, "
              f"首次成功={stats.get('first_try_success', '?')}, "
              f"修正后成功={stats.get('after_retry_success', '?')}")

    # 不可解题列表
    unsolvable = [r for r in report["details"] if not r.get("solvable", True)]
    if unsolvable:
        print(f"\n--- 不可解题（数据模型限制，已从主指标中剔除） ---")
        for r in unsolvable:
            print(f"  [{r['id']}] {r['query'][:60]}...")
            ut = [t for t in SQL_TEST_QUERIES if t["id"] == r["id"]]
            if ut:
                print(f"    原因: {ut[0].get('unsolvable_reason', 'N/A')}")

    # 最终失败案例（仅可解题）
    failures = [r for r in report["details"]
                if not r.get("after_retry_success") and r.get("sql") and r.get("solvable", True)]
    if failures:
        print(f"\n--- 最终失败案例（可解但未成功） ---")
        for r in failures:
            print(f"  [{r['id']}/{r['difficulty']}] {r['query'][:60]}...")
            print(f"    重试次数: {r.get('retries_used', 'N/A')}")
            print(f"    Error: {r['error'][:150]}")
            for rd in r.get("retry_detail", []):
                print(f"    [重试{rd['retry']}] {rd['phase']}: {rd.get('error', rd.get('reason', ''))[:100]}")


def main():
    print("=" * 60)
    print("SQL Agent 评估")
    print("=" * 60)

    report = run_sql_evaluation()
    print_summary(report)

    # 保存报告
    output_dir = Path(__file__).resolve().parent.parent / "data" / "processed"
    output_dir.mkdir(exist_ok=True)
    report_path = output_dir / "eval_sql_report.json"

    # 清理不可序列化的内容
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n[OK] 评估报告已保存: {report_path}")


if __name__ == "__main__":
    main()
