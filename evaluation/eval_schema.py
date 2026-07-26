"""
Schema Agent 检索准确率评估。

评估指标：
- Top-1 / Top-3 / Top-5 检索准确率（ChromaDB 粗排）
- LLM 精排后准确率（需要 API key）

测试集：20 条中文自然语言查询 + 人工标注的期望表/字段
"""

import json
import sys
from pathlib import Path
from typing import List, Dict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from storage.chromadb.embedder import get_embedder


# ============================================================
# 测试查询集（20条）
# ============================================================

# 每条: query + 期望匹配的字段列表（table.column）
TEST_QUERIES: List[Dict] = [
    # ---- 简单查询 ----
    {
        "id": 1,
        "query": "中国区GMV最高的前10个客户是谁",
        "difficulty": "easy",
        "expected_fields": [
            "customers.total_spend_usd",
            "customers.country",
            "customers.customer_id",
        ],
        "expected_tables": ["customers"],
    },
    {
        "id": 2,
        "query": "各个国家的客户数量分布",
        "difficulty": "easy",
        "expected_fields": [
            "customers.country",
            "customers.customer_id",
        ],
        "expected_tables": ["customers"],
    },
    {
        "id": 3,
        "query": "已流失的客户有哪些",
        "difficulty": "easy",
        "expected_fields": [
            "customers.churned",
            "customers.customer_id",
        ],
        "expected_tables": ["customers"],
    },
    {
        "id": 4,
        "query": "Gold会员的平均消费金额是多少",
        "difficulty": "easy",
        "expected_fields": [
            "customers.membership_tier",
            "customers.total_spend_usd",
        ],
        "expected_tables": ["customers"],
    },
    {
        "id": 5,
        "query": "每个品类的商品数量统计",
        "difficulty": "easy",
        "expected_fields": [
            "orders.category",
            "product_summary.category",
        ],
        "expected_tables": ["orders", "product_summary"],
    },
    # ---- 中等查询 ----
    {
        "id": 6,
        "query": "上个月订单量最多的5个商品是什么",
        "difficulty": "medium",
        "expected_fields": [
            "orders.order_date",
            "orders.product_name",
            "orders.order_id",
        ],
        "expected_tables": ["orders"],
    },
    {
        "id": 7,
        "query": "各会员等级的平均客单价对比",
        "difficulty": "medium",
        "expected_fields": [
            "customers.membership_tier",
            "customers.avg_order_value_usd",
        ],
        "expected_tables": ["customers"],
    },
    {
        "id": 8,
        "query": "通过社交媒体渠道获客的客户平均退货率",
        "difficulty": "medium",
        "expected_fields": [
            "customers.acquisition_channel",
            "customers.returns_made",
        ],
        "expected_tables": ["customers"],
    },
    {
        "id": 9,
        "query": "2025年Q4的总营收是多少",
        "difficulty": "medium",
        "expected_fields": [
            "monthly_revenue.revenue_usd",
            "monthly_revenue.year",
            "monthly_revenue.quarter",
        ],
        "expected_tables": ["monthly_revenue"],
    },
    {
        "id": 10,
        "query": "各个国家每月的新增客户趋势",
        "difficulty": "medium",
        "expected_fields": [
            "customers.country",
            "customers.registration_date",
            "monthly_revenue.new_customers",
        ],
        "expected_tables": ["customers", "monthly_revenue"],
    },
    # ---- 困难查询 ----
    {
        "id": 11,
        "query": "复购率最高的品类中，客单价超过100美元的产品有哪些",
        "difficulty": "hard",
        "expected_fields": [
            "orders.is_repeat_customer",
            "orders.category",
            "product_summary.avg_price",
            "product_summary.product_name",
        ],
        "expected_tables": ["orders", "product_summary"],
    },
    {
        "id": 12,
        "query": "最近30天内有购买但之前已流失的客户名单",
        "difficulty": "hard",
        "expected_fields": [
            "orders.order_date",
            "customers.churned",
            "customers.customer_id",
        ],
        "expected_tables": ["customers", "orders"],
    },
    {
        "id": 13,
        "query": "使用移动端下单且配送天数超过5天的订单退货率",
        "difficulty": "hard",
        "expected_fields": [
            "orders.device_used",
            "orders.delivery_days",
            "orders.returned",
        ],
        "expected_tables": ["orders"],
    },
    {
        "id": 14,
        "query": "客单价Top20%的客户的共同特征是什么",
        "difficulty": "hard",
        "expected_fields": [
            "customers.avg_order_value_usd",
            "customers.preferred_category",
            "customers.membership_tier",
            "customers.preferred_device",
        ],
        "expected_tables": ["customers"],
    },
    {
        "id": 15,
        "query": "折扣力度最大的月份对应的营收和退货率变化",
        "difficulty": "hard",
        "expected_fields": [
            "monthly_revenue.avg_discount_pct",
            "monthly_revenue.revenue_usd",
            "monthly_revenue.return_rate",
            "monthly_revenue.month",
        ],
        "expected_tables": ["monthly_revenue"],
    },
    # ---- 术语映射测试 ----
    {
        "id": 16,
        "query": "哪个品类的GMV最大",
        "difficulty": "easy",
        "expected_fields": [
            "orders.category",
            "orders.subtotal_usd",
            "product_summary.total_revenue_usd",
        ],
        "expected_tables": ["orders", "product_summary"],
    },
    {
        "id": 17,
        "query": "各渠道的获客成本对比（用客单价衡量）",
        "difficulty": "medium",
        "expected_fields": [
            "customers.acquisition_channel",
            "customers.avg_order_value_usd",
        ],
        "expected_tables": ["customers"],
    },
    {
        "id": 18,
        "query": "哪种支付方式的复购率最高",
        "difficulty": "medium",
        "expected_fields": [
            "customers.preferred_payment_method",
            "orders.is_repeat_customer",
            "orders.payment_method",
        ],
        "expected_tables": ["customers", "orders"],
    },
    {
        "id": 19,
        "query": "物流时间最长的10个商品及其退货率",
        "difficulty": "medium",
        "expected_fields": [
            "product_summary.avg_delivery_days",
            "product_summary.return_rate",
            "product_summary.product_name",
        ],
        "expected_tables": ["product_summary"],
    },
    {
        "id": 20,
        "query": "评分低于3分的订单对应的客户有什么共同特征",
        "difficulty": "hard",
        "expected_fields": [
            "orders.customer_rating",
            "customers.age",
            "customers.membership_tier",
            "customers.preferred_device",
        ],
        "expected_tables": ["orders", "customers"],
    },
]


# ============================================================
# 评估逻辑
# ============================================================

def evaluate_retrieval(embedder, top_k: int = 10) -> dict:
    """
    对每条测试查询做检索，计算准确率。

    准确率定义：
    - Top-1 命中率: 期望字段中至少一个出现在检索结果前1位
    - Top-3 命中率: 期望字段中至少一个出现在检索结果前3位
    - 表级召回率: 期望的表是否都被检索到（在前top_k中）
    """
    results = []
    hit_at_1 = 0
    hit_at_3 = 0
    hit_at_5 = 0
    table_recall_total = 0.0

    for test in TEST_QUERIES:
        query = test["query"]
        expected_fields = set(test["expected_fields"])
        expected_tables = set(test["expected_tables"])

        # 检索
        candidates = embedder.search(query, top_k=top_k)

        # 命中检查
        retrieved_fields = set()
        retrieved_tables = set()
        for i, c in enumerate(candidates):
            field_key = f"{c['table_name']}.{c['column_name']}"
            retrieved_fields.add(field_key)
            retrieved_tables.add(c["table_name"])

            # Top-1
            if i == 0 and field_key in expected_fields:
                hit_at_1 += 1
            # Top-3
            if i == 2 and any(
                f"{c2['table_name']}.{c2['column_name']}" in expected_fields
                for c2 in candidates[:3]
            ):
                hit_at_3 += 1
            # Top-5
            if i == 4 and any(
                f"{c2['table_name']}.{c2['column_name']}" in expected_fields
                for c2 in candidates[:5]
            ):
                hit_at_5 += 1

        # 表级召回
        table_recall = (
            len(expected_tables & retrieved_tables) / len(expected_tables)
            if expected_tables else 1.0
        )
        table_recall_total += table_recall

        # 字段级命中数
        field_hits = expected_fields & retrieved_fields
        field_recall = (
            len(field_hits) / len(expected_fields) if expected_fields else 1.0
        )

        results.append({
            "id": test["id"],
            "query": query,
            "difficulty": test["difficulty"],
            "expected_tables": list(expected_tables),
            "retrieved_tables": list(retrieved_tables)[:5],
            "field_recall": round(field_recall, 3),
            "table_recall": round(table_recall, 3),
            "hits": list(field_hits)[:5],
            "misses": list(expected_fields - retrieved_fields),
        })

    n = len(TEST_QUERIES)
    summary = {
        "total_queries": n,
        "top_k": top_k,
        "top1_hit_rate": round(hit_at_1 / n, 3),
        "top3_hit_rate": round(hit_at_3 / n, 3),
        "top5_hit_rate": round(hit_at_5 / n, 3),
        "avg_table_recall": round(table_recall_total / n, 3),
        "by_difficulty": {},
    }

    # 按难度分组统计
    for diff in ["easy", "medium", "hard"]:
        subset = [r for r in results if r["difficulty"] == diff]
        if subset:
            avg_field_recall = sum(r["field_recall"] for r in subset) / len(subset)
            avg_table_recall = sum(r["table_recall"] for r in subset) / len(subset)
            summary["by_difficulty"][diff] = {
                "count": len(subset),
                "avg_field_recall": round(avg_field_recall, 3),
                "avg_table_recall": round(avg_table_recall, 3),
            }

    return {"summary": summary, "details": results}


def main():
    print("=" * 60)
    print("Schema Agent 检索准确率评估")
    print("=" * 60)

    embedder = get_embedder()
    if embedder.collection is None:
        print("[WARN] ChromaDB collection 不存在，正在构建...")
        embedder.build()

    report = evaluate_retrieval(embedder, top_k=10)

    # 打印摘要
    s = report["summary"]
    print(f"\n总查询数: {s['total_queries']}")
    print(f"检索 Top-K: {s['top_k']}")
    print(f"Top-1 命中率: {s['top1_hit_rate']:.1%}")
    print(f"Top-3 命中率: {s['top3_hit_rate']:.1%}")
    print(f"Top-5 命中率: {s['top5_hit_rate']:.1%}")
    print(f"平均表级召回率: {s['avg_table_recall']:.1%}")

    print("\n按难度分组:")
    for diff, stats in s["by_difficulty"].items():
        print(f"  {diff} ({stats['count']}条): "
              f"字段召回={stats['avg_field_recall']:.1%}, "
              f"表召回={stats['avg_table_recall']:.1%}")

    # 打印详细错误
    print("\n--- 未命中的字段（前5条） ---")
    for r in report["details"][:5]:
        if r["misses"]:
            print(f"  [{r['id']}] {r['query'][:40]}...")
            print(f"    未命中: {r['misses']}")

    # 保存报告
    output_dir = Path(__file__).resolve().parent.parent / "data" / "processed"
    output_dir.mkdir(exist_ok=True)
    report_path = output_dir / "eval_schema_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\n[OK] 评估报告已保存: {report_path}")


if __name__ == "__main__":
    main()
