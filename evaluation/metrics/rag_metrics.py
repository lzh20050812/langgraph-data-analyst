"""
RAG Schema 检索实验专用评价指标。

指标定义：
- Recall@K : Top-K 检索结果中是否包含正确表/字段的命中率
- MRR (Mean Reciprocal Rank) : 第一个正确结果排名的倒数均值
- Precision@K : Top-K 结果中正确结果的比例
- avg_retrieval_time : 平均检索耗时

计算公式：
- Table Recall@K = 命中正确表的样本数 / 总样本数
- Field Recall@K = 命中正确字段的样本数 / 总样本数
- MRR = (1/N) * Σ(1/rank_of_first_correct)
"""

from typing import Any, Dict, List


# ============================================================
# 检索结果评价
# ============================================================

def compute_table_recall_at_k(
    details: List[Dict[str, Any]], k: int
) -> float:
    """计算 Table Recall@K。

    对于每条样本，检查 Top-K 检索结果中是否包含任意一个 expected_table。
    """
    if not details:
        return 0.0
    hits = 0
    for d in details:
        expected = set(d.get("expected_tables", []))
        retrieved = set(d.get(f"retrieved_tables_top{k}", []))
        if expected & retrieved:
            hits += 1
    return hits / len(details)


def compute_field_recall_at_k(
    details: List[Dict[str, Any]], k: int
) -> float:
    """计算 Field Recall@K。

    对于每条样本，检查 Top-K 检索结果中是否包含任意一个 expected_field。
    expected_field 格式: "table.column"
    retrieved 格式: "table.column"
    """
    if not details:
        return 0.0
    hits = 0
    for d in details:
        expected = set(d.get("expected_fields", []))
        retrieved = set(d.get(f"retrieved_fields_top{k}", []))
        if expected & retrieved:
            hits += 1
    return hits / len(details)


def compute_precision_at_k(
    details: List[Dict[str, Any]], k: int
) -> float:
    """计算 Field Precision@K。

    Top-K 结果中属于正确字段的比例。
    """
    if not details:
        return 0.0
    precisions = []
    for d in details:
        expected = set(d.get("expected_fields", []))
        retrieved = d.get(f"retrieved_fields_top{k}", [])
        if not retrieved:
            precisions.append(0.0)
            continue
        correct = sum(1 for f in retrieved if f in expected)
        precisions.append(correct / len(retrieved))
    return sum(precisions) / len(precisions)


def compute_mrr(details: List[Dict[str, Any]]) -> float:
    """计算 Mean Reciprocal Rank (MRR)。

    对于每条样本，找到第一个正确字段在检索结果中的排名，
    取其倒数的平均值。
    """
    if not details:
        return 0.0
    reciprocals = []
    for d in details:
        expected = set(d.get("expected_fields", []))
        retrieved = d.get("retrieved_fields", [])
        for rank, field in enumerate(retrieved, 1):
            if field in expected:
                reciprocals.append(1.0 / rank)
                break
        else:
            reciprocals.append(0.0)
    return sum(reciprocals) / len(reciprocals)


def compute_avg_retrieval_time(details: List[Dict[str, Any]]) -> float:
    """计算平均检索耗时（秒）。"""
    if not details:
        return 0.0
    times = [d.get("retrieval_time_seconds", 0.0) for d in details]
    return sum(times) / len(times)


def compute_hit_rate_at_k(
    details: List[Dict[str, Any]], k: int
) -> float:
    """综合命中率：表或字段任一命中即算成功。"""
    if not details:
        return 0.0
    hits = 0
    for d in details:
        exp_tables = set(d.get("expected_tables", []))
        exp_fields = set(d.get("expected_fields", []))
        ret_tables = set(d.get(f"retrieved_tables_top{k}", []))
        ret_fields = set(d.get(f"retrieved_fields_top{k}", []))
        if (exp_tables & ret_tables) or (exp_fields & ret_fields):
            hits += 1
    return hits / len(details)


# ============================================================
# 按业务类别统计
# ============================================================

def compute_metrics_by_category(
    details: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """按业务类别分层计算各项 Recall 指标。"""
    by_cat: Dict[str, List[Dict]] = {}
    for d in details:
        cat = d.get("category", "unknown")
        by_cat.setdefault(cat, []).append(d)

    result = {}
    for cat, items in sorted(by_cat.items()):
        result[cat] = {
            "count": len(items),
            "table_recall_at_1": round(compute_table_recall_at_k(items, 1), 4),
            "table_recall_at_3": round(compute_table_recall_at_k(items, 3), 4),
            "table_recall_at_5": round(compute_table_recall_at_k(items, 5), 4),
            "field_recall_at_1": round(compute_field_recall_at_k(items, 1), 4),
            "field_recall_at_3": round(compute_field_recall_at_k(items, 3), 4),
            "field_recall_at_5": round(compute_field_recall_at_k(items, 5), 4),
            "mrr": round(compute_mrr(items), 4),
            "avg_retrieval_time": round(compute_avg_retrieval_time(items), 4),
        }
    return result


def compute_all_rag_metrics(details: List[Dict[str, Any]]) -> Dict[str, Any]:
    """一键计算所有 RAG Schema 检索实验指标。"""
    total = len(details)
    if total == 0:
        return {}

    return {
        "total_samples": total,
        # Table Recall
        "table_recall_at_1": round(compute_table_recall_at_k(details, 1), 4),
        "table_recall_at_3": round(compute_table_recall_at_k(details, 3), 4),
        "table_recall_at_5": round(compute_table_recall_at_k(details, 5), 4),
        # Field Recall
        "field_recall_at_1": round(compute_field_recall_at_k(details, 1), 4),
        "field_recall_at_3": round(compute_field_recall_at_k(details, 3), 4),
        "field_recall_at_5": round(compute_field_recall_at_k(details, 5), 4),
        # Precision
        "field_precision_at_5": round(compute_precision_at_k(details, 5), 4),
        # MRR
        "mrr": round(compute_mrr(details), 4),
        # 耗时
        "avg_retrieval_time_seconds": round(compute_avg_retrieval_time(details), 4),
        # 综合命中
        "hit_rate_at_5": round(compute_hit_rate_at_k(details, 5), 4),
        # 按类别
        "by_category": compute_metrics_by_category(details),
    }


# ============================================================
# 技术评测表格生成
# ============================================================

def generate_table_overall(metrics: Dict[str, Any]) -> str:
    """表1：Schema 检索总体结果。"""
    return f"""### 表1：Schema 语义检索总体结果

| 指标 | 数值 |
|------|------|
| 测试样本数量 | {metrics.get('total_samples', 'N/A')} |
| Table Recall@1 | {_fmt_pct(metrics.get('table_recall_at_1'))} |
| Table Recall@3 | {_fmt_pct(metrics.get('table_recall_at_3'))} |
| Table Recall@5 | {_fmt_pct(metrics.get('table_recall_at_5'))} |
| Field Recall@1 | {_fmt_pct(metrics.get('field_recall_at_1'))} |
| Field Recall@3 | {_fmt_pct(metrics.get('field_recall_at_3'))} |
| Field Recall@5 | {_fmt_pct(metrics.get('field_recall_at_5'))} |
| Field Precision@5 | {_fmt_pct(metrics.get('field_precision_at_5'))} |
| MRR | {metrics.get('mrr', 'N/A')} |
| 综合命中率@5 | {_fmt_pct(metrics.get('hit_rate_at_5'))} |
| 平均检索耗时 | {metrics.get('avg_retrieval_time_seconds', 'N/A')}s |
"""


def generate_table_by_category(metrics: Dict[str, Any]) -> str:
    """表2：不同业务类别检索效果。"""
    by_cat = metrics.get("by_category", {})
    if not by_cat:
        return "### 表2：不同业务类别检索效果\n\n（无数据）\n"

    lines = [
        "### 表2：不同业务类别检索效果",
        "",
        "| 业务类别 | 样本数 | Table R@1 | Table R@3 | Field R@1 | Field R@3 | Field R@5 | MRR | 均耗时(s) |",
        "|----------|--------|-----------|-----------|-----------|-----------|-----------|-----|----------|",
    ]
    for cat, d in by_cat.items():
        lines.append(
            f"| {cat} | {d['count']} | "
            f"{_fmt_pct(d.get('table_recall_at_1'))} | "
            f"{_fmt_pct(d.get('table_recall_at_3'))} | "
            f"{_fmt_pct(d.get('field_recall_at_1'))} | "
            f"{_fmt_pct(d.get('field_recall_at_3'))} | "
            f"{_fmt_pct(d.get('field_recall_at_5'))} | "
            f"{d.get('mrr', 'N/A')} | "
            f"{d.get('avg_retrieval_time', 'N/A')} |"
        )
    lines.append("")
    return "\n".join(lines)


def generate_table_failure_analysis(details: List[Dict[str, Any]]) -> str:
    """表3：失败案例分析（Field Recall@5 = 0 的样本）。"""
    failures = [
        d for d in details
        if not (
            set(d.get("expected_fields", [])) &
            set(d.get("retrieved_fields_top5", []))
        )
    ]

    if not failures:
        return "### 表3：失败案例分析\n\n✅ 所有样本 Field Recall@5 = 100%，无失败案例。\n"

    lines = [
        "### 表3：失败案例分析",
        "",
        "| ID | 用户问题 | 期望字段 | 实际Top-5字段 | 失败原因 |",
        "|----|----------|----------|-------------|---------|",
    ]
    for d in failures[:15]:
        exp = ", ".join(d.get("expected_fields", []))
        ret = ", ".join(d.get("retrieved_fields_top5", [])[:5])
        reason = d.get("failure_reason", "检索排名不足")
        lines.append(
            f"| {d.get('id', '?')} | {d.get('query', 'N/A')[:40]} | "
            f"{exp[:60]} | {ret[:60]} | {reason[:50]} |"
        )
    lines.append("")
    return "\n".join(lines)


# ============================================================
# 实验报告
# ============================================================

def generate_rag_experiment_report(
    experiment_name: str,
    metrics: Dict[str, Any],
    details: List[Dict[str, Any]],
    duration_seconds: float,
) -> str:
    """生成 RAG Schema 检索实验 Markdown 报告。"""
    from datetime import datetime, timezone

    report = f"""# RAG Schema 检索实验报告

**实验名称**: {experiment_name}
**生成时间**: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}
**总耗时**: {duration_seconds:.1f}s

---

## 实验概览

| 项目 | 数值 |
|------|------|
| 测试集规模 | {metrics.get('total_samples', 'N/A')} 条 |
| Table Recall@5 | {_fmt_pct(metrics.get('table_recall_at_5'))} |
| Field Recall@5 | {_fmt_pct(metrics.get('field_recall_at_5'))} |
| MRR | {metrics.get('mrr', 'N/A')} |
| 平均检索耗时 | {metrics.get('avg_retrieval_time_seconds', 'N/A')}s |

---

{generate_table_overall(metrics)}

{generate_table_by_category(metrics)}

{generate_table_failure_analysis(details)}

---
*本报告由 RAG Schema 检索实验框架自动生成。*
"""
    return report


# ============================================================
# 辅助
# ============================================================

def _fmt_pct(val) -> str:
    """格式化百分比。"""
    if val is None:
        return "N/A"
    if isinstance(val, float):
        return f"{val * 100:.1f}%"
    return str(val)
