"""
Text2SQL 实验专用指标计算模块。

提供的指标：
- sql_generation_success_rate : SQL 生成成功率（LLM 成功输出了可解析 SQL）
- execution_success_rate     : SQL 执行成功率（生成SQL能在数据库正常执行）
- execution_accuracy (EX)    : 执行准确率（生成SQL执行结果与标准答案一致的比例）
- avg_response_time          : 平均响应时间（秒）
- avg_retry_count            : 平均重试次数
- error_type_distribution    : SQL 错误类型分布统计
- by_difficulty              : 按难度分层统计

SQL 错误类型分类：
- syntax_error      : 语法错误（SQL 语法不正确，数据库拒绝执行）
- column_not_found  : 字段不存在（SQL 引用了不存在的列）
- table_not_found   : 表不存在（SQL 引用了不存在的表）
- execution_error   : 执行异常（数据库执行过程中的其他错误）
- timeout           : 超时
- logical_error     : 逻辑错误（SQL 可执行但结果与预期不符）
- no_sql_generated  : 未生成 SQL（LLM 返回空或无法解析）
- security_blocked  : 安全拦截（触发了 SELECT-only 校验）
- unknown           : 未知错误
"""

from typing import Any, Dict, List, Optional


# ============================================================
# 结果集比对
# ============================================================

def normalize_value(val: Any) -> str:
    """将值归一化为可比较的字符串。"""
    if val is None:
        return "__NULL__"
    if isinstance(val, float):
        return f"{val:.4f}"
    if isinstance(val, int):
        return str(val)
    return str(val)


def normalize_row(row: Dict[str, Any], columns: List[str]) -> tuple:
    """将一行数据归一化为可哈希的元组。"""
    return tuple(normalize_value(row.get(col)) for col in columns)


def compare_result_sets(
    generated_result: Optional[List[Dict[str, Any]]],
    expected_result: Optional[List[Dict[str, Any]]],
) -> bool:
    """比对两个查询结果集是否等价。"""
    if generated_result is None and expected_result is None:
        return True
    if generated_result is None or expected_result is None:
        return False
    if len(generated_result) != len(expected_result):
        return False
    if len(generated_result) == 0:
        return True

    gen_cols = list(generated_result[0].keys())
    exp_cols = list(expected_result[0].keys())

    # 特殊情况：单行单列聚合查询（COUNT/SUM/AVG等），直接比对数值
    # 列名可能因别名不同（如 "cnt" vs "churned_customers"），但只有值有意义
    if len(gen_cols) == 1 and len(exp_cols) == 1 and len(generated_result) == 1 and len(expected_result) == 1:
        gv = normalize_value(list(generated_result[0].values())[0])
        ev = normalize_value(list(expected_result[0].values())[0])
        return gv == ev

    if len(gen_cols) != len(exp_cols):
        common_cols = [c for c in exp_cols if c in gen_cols]
        if len(common_cols) != len(exp_cols):
            return False
        compare_cols = common_cols
    elif set(gen_cols) != set(exp_cols):
        # SQL 列别名不影响执行语义。列数一致但别名不同的情况下，按 SELECT
        # 输出位置比较值；Python dict 保留数据库结果的列顺序。
        generated_set = {
            tuple(normalize_value(value) for value in row.values())
            for row in generated_result
        }
        expected_set = {
            tuple(normalize_value(value) for value in row.values())
            for row in expected_result
        }
        return generated_set == expected_set
    else:
        compare_cols = exp_cols

    expected_set = set()
    for row in expected_result:
        expected_set.add(normalize_row(row, compare_cols))

    for row in generated_result:
        key = normalize_row(row, compare_cols)
        if key not in expected_set:
            return False
        expected_set.discard(key)

    return True


# ============================================================
# SQL 错误类型分类
# ============================================================

def classify_sql_error(error_message: Optional[str], generated_sql: Optional[str]) -> str:
    """根据错误信息自动分类 SQL 错误类型。

    Args:
        error_message: 数据库返回的错误信息
        generated_sql: 生成的 SQL 语句

    Returns:
        错误类型标签: syntax_error | column_not_found | table_not_found |
                     execution_error | timeout | no_sql_generated |
                     security_blocked | unknown
    """
    if not generated_sql or not generated_sql.strip():
        return "no_sql_generated"

    if not error_message:
        return "unknown"

    error_lower = error_message.lower()

    # 按匹配优先级判断
    # 安全拦截（本项目特有）
    if any(kw in error_lower for kw in ["安全校验", "危险关键词", "select only", "仅允许 select"]):
        return "security_blocked"

    # 表不存在
    if any(kw in error_lower for kw in [
        "table", "doesn't exist", "does not exist", "no such table",
        "表", "relation", "doesn't exist"
    ]):
        return "table_not_found"

    # 字段不存在
    if any(kw in error_lower for kw in [
        "column", "unknown column", "doesn't exist",
        "字段", "列", "unresolved",
    ]):
        return "column_not_found"

    # 语法错误
    if any(kw in error_lower for kw in [
        "syntax error", "sql syntax", "parse error",
        "语法错误", "near", "at line",
        "you have an error in your sql syntax",
    ]):
        return "syntax_error"

    # 超时
    if any(kw in error_lower for kw in [
        "timeout", "timed out", "超时", "lock wait timeout",
    ]):
        return "timeout"

    # 其他执行错误
    return "execution_error"


def compute_error_distribution(details: List[Dict[str, Any]]) -> Dict[str, int]:
    """统计各错误类型的分布。

    Args:
        details: 逐样本详情列表，每条需包含 "error_type" 字段

    Returns:
        {error_type: count} 字典
    """
    dist: Dict[str, int] = {}
    for d in details:
        # 对于成功的样本，不统计错误
        if d.get("execution_success", False) and d.get("result_correct", False):
            continue
        # 对于执行成功但结果错误的 → logical_error
        if d.get("execution_success", False) and not d.get("result_correct", False):
            dist["logical_error"] = dist.get("logical_error", 0) + 1
            continue
        # 其他错误按 error_type 分类
        etype = d.get("error_type", "unknown")
        dist[etype] = dist.get(etype, 0) + 1

    return dict(sorted(dist.items(), key=lambda x: x[1], reverse=True))


# ============================================================
# 基础指标
# ============================================================

def compute_sql_generation_success_rate(details: List[Dict[str, Any]]) -> float:
    """SQL 生成成功率：LLM 成功输出了非空 SQL。"""
    if not details:
        return 0.0
    generated = sum(1 for d in details
                    if d.get("generated_sql") and d.get("generated_sql").strip()
                    and d.get("error_type") != "no_sql_generated"
                    and d.get("error_type") != "security_blocked")
    return generated / len(details)


def compute_execution_success_rate(details: List[Dict[str, Any]]) -> float:
    """SQL 执行成功率：SQL 在数据库正常执行。"""
    if not details:
        return 0.0
    success = sum(1 for d in details if d.get("execution_success", False))
    return success / len(details)


def compute_execution_accuracy(details: List[Dict[str, Any]]) -> float:
    """Execution Accuracy (EX)：执行结果与标准答案一致。"""
    if not details:
        return 0.0
    correct = sum(1 for d in details if d.get("result_correct", False))
    return correct / len(details)


def compute_avg_response_time(details: List[Dict[str, Any]]) -> float:
    """平均响应时间（秒）。"""
    if not details:
        return 0.0
    times = [d.get("duration_seconds", 0.0) for d in details]
    return sum(times) / len(times)


def compute_avg_retry_count(details: List[Dict[str, Any]]) -> float:
    """平均 SQL 重试次数。"""
    if not details:
        return 0.0
    retries = [d.get("retry_count", 0) for d in details]
    return sum(retries) / len(retries)


# ============================================================
# 分层统计
# ============================================================

def compute_metrics_by_difficulty(
    details: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """按难度分层计算各项指标。"""
    by_diff: Dict[str, List[Dict]] = {}
    for d in details:
        diff = d.get("difficulty", "unknown")
        by_diff.setdefault(diff, []).append(d)

    result = {}
    for diff in ["easy", "medium", "hard", "edge"]:
        items = by_diff.get(diff, [])
        if not items:
            continue
        result[diff] = {
            "count": len(items),
            "generation_success_rate": round(compute_sql_generation_success_rate(items), 4),
            "execution_success_rate": round(compute_execution_success_rate(items), 4),
            "execution_accuracy": round(compute_execution_accuracy(items), 4),
            "avg_response_time": round(compute_avg_response_time(items), 4),
            "avg_retry_count": round(compute_avg_retry_count(items), 4),
        }
    return result


def compute_all_metrics(details: List[Dict[str, Any]]) -> Dict[str, Any]:
    """一键计算所有 Text2SQL 实验指标。"""
    total = len(details)
    exec_success = sum(1 for d in details if d.get("execution_success", False))
    exec_correct = sum(1 for d in details if d.get("result_correct", False))
    gen_success = sum(1 for d in details
                      if d.get("generated_sql") and d.get("generated_sql").strip()
                      and d.get("error_type") != "no_sql_generated"
                      and d.get("error_type") != "security_blocked")

    return {
        "total_samples": total,
        "sql_generation_success": gen_success,
        "sql_generation_failure": total - gen_success,
        "execution_success": exec_success,
        "execution_failure": total - exec_success,
        "execution_correct": exec_correct,
        "sql_generation_success_rate": round(compute_sql_generation_success_rate(details), 4),
        "execution_success_rate": round(compute_execution_success_rate(details), 4),
        "execution_accuracy": round(compute_execution_accuracy(details), 4),
        "sql_error_rate": round(
            sum(1 for d in details if d.get("error_type", "") not in ("", None)
                and not d.get("result_correct", False)) / total, 4
        ) if total > 0 else 0.0,
        "avg_response_time_seconds": round(compute_avg_response_time(details), 4),
        "avg_retry_count": round(compute_avg_retry_count(details), 4),
        "error_distribution": compute_error_distribution(details),
        "by_difficulty": compute_metrics_by_difficulty(details),
    }


# ============================================================
# 技术评测表格生成
# ============================================================

def generate_table_overall(metrics: Dict[str, Any]) -> str:
    """表1：Text2SQL 总体实验结果。"""
    return f"""### 表1：Text2SQL 总体实验结果

| 指标 | 数值 |
|------|------|
| 测试样本数量 | {metrics.get('total_samples', 'N/A')} |
| SQL 生成成功数 | {metrics.get('sql_generation_success', 'N/A')} |
| SQL 生成成功率 | {_fmt_pct(metrics.get('sql_generation_success_rate'))} |
| SQL 执行成功数 | {metrics.get('execution_success', 'N/A')} |
| SQL 执行成功率 (EXE) | {_fmt_pct(metrics.get('execution_success_rate'))} |
| 执行准确数 | {metrics.get('execution_correct', 'N/A')} |
| 执行准确率 (EX) | {_fmt_pct(metrics.get('execution_accuracy'))} |
| SQL 错误率 | {_fmt_pct(metrics.get('sql_error_rate'))} |
| 平均响应时间 | {metrics.get('avg_response_time_seconds', 'N/A')}s |
| 平均重试次数 | {metrics.get('avg_retry_count', 'N/A')} |
"""


def generate_table_by_difficulty(metrics: Dict[str, Any]) -> str:
    """表2：不同查询难度实验结果。"""
    lines = [
        "### 表2：不同查询难度实验结果",
        "",
        "| 难度 | 样本数 | SQL生成成功率 | SQL执行成功率 | 执行准确率(EX) | 平均耗时(s) |",
        "|------|--------|-------------|-------------|---------------|------------|",
    ]
    for diff in ["easy", "medium", "hard", "edge"]:
        d = metrics.get("by_difficulty", {}).get(diff)
        if d:
            lines.append(
                f"| {diff} | {d['count']} | "
                f"{_fmt_pct(d.get('generation_success_rate'))} | "
                f"{_fmt_pct(d.get('execution_success_rate'))} | "
                f"{_fmt_pct(d.get('execution_accuracy'))} | "
                f"{d.get('avg_response_time', 'N/A')} |"
            )
    lines.append("")
    return "\n".join(lines)


def generate_table_error_types(metrics: Dict[str, Any]) -> str:
    """表3：SQL 错误类型统计。"""
    err_dist = metrics.get("error_distribution", {})
    if not err_dist:
        return "### 表3：SQL 错误类型统计\n\n（无错误记录）\n"

    total_errors = sum(err_dist.values())
    error_labels = {
        "syntax_error": "语法错误",
        "column_not_found": "字段不存在",
        "table_not_found": "表不存在",
        "execution_error": "执行异常",
        "timeout": "超时",
        "logical_error": "逻辑错误（可执行但结果不符）",
        "no_sql_generated": "未生成SQL",
        "security_blocked": "安全拦截",
        "unknown": "未知错误",
    }
    lines = [
        "### 表3：SQL 错误类型统计",
        "",
        "| 错误类型 | 数量 | 占比 |",
        "|----------|------|------|",
    ]
    for etype, count in err_dist.items():
        label = error_labels.get(etype, etype)
        pct = f"{count / total_errors * 100:.1f}%" if total_errors > 0 else "0%"
        lines.append(f"| {label} | {count} | {pct} |")
    lines.append(f"| **合计** | **{total_errors}** | **100%** |")
    lines.append("")
    return "\n".join(lines)


# ============================================================
# 实验报告生成
# ============================================================

def generate_experiment_report(
    experiment_name: str,
    metrics: Dict[str, Any],
    details: List[Dict[str, Any]],
    duration_seconds: float,
) -> str:
    """生成实验总结 Markdown 报告。

    Args:
        experiment_name: 实验名称
        metrics: 指标字典（来自 compute_all_metrics）
        details: 逐样本详情列表
        duration_seconds: 实验总耗时

    Returns:
        Markdown 格式的实验报告
    """
    from datetime import datetime, timezone

    fail_count = sum(1 for d in details if not d.get("result_correct", False))
    err_dist = metrics.get("error_distribution", {})
    main_error = ""
    if err_dist:
        error_labels = {
            "syntax_error": "语法错误", "column_not_found": "字段不存在",
            "table_not_found": "表不存在", "execution_error": "执行异常",
            "timeout": "超时", "logical_error": "逻辑错误",
            "no_sql_generated": "未生成SQL", "security_blocked": "安全拦截",
            "unknown": "未知错误",
        }
        top_err = list(err_dist.keys())[0]
        main_error = error_labels.get(top_err, top_err)

    report = f"""# Text2SQL 实验报告

**实验名称**: {experiment_name}
**生成时间**: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}
**总耗时**: {duration_seconds:.1f}s

---

## 实验概览

| 项目 | 数值 |
|------|------|
| 测试集规模 | {metrics.get('total_samples', 'N/A')} 条 |
| SQL 生成成功率 | {_fmt_pct(metrics.get('sql_generation_success_rate'))} |
| SQL 执行成功率 (EXE) | {_fmt_pct(metrics.get('execution_success_rate'))} |
| 执行准确率 (EX) | {_fmt_pct(metrics.get('execution_accuracy'))} |
| 失败案例数量 | {fail_count} |
| 平均响应时间 | {metrics.get('avg_response_time_seconds', 'N/A')}s |
| 平均重试次数 | {metrics.get('avg_retry_count', 'N/A')} |
| 主要错误类型 | {main_error or '无'} |

---

{generate_table_overall(metrics)}

{generate_table_by_difficulty(metrics)}

{generate_table_error_types(metrics)}

## 失败案例分析

"""
    # 添加失败案例详情（最多10条）
    failures = [d for d in details if not d.get("result_correct", False)][:10]
    if failures:
        for f in failures:
            report += f"""### Q{f.get('id', '?')} [{f.get('difficulty', '?')}]

- **问题**: {f.get('query', 'N/A')}
- **生成SQL**: `{f.get('generated_sql', 'N/A')}`
- **错误类型**: {error_labels.get(f.get('error_type', ''), f.get('error_type', 'N/A')) if error_labels else f.get('error_type', 'N/A')}
- **错误信息**: {f.get('error_message', '无')[:200]}
- **耗时**: {f.get('duration_seconds', 'N/A')}s

"""
    else:
        report += "（无失败案例）\n"

    report += "\n---\n*本报告由 Text2SQL 实验框架自动生成。*\n"
    return report


# ============================================================
# 辅助函数
# ============================================================

def _fmt_pct(val) -> str:
    """格式化百分比。"""
    if val is None or val == "N/A":
        return "N/A"
    if isinstance(val, float):
        return f"{val * 100:.1f}%"
    return str(val)


# Re-export error labels for external use
error_labels = {
    "syntax_error": "语法错误",
    "column_not_found": "字段不存在",
    "table_not_found": "表不存在",
    "execution_error": "执行异常",
    "timeout": "超时",
    "logical_error": "逻辑错误（可执行但结果不符）",
    "no_sql_generated": "未生成SQL",
    "security_blocked": "安全拦截",
    "unknown": "未知错误",
}
