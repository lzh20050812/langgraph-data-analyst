"""
Report Agent 可读性评估脚本。

评估方式：
1. LLM-as-Judge：用 LLM 对报告从完整性、洞察深度、可执行性、可读性四个维度打分
2. 结构完整性检查：验证报告是否包含规定的六大章节
3. 启发式质量指标：字数、段落数、建议数量等

测试集：5 条综合查询，触发 mixed 意图（analysis + prediction → report）

输出格式：遵循 eval_schema / eval_sql / eval_prediction 的评估报告格式
  {"summary": {...}, "details": [...]}
"""

import json
import re
import sys
import time
from pathlib import Path
from typing import List, Dict

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ============================================================
# 测试查询集（5条综合查询，触发 report 生成）
# ============================================================

REPORT_TEST_QUERIES: List[Dict] = [
    {
        "id": 1,
        "query": "分析客户价值并进行客户聚类，预测流失风险，生成综合经营报告",
        "description": "全链路综合分析",
        "expected_sections": ["核心发现", "客户价值", "运营绩效", "预测", "策略建议"],
    },
    {
        "id": 2,
        "query": "对当前经营状况做全面诊断，给出改善建议",
        "description": "经营诊断",
        "expected_sections": ["核心发现", "运营绩效", "策略建议"],
    },
    {
        "id": 3,
        "query": "做客户流失分析并预测未来销售趋势，输出详细报告",
        "description": "流失+销售预测",
        "expected_sections": ["核心发现", "预测", "策略建议"],
    },
    {
        "id": 4,
        "query": "综合评估客户生命周期价值，给出提升复购率的策略报告",
        "description": "客户生命周期+策略",
        "expected_sections": ["核心发现", "客户价值", "策略建议"],
    },
    {
        "id": 5,
        "query": "生成一份包含客户分群、运营指标和未来预测的完整经营分析报告",
        "description": "完整经营分析报告",
        "expected_sections": ["核心发现", "客户价值", "运营绩效", "预测", "策略建议"],
    },
]

# ============================================================
# LLM-as-Judge Prompt
# ============================================================

JUDGE_SYSTEM_PROMPT = """你是一位报告质量评审专家。请对以下由 AI 生成的经营分析报告进行多维度评分。

评分维度（每个维度 1-10 分）：
1. **完整性（completeness）**：报告是否覆盖了用户问题的核心需求？是否遗漏关键分析？
2. **洞察深度（insightfulness）**：是否解释了数据背后的"为什么"？是否提供了深层次的业务解读？
3. **可执行性（actionability）**：策略建议是否具体、可落地？是否给出了量化的预期效果？
4. **可读性（readability）**：结构是否清晰？语言是否流畅？是否适合管理层阅读？
5. **数据引用准确性（data_accuracy）**：引用的数字是否与输入数据一致？是否存在编造数据？

评分标准：
- 9-10: 优秀，可直接用于商业决策
- 7-8: 良好，需要少量修改
- 5-6: 一般，存在明显不足
- 3-4: 较差，关键维度未满足
- 1-2: 不可用，严重质量问题

输出 JSON 格式：
{
  "completeness": <int>,
  "insightfulness": <int>,
  "actionability": <int>,
  "readability": <int>,
  "data_accuracy": <int>,
  "overall_comment": "<一句话总评>",
  "strengths": ["<优点1>", "<优点2>"],
  "weaknesses": ["<不足1>", "<不足2>"]
}
"""


def judge_report(report_text: str, query: str) -> dict:
    """
    使用 LLM-as-Judge 对报告进行多维度评分。

    Returns:
        dict with completeness/insightfulness/actionability/readability/data_accuracy scores
    """
    from agents.llm import chat_with_json_output

    # 截断过长报告（节省 token）
    truncated = report_text[:4000] if len(report_text) > 4000 else report_text

    messages = [
        {"role": "system", "content": JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": f"""## 用户原始问题
{query}

## AI 生成的报告
{truncated}

## 任务
请对以上报告进行多维度评分，以 JSON 格式输出。"""},
    ]

    try:
        response = chat_with_json_output(messages, temperature=0.2, max_tokens=1024)
        # 尝试从响应中提取 JSON
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            scores = json.loads(json_match.group())
            return scores
        return {"error": "无法解析 LLM 评分响应", "raw": response[:500]}
    except Exception as e:
        return {"error": f"LLM-as-Judge 评分失败: {e}"}


# ============================================================
# 结构完整性检查
# ============================================================

def check_structure(report_text: str) -> dict:
    """
    检查报告是否包含规定章节。

    Returns:
        dict with section_presence, total_sections, found_sections
    """
    required_sections = [
        ("核心发现", ["核心发现", "摘要"]),
        ("客户价值分析", ["客户价值", "客户分", "RFM"]),
        ("运营绩效", ["运营绩效", "运营指标", "经营指标"]),
        ("预测预警", ["预测", "预警", "趋势"]),
        ("策略建议", ["策略建议", "建议", "措施"]),
        ("数据质量", ["数据质量", "质量说明", "可信度"]),
    ]

    found = []
    for section_name, keywords in required_sections:
        matched = any(kw in report_text for kw in keywords)
        if matched:
            found.append(section_name)

    return {
        "total_expected": len(required_sections),
        "found_count": len(found),
        "found_sections": found,
        "missing_sections": [s for s, _ in required_sections if s not in found],
        "completeness_pct": round(len(found) / len(required_sections) * 100, 1),
    }


# ============================================================
# 启发式指标
# ============================================================

def compute_heuristics(report_text: str) -> dict:
    """
    计算报告的启发式质量指标。

    Returns:
        dict with char_count, paragraph_count, suggestion_count, number_mentions
    """
    paragraphs = [p.strip() for p in report_text.split("\n\n") if p.strip()]
    suggestion_count = len(re.findall(r'建议|措施|策略|方案|行动', report_text))
    number_count = len(re.findall(r'\d+\.?\d*%?', report_text))

    return {
        "char_count": len(report_text),
        "paragraph_count": len(paragraphs),
        "suggestion_count": suggestion_count,
        "number_mentions": number_count,
        # 基础质量检查
        "too_short": len(report_text) < 500,
        "too_long": len(report_text) > 5000,
        "has_numbers": number_count >= 5,
    }


# ============================================================
# 主评估函数
# ============================================================

def run_report_evaluation(skip_llm_judge: bool = False) -> dict:
    """
    运行 Report Agent 评估。

    Args:
        skip_llm_judge: 如果为 True，跳过 LLM-as-Judge 评分（只做结构+启发式评估）

    Returns:
        {"summary": {...}, "details": [...]}
    """
    from agents.planner import run_query
    from config.settings import get_settings

    settings = get_settings()
    llm_available = bool(settings.LLM_API_KEY and "your_" not in settings.LLM_API_KEY)

    if skip_llm_judge or not llm_available:
        print("[WARN] 跳过 LLM-as-Judge 评分（LLM 不可用或手动跳过）")
        skip_llm_judge = True

    results = []
    stats = {
        "total": len(REPORT_TEST_QUERIES),
        "generated": 0,
        "avg_char_count": 0,
        "avg_structure_completeness": 0,
        "judge_scores": {} if not skip_llm_judge else "skipped",
    }

    all_scores = {
        "completeness": [],
        "insightfulness": [],
        "actionability": [],
        "readability": [],
        "data_accuracy": [],
    }

    print("=" * 60)
    print("Report Agent 评估")
    print("=" * 60)

    for i, test in enumerate(REPORT_TEST_QUERIES):
        print(f"\n[{i+1}/{len(REPORT_TEST_QUERIES)}] {test['description']}")
        print(f"  查询: {test['query'][:60]}...")

        result = {
            "id": test["id"],
            "query": test["query"],
            "description": test["description"],
            "report": None,
            "error": None,
            "structure": None,
            "heuristics": None,
            "judge_scores": None,
        }

        try:
            # 运行完整链路
            print("  执行 LangGraph 链路...")
            state = run_query(test["query"])

            if state.get("error"):
                result["error"] = state["error"]
                print(f"  ✗ Error: {state['error'][:100]}")
                results.append(result)
                continue

            report_text = state.get("report", "")
            if not report_text or len(report_text) < 100:
                result["error"] = "报告为空或过短"
                print(f"  ✗ 报告过短: {len(report_text)} 字符")
                results.append(result)
                continue

            result["report"] = report_text[:500] + "..."  # 只保留前 500 字符用于存储
            stats["generated"] += 1

            # 结构检查
            structure = check_structure(report_text)
            result["structure"] = structure
            print(f"  结构完整性: {structure['completeness_pct']}% "
                  f"({structure['found_count']}/{structure['total_expected']})")

            # 启发式指标
            heuristics = compute_heuristics(report_text)
            result["heuristics"] = heuristics
            print(f"  字数: {heuristics['char_count']}, "
                  f"段落: {heuristics['paragraph_count']}, "
                  f"建议提及: {heuristics['suggestion_count']}")

            # LLM-as-Judge
            if not skip_llm_judge:
                print("  LLM-as-Judge 评分中...")
                judge = judge_report(report_text, test["query"])
                result["judge_scores"] = judge
                if "error" not in judge:
                    for dim in all_scores:
                        if dim in judge and isinstance(judge[dim], (int, float)):
                            all_scores[dim].append(judge[dim])
                    avg = sum(
                        judge.get(d, 0) for d in all_scores
                        if isinstance(judge.get(d), (int, float))
                    ) / max(1, sum(1 for d in all_scores if isinstance(judge.get(d), (int, float))))
                    print(f"  LLM 评分: avg={avg:.1f}/10 | {judge.get('overall_comment', '')[:80]}")
                else:
                    print(f"  LLM 评分失败: {judge['error'][:80]}")

            # 避免频繁调用 API
            time.sleep(1)

        except Exception as e:
            result["error"] = str(e)[:300]
            print(f"  ✗ Exception: {e}")

        results.append(result)

    # 汇总统计
    if all_scores["completeness"]:
        stats["judge_scores"] = {
            dim: {
                "mean": round(sum(vals) / len(vals), 2) if vals else 0,
                "min": min(vals) if vals else 0,
                "max": max(vals) if vals else 0,
                "count": len(vals),
            }
            for dim, vals in all_scores.items()
        }

    structure_rates = [
        r["structure"]["completeness_pct"]
        for r in results if r["structure"]
    ]
    if structure_rates:
        stats["avg_structure_completeness"] = round(
            sum(structure_rates) / len(structure_rates), 1
        )

    char_counts = [
        r["heuristics"]["char_count"]
        for r in results if r["heuristics"]
    ]
    if char_counts:
        stats["avg_char_count"] = int(sum(char_counts) / len(char_counts))

    return {"summary": stats, "details": results}


def print_summary(report: dict) -> None:
    """打印可读的评估摘要。"""
    s = report["summary"]
    details = report["details"]

    print("\n" + "=" * 60)
    print("Report Agent 评估结果")
    print("=" * 60)
    print(f"  测试查询数: {s['total']}")
    print(f"  成功生成: {s['generated']}")
    print(f"  平均字数: {s.get('avg_char_count', 'N/A')}")
    print(f"  平均结构完整度: {s.get('avg_structure_completeness', 'N/A')}%")

    # LLM-as-Judge 评分摘要
    judge = s.get("judge_scores", {})
    if judge and judge != "skipped":
        print(f"\n--- LLM-as-Judge 多维度评分 (1-10) ---")
        dim_labels = {
            "completeness": "完整性",
            "insightfulness": "洞察深度",
            "actionability": "可执行性",
            "readability": "可读性",
            "data_accuracy": "数据引用准确性",
        }
        for dim, label in dim_labels.items():
            if dim in judge:
                d = judge[dim]
                print(f"  {label}: 均值={d['mean']}, 范围=[{d['min']}-{d['max']}] (n={d['count']})")
    elif judge == "skipped":
        print("\n  LLM-as-Judge: 已跳过")

    # 各用例详情
    print(f"\n--- 各用例详情 ---")
    for r in details:
        status = "✓" if r["report"] else "✗"
        struct = r.get("structure", {})
        heur = r.get("heuristics", {})
        judge_s = r.get("judge_scores", {})
        print(f"  [{status}] [{r['id']}] {r['description']}")

        if struct:
            missing = struct.get("missing_sections", [])
            if missing:
                print(f"     缺失章节: {', '.join(missing)}")
        if heur:
            print(f"     字数={heur.get('char_count', '?')}, "
                  f"建议提及={heur.get('suggestion_count', '?')}")
        if judge_s and "error" not in judge_s:
            avg = sum(
                v for k, v in judge_s.items()
                if k in ("completeness", "insightfulness", "actionability",
                         "readability", "data_accuracy")
            ) / 5
            print(f"     LLM评分: {avg:.1f}/10 — {judge_s.get('overall_comment', '')[:80]}")
        if r.get("error"):
            print(f"     错误: {r['error'][:120]}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Report Agent 可读性评估")
    parser.add_argument("--skip-judge", action="store_true",
                        help="跳过 LLM-as-Judge 评分（只做结构+启发式）")
    parser.add_argument("--output", type=str, default=None,
                        help="输出报告路径")
    args = parser.parse_args()

    report = run_report_evaluation(skip_llm_judge=args.skip_judge)
    print_summary(report)

    # 保存报告
    output_dir = Path(__file__).resolve().parent.parent / "data" / "processed"
    output_dir.mkdir(exist_ok=True)
    output_path = args.output or str(output_dir / "eval_report.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n[OK] 评估报告已保存: {output_path}")


if __name__ == "__main__":
    main()
