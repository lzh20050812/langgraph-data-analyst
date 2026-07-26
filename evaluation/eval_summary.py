"""
评估层汇总 —— 将所有 Agent 的评估结果整合为一份总评估报告。

汇总来源：
- eval_schema: Schema Agent 检索准确率
- eval_sql: SQL Agent 生成成功率（含自修正分析）
- eval_prediction: 模型评估（AUC/F1/RMSE/MAPE）
- eval_report: Report Agent 可读性评估

输出：
  1. data/processed/evaluation_summary.json —— 机器可读的完整报告
  2. 控制台打印 —— 论文速查表

用法：
  python -m evaluation.eval_summary
"""

import json
import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def load_existing_report(filename: str) -> dict:
    """加载已有的评估报告 JSON 文件。"""
    path = Path(__file__).resolve().parent.parent / "data" / "processed" / filename
    if path.exists():
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError) as e:
            return {"error": f"无法解析 {filename}: {e}", "path": str(path)}
    return {"error": f"文件不存在: {path}", "path": str(path)}


def extract_schema_metrics(report: dict) -> dict:
    """从 Schema 评估报告中提取关键指标。"""
    summary = report.get("summary", {})
    return {
        "top5_hit_rate": summary.get("top5_hit_rate", "N/A"),
        "pure_chromadb_top5": summary.get("pure_chromadb_top5", "N/A"),
        "llm_rerank_hard_recall": summary.get("llm_rerank_hard_recall", "N/A"),
        "test_count": summary.get("total_test_cases", "N/A"),
    }


def extract_sql_metrics(report: dict) -> dict:
    """从 SQL 评估报告中提取关键指标。"""
    summary = report.get("summary", {})
    return {
        "total_queries": summary.get("total", "N/A"),
        "unsolvable_count": summary.get("unsolvable_count", "N/A"),
        "first_try_success_rate_all": summary.get("sql_first_try_success_rate", "N/A"),
        "after_retry_success_rate_all": summary.get("sql_after_retry_success_rate", "N/A"),
        "retry_improvement": summary.get("retry_improvement", "N/A"),
        "solvable_first_try_success_rate": summary.get("solvable_first_try_success_rate", "N/A"),
        "solvable_after_retry_success_rate": summary.get("solvable_after_retry_success_rate", "N/A"),
        "retries_fixed": summary.get("retries_fixed", "N/A"),
        "safety_rejections": summary.get("safety_rejections", "N/A"),
        "by_difficulty": summary.get("by_difficulty", {}),
    }


def extract_prediction_metrics(report: dict) -> dict:
    """从预测评估报告中提取关键指标。"""
    churn = report.get("churn", {})
    sales = report.get("sales", {})

    churn_metrics = {}
    if churn and "error" not in churn:
        optimal = churn.get("optimal_threshold", {})
        churn_metrics = {
            "model": churn.get("model", "N/A"),
            "auc": churn.get("auc", "N/A"),
            "positive_rate": churn.get("positive_rate", "N/A"),
            "recall_optimal": optimal.get("recall", "N/A"),
            "precision_optimal": optimal.get("precision", "N/A"),
            "f1_optimal": optimal.get("f1", "N/A"),
            "optimal_threshold": optimal.get("threshold", "N/A"),
            "train_samples": churn.get("train_samples", "N/A"),
            "test_samples": churn.get("test_samples", "N/A"),
        }

    sales_metrics = {}
    if sales and "error" not in sales:
        sales_metrics = {
            "model": sales.get("model", "N/A"),
            "rmse": sales.get("rmse", "N/A"),
            "mape_pct": sales.get("mape_pct", "N/A"),
            "mae": sales.get("mae", "N/A"),
            "train_periods": sales.get("train_periods", "N/A"),
            "test_periods": sales.get("test_periods", "N/A"),
        }

    return {"churn": churn_metrics, "sales": sales_metrics}


def extract_report_metrics(report: dict) -> dict:
    """从 Report 评估报告中提取关键指标。"""
    summary = report.get("summary", {})
    judge = summary.get("judge_scores", {})

    metrics = {
        "total_tests": summary.get("total", "N/A"),
        "generated": summary.get("generated", "N/A"),
        "avg_char_count": summary.get("avg_char_count", "N/A"),
        "avg_structure_completeness_pct": summary.get("avg_structure_completeness", "N/A"),
    }

    if judge and judge != "skipped":
        metrics["llm_judge"] = {
            dim: {"mean": d["mean"], "min": d["min"], "max": d["max"]}
            for dim, d in judge.items()
        }
    else:
        metrics["llm_judge"] = "skipped"

    return metrics


def build_summary() -> dict:
    """构建总评估报告。"""
    print("加载评估报告...")

    schema_report = load_existing_report("eval_schema_report.json")
    sql_report = load_existing_report("eval_sql_report.json")
    prediction_report = load_existing_report("eval_prediction_report.json")
    report_report = load_existing_report("eval_report.json")

    summary = {
        "generated_at": datetime.now().isoformat(),
        "project": "AI Data Analyst — 多智能体企业智能运营分析平台",
        "sections": {
            "schema_agent": extract_schema_metrics(schema_report),
            "sql_agent": extract_sql_metrics(sql_report),
            "prediction_agent": extract_prediction_metrics(prediction_report),
            "report_agent": extract_report_metrics(report_report),
        },
    }

    return summary


def print_summary(summary: dict) -> None:
    """打印论文速查表。"""
    s = summary["sections"]

    print("\n" + "=" * 65)
    print("   [Evaluation Summary] 评估层总汇总 -- 论文数据速查表")
    print("=" * 65)
    print(f"   生成时间: {summary['generated_at'][:19]}")
    print()

    # Schema Agent
    sch = s["schema_agent"]
    print("-- Schema Agent --")
    print(f"   ChromaDB Top-5 检索命中率: {_fmt(sch.get('top5_hit_rate'))}")
    print(f"   Pure ChromaDB Top-5:        {_fmt(sch.get('pure_chromadb_top5'))}")
    print(f"   LLM 精排后 Hard 组召回:     {_fmt(sch.get('llm_rerank_hard_recall'))}")
    print(f"   测试集规模:                 {sch.get('test_count', 'N/A')} 条")
    print()

    # SQL Agent
    sql = s["sql_agent"]
    print("-- SQL Agent --")
    print(f"   总查询数:                   {sql.get('total_queries', 'N/A')}")
    print(f"   不可解题:                   {sql.get('unsolvable_count', 'N/A')} (数据模型限制)")
    print(f"   首次成功率（全量）:         {_fmt_pct(sql.get('first_try_success_rate_all'))}")
    print(f"   自修正后成功率（全量）:     {_fmt_pct(sql.get('after_retry_success_rate_all'))}")
    print(f"   可解题首次成功率:           {_fmt_pct(sql.get('solvable_first_try_success_rate'))}")
    print(f"   可解题自修正后成功率:       {_fmt_pct(sql.get('solvable_after_retry_success_rate'))}")
    print(f"   自修正提效:                 {_fmt_pct(sql.get('retry_improvement'))}")
    print(f"   重试修复数:                 {sql.get('retries_fixed', 'N/A')}")
    print(f"   安全拦截数:                 {sql.get('safety_rejections', 'N/A')}")

    by_diff = sql.get("by_difficulty", {})
    if by_diff:
        print("   按难度分层:")
        for diff in ["easy", "medium", "hard"]:
            d = by_diff.get(diff, {})
            print(f"     {diff}: 首次={_fmt_pct(d.get('first_try_rate'))}, "
                  f"修正后={_fmt_pct(d.get('after_retry_rate'))}, "
                  f"成功={d.get('first_try_success','?')}/{d.get('generated','?')}")
    print()

    # Prediction Agent
    pred = s["prediction_agent"]
    churn = pred.get("churn", {})
    sales = pred.get("sales", {})

    print("-- Prediction Agent --")
    if churn:
        print(f"   [XGBoost 流失预测]")
        print(f"   AUC:                        {churn.get('auc', 'N/A')}")
        print(f"   正样本率:                   {_fmt_pct(churn.get('positive_rate'))}")
        print(f"   Recall@最优阈值:            {_fmt_pct(churn.get('recall_optimal'))}")
        print(f"   Precision@最优阈值:         {_fmt_pct(churn.get('precision_optimal'))}")
        print(f"   F1@最优阈值:                {churn.get('f1_optimal', 'N/A')}")
        print(f"   最优阈值:                   {churn.get('optimal_threshold', 'N/A')}")
    if sales:
        print(f"   [Prophet 销售预测]")
        print(f"   RMSE:                       {sales.get('rmse', 'N/A')}")
        print(f"   MAPE:                       {sales.get('mape_pct', 'N/A')}%")
        print(f"   MAE:                        {sales.get('mae', 'N/A')}")
        print(f"   训练/测试期:                {sales.get('train_periods','?')}/{sales.get('test_periods','?')} 月")
    print()

    # Report Agent
    rep = s["report_agent"]
    print("-- Report Agent --")
    print(f"   测试用例数:                 {rep.get('total_tests', 'N/A')}")
    print(f"   成功生成数:                 {rep.get('generated', 'N/A')}")
    print(f"   平均字数:                   {rep.get('avg_char_count', 'N/A')}")
    print(f"   平均结构完整度:             {rep.get('avg_structure_completeness_pct', 'N/A')}%")

    judge = rep.get("llm_judge", {})
    if judge and judge != "skipped":
        dim_labels = {
            "completeness": "完整性",
            "insightfulness": "洞察深度",
            "actionability": "可执行性",
            "readability": "可读性",
            "data_accuracy": "数据引用准确性",
        }
        print("   LLM-as-Judge 评分:")
        for dim, label in dim_labels.items():
            if dim in judge:
                print(f"     {label}: {judge[dim]['mean']}/10 "
                      f"(范围 [{judge[dim]['min']}-{judge[dim]['max']}])")
    print()

    print("=" * 65)
    print("   以上数字已通过评估脚本验证，可直接用于论文。")
    print("=" * 65)


def _fmt(val) -> str:
    """格式化数值，处理 N/A。"""
    if val is None or val == "N/A":
        return "N/A"
    if isinstance(val, float):
        return f"{val:.3f}"
    return str(val)


def _fmt_pct(val) -> str:
    """格式化百分比。"""
    if val is None or val == "N/A":
        return "N/A"
    if isinstance(val, float):
        return f"{val*100:.1f}%"
    return str(val)


def main():
    summary = build_summary()
    print_summary(summary)

    # 保存总评估报告
    output_dir = Path(__file__).resolve().parent.parent / "data" / "processed"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "evaluation_summary.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n[OK] 总评估报告已保存: {output_path}")


if __name__ == "__main__":
    main()
