"""
Text2SQL 自动评估实验 —— 技术评测就绪版本。

评估 SQL Agent 在 50 条不同难度自然语言查询上的表现。

运行方式：
    python -m evaluation.experiments.text2sql_experiment
    python -m evaluation.experiments.text2sql_experiment --max-samples 10
    python -m evaluation.experiments.text2sql_experiment --test-data path/to/queries.json

架构设计（支持对比实验）：
    - self.run_method 切换调用方式
    - 子类覆盖 _call_agent() 接入不同方案
"""

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from evaluation.framework.base import ExperimentBase, ExperimentResult
from evaluation.framework.result import ResultManager, save_json, load_json, save_csv
from evaluation.metrics.text2sql_metrics import (
    classify_sql_error,
    compare_result_sets,
    compute_all_metrics,
    generate_table_overall,
    generate_table_by_difficulty,
    generate_table_error_types,
    generate_experiment_report,
)


class Text2SQLExperiment(ExperimentBase):
    """Text2SQL 生成准确率与执行准确率实验（技术评测就绪版）。"""

    name = "text2sql_accuracy"
    description = "评估 SQL Agent 在 50 条不同难度自然语言查询上的 SQL 生成准确率与执行准确率"
    version = "1.0"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.test_data_path: Optional[Path] = None
        self.test_queries: List[Dict[str, Any]] = []
        self.run_method: str = "full_pipeline"
        self._adapter = None
        self._db_available: bool = False

    # ---- 数据加载 ----

    def load_data(self) -> int:
        data_path = self.config.get("data.test_data_path") or str(
            Path(__file__).resolve().parent.parent / "data" / "text2sql" / "test_queries.json"
        )
        self.test_data_path = Path(data_path)
        if not self.test_data_path.exists():
            raise FileNotFoundError(f"测试数据文件不存在: {self.test_data_path}")

        self.test_queries = load_json(self.test_data_path)
        self.logger.info(f"加载测试数据: {self.test_data_path}")

        # 应用最大样本限制
        limits = self.config.get("limits", {})
        max_n = limits.get("max_samples", 0) if isinstance(limits, dict) else 0
        if max_n <= 0:
            max_n = self.config.get("limits.max_samples", 0)
        if max_n > 0 and len(self.test_queries) > max_n:
            self.test_queries = self.test_queries[:max_n]
            self.logger.info(f"限制样本数为 {max_n}")

        return len(self.test_queries)

    # ---- 准备 ----

    def setup(self) -> None:
        from config.settings import get_settings
        from storage.db_adapter import get_available_adapter

        try:
            self._adapter = get_available_adapter()
            self._db_available = self._adapter.check_connection()
            self.logger.info(f"数据库: {'MySQL' if self._db_available else 'DuckDB(降级)'}")
        except Exception as e:
            self.logger.warning(f"数据库初始化异常: {e}")
            self._adapter = None
            self._db_available = False

        settings = get_settings()
        llm_ok = bool(settings.LLM_API_KEY and "your_" not in settings.LLM_API_KEY)
        if not llm_ok:
            raise RuntimeError("LLM API key 未配置")
        self.logger.info(f"LLM: {settings.LLM_MODEL}")

        # 记录测试集构成
        diffs = {}
        for q in self.test_queries:
            d = q.get("difficulty", "unknown")
            diffs[d] = diffs.get(d, 0) + 1
        self.logger.info(f"测试集构成: {diffs}")

    # ---- 执行 ----

    def run(self) -> None:
        total = len(self.test_queries)

        for idx, test in enumerate(self.test_queries):
            qid = test["id"]
            query = test["query"]
            difficulty = test.get("difficulty", "unknown")
            expected_sql = test.get("expected_sql", "")

            self.logger.info(f"[{idx + 1}/{total}] Q{qid} [{difficulty}] {query[:60]}...")

            detail = {
                "id": qid,
                "query": query,
                "difficulty": difficulty,
                "category": test.get("category", ""),
                "generated_sql": None,
                "expected_sql": expected_sql,
                "execution_success": False,
                "result_correct": False,
                "error_message": None,
                "error_type": None,
                "duration_seconds": 0.0,
                "retry_count": 0,
                "row_count_generated": 0,
                "row_count_expected": 0,
                "query_result_sample": None,
                "expected_result_sample": None,
            }

            t_start = time.time()

            try:
                # ---- Step 1: 调用完整 LangGraph 流程 ----
                agent_output = self._call_agent(query)
                detail["generated_sql"] = agent_output.get("sql")
                detail["retry_count"] = agent_output.get("retry_count", 0)

                if agent_output.get("error"):
                    # Agent 返回了错误（SQL 生成或执行失败）
                    error_msg = agent_output["error"]
                    detail["error_message"] = error_msg
                    detail["error_type"] = classify_sql_error(error_msg, detail["generated_sql"])
                    self.logger.record_sample(qid, False, f"Error: {detail['error_type']}")

                elif agent_output.get("query_result") is not None:
                    # SQL 执行成功
                    detail["execution_success"] = True
                    gen_result = agent_output["query_result"]
                    detail["row_count_generated"] = len(gen_result)
                    detail["query_result_sample"] = _sample_rows(gen_result, 5)

                    # ---- Step 2: 执行 expected_sql 获取标准答案 ----
                    if expected_sql and self._adapter and self._db_available:
                        try:
                            exp_result = self._adapter.execute_sql(expected_sql)
                            detail["row_count_expected"] = len(exp_result) if exp_result else 0
                            detail["expected_result_sample"] = _sample_rows(exp_result, 5)

                            # ---- Step 3: 比对结果集 ----
                            is_correct = compare_result_sets(gen_result, exp_result)
                            detail["result_correct"] = is_correct
                            if not is_correct:
                                detail["error_type"] = "logical_error"
                                detail["error_message"] = (
                                    f"结果不匹配: 生成{len(gen_result)}行, 期望{len(exp_result) if exp_result else 0}行"
                                )
                        except Exception as e:
                            self.logger.warning(f"Expected SQL 执行失败: {e}")
                            detail["error_message"] = f"Expected SQL error: {e}"
                            detail["error_type"] = classify_sql_error(str(e), expected_sql)
                    elif not self._db_available:
                        detail["error_message"] = "数据库不可用，无法执行结果比对"
                        detail["error_type"] = "execution_error"

                    self.logger.record_sample(
                        qid,
                        detail["result_correct"],
                        f"gen={detail['row_count_generated']}r exp={detail['row_count_expected']}r",
                    )

                else:
                    # 既没有 error 也没有 query_result
                    detail["error_message"] = "Agent 未生成 SQL 且未返回错误"
                    detail["error_type"] = "no_sql_generated"
                    self.logger.record_sample(qid, False, "No SQL and no error")

            except Exception as e:
                detail["error_message"] = str(e)
                detail["error_type"] = classify_sql_error(str(e), detail.get("generated_sql"))
                self.logger.record_sample(qid, False, f"Exception: {type(e).__name__}")

            detail["duration_seconds"] = round(time.time() - t_start, 3)
            self.result.details.append(detail)

        # 汇总
        self.result.sample_count = total
        self.result.success_count = sum(1 for d in self.result.details if d["result_correct"])
        self.result.failure_count = total - self.result.success_count

    # ---- 指标 ----

    def evaluate(self) -> Dict[str, Any]:
        self.result.metrics = compute_all_metrics(self.result.details)

        m = self.result.metrics
        self.logger.info(f"SQL生成成功率: {m['sql_generation_success_rate']:.2%}")
        self.logger.info(f"SQL执行成功率: {m['execution_success_rate']:.2%}")
        self.logger.info(f"执行准确率(EX): {m['execution_accuracy']:.2%}")
        self.logger.info(f"平均响应时间: {m['avg_response_time_seconds']:.2f}s")
        self.logger.info(f"错误分布: {m['error_distribution']}")

        for diff, stats in m.get("by_difficulty", {}).items():
            self.logger.info(
                f"  [{diff}] count={stats['count']} "
                f"GEN={stats['generation_success_rate']:.1%} "
                f"EXE={stats['execution_success_rate']:.1%} "
                f"EX={stats['execution_accuracy']:.1%}"
            )

        return self.result.metrics

    # ---- 结果保存（覆盖以生成技术评测表格和实验报告） ----

    def save_results(
        self,
        output_dir: Optional[Path] = None,
        formats: Optional[List[str]] = None,
    ) -> Dict[str, Path]:
        # 确定目录
        if output_dir is None:
            output_dir = Path(__file__).resolve().parent.parent / "results"
        raw_dir = Path(output_dir) / "raw"
        tables_dir = Path(output_dir) / "tables"

        saved = {}

        # 1. JSON 完整结果
        jpath = raw_dir / f"{self.name}_{_timestamp()}.json"
        save_json(self.result.to_dict(), jpath)
        saved["json"] = jpath

        # 2. 逐样本 CSV
        cpath = raw_dir / f"{self.name}_details_{_timestamp()}.csv"
        save_csv(self.result.details, cpath)
        saved["details_csv"] = cpath

        # 3. 技术评测表格（3个表）
        import pandas as pd

        # 表1: 总体结果
        m = self.result.metrics
        df_overall = pd.DataFrame([
            {"指标": "测试样本数量", "数值": str(m.get("total_samples", ""))},
            {"指标": "SQL生成成功率", "数值": _fmt_pct(m.get("sql_generation_success_rate"))},
            {"指标": "SQL执行成功率(EXE)", "数值": _fmt_pct(m.get("execution_success_rate"))},
            {"指标": "执行准确率(EX)", "数值": _fmt_pct(m.get("execution_accuracy"))},
            {"指标": "平均响应时间", "数值": f"{m.get('avg_response_time_seconds', '')}s"},
        ])
        t1 = tables_dir / f"{self.name}_table1_overall_{_timestamp()}.csv"
        df_overall.to_csv(t1, index=False, encoding="utf-8-sig")
        saved["table1_overall"] = t1

        # 表2: 按难度分层
        by_diff = m.get("by_difficulty", {})
        diff_rows = []
        for diff in ["easy", "medium", "hard", "edge"]:
            d = by_diff.get(diff)
            if d:
                diff_rows.append({
                    "难度": diff, "样本数": d["count"],
                    "SQL生成成功率": _fmt_pct(d.get("generation_success_rate")),
                    "SQL执行成功率": _fmt_pct(d.get("execution_success_rate")),
                    "执行准确率(EX)": _fmt_pct(d.get("execution_accuracy")),
                    "平均耗时(s)": d.get("avg_response_time", ""),
                })
        if diff_rows:
            df_diff = pd.DataFrame(diff_rows)
            t2 = tables_dir / f"{self.name}_table2_difficulty_{_timestamp()}.csv"
            df_diff.to_csv(t2, index=False, encoding="utf-8-sig")
            saved["table2_difficulty"] = t2

        # 表3: 错误类型统计
        err_dist = m.get("error_distribution", {})
        if err_dist:
            error_labels = {
                "syntax_error": "语法错误", "column_not_found": "字段不存在",
                "table_not_found": "表不存在", "execution_error": "执行异常",
                "timeout": "超时", "logical_error": "逻辑错误",
                "no_sql_generated": "未生成SQL", "security_blocked": "安全拦截",
                "unknown": "未知错误",
            }
            err_rows = [
                {"错误类型": error_labels.get(e, e), "数量": c,
                 "占比": f"{c / sum(err_dist.values()) * 100:.1f}%"}
                for e, c in err_dist.items()
            ]
            df_err = pd.DataFrame(err_rows)
            t3 = tables_dir / f"{self.name}_table3_errors_{_timestamp()}.csv"
            df_err.to_csv(t3, index=False, encoding="utf-8-sig")
            saved["table3_errors"] = t3

        # 4. 实验报告 (Markdown)
        # 计算真实耗时（base.execute() 的 finally 在 save_results 之后才设置 duration_seconds）
        from datetime import datetime
        elapsed = 0.0
        try:
            t0 = datetime.fromisoformat(self.result.started_at)
            elapsed = (datetime.now(t0.tzinfo) - t0).total_seconds()
        except Exception:
            elapsed = sum(d.get("duration_seconds", 0) for d in self.result.details)
        report_md = generate_experiment_report(
            self.name, m, self.result.details, round(elapsed, 1),
        )
        rpath = raw_dir / f"{self.name}_report_{_timestamp()}.md"
        rpath.write_text(report_md, encoding="utf-8")
        saved["report_md"] = rpath

        # 5. 技术评测 Markdown 表格合集
        md_tables = "\n".join([
            generate_table_overall(m),
            generate_table_by_difficulty(m),
            generate_table_error_types(m),
        ])
        mdpath = raw_dir / f"{self.name}_benchmark_tables_{_timestamp()}.md"
        mdpath.write_text(md_tables, encoding="utf-8")
        saved["benchmark_tables_md"] = mdpath

        self.logger.info(f"结果已保存: {len(saved)} 个文件")
        for k, v in saved.items():
            self.logger.info(f"  [{k}] {v.name}")

        return saved

    # ---- Agent 调用接口（可覆盖） ----

    def _call_agent(self, user_query: str) -> Dict[str, Any]:
        if self.run_method == "full_pipeline":
            return self._call_via_langgraph(user_query)
        raise ValueError(f"Unknown run_method: {self.run_method}")

    def _call_via_langgraph(self, user_query: str) -> Dict[str, Any]:
        from agents.planner import run_query
        state = run_query(user_query)
        return {
            "sql": state.get("sql"),
            "query_result": state.get("query_result"),
            "error": state.get("error"),
            "retry_count": state.get("sql_retries", 0),
            "intent": state.get("intent", ""),
        }


# ---- 辅助 ----

def _timestamp() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _sample_rows(rows: list, n: int = 5) -> list:
    if not rows:
        return []
    return rows[:n]


def _fmt_pct(val) -> str:
    if val is None:
        return ""
    if isinstance(val, float):
        return f"{val * 100:.1f}%"
    return str(val)


# ---- 入口 ----

def run_text2sql_experiment(
    test_data_path: Optional[str] = None,
    max_samples: int = 0,
    run_method: str = "full_pipeline",
) -> ExperimentResult:
    config = {
        "experiment": {"name": "text2sql_accuracy", "version": "1.0"},
        "run_method": run_method,
        "limits": {"max_samples": max_samples},
    }
    if test_data_path:
        config["data"] = {"test_data_path": test_data_path}
    return Text2SQLExperiment(config).execute()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Text2SQL 自动评估实验")
    parser.add_argument("--test-data", type=str, default=None)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--method", type=str, default="full_pipeline")
    parser.add_argument("--output-dir", type=str, default=None)
    args = parser.parse_args()

    if args.max_samples > 0:
        print(f"[INFO] 限制样本数: {args.max_samples}（快速验证模式）")

    result = run_text2sql_experiment(
        test_data_path=args.test_data,
        max_samples=args.max_samples,
        run_method=args.method,
    )

    print()
    print("=" * 60)
    print(f"Text2SQL 实验: {result.status}")
    print(f"耗时: {result.duration_seconds:.1f}s")
    m = result.metrics
    print(f"SQL生成成功率: {_fmt_pct(m.get('sql_generation_success_rate'))}")
    print(f"SQL执行成功率: {_fmt_pct(m.get('execution_success_rate'))}")
    print(f"执行准确率(EX): {_fmt_pct(m.get('execution_accuracy'))}")
    print(f"平均响应时间: {m.get('avg_response_time_seconds', 'N/A')}s")
    print(f"错误分布: {m.get('error_distribution', {})}")
    print("=" * 60)

    return 0 if result.status == "success" else 1


if __name__ == "__main__":
    sys.exit(main())
