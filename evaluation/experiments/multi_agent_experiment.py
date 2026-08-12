"""
Multi-Agent 协同效果验证实验 —— 技术评测正式评测核心实验。

对比三种方案：
  - Single LLM: 纯 LLM，无 Agent，无数据库
  - LLM + RAG: Schema RAG 检索 + LLM
  - Full Multi-Agent: 完整 LangGraph 多智能体流程

支持消融实验和 LLM-as-Judge 报告质量评估。

运行方式：
  # 单模式
  python -m evaluation.experiments.multi_agent_experiment --mode full_multi_agent
  python -m evaluation.experiments.multi_agent_experiment --mode single_llm
  python -m evaluation.experiments.multi_agent_experiment --mode rag_llm

  # 三方案对比 (含LLM-as-Judge)
  python -m evaluation.experiments.multi_agent_experiment --compare --judge

  # 消融实验
  python -m evaluation.experiments.multi_agent_experiment --ablation

  # 完整技术评测实验 (三方案+消融+报告)
  python -m evaluation.experiments.multi_agent_experiment --full
"""

import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from evaluation.framework.base import ExperimentBase, ExperimentResult
from evaluation.framework.result import save_json, load_json, save_csv
from evaluation.metrics.multi_agent_metrics import (
    compute_all_multi_agent_metrics,
    evaluate_all_reports,
    generate_table1_comparison,
    generate_table2_task_types,
    generate_table3_resources,
    generate_table4_ablation,
    generate_full_experiment_report,
)
from evaluation.metrics.text2sql_metrics import compare_result_sets


# ============================================================
# Multi-Agent 协同实验
# ============================================================

class MultiAgentExperiment(ExperimentBase):
    """Multi-Agent 协同效果验证实验。

    支持三种运行模式:
        - single_llm: 纯 LLM 直接回答（Baseline 1）
        - rag_llm: Schema RAG + LLM（Baseline 2）
        - full_multi_agent: 完整 LangGraph 多智能体流程（本文方案）

    消融变体通过 ablation 参数控制:
        - no_rag: 多智能体但无 Schema 检索
        - no_self_correction: 多智能体但 SQL 不自修正
    """

    name = "multi_agent_comparison"
    description = "Multi-Agent 协同 vs Single LLM vs LLM+RAG 对比实验"
    version = "1.0"

    # 合法的模式名
    VALID_MODES = ("single_llm", "rag_llm", "full_multi_agent")
    VALID_ABLATIONS = (None, "no_rag", "no_multi_agent", "no_self_correction")

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.test_queries: List[Dict[str, Any]] = []
        # 运行模式
        self.mode: str = self.config.get("mode", "full_multi_agent")
        self.ablation: Optional[str] = self.config.get("ablation", None)
        # 依赖
        self._embedder = None
        self._adapter = None
        self._db_available = False

        # 根据 ablation 调整 name
        if self.ablation:
            self.name = f"multi_agent_ablation_{self.ablation}"
        else:
            self.name = f"multi_agent_{self.mode}"

    # ---- 数据加载 ----

    def load_data(self) -> int:
        data_path = self.config.get(
            "data.test_data_path",
            str(Path(__file__).resolve().parent.parent / "data" / "multi_agent" / "test_queries.json"),
        )
        path = Path(data_path)
        if not path.exists():
            raise FileNotFoundError(f"Multi-Agent 测试数据文件不存在: {path}")

        self.test_queries = load_json(path)
        self.logger.info(f"加载 Multi-Agent 测试数据: {path}")

        limits = self.config.get("limits", {})
        max_n = limits.get("max_samples", 0) if isinstance(limits, dict) else 0
        if max_n > 0 and len(self.test_queries) > max_n:
            self.test_queries = self.test_queries[:max_n]
            self.logger.info(f"限制样本数为 {max_n}")

        # 统计
        cats = {}
        types = {}
        for q in self.test_queries:
            c = q.get("category", "unknown")
            cats[c] = cats.get(c, 0) + 1
            t = q.get("task_type", "unknown")
            types[t] = types.get(t, 0) + 1
        self.logger.info(f"类别构成: {cats}")
        self.logger.info(f"任务类型构成: {types}")

        return len(self.test_queries)

    # ---- 准备 ----

    def setup(self) -> None:
        from config.settings import get_settings
        from storage.db_adapter import get_available_adapter

        # LLM 配置检查
        settings = get_settings()
        llm_ok = bool(settings.LLM_API_KEY and "your_" not in settings.LLM_API_KEY)
        if not llm_ok:
            raise RuntimeError("LLM API key 未配置")

        self.logger.info(f"运行模式: {self.mode}")
        self.logger.info(f"消融配置: {self.ablation or '无'}")
        self.logger.info(f"LLM: {settings.LLM_MODEL}")

        # RAG 模式或 Full 模式需要 embedder
        if self.mode in ("rag_llm", "full_multi_agent") and self.ablation != "no_rag":
            from storage.chromadb.embedder import get_embedder
            self._embedder = get_embedder()
            if self._embedder.collection is None:
                self._embedder.build()

        # Full 模式需要数据库
        if self.mode == "full_multi_agent":
            try:
                self._adapter = get_available_adapter()
                self._db_available = self._adapter.check_connection()
                self.logger.info(f"数据库连接: {'成功' if self._db_available else '失败'}")
            except Exception as e:
                self.logger.warning(f"数据库初始化异常: {e}")
                self._db_available = False

    # ---- 执行 ----

    def run(self) -> None:
        total = len(self.test_queries)

        for idx, test in enumerate(self.test_queries):
            qid = test["id"]
            query = test["query"]
            task_type = test.get("task_type", "sql_query")
            category = test.get("category", "unknown")

            self.logger.info(f"[{idx + 1}/{total}] Q{qid} [{task_type}] {query[:60]}...")

            detail = {
                "id": qid,
                "query": query,
                "category": category,
                "task_type": task_type,
                "mode": self.mode,
                "ablation": self.ablation,
                "expected_tables": test.get("expected_tables", []),
                "expected_fields": test.get("expected_fields", []),
                "expected_sql": test.get("expected_sql"),
                "expected_insights": test.get("expected_insights", []),
                # 结果
                "success": False,
                "error": None,
                "duration_seconds": 0.0,
                # Agent 调用追踪
                "agent_call_count": 0,
                "llm_call_count": 0,
                "agent_trace": [],
                "agents_invoked": [],
                # SQL 相关
                "generated_sql": None,
                "execution_success": False,
                "query_result": None,
                "result_correct": False,
                # 分析/预测
                "analysis_result": None,
                "prediction_result": None,
                "governance_result": None,
                # 报告
                "report": None,
                "llm_output": None,
                "report_quality_score": None,
            }

            t_start = time.time()

            try:
                if self.ablation == "no_multi_agent":
                    # 消融：退化为单 Agent = Single LLM
                    self._run_single_llm(query, detail)
                elif self.mode == "single_llm":
                    self._run_single_llm(query, detail)
                elif self.mode == "rag_llm":
                    self._run_rag_llm(query, detail)
                elif self.mode == "full_multi_agent":
                    self._run_full_multi_agent(query, task_type, detail)
                else:
                    detail["error"] = f"未知运行模式: {self.mode}"

                # 判定成功
                detail["success"] = self._judge_success(detail, test)

            except Exception as e:
                detail["error"] = f"{type(e).__name__}: {str(e)[:200]}"
                detail["success"] = False
                self.logger.record_sample(qid, False, f"Exception: {e}")

            detail["duration_seconds"] = round(time.time() - t_start, 3)

            status = "OK" if detail["success"] else "FAIL"
            extra = f"time={detail['duration_seconds']}s"
            if detail.get("error"):
                extra += f" err={detail['error'][:40]}"
            self.logger.record_sample(qid, detail["success"], extra)

            self.result.details.append(detail)

        # 汇总
        self.result.sample_count = total
        self.result.success_count = sum(1 for d in self.result.details if d["success"])
        self.result.failure_count = total - self.result.success_count

    # ---- 模式 1: Single LLM ----

    def _run_single_llm(self, query: str, detail: Dict[str, Any]) -> None:
        """纯 LLM 模式：不调用任何 Agent，不访问数据库。"""
        from agents.llm import chat

        detail["agent_call_count"] = 0
        detail["agents_invoked"] = []
        detail["agent_trace"].append("single_llm: LLM 直接回答")

        prompt = f"""你是一位电商数据分析专家。请根据你的知识回答以下数据分析问题。

注意：
- 你无法访问数据库，请基于电商行业常识给出分析
- 如果是数据查询类问题，说明需要查询哪些数据
- 如果是分析预测类问题，给出分析框架和方法建议

问题：{query}

请给出专业、结构化的回答。"""

        messages = [
            {"role": "system", "content": "你是一位专业的电商数据分析专家，请给出结构化的分析回答。"},
            {"role": "user", "content": prompt},
        ]

        output = chat(messages, temperature=0.3, max_tokens=2048)
        detail["llm_output"] = output
        detail["llm_call_count"] = 1

    # ---- 模式 2: LLM + RAG ----

    def _run_rag_llm(self, query: str, detail: Dict[str, Any]) -> None:
        """LLM + Schema RAG 模式：检索表结构作为上下文，然后 LLM 回答。"""
        from agents.llm import chat

        detail["agents_invoked"] = ["rag_retrieval"]
        detail["agent_trace"].append("rag_llm: Schema RAG 检索 → LLM 回答")

        # 1. Schema RAG 检索
        rag_start = time.time()
        try:
            if self._embedder:
                schema_results = self._embedder.search(query, top_k=10)
            else:
                from storage.chromadb.embedder import get_embedder
                schema_results = get_embedder().search(query, top_k=10)
        except Exception as e:
            schema_results = []
            detail["agent_trace"].append(f"RAG 检索失败: {e}")

        rag_time = round(time.time() - rag_start, 3)

        # 2. 构建带有 Schema 上下文的 Prompt
        schema_context = ""
        if schema_results:
            tables_seen = set()
            for r in schema_results:
                t = r["table_name"]
                if t not in tables_seen:
                    schema_context += f"\n[{t}]"
                    tables_seen.add(t)
                schema_context += (
                    f"\n  {t}.{r['column_name']} ({r.get('dtype', '')})"
                    f" — {r.get('business_term', '')}"
                )

        prompt = f"""你是一位电商数据分析专家。以下是数据库中可用的表结构信息：

{schema_context if schema_context else '（无Schema信息可用）'}

请基于以上表结构回答用户问题。如果需要SQL查询，请写出对应的SQL语句。
如果需要对数据进行复杂分析（如RFM、预测等），请说明分析方法和预期结果。

用户问题：{query}

请给出专业、结构化的分析回答。"""

        messages = [
            {"role": "system", "content": "你是一位专业的电商数据分析专家，基于提供的数据库结构信息回答分析问题。"},
            {"role": "user", "content": prompt},
        ]

        output = chat(messages, temperature=0.3, max_tokens=2048)
        detail["llm_output"] = output
        detail["llm_call_count"] = 1
        detail["agent_call_count"] = 1

        # 记录检索耗时（不算在总响应时间里，因为只是上下文准备）
        detail["_rag_retrieval_time"] = rag_time

    # ---- 模式 3: Full Multi-Agent ----

    def _run_full_multi_agent(self, query: str, task_type: str,
                              detail: Dict[str, Any]) -> None:
        """完整 LangGraph 多智能体流程。"""
        from agents.planner import run_query

        # 消融：调整 SQL 自修正次数
        original_max_retries = None
        if self.ablation == "no_self_correction":
            from config.settings import get_settings
            settings = get_settings()
            original_max_retries = settings.SQL_MAX_RETRIES
            settings.SQL_MAX_RETRIES = 0
            detail["agent_trace"].append("ablation: SQL自修正已禁用 (SQL_MAX_RETRIES=0)")

        # 消融：不提供 RAG（通过环境变量 RAG_DISABLED=1 控制）
        if self.ablation == "no_rag":
            os.environ["RAG_DISABLED"] = "1"
            detail["agent_trace"].append("ablation: Schema RAG 已禁用，Schema Agent 将返回全部 54 个字段")

        try:
            state = run_query(query)

            # 提取执行轨迹
            messages = state.get("messages", [])
            detail["agent_call_count"] = self._count_agents(messages)
            detail["llm_call_count"] = self._count_llm_calls(messages)
            detail["agents_invoked"] = self._extract_agents(messages)
            detail["agent_trace"] = [m for m in messages if m.startswith("[")]

            # SQL
            detail["generated_sql"] = state.get("sql")
            detail["query_result"] = state.get("query_result")
            detail["execution_success"] = state.get("query_result") is not None
            detail["error"] = state.get("error")

            # 分析/预测/治理
            if state.get("analysis_result"):
                detail["analysis_result"] = _safe_summary(state["analysis_result"])
            if state.get("prediction_result"):
                detail["prediction_result"] = _safe_summary(state["prediction_result"])
            if state.get("governance_result"):
                detail["governance_result"] = _safe_summary(state["governance_result"])

            # 报告
            detail["report"] = state.get("report")

            # SQL 结果对比
            expected_sql = detail.get("expected_sql")
            if expected_sql and detail["execution_success"] and self._db_available:
                try:
                    expected_result = self._adapter.execute_sql(expected_sql)
                    detail["result_correct"] = compare_result_sets(
                        state["query_result"], expected_result
                    )
                except Exception:
                    # 如果 expected SQL 本身有错，跳过对比
                    detail["result_correct"] = None  # 标记为无法验证

        finally:
            # 恢复原始设置
            if original_max_retries is not None:
                from config.settings import get_settings
                get_settings().SQL_MAX_RETRIES = original_max_retries
            if self.ablation == "no_rag":
                os.environ.pop("RAG_DISABLED", None)

    # ---- 成功判定 ----

    def _judge_success(self, detail: Dict[str, Any],
                       test: Dict[str, Any]) -> bool:
        """根据任务类型和模式判断任务是否成功完成。"""
        if detail.get("error"):
            return False

        task_type = test.get("task_type", "sql_query")
        criteria = test.get("success_criteria", "")

        if self.mode in ("single_llm", "rag_llm") or self.ablation == "no_multi_agent":
            # LLM 模式：输出非空且有意义
            output = detail.get("llm_output", "")
            return len(output.strip()) > 50
        else:
            # Multi-Agent 模式
            if task_type == "sql_query":
                # SQL 任务：执行成功
                return detail.get("execution_success", False)
            elif task_type == "analysis":
                # 分析任务：有分析结果
                return detail.get("analysis_result") is not None
            elif task_type == "prediction":
                # 预测任务：有预测结果且 AUC > 0.5
                pred = detail.get("prediction_result", {}) or {}
                auc = pred.get("churn", {}).get("auc", 0)
                has_sales = bool(pred.get("sales"))
                return (auc > 0.5) or has_sales
            elif task_type == "mixed":
                # 综合任务：有报告且长度超过 100 字符
                report = detail.get("report") or ""
                return len(report.strip()) > 100
            else:
                return not detail.get("error")

    # ---- Agent 轨迹分析 ----

    @staticmethod
    def _count_agents(messages: List[str]) -> int:
        """从消息日志中统计被调用的 Agent 数量。"""
        agents_seen = set()
        for m in messages:
            if m.startswith("[") and "]" in m[:40]:
                name = m[1:].split("]")[0].strip()
                if name not in ("Init", "Planner"):
                    agents_seen.add(name)
        return len(agents_seen)

    @staticmethod
    def _count_llm_calls(messages: List[str]) -> int:
        """估算 LLM 调用次数（Schema + SQL + SQL Retry + Report）。"""
        count = 0
        for m in messages:
            if "重试" in m or "生成" in m or "调用 LLM" in m:
                count += 1
        # 默认：schema(1) + sql(1+retries) + report(1 if exists)
        count = max(count, 1)
        return count

    @staticmethod
    def _extract_agents(messages: List[str]) -> List[str]:
        """提取被调用的 Agent 列表（按执行顺序）。"""
        agents = []
        for m in messages:
            if m.startswith("[") and "]" in m[:40]:
                name = m[1:].split("]")[0].strip()
                if name not in ("Init",) and name not in agents:
                    agents.append(name)
        return agents

    # ---- 评价指标 ----

    def evaluate(self) -> Dict[str, Any]:
        self.result.metrics = compute_all_multi_agent_metrics(self.result.details)

        m = self.result.metrics
        self.logger.info(f"任务成功率: {_fmt(m.get('task_success_rate'))}")
        self.logger.info(f"SQL准确率: {_fmt(m.get('sql_accuracy'))}")
        self.logger.info(f"报告质量分: {m.get('report_score', 'N/A')}")
        self.logger.info(f"平均响应时间: {m.get('avg_response_time_seconds', 'N/A')}s")
        self.logger.info(f"平均Agent调用: {m.get('avg_agent_calls', 'N/A')}")

        for tt, d in m.get("by_task_type", {}).items():
            if isinstance(d, dict):
                self.logger.info(
                    f"  [{tt}] count={d.get('count', '?')} "
                    f"TSR={_fmt(d.get('task_success_rate'))}"
                )

        return self.result.metrics

    def evaluate_reports(self) -> int:
        """LLM-as-Judge 报告质量评估（在 run() 之后调用）。

        对所有非 sql_query 任务调用 LLM 评分，耗时较长（~2s/条）。
        """
        self.logger.info("开始 LLM-as-Judge 报告质量评估...")
        count_before = sum(
            1 for d in self.result.details
            if d.get("report_quality_score") is not None
        )

        self.result.details = evaluate_all_reports(
            self.result.details, skip_sql_query=True
        )

        count_after = sum(
            1 for d in self.result.details
            if d.get("report_quality_score") is not None
        )
        self.logger.info(f"报告评估完成: {count_after} 份报告已评分")

        # 更新指标
        scores = [d["report_quality_score"] for d in self.result.details
                  if d.get("report_quality_score") is not None]
        if scores:
            self.result.metrics["report_score"] = round(sum(scores) / len(scores), 1)
            self.result.metrics["report_score_count"] = len(scores)

        return count_after

    # ---- 结果保存 ----

    def save_results(
        self, output_dir: Optional[Path] = None, formats=None
    ) -> Dict[str, Path]:
        if output_dir is None:
            output_dir = Path(__file__).resolve().parent.parent / "results"
        raw_dir = Path(output_dir) / "raw"
        tables_dir = Path(output_dir) / "tables"

        saved = {}
        tag = self.ablation or self.mode

        # 1. JSON
        jpath = raw_dir / f"{self.name}_{_timestamp()}.json"
        save_json(self.result.to_dict(), jpath)
        saved["json"] = jpath

        # 2. CSV 明细
        cpath = raw_dir / f"{self.name}_details_{_timestamp()}.csv"
        csv_details = []
        for d in self.result.details:
            csv_details.append({
                "id": d.get("id"),
                "query": d.get("query", "")[:80],
                "category": d.get("category"),
                "task_type": d.get("task_type"),
                "success": d.get("success"),
                "generated_sql": (d.get("generated_sql") or "")[:120],
                "execution_success": d.get("execution_success"),
                "result_correct": d.get("result_correct"),
                "report_quality_score": d.get("report_quality_score"),
                "duration_seconds": d.get("duration_seconds"),
                "agent_call_count": d.get("agent_call_count"),
                "error": (d.get("error") or "")[:80],
            })
        save_csv(csv_details, cpath)
        saved["details_csv"] = cpath

        # 3. 指标汇总 CSV
        import pandas as pd
        m = self.result.metrics
        summary_rows = [
            {"指标": "测试样本数", "数值": str(m.get("total_samples", ""))},
            {"指标": "任务成功率", "数值": _fmt(m.get("task_success_rate"))},
            {"指标": "SQL准确率", "数值": _fmt(m.get("sql_accuracy"))},
            {"指标": "SQL执行成功率", "数值": _fmt(m.get("sql_execution_rate"))},
            {"指标": "报告质量分", "数值": str(m.get("report_quality_score", ""))},
            {"指标": "平均响应时间", "数值": f"{m.get('avg_response_time_seconds', '')}s"},
            {"指标": "平均Agent调用", "数值": str(m.get("avg_agent_calls", ""))},
            {"指标": "平均LLM调用", "数值": str(m.get("avg_llm_calls", ""))},
        ]
        df = pd.DataFrame(summary_rows)
        sp = tables_dir / f"{self.name}_summary_{_timestamp()}.csv"
        df.to_csv(sp, index=False, encoding="utf-8-sig")
        saved["summary_table"] = sp

        # 4. 按任务类型 CSV
        by_type = m.get("by_task_type", {})
        if by_type:
            type_rows = []
            for tt, d in by_type.items():
                type_rows.append({
                    "任务类型": tt, "样本数": d["count"],
                    "任务成功率": _fmt(d.get("task_success_rate")),
                    "SQL准确率": _fmt(d.get("sql_accuracy")) if "sql_accuracy" in d else "N/A",
                    "报告质量分": d.get("report_quality_score", "N/A"),
                    "平均响应时间": d.get("avg_response_time", "N/A"),
                })
            df2 = pd.DataFrame(type_rows)
            tp = tables_dir / f"{self.name}_by_type_{_timestamp()}.csv"
            df2.to_csv(tp, index=False, encoding="utf-8-sig")
            saved["by_type_table"] = tp

        self.logger.info(f"结果已保存: {len(saved)} 个文件")
        for k, v in saved.items():
            self.logger.info(f"  [{k}] {v.name}")

        return saved


# ============================================================
# 消融实验协调器
# ============================================================

ABLATION_VARIANTS = [
    ("full_model", None),
    ("no_rag", "no_rag"),
    ("no_multi_agent", "no_multi_agent"),
    ("no_self_correction", "no_self_correction"),
]


def run_ablation_experiment(
    test_data_path: Optional[str] = None,
    max_samples: int = 0,
) -> Dict[str, ExperimentResult]:
    """运行全部消融实验（4个版本）。"""
    config_base = {
        "experiment": {"name": "multi_agent_ablation", "version": "1.0"},
        "limits": {"max_samples": max_samples},
        "mode": "full_multi_agent",
    }
    if test_data_path:
        config_base["data"] = {"test_data_path": test_data_path}

    results = {}
    for label, ablation in ABLATION_VARIANTS:
        print(f"\n{'=' * 60}")
        print(f"消融实验: {label}")
        print(f"{'=' * 60}")

        config = {**config_base, "ablation": ablation}
        exp = MultiAgentExperiment(config)
        result = exp.execute()
        results[label] = result

        m = result.metrics
        print(f"[{label}] TSR={_fmt(m.get('task_success_rate'))}, "
              f"SQL_ACC={_fmt(m.get('sql_accuracy'))}, "
              f"RPT={m.get('report_quality_score', 'N/A')}")

    return results


def run_full_comparison(
    test_data_path: Optional[str] = None,
    max_samples: int = 0,
    with_judge: bool = False,
) -> Dict[str, ExperimentResult]:
    """运行完整的三方案对比实验。"""
    results = {}

    for mode in ("single_llm", "rag_llm", "full_multi_agent"):
        print(f"\n{'=' * 60}")
        print(f"Baseline 对比: {mode}")
        print(f"{'=' * 60}")

        config = {
            "experiment": {"name": f"multi_agent_{mode}", "version": "1.0"},
            "limits": {"max_samples": max_samples},
            "mode": mode,
        }
        if test_data_path:
            config["data"] = {"test_data_path": test_data_path}

        exp = MultiAgentExperiment(config)
        result = exp.execute()

        # LLM-as-Judge 评估（可选，耗时较长）
        if with_judge:
            exp.evaluate_reports()

        results[mode] = result

    return results


# ============================================================
# 辅助函数
# ============================================================

def _timestamp() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _fmt(val) -> str:
    if val is None:
        return "N/A"
    if isinstance(val, float):
        return f"{val * 100:.1f}%"
    return str(val)


def _safe_summary(data: Any, maxlen: int = 300) -> Any:
    """对分析/预测结果做安全截断（避免存储过大）。"""
    if data is None:
        return None
    s = str(data)
    if len(s) <= maxlen * 2:
        return data
    return {"_summary": s[:maxlen] + "...", "_type": type(data).__name__}


# ============================================================
# CLI 入口
# ============================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Multi-Agent 协同效果验证实验")
    parser.add_argument("--test-data", type=str, default=None)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--mode", type=str, default="full_multi_agent",
                        choices=["single_llm", "rag_llm", "full_multi_agent"],
                        help="运行模式 (default: full_multi_agent)")
    parser.add_argument("--ablation", action="store_true",
                        help="运行全部消融实验（4个版本）")
    parser.add_argument("--compare", action="store_true",
                        help="运行完整的三方案对比实验")
    parser.add_argument("--judge", action="store_true",
                        help="启用 LLM-as-Judge 报告质量评估")
    parser.add_argument("--full", action="store_true",
                        help="完整技术评测实验：三方案对比+消融+LLM-Judge+报告生成")
    args = parser.parse_args()

    if args.full:
        run_full_paper_experiment(
            test_data_path=args.test_data,
            max_samples=args.max_samples,
        )
    elif args.ablation:
        print("\n" + "=" * 60)
        print("消融实验: 验证各模块贡献")
        print("=" * 60)
        results = run_ablation_experiment(
            test_data_path=args.test_data,
            max_samples=args.max_samples,
        )
        print_summary_ablation(results)
    elif args.compare:
        print("\n" + "=" * 60)
        print("三方案 Baseline 对比实验")
        print("=" * 60)
        results = run_full_comparison(
            test_data_path=args.test_data,
            max_samples=args.max_samples,
            with_judge=args.judge,
        )
        print_summary_comparison(results)
    else:
        # 单模式运行
        config = {
            "experiment": {"name": f"multi_agent_{args.mode}", "version": "1.0"},
            "limits": {"max_samples": args.max_samples},
            "mode": args.mode,
        }
        if args.test_data:
            config["data"] = {"test_data_path": args.test_data}

        exp = MultiAgentExperiment(config)
        result = exp.execute()

        if args.judge:
            exp.evaluate_reports()
            exp.evaluate()

        print()
        print("=" * 60)
        print(f"Multi-Agent 实验 [{args.mode}]: {result.status}")
        print(f"耗时: {result.duration_seconds:.1f}s")
        m = result.metrics
        print(f"任务成功率: {_fmt(m.get('task_success_rate'))}")
        print(f"SQL准确率: {_fmt(m.get('sql_accuracy'))}")
        print(f"报告质量分: {m.get('report_score', 'N/A')}")
        print(f"平均响应时间: {m.get('avg_response_time_seconds', 'N/A')}s")
        print(f"平均Agent调用: {m.get('avg_agent_calls', 'N/A')}")
        print("=" * 60)

    return 0


def run_full_paper_experiment(
    test_data_path: Optional[str] = None,
    max_samples: int = 0,
) -> None:
    """运行完整技术评测实验：三方案对比 + 消融 + LLM-Judge + 报告生成。"""
    output_dir = Path(__file__).resolve().parent.parent / "results"

    # ---- Step 1: 三方案对比 ----
    print("\n" + "=" * 70)
    print("Phase 1/3: 三方案 Baseline 对比实验 (50 queries × 3 modes)")
    print("=" * 70)
    comparison_results = run_full_comparison(
        test_data_path=test_data_path,
        max_samples=max_samples,
        with_judge=True,  # 启用 LLM-as-Judge
    )
    print_summary_comparison(comparison_results)

    # ---- Step 2: 消融实验 ----
    print("\n" + "=" * 70)
    print("Phase 2/3: 消融实验 (4 variants)")
    print("=" * 70)
    ablation_results = run_ablation_experiment(
        test_data_path=test_data_path,
        max_samples=max_samples,
    )
    print_summary_ablation(ablation_results)

    # ---- Step 3: 生成技术评测报告 ----
    print("\n" + "=" * 70)
    print("Phase 3/3: 生成技术评测报告")
    print("=" * 70)

    cm = {k: v.metrics for k, v in comparison_results.items()}
    am = {k: v.metrics for k, v in ablation_results.items()}
    multi_details = comparison_results.get("full_multi_agent",
                                           ExperimentResult("")).details

    report_md = generate_full_experiment_report(
        single_metrics=cm.get("single_llm", {}),
        rag_metrics=cm.get("rag_llm", {}),
        multi_metrics=cm.get("full_multi_agent", {}),
        ablation_results=am,
        multi_details=multi_details,
        duration_seconds=sum(r.duration_seconds for r in comparison_results.values()),
    )

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    report_path = output_dir / "raw" / f"multi_agent_full_report_{ts}.md"
    report_path.write_text(report_md, encoding="utf-8")
    print(f"技术评测报告已保存: {report_path}")

    # 保存表格 CSV
    tables_dir = output_dir / "tables"
    import pandas as pd

    # 表1
    if cm:
        rows1 = []
        labels1 = {"single_llm": "Single LLM", "rag_llm": "LLM+RAG",
                   "full_multi_agent": "Multi-Agent(Ours)"}
        for key, label in labels1.items():
            m = cm.get(key, {})
            if m:
                rows1.append({
                    "方案": label,
                    "任务成功率": _fmt(m.get("task_success_rate")),
                    "Report Score": str(m.get("report_score", "N/A")),
                    "SQL准确率": _fmt(m.get("sql_accuracy")),
                    "平均耗时(s)": str(m.get("avg_response_time_seconds", "N/A")),
                    "Agent调用": str(m.get("avg_agent_calls", "N/A")),
                    "LLM调用": str(m.get("avg_llm_calls", "N/A")),
                })
        if rows1:
            pd.DataFrame(rows1).to_csv(
                tables_dir / f"table1_comparison_{ts}.csv",
                index=False, encoding="utf-8-sig")

    # 表2: 按任务类型
    if cm:
        for mode_key in ("single_llm", "rag_llm", "full_multi_agent"):
            m = cm.get(mode_key, {})
            by_type = m.get("by_task_type", {})
            if by_type:
                rows2 = []
                for tt, d in by_type.items():
                    rows2.append({
                        "任务类型": tt, "样本数": d["count"],
                        "任务成功率": _fmt(d.get("task_success_rate")),
                        "SQL准确率": _fmt(d.get("sql_accuracy")) if "sql_accuracy" in d else "N/A",
                        "报告质量分": str(d.get("report_quality_score", "N/A")),
                        "平均响应时间(s)": str(d.get("avg_response_time", "N/A")),
                    })
                pd.DataFrame(rows2).to_csv(
                    tables_dir / f"table2_by_type_{mode_key}_{ts}.csv",
                    index=False, encoding="utf-8-sig")

    # 表4: 消融
    if am:
        rows4 = []
        labels4 = {"full_model": "完整模型", "no_rag": "去除RAG",
                   "no_multi_agent": "去除Multi-Agent", "no_self_correction": "去除SQL自修正"}
        for key, label in labels4.items():
            m = am.get(key, {})
            if m:
                rows4.append({
                    "模型版本": label,
                    "任务成功率": _fmt(m.get("task_success_rate")),
                    "Report Score": str(m.get("report_score", "N/A")),
                    "SQL准确率": _fmt(m.get("sql_accuracy")),
                    "平均响应时间(s)": str(m.get("avg_response_time_seconds", "N/A")),
                    "Agent调用": str(m.get("avg_agent_calls", "N/A")),
                })
        if rows4:
            pd.DataFrame(rows4).to_csv(
                tables_dir / f"table4_ablation_{ts}.csv",
                index=False, encoding="utf-8-sig")

    print(f"\n所有结果已保存到: {output_dir}")
    print("=" * 70)


def print_summary_comparison(results: Dict[str, ExperimentResult]) -> None:
    """打印三方案对比摘要。"""
    print("\n" + "=" * 70)
    print("三方案 Baseline 对比结果")
    print("=" * 70)
    print(f"{'方案':<20} {'TSR':<10} {'RPT':<8} {'SQL ACC':<10} {'Time(s)':<10} {'Agents':<8}")
    print("-" * 70)
    labels = {"single_llm": "Single LLM", "rag_llm": "LLM + RAG",
              "full_multi_agent": "Multi-Agent(Ours)"}
    for key in ("single_llm", "rag_llm", "full_multi_agent"):
        r = results.get(key)
        if r:
            m = r.metrics
            print(f"{labels[key]:<20} {_fmt(m.get('task_success_rate')):<10} "
                  f"{str(m.get('report_score', 'N/A')):<8} "
                  f"{_fmt(m.get('sql_accuracy')):<10} "
                  f"{str(m.get('avg_response_time_seconds', 'N/A')):<10} "
                  f"{str(m.get('avg_agent_calls', 'N/A')):<8}")
    print("=" * 70)


def print_summary_ablation(results: Dict[str, ExperimentResult]) -> None:
    """打印消融实验摘要。"""
    print("\n" + "=" * 70)
    print("消融实验结果")
    print("=" * 70)
    print(f"{'版本':<25} {'TSR':<10} {'RPT':<8} {'SQL ACC':<10} {'Time(s)':<10}")
    print("-" * 70)
    labels = {
        "full_model": "完整模型",
        "no_rag": "去除RAG",
        "no_multi_agent": "去除Multi-Agent",
        "no_self_correction": "去除SQL自修正",
    }
    for key in ("full_model", "no_rag", "no_multi_agent", "no_self_correction"):
        r = results.get(key)
        if r:
            m = r.metrics
            print(f"{labels[key]:<25} {_fmt(m.get('task_success_rate')):<10} "
                  f"{str(m.get('report_score', 'N/A')):<8} "
                  f"{_fmt(m.get('sql_accuracy')):<10} "
                  f"{str(m.get('avg_response_time_seconds', 'N/A')):<10}")
    print("=" * 70)


if __name__ == "__main__":
    sys.exit(main())
