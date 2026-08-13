"""
RAG Schema 检索效果评估实验 —— 技术评测正式评测 RAG 实验。

评估 ChromaDB + Embedding 的 Schema 语义检索能力。

运行方式：
    # 纯检索评估
    python -m evaluation.experiments.rag_schema_experiment

    # RAG vs No-RAG 对比实验
    python -m evaluation.experiments.rag_schema_experiment --comparison

架构设计：
    - RAGSchemaExperiment: 纯检索 Recall@K 评估
    - RAGComparisonExperiment: RAG vs No-RAG Text2SQL 对比
"""

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from evaluation.framework.base import ExperimentBase, ExperimentResult
from evaluation.framework.result import save_json, load_json, save_csv
from evaluation.metrics.rag_metrics import (
    compute_all_rag_metrics,
    generate_table_overall,
    generate_table_by_category,
    generate_table_failure_analysis,
    generate_rag_experiment_report,
)
from evaluation.metrics.text2sql_metrics import (
    compute_all_metrics as compute_text2sql_metrics,
    generate_table_overall as text2sql_table_overall,
)


# ============================================================
# RAG Schema 检索实验
# ============================================================

class RAGSchemaExperiment(ExperimentBase):
    """Schema 语义检索效果评估实验。

    直接调用 ChromaDB Embedder.search() 获取 Top-K 结果，
    与标准答案比较，计算 Recall@K 等指标。
    """

    name = "rag_schema_retrieval"
    description = "评估 ChromaDB + BGE Embedding 的 Schema 语义检索 Recall@K"
    version = "1.0"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.test_queries: List[Dict[str, Any]] = []
        self.top_k: int = 5
        self._embedder = None

    # ---- 数据加载 ----

    def load_data(self) -> int:
        data_path = self.config.get(
            "data.test_data_path",
            str(Path(__file__).resolve().parent.parent / "data" / "rag_schema" / "test_queries.json"),
        )
        path = Path(data_path)
        if not path.exists():
            raise FileNotFoundError(f"RAG 测试数据文件不存在: {path}")

        self.test_queries = load_json(path)
        self.logger.info(f"加载 RAG 测试数据: {path}")

        limits = self.config.get("limits", {})
        max_n = limits.get("max_samples", 0) if isinstance(limits, dict) else 0
        if max_n > 0 and len(self.test_queries) > max_n:
            self.test_queries = self.test_queries[:max_n]
            self.logger.info(f"限制样本数为 {max_n}")

        return len(self.test_queries)

    # ---- 准备 ----

    def setup(self) -> None:
        from storage.chromadb.embedder import get_embedder

        self._embedder = get_embedder()
        if self._embedder.collection is None:
            self.logger.info("ChromaDB collection 未初始化，正在构建索引...")
            self._embedder.build()

        count = self._embedder.collection.count()
        self.logger.info(f"ChromaDB 就绪: {count} 条 schema 记录")

        # 统计测试集构成
        cats = {}
        for q in self.test_queries:
            c = q.get("category", "unknown")
            cats[c] = cats.get(c, 0) + 1
        self.logger.info(f"测试集构成: {cats}")

    # ---- 执行 ----

    def run(self) -> None:
        total = len(self.test_queries)

        for idx, test in enumerate(self.test_queries):
            qid = test["id"]
            query = test["query"]
            category = test.get("category", "unknown")
            expected_tables = test.get("expected_tables", [])
            expected_fields = test.get("expected_fields", [])

            self.logger.info(f"[{idx + 1}/{total}] Q{qid} [{category}] {query[:60]}...")

            detail = {
                "id": qid,
                "query": query,
                "category": category,
                "expected_tables": expected_tables,
                "expected_fields": expected_fields,
                "retrieved_tables": [],
                "retrieved_fields": [],
                "retrieved_tables_top1": [],
                "retrieved_tables_top3": [],
                "retrieved_tables_top5": [],
                "retrieved_fields_top1": [],
                "retrieved_fields_top3": [],
                "retrieved_fields_top5": [],
                "retrieval_time_seconds": 0.0,
                "top5_details": [],
            }

            t_start = time.time()

            try:
                # 调用 ChromaDB 语义检索
                results = self._embedder.search(query, top_k=self.top_k)
                detail["retrieval_time_seconds"] = round(time.time() - t_start, 4)

                # 提取检索结果的表和字段
                retrieved_tables = []
                retrieved_fields = []
                for r in results:
                    table = r["table_name"]
                    field = f"{table}.{r['column_name']}"
                    if table not in retrieved_tables:
                        retrieved_tables.append(table)
                    retrieved_fields.append(field)
                    detail["top5_details"].append({
                        "rank": r["rank"],
                        "table_name": table,
                        "column_name": r["column_name"],
                        "score": r["score"],
                    })

                detail["retrieved_tables"] = retrieved_tables
                detail["retrieved_fields"] = retrieved_fields
                detail["retrieved_tables_top1"] = retrieved_tables[:1]
                detail["retrieved_tables_top3"] = retrieved_tables[:3]
                detail["retrieved_tables_top5"] = retrieved_tables[:5]
                detail["retrieved_fields_top1"] = retrieved_fields[:1]
                detail["retrieved_fields_top3"] = retrieved_fields[:3]
                detail["retrieved_fields_top5"] = retrieved_fields[:5]

                # 判断是否命中
                table_hit = bool(set(expected_tables) & set(retrieved_tables[:5]))
                field_hit = bool(set(expected_fields) & set(retrieved_fields[:5]))

                if field_hit:
                    self.logger.record_sample(qid, True, f"Tables: {retrieved_tables[:3]}...")
                else:
                    reason = "表未命中" if not table_hit else "字段排名不足"
                    detail["failure_reason"] = reason
                    self.logger.record_sample(
                        qid, False,
                        f"Expected: {expected_fields}, Got top-5: {retrieved_fields[:5]}"
                    )

            except Exception as e:
                detail["retrieval_time_seconds"] = round(time.time() - t_start, 4)
                detail["failure_reason"] = f"检索异常: {str(e)[:100]}"
                self.logger.record_sample(qid, False, f"Exception: {type(e).__name__}: {e}")

            self.result.details.append(detail)

        # 汇总
        self.result.sample_count = total
        field_hits = sum(
            1 for d in self.result.details
            if set(d.get("expected_fields", [])) & set(d.get("retrieved_fields_top5", []))
        )
        self.result.success_count = field_hits
        self.result.failure_count = total - field_hits

    # ---- 指标 ----

    def evaluate(self) -> Dict[str, Any]:
        self.result.metrics = compute_all_rag_metrics(self.result.details)

        m = self.result.metrics
        self.logger.info(f"Table Recall@1: {_fmt(m.get('table_recall_at_1'))}")
        self.logger.info(f"Table Recall@3: {_fmt(m.get('table_recall_at_3'))}")
        self.logger.info(f"Table Recall@5: {_fmt(m.get('table_recall_at_5'))}")
        self.logger.info(f"Field Recall@1: {_fmt(m.get('field_recall_at_1'))}")
        self.logger.info(f"Field Recall@3: {_fmt(m.get('field_recall_at_3'))}")
        self.logger.info(f"Field Recall@5: {_fmt(m.get('field_recall_at_5'))}")
        self.logger.info(f"MRR: {m.get('mrr', 'N/A')}")
        self.logger.info(f"平均检索耗时: {m.get('avg_retrieval_time_seconds', 'N/A')}s")

        for cat, d in m.get("by_category", {}).items():
            self.logger.info(
                f"  [{cat}] count={d['count']} "
                f"TR@5={_fmt(d.get('table_recall_at_5'))} "
                f"FR@5={_fmt(d.get('field_recall_at_5'))}"
            )

        return self.result.metrics

    # ---- 结果保存 ----

    def save_results(
        self,
        output_dir: Optional[Path] = None,
        formats: Optional[List[str]] = None,
    ) -> Dict[str, Path]:
        if output_dir is None:
            output_dir = Path(__file__).resolve().parent.parent / "results"
        raw_dir = Path(output_dir) / "raw"
        tables_dir = Path(output_dir) / "tables"

        saved = {}

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
                "query": d.get("query"),
                "category": d.get("category"),
                "expected_tables": ", ".join(d.get("expected_tables", [])),
                "expected_fields": ", ".join(d.get("expected_fields", [])),
                "retrieved_tables_top5": ", ".join(d.get("retrieved_tables_top5", [])),
                "retrieved_fields_top5": ", ".join(d.get("retrieved_fields_top5", [])),
                "retrieval_time_s": d.get("retrieval_time_seconds"),
            })
        save_csv(csv_details, cpath)
        saved["details_csv"] = cpath

        # 3. 技术评测表格
        import pandas as pd
        m = self.result.metrics

        # 表1: 总体结果
        df1 = pd.DataFrame([
            {"指标": "测试样本数量", "数值": str(m.get("total_samples", ""))},
            {"指标": "Table Recall@1", "数值": _fmt(m.get("table_recall_at_1"))},
            {"指標": "Table Recall@3", "数值": _fmt(m.get("table_recall_at_3"))},
            {"指标": "Table Recall@5", "数值": _fmt(m.get("table_recall_at_5"))},
            {"指标": "Field Recall@1", "数值": _fmt(m.get("field_recall_at_1"))},
            {"指标": "Field Recall@3", "数值": _fmt(m.get("field_recall_at_3"))},
            {"指标": "Field Recall@5", "数值": _fmt(m.get("field_recall_at_5"))},
            {"指标": "Field Precision@5", "数值": _fmt(m.get("field_precision_at_5"))},
            {"指标": "MRR", "数值": str(m.get("mrr", ""))},
            {"指标": "综合命中率@5", "数值": _fmt(m.get("hit_rate_at_5"))},
            {"指标": "平均检索耗时", "数值": f"{m.get('avg_retrieval_time_seconds', '')}s"},
        ])
        t1 = tables_dir / f"{self.name}_table1_overall_{_timestamp()}.csv"
        df1.to_csv(t1, index=False, encoding="utf-8-sig")
        saved["table1_overall"] = t1

        # 表2: 按业务类别
        by_cat = m.get("by_category", {})
        if by_cat:
            cat_rows = []
            for cat, d in by_cat.items():
                cat_rows.append({
                    "业务类别": cat, "样本数": d["count"],
                    "Table R@1": _fmt(d.get("table_recall_at_1")),
                    "Table R@3": _fmt(d.get("table_recall_at_3")),
                    "Table R@5": _fmt(d.get("table_recall_at_5")),
                    "Field R@1": _fmt(d.get("field_recall_at_1")),
                    "Field R@3": _fmt(d.get("field_recall_at_3")),
                    "Field R@5": _fmt(d.get("field_recall_at_5")),
                    "MRR": str(d.get("mrr", "")),
                    "均耗时(s)": str(d.get("avg_retrieval_time", "")),
                })
            df2 = pd.DataFrame(cat_rows)
            t2 = tables_dir / f"{self.name}_table2_category_{_timestamp()}.csv"
            df2.to_csv(t2, index=False, encoding="utf-8-sig")
            saved["table2_category"] = t2

        # 4. Markdown 报告
        elapsed = sum(d.get("retrieval_time_seconds", 0) for d in self.result.details)
        report_md = generate_rag_experiment_report(
            self.name, m, self.result.details, round(elapsed, 1),
        )
        rpath = raw_dir / f"{self.name}_report_{_timestamp()}.md"
        rpath.write_text(report_md, encoding="utf-8")
        saved["report_md"] = rpath

        # 5. 技术评测 Markdown 表格
        md_tables = "\n".join([
            generate_table_overall(m),
            generate_table_by_category(m),
            generate_table_failure_analysis(self.result.details),
        ])
        mdpath = raw_dir / f"{self.name}_benchmark_tables_{_timestamp()}.md"
        mdpath.write_text(md_tables, encoding="utf-8")
        saved["benchmark_tables_md"] = mdpath

        self.logger.info(f"结果已保存: {len(saved)} 个文件")
        for k, v in saved.items():
            self.logger.info(f"  [{k}] {v.name}")

        return saved


# ============================================================
# RAG vs No-RAG 对比实验
# ============================================================

class RAGComparisonExperiment(ExperimentBase):
    """RAG vs No-RAG Schema 检索对比实验。

    两个模式：
    - Schema-RAG: 调用 ChromaDB 检索 + LLM 精排（正常流程）
    - No-RAG: 不检索，提供全部 54 个字段的 Schema dump
    """

    name = "rag_comparison"
    description = "RAG vs No-RAG Schema 检索对 Text2SQL 准确率的影响对比"
    version = "1.0"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.test_queries: List[Dict[str, Any]] = []
        self._adapter = None
        self._db_available = False
        # 从配置读取是否运行 RAG 模式
        self.use_rag: bool = self.config.get("use_rag", True)

    def load_data(self) -> int:
        data_path = self.config.get(
            "data.test_data_path",
            str(Path(__file__).resolve().parent.parent / "data" / "text2sql" / "test_queries.json"),
        )
        path = Path(data_path)
        if not path.exists():
            raise FileNotFoundError(f"测试数据文件不存在: {path}")

        self.test_queries = load_json(path)
        limits = self.config.get("limits", {})
        max_n = limits.get("max_samples", 0) if isinstance(limits, dict) else 0
        if max_n > 0 and len(self.test_queries) > max_n:
            self.test_queries = self.test_queries[:max_n]
        return len(self.test_queries)

    def setup(self) -> None:
        self.name = f"rag_comparison_{'rag' if self.use_rag else 'no_rag'}"
        from config.settings import get_settings
        from storage.db_adapter import get_available_adapter

        try:
            self._adapter = get_available_adapter()
            self._db_available = self._adapter.check_connection()
        except Exception as e:
            self.logger.warning(f"数据库初始化异常: {e}")
            self._adapter = None
            self._db_available = False

        settings = get_settings()
        llm_ok = bool(settings.LLM_API_KEY and "your_" not in settings.LLM_API_KEY)
        if not llm_ok:
            raise RuntimeError("LLM API key 未配置")
        self.logger.info(f"模式: {'Schema-RAG' if self.use_rag else 'No-RAG (全部Schema)'}")
        self.logger.info(f"LLM: {settings.LLM_MODEL}")

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
                "mode": "rag" if self.use_rag else "no_rag",
                "generated_sql": None,
                "expected_sql": expected_sql,
                "execution_success": False,
                "result_correct": False,
                "error_message": None,
                "duration_seconds": 0.0,
            }

            t_start = time.time()

            try:
                if self.use_rag:
                    output = self._run_with_rag(query)
                else:
                    output = self._run_without_rag(query)

                detail["generated_sql"] = output.get("sql")
                detail["execution_success"] = output.get("execution_success", False)

                if output.get("query_result") is not None and self._db_available:
                    gen_result = output["query_result"]
                    if expected_sql:
                        try:
                            from evaluation.metrics.text2sql_metrics import compare_result_sets
                            exp_result = self._adapter.execute_sql(expected_sql)
                            detail["result_correct"] = compare_result_sets(gen_result, exp_result)
                        except Exception as e:
                            detail["error_message"] = f"Expected SQL error: {e}"
                    self.logger.record_sample(qid, detail.get("result_correct", False),
                                              f"SQL={'OK' if output.get('sql') else 'FAIL'}")
                elif output.get("error"):
                    detail["error_message"] = output["error"]
                    self.logger.record_sample(qid, False, f"Error: {output['error'][:50]}")

            except Exception as e:
                detail["error_message"] = str(e)
                self.logger.record_sample(qid, False, f"Exception: {type(e).__name__}")

            detail["duration_seconds"] = round(time.time() - t_start, 3)
            self.result.details.append(detail)

        self.result.sample_count = total
        self.result.success_count = sum(1 for d in self.result.details if d["result_correct"])
        self.result.failure_count = total - self.result.success_count

    def _run_with_rag(self, user_query: str) -> Dict[str, Any]:
        """通过完整 LangGraph 流程运行（含 Schema RAG）。"""
        from agents.planner import run_query
        state = run_query(user_query)
        return {
            "sql": state.get("sql"),
            "query_result": state.get("query_result"),
            "error": state.get("error"),
            "execution_success": state.get("query_result") is not None,
        }

    def _run_without_rag(self, user_query: str) -> Dict[str, Any]:
        """No-RAG 模式：跳过 Schema 检索，使用全部 54 字段的完整 Schema dump。"""
        from storage.chromadb.schema_metadata import SCHEMA_FIELDS
        from agents.sql_agent import _generate_sql, _execute_sql, validate_select_only

        # 构造全部字段的 schema info（模拟无 RAG 过滤的情况）
        schema_lines = []
        seen_tables = set()
        for field in SCHEMA_FIELDS:
            t = field["table_name"]
            if t not in seen_tables:
                schema_lines.append(f"\n[{t}]")
                seen_tables.add(t)
            schema_lines.append(
                f"  {t}.{field['column_name']} ({field['dtype']}) — {field['business_term']}"
            )
        schema_info = "\n".join(schema_lines)

        try:
            sql = _generate_sql(user_query, schema_info)
            is_safe, reason = validate_select_only(sql)
            if not is_safe:
                return {"sql": sql, "query_result": None, "error": f"安全校验: {reason}",
                        "execution_success": False}

            rows, error = _execute_sql(sql, self._adapter)
            if error:
                return {"sql": sql, "query_result": None, "error": error,
                        "execution_success": False}

            return {"sql": sql, "query_result": rows, "error": None, "execution_success": True}
        except Exception as e:
            return {"sql": None, "query_result": None, "error": str(e), "execution_success": False}

    def evaluate(self) -> Dict[str, Any]:
        self.result.metrics = compute_text2sql_metrics(self.result.details)
        m = self.result.metrics
        self.logger.info(f"[{self.name}] EX={_fmt(m.get('execution_accuracy'))} "
                         f"EXE={_fmt(m.get('execution_success_rate'))}")
        return self.result.metrics

    def save_results(
        self, output_dir: Optional[Path] = None, formats=None
    ) -> Dict[str, Path]:
        if output_dir is None:
            output_dir = Path(__file__).resolve().parent.parent / "results"
        raw_dir = Path(output_dir) / "raw"
        tables_dir = Path(output_dir) / "tables"

        saved = {}
        tag = "rag" if self.use_rag else "no_rag"

        jpath = raw_dir / f"{self.name}_{_timestamp()}.json"
        save_json(self.result.to_dict(), jpath)
        saved["json"] = jpath

        cpath = raw_dir / f"{self.name}_details_{_timestamp()}.csv"
        save_csv(self.result.details, cpath)
        saved["details_csv"] = cpath

        # 对比表
        import pandas as pd
        m = self.result.metrics
        df = pd.DataFrame([
            {"指标": "SQL生成成功率", tag: _fmt(m.get("sql_generation_success_rate"))},
            {"指标": "SQL执行成功率(EXE)", tag: _fmt(m.get("execution_success_rate"))},
            {"指标": "执行准确率(EX)", tag: _fmt(m.get("execution_accuracy"))},
            {"指标": "平均响应时间", tag: f"{m.get('avg_response_time_seconds', '')}s"},
        ])
        tp = tables_dir / f"{self.name}_summary_{_timestamp()}.csv"
        df.to_csv(tp, index=False, encoding="utf-8-sig")
        saved["summary_table"] = tp

        self.logger.info(f"结果已保存: {len(saved)} 个文件")
        return saved


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


def _sample_rows(rows: list, n: int = 5) -> list:
    if not rows:
        return []
    return rows[:n]


# ============================================================
# CLI 入口
# ============================================================

def run_rag_experiment(
    test_data_path: Optional[str] = None,
    max_samples: int = 0,
) -> ExperimentResult:
    """运行 RAG Schema 检索实验。"""
    config = {
        "experiment": {"name": "rag_schema_retrieval", "version": "1.0"},
        "limits": {"max_samples": max_samples},
    }
    if test_data_path:
        config["data"] = {"test_data_path": test_data_path}
    return RAGSchemaExperiment(config).execute()


def run_rag_comparison(
    test_data_path: Optional[str] = None,
    max_samples: int = 0,
    use_rag: bool = True,
) -> ExperimentResult:
    """运行 RAG vs No-RAG 对比实验。"""
    config = {
        "experiment": {"name": "rag_comparison", "version": "1.0"},
        "limits": {"max_samples": max_samples},
        "use_rag": use_rag,
    }
    if test_data_path:
        config["data"] = {"test_data_path": test_data_path}
    return RAGComparisonExperiment(config).execute()


def main():
    import argparse
    parser = argparse.ArgumentParser(description="RAG Schema 检索评估实验")
    parser.add_argument("--test-data", type=str, default=None)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--comparison", action="store_true",
                        help="运行 RAG vs No-RAG 对比实验")
    parser.add_argument("--no-rag", action="store_true",
                        help="对比模式：不启用 RAG（需配合 --comparison）")
    args = parser.parse_args()

    if args.comparison:
        print(f"\n{'=' * 60}")
        mode = "No-RAG" if args.no_rag else "Schema-RAG"
        print(f"RAG 对比实验: {mode} 模式")
        print(f"{'=' * 60}")

        if args.max_samples > 0:
            print(f"[INFO] 限制样本数: {args.max_samples}")

        result = run_rag_comparison(
            test_data_path=args.test_data,
            max_samples=args.max_samples,
            use_rag=not args.no_rag,
        )

        print(f"\n{'=' * 60}")
        print(f"RAG 对比 [{mode}]: {result.status}")
        print(f"耗时: {result.duration_seconds:.1f}s")
        m = result.metrics
        print(f"SQL生成成功率: {_fmt(m.get('sql_generation_success_rate'))}")
        print(f"SQL执行成功率: {_fmt(m.get('execution_success_rate'))}")
        print(f"执行准确率(EX): {_fmt(m.get('execution_accuracy'))}")
        print(f"平均响应时间: {m.get('avg_response_time_seconds', 'N/A')}s")
        print(f"{'=' * 60}")
    else:
        # 纯检索实验
        if args.max_samples > 0:
            print(f"[INFO] 限制样本数: {args.max_samples}")

        result = run_rag_experiment(
            test_data_path=args.test_data,
            max_samples=args.max_samples,
        )

        print()
        print("=" * 60)
        print(f"RAG Schema 检索实验: {result.status}")
        print(f"耗时: {result.duration_seconds:.1f}s")
        m = result.metrics
        print(f"Table Recall@1: {_fmt(m.get('table_recall_at_1'))}")
        print(f"Table Recall@3: {_fmt(m.get('table_recall_at_3'))}")
        print(f"Table Recall@5: {_fmt(m.get('table_recall_at_5'))}")
        print(f"Field Recall@1: {_fmt(m.get('field_recall_at_1'))}")
        print(f"Field Recall@3: {_fmt(m.get('field_recall_at_3'))}")
        print(f"Field Recall@5: {_fmt(m.get('field_recall_at_5'))}")
        print(f"MRR: {m.get('mrr', 'N/A')}")
        print(f"平均检索耗时: {m.get('avg_retrieval_time_seconds', 'N/A')}s")
        print("=" * 60)

    return 0 if result.status == "success" else 0  # partial 也正常退出


if __name__ == "__main__":
    sys.exit(main())
