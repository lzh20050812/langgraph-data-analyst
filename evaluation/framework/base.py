"""
实验基类 —— 定义所有技术评测实验的统一接口。

后续所有具体实验（Text2SQL / RAG / 协同对比 / Memory / 预测 / 性能）
均需继承 ExperimentBase 并实现抽象方法，确保实验流程一致、结果可复现。

统一实验流程：
    experiment = MyExperiment(config)
    experiment.execute()                    # 完整流程
    # 或分步执行：
    experiment.load_data()                  # 1. 加载测试数据
    experiment.run()                        # 2. 执行实验
    experiment.evaluate()                   # 3. 计算指标
    experiment.save_results()               # 4. 保存结果
"""

import time
import traceback
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


# ============================================================
# 实验结果数据结构
# ============================================================

@dataclass
class ExperimentResult:
    """单次实验运行的完整结果。

    设计为 dataclass 以支持 JSON 序列化和跨实验对比。

    Attributes:
        experiment_name: 实验名称（如 "text2sql_accuracy"）
        status: 运行状态（"success" / "partial" / "failed"）
        started_at: 开始时间（ISO 8601 字符串）
        finished_at: 结束时间（ISO 8601 字符串）
        duration_seconds: 运行耗时（秒）
        metrics: 评价指标字典（如 {"accuracy": 0.92, "f1": 0.88}）
        details: 逐样本详细结果列表
        summary: 实验摘要文本（可选的文字总结）
        sample_count: 测试样本总数
        success_count: 成功样本数
        failure_count: 失败样本数
        errors: 异常信息列表
        metadata: 扩展元数据（可存放 LLM 调用次数、Token 消耗等）
    """

    experiment_name: str
    status: str = "pending"  # pending | running | success | partial | failed
    started_at: str = ""
    finished_at: str = ""
    duration_seconds: float = 0.0

    # 指标
    metrics: Dict[str, Any] = field(default_factory=dict)

    # 明细
    details: List[Dict[str, Any]] = field(default_factory=list)

    # 摘要
    summary: str = ""

    # 计数
    sample_count: int = 0
    success_count: int = 0
    failure_count: int = 0

    # 异常
    errors: List[Dict[str, str]] = field(default_factory=list)

    # 扩展元数据（LLM 调用次数、Token 消耗、Agent 调用次数等）
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转为可 JSON 序列化的字典。"""
        return {
            "experiment_name": self.experiment_name,
            "status": self.status,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_seconds": self.duration_seconds,
            "metrics": self.metrics,
            "details": self.details,
            "summary": self.summary,
            "sample_count": self.sample_count,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "errors": self.errors,
            "metadata": self.metadata,
        }


# ============================================================
# 实验基类
# ============================================================

class ExperimentBase(ABC):
    """技术评测实验统一基类。

    子类必须实现:
        - load_data(): 加载测试数据
        - run(): 执行实验核心逻辑
        - evaluate(): 计算评价指标

    子类可选覆盖:
        - save_results(): 保存实验结果（已有默认实现）
        - setup(): 实验前准备（如初始化模型、连接数据库）
        - teardown(): 实验后清理

    使用示例:
        class Text2SQLExperiment(ExperimentBase):
            name = "text2sql_accuracy"
            description = "评估 SQL Agent 在不同难度下的生成准确率"

            def load_data(self):
                # 加载测试集
                self.test_queries = load_test_queries()
                return len(self.test_queries)

            def run(self):
                # 逐条执行并记录
                for query in self.test_queries:
                    result = run_sql_agent(query)
                    self.result.details.append(result)

            def evaluate(self):
                # 计算准确率等指标
                correct = sum(1 for d in self.result.details if d["correct"])
                self.result.metrics["accuracy"] = correct / len(self.result.details)
    """

    # ---- 子类必须设置的元数据 ----
    name: str = ""            # 实验名称（如 "text2sql_accuracy"）
    description: str = ""     # 实验描述
    version: str = "1.0"      # 实验版本

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Args:
            config: 实验配置字典。子类可在 setup() 中读取具体配置项。
        """
        self.config = config or {}
        self.result = ExperimentResult(experiment_name=self.name)
        self._logger = None  # 延迟初始化，子类通过 self.logger 属性访问

    # ---- 日志 ----

    @property
    def logger(self):
        """获取实验日志记录器（延迟导入，避免循环依赖）。"""
        if self._logger is None:
            from .logger import ExperimentLogger
            self._logger = ExperimentLogger(self.name)
        return self._logger

    # ---- 抽象方法（子类必须实现） ----

    @abstractmethod
    def load_data(self) -> int:
        """加载测试数据。

        子类实现应：
        1. 从 evaluation/data/ 或外部加载测试集
        2. 将数据存储在 self 的属性中（如 self.test_queries）
        3. 验证数据完整性和格式

        Returns:
            加载的测试样本数量
        """
        ...

    @abstractmethod
    def run(self) -> None:
        """执行实验核心逻辑。

        子类实现应：
        1. 遍历测试样本，调用对应的 Agent 或功能模块
        2. 将每条样本的执行结果追加到 self.result.details
        3. 更新 self.result.success_count / failure_count
        4. 捕获并记录异常到 self.result.errors
        5. 通过 self.logger 记录关键过程信息

        注意：不要修改已有 Agent 的调用方式，仅作为外部调用者使用。
        """
        ...

    @abstractmethod
    def evaluate(self) -> Dict[str, Any]:
        """计算评价指标。

        子类实现应：
        1. 从 self.result.details 中读取逐样本结果
        2. 计算所有需要的评价指标
        3. 将指标写入 self.result.metrics
        4. 生成文字摘要写入 self.result.summary

        Returns:
            评价指标字典
        """
        ...

    # ---- 可选覆盖方法 ----

    def setup(self) -> None:
        """实验前准备。子类可覆盖以实现模型加载、数据库连接等初始化操作。"""
        pass

    def teardown(self) -> None:
        """实验后清理。子类可覆盖以实现资源释放、连接关闭等操作。"""
        pass

    def save_results(
        self,
        output_dir: Optional[Path] = None,
        formats: Optional[List[str]] = None,
    ) -> Dict[str, Path]:
        """保存实验结果。

        默认保存为 JSON 格式到 evaluation/results/raw/ 目录。
        子类可覆盖以实现自定义保存逻辑。

        Args:
            output_dir: 输出目录，默认为 evaluation/results/raw/
            formats: 保存格式列表，默认 ["json"]，可选 ["json", "csv"]

        Returns:
            {format: file_path} 字典
        """
        from .result import ResultManager

        if output_dir is None:
            output_dir = Path(__file__).resolve().parent.parent / "results" / "raw"

        if formats is None:
            formats = ["json"]

        manager = ResultManager(output_dir)
        saved = {}

        if "json" in formats:
            saved["json"] = manager.save_result_json(self.result)
        if "csv" in formats:
            saved["csv"] = manager.save_result_csv(self.result)

        self.logger.info(f"结果已保存: {saved}")
        return saved

    # ---- 完整流程 ----

    def execute(self) -> ExperimentResult:
        """执行完整实验流程（一键运行）。

        流程: setup → load_data → run → evaluate → save_results → teardown

        Returns:
            ExperimentResult 对象，包含完整的实验结果
        """
        self.result.status = "running"
        self.result.started_at = _now_iso()

        try:
            # 1. 准备
            self.logger.info(f"========== 实验开始: {self.name} v{self.version} ==========")
            self.logger.info(f"描述: {self.description}")
            self.setup()

            # 2. 加载数据
            self.logger.info("正在加载测试数据...")
            self.result.sample_count = self.load_data()
            self.logger.info(f"加载完成，共 {self.result.sample_count} 条测试样本")

            # 3. 执行实验
            self.logger.info("正在执行实验...")
            self.run()
            self.logger.info(
                f"执行完成: 成功 {self.result.success_count}, "
                f"失败 {self.result.failure_count}"
            )

            # 4. 计算指标
            self.logger.info("正在计算评价指标...")
            self.evaluate()
            self.logger.info(f"指标: {self.result.metrics}")

            # 5. 设置最终状态
            self.result.status = (
                "success" if self.result.failure_count == 0 else "partial"
            )

            # 6. 保存结果
            self.logger.info("正在保存实验结果...")
            self.save_results()

        except Exception as e:
            self.result.status = "failed"
            self.result.errors.append({
                "phase": "execute",
                "error": str(e),
                "traceback": traceback.format_exc(),
            })
            self.logger.error(f"实验失败: {e}")
            self.logger.error(traceback.format_exc())

        finally:
            self.result.finished_at = _now_iso()
            self.result.duration_seconds = round(
                _parse_iso_diff(self.result.finished_at, self.result.started_at), 2
            )
            self.logger.info(
                f"========== 实验结束: {self.result.status} "
                f"(耗时 {self.result.duration_seconds}s) =========="
            )
            self.teardown()

        return self.result


# ============================================================
# 辅助函数
# ============================================================

def _now_iso() -> str:
    """返回当前时间的 ISO 8601 字符串。"""
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def _parse_iso_diff(end: str, start: str) -> float:
    """计算两个 ISO 时间字符串之间的秒数差。"""
    from datetime import datetime
    try:
        t_end = datetime.fromisoformat(end)
        t_start = datetime.fromisoformat(start)
        return (t_end - t_start).total_seconds()
    except Exception:
        return 0.0
