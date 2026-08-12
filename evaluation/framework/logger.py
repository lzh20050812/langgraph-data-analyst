"""
实验日志记录器 —— 独立的日志工具，不依赖已有 Agent 代码。

功能：
1. 将实验运行过程的日志同时输出到控制台和文件
2. 日志格式包含时间戳、日志级别、实验名称
3. 自动创建日志目录和文件
4. 提供扩展接口（agent_calls, llm_calls, token_usage）供后续接入

设计约束：
- 不修改任何已有 Agent 代码
- 不读取已有 Agent 内部状态（除非 Agent 自身暴露了公开接口）
- 后续可通过 monkey-patch 或 wrapper 方式采集 Agent/LLM 调用信息
"""

import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


# ============================================================
# 日志格式化器
# ============================================================

class ExperimentFormatter(logging.Formatter):
    """实验日志专用格式化器。

    输出格式:
        [2026-08-10 19:30:00] [text2sql_accuracy] [INFO] 正在加载测试数据...
    """

    def __init__(self, experiment_name: str = ""):
        super().__init__()
        self.experiment_name = experiment_name

    def format(self, record: logging.LogRecord) -> str:
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        exp_tag = f"[{self.experiment_name}]" if self.experiment_name else ""
        level = record.levelname
        msg = record.getMessage()
        return f"[{ts}] {exp_tag} [{level}] {msg}"


# ============================================================
# 实验日志记录器
# ============================================================

class ExperimentLogger:
    """技术评测实验专用日志记录器。

    每个实验实例化一个独立的 Logger，日志同时输出到：
    - 控制台（stdout）
    - 日志文件（evaluation/logs/{experiment_name}_{timestamp}.log）

    使用示例:
        logger = ExperimentLogger("text2sql_accuracy")
        logger.info("加载测试数据...")
        logger.record_sample(1, success=True, detail="SQL 执行成功")
        logger.record_metric("accuracy", 0.92)
        logger.summary()
    """

    def __init__(self, experiment_name: str, log_dir: Optional[Path] = None):
        """
        Args:
            experiment_name: 实验名称，用于日志文件命名
            log_dir: 日志目录，默认为 evaluation/logs/
        """
        self.experiment_name = experiment_name

        # 日志目录
        if log_dir is None:
            log_dir = Path(__file__).resolve().parent.parent / "logs"
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # 日志文件路径
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self.log_file = self.log_dir / f"{experiment_name}_{ts}.log"

        # Python logging 配置
        self._logger = logging.getLogger(f"experiment.{experiment_name}")
        self._logger.setLevel(logging.DEBUG)
        self._logger.handlers.clear()
        self._logger.propagate = False

        # 控制台 handler
        console = logging.StreamHandler(sys.stdout)
        console.setLevel(logging.INFO)
        console.setFormatter(ExperimentFormatter(experiment_name))
        self._logger.addHandler(console)

        # 文件 handler
        file_handler = logging.FileHandler(self.log_file, encoding="utf-8")
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(ExperimentFormatter(experiment_name))
        self._logger.addHandler(file_handler)

        # 运行统计
        self._sample_records: List[Dict[str, Any]] = []
        self._start_time: Optional[datetime] = None

        # ---- 扩展接口：Agent/LLM 调用追踪 ----
        # 当前为预留字段，后续通过外部 wrapper 注入数据
        self._agent_calls: Dict[str, int] = {}    # {"schema_agent": 25, "sql_agent": 25}
        self._llm_calls: int = 0                   # LLM API 总调用次数
        self._token_usage: Dict[str, int] = {}     # {"prompt": 50000, "completion": 15000}

    # ---- 标准日志方法 ----

    def debug(self, msg: str) -> None:
        self._logger.debug(msg)

    def info(self, msg: str) -> None:
        self._logger.info(msg)

    def warning(self, msg: str) -> None:
        self._logger.warning(msg)

    def error(self, msg: str) -> None:
        self._logger.error(msg)

    # ---- 实验专用记录方法 ----

    def start_timer(self) -> None:
        """开始计时。"""
        self._start_time = datetime.now(timezone.utc)
        self.info("计时开始")

    def elapsed(self) -> float:
        """获取已用时间（秒）。"""
        if self._start_time is None:
            return 0.0
        return (datetime.now(timezone.utc) - self._start_time).total_seconds()

    def record_sample(
        self,
        sample_id: Any,
        success: bool,
        detail: str = "",
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """记录单条样本的执行结果。

        Args:
            sample_id: 样本标识
            success: 是否成功
            detail: 详细说明
            extra: 扩展信息（可存放 SQL、预测结果、耗时等）
        """
        record = {
            "sample_id": str(sample_id),
            "success": success,
            "detail": detail,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        if extra:
            record["extra"] = extra

        self._sample_records.append(record)
        status = "✓" if success else "✗"
        self.debug(f"  [{status}] sample={sample_id} | {detail}")

    def record_metric(self, name: str, value: Any) -> None:
        """记录一项评价指标。"""
        self.info(f"  指标 [{name}] = {value}")

    def record_agent_call(self, agent_name: str, count: int = 1) -> None:
        """记录 Agent 调用次数（扩展接口）。

        后续可通过 wrapper 在 Agent 节点函数调用前后自动记录。
        当前阶段仅为预留接口，不实际接入已有代码。
        """
        self._agent_calls[agent_name] = self._agent_calls.get(agent_name, 0) + count

    def record_llm_call(self, count: int = 1) -> None:
        """记录 LLM 调用次数（扩展接口）。"""
        self._llm_calls += count

    def record_token_usage(self, prompt_tokens: int = 0, completion_tokens: int = 0) -> None:
        """记录 Token 消耗（扩展接口）。"""
        self._token_usage["prompt"] = self._token_usage.get("prompt", 0) + prompt_tokens
        self._token_usage["completion"] = self._token_usage.get("completion", 0) + completion_tokens

    # ---- 汇总与持久化 ----

    def get_stats(self) -> Dict[str, Any]:
        """获取当前实验的运行统计。"""
        total = len(self._sample_records)
        success = sum(1 for r in self._sample_records if r["success"])
        return {
            "experiment_name": self.experiment_name,
            "total_samples": total,
            "success_count": success,
            "failure_count": total - success,
            "success_rate": round(success / total, 4) if total > 0 else 0,
            "elapsed_seconds": round(self.elapsed(), 2),
            "agent_calls": dict(self._agent_calls),
            "llm_calls": self._llm_calls,
            "token_usage": dict(self._token_usage),
        }

    def summary(self) -> None:
        """打印实验运行汇总。"""
        stats = self.get_stats()
        self.info("─" * 50)
        self.info(f"实验名称: {stats['experiment_name']}")
        self.info(f"总样本数: {stats['total_samples']}")
        self.info(f"成功: {stats['success_count']} | 失败: {stats['failure_count']}")
        self.info(f"成功率: {stats['success_rate']:.2%}")
        self.info(f"运行耗时: {stats['elapsed_seconds']:.1f}s")
        if stats["agent_calls"]:
            self.info(f"Agent 调用: {stats['agent_calls']}")
        if stats["llm_calls"]:
            self.info(f"LLM 调用次数: {stats['llm_calls']}")
        if stats["token_usage"]:
            self.info(f"Token 消耗: {stats['token_usage']}")
        self.info("─" * 50)

    def save_sample_log(self) -> Path:
        """将逐样本执行记录保存为 JSON 文件。"""
        path = self.log_dir / f"{self.experiment_name}_samples.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "experiment_name": self.experiment_name,
                    "stats": self.get_stats(),
                    "samples": self._sample_records,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )
        self.info(f"逐样本日志已保存: {path}")
        return path
