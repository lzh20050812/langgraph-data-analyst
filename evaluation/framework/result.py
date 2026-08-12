"""
实验结果管理器 —— 统一的实验结果保存与加载工具。

功能：
1. 将实验结果保存为 JSON / CSV 格式
2. 支持原始实验结果（raw/）和技术评测统计表格（tables/）分类存储
3. 提供公共工具函数，方便实验脚本直接调用

设计约束：
- 不依赖任何已有业务代码
- 仅使用标准库 + pandas（项目已有依赖）
"""

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd


# ============================================================
# 基础工具函数
# ============================================================

def save_json(data: Any, path: Path, indent: int = 2) -> Path:
    """将任意数据保存为 JSON 文件。

    Args:
        data: 要保存的数据（需可 JSON 序列化）
        path: 输出文件路径
        indent: JSON 缩进

    Returns:
        输出文件的 Path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, ensure_ascii=False, default=str)
    return path


def load_json(path: Path) -> Any:
    """从 JSON 文件加载数据。"""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_csv(data: List[Dict[str, Any]], path: Path) -> Path:
    """将字典列表保存为 CSV 文件。

    Args:
        data: 字典列表，每个字典对应一行
        path: 输出文件路径

    Returns:
        输出文件的 Path
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(data)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def load_csv(path: Path) -> List[Dict[str, Any]]:
    """从 CSV 文件加载数据，返回字典列表。"""
    df = pd.read_csv(path, encoding="utf-8-sig")
    return df.to_dict(orient="records")


# ============================================================
# 实验结果管理器
# ============================================================

class ResultManager:
    """实验结果管理器。

    管理两类输出目录：
    - raw/:    原始实验结果（JSON/CSV），用于后续分析和复现
    - tables/: 技术评测统计表格（CSV），可直接插入技术评测或生成 LaTeX 表格

    使用示例:
        manager = ResultManager()
        manager.save_result_json(result)      # 保存为 JSON
        manager.save_result_csv(result)       # 保存为 CSV（指标摘要）
        manager.save_table(df, "table6_1")    # 保存技术评测表格
    """

    def __init__(
        self,
        raw_dir: Optional[Path] = None,
        tables_dir: Optional[Path] = None,
    ):
        """
        Args:
            raw_dir: 原始结果目录，默认 evaluation/results/raw/
            tables_dir: 技术评测表格目录，默认 evaluation/results/tables/
        """
        base = Path(__file__).resolve().parent.parent / "results"

        self.raw_dir = Path(raw_dir) if raw_dir else base / "raw"
        self.tables_dir = Path(tables_dir) if tables_dir else base / "tables"

        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.tables_dir.mkdir(parents=True, exist_ok=True)

    # ---- 原始结果保存 ----

    def save_result_json(
        self,
        result,
        filename: Optional[str] = None,
    ) -> Path:
        """将 ExperimentResult 保存为 JSON 文件。

        Args:
            result: ExperimentResult 对象或字典
            filename: 自定义文件名（不含扩展名），默认自动生成

        Returns:
            输出文件路径
        """
        if filename is None:
            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            name = getattr(result, "experiment_name", "unknown")
            filename = f"{name}_{ts}"

        data = result.to_dict() if hasattr(result, "to_dict") else result
        path = self.raw_dir / f"{filename}.json"
        return save_json(data, path)

    def save_result_csv(
        self,
        result,
        filename: Optional[str] = None,
    ) -> Path:
        """将实验结果指标保存为 CSV（一行一条实验，方便对比）。

        Args:
            result: ExperimentResult 对象或字典
            filename: 自定义文件名，默认自动生成

        Returns:
            输出文件路径
        """
        if filename is None:
            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            name = getattr(result, "experiment_name", "unknown")
            filename = f"{name}_{ts}"

        if hasattr(result, "to_dict"):
            d = result.to_dict()
        else:
            d = result

        # 展平为一维行（metrics 展开为独立列）
        row = {
            "experiment_name": d.get("experiment_name", ""),
            "status": d.get("status", ""),
            "started_at": d.get("started_at", ""),
            "duration_seconds": d.get("duration_seconds", 0),
            "sample_count": d.get("sample_count", 0),
            "success_count": d.get("success_count", 0),
            "failure_count": d.get("failure_count", 0),
        }
        # 展开 metrics
        for k, v in d.get("metrics", {}).items():
            row[f"metric_{k}"] = v

        path = self.raw_dir / f"{filename}.csv"
        return save_csv([row], path)

    # ---- 技术评测表格保存 ----

    def save_table(
        self,
        df: pd.DataFrame,
        table_name: str,
        formats: Optional[List[str]] = None,
    ) -> Dict[str, Path]:
        """保存技术评测统计表格。

        一次调用可同时保存 CSV 和 JSON 两种格式。

        Args:
            df: pandas DataFrame
            table_name: 表格名称（如 "table6_1_text2sql_results"）
            formats: 格式列表，默认 ["csv"]

        Returns:
            {format: file_path} 字典
        """
        if formats is None:
            formats = ["csv"]

        saved = {}
        for fmt in formats:
            path = self.tables_dir / f"{table_name}.{fmt}"
            if fmt == "csv":
                df.to_csv(path, index=False, encoding="utf-8-sig")
            elif fmt == "json":
                df.to_json(path, orient="records", indent=2, force_ascii=False)
            saved[fmt] = path

        return saved

    # ---- 结果加载与聚合 ----

    def load_all_results(self) -> List[Dict[str, Any]]:
        """加载 raw/ 目录下所有 JSON 实验结果。"""
        results = []
        for f in sorted(self.raw_dir.glob("*.json")):
            try:
                results.append(load_json(f))
            except Exception:
                continue
        return results

    def aggregate_to_dataframe(self) -> pd.DataFrame:
        """将 raw/ 目录下所有实验结果聚合为一个 DataFrame。

        Returns:
            每行一次实验，列为 metrics 展开后的各维度
        """
        all_results = self.load_all_results()
        rows = []
        for r in all_results:
            row = {
                "experiment_name": r.get("experiment_name", ""),
                "status": r.get("status", ""),
                "duration_seconds": r.get("duration_seconds", 0),
                "sample_count": r.get("sample_count", 0),
                "success_count": r.get("success_count", 0),
                "failure_count": r.get("failure_count", 0),
            }
            for k, v in r.get("metrics", {}).items():
                row[f"metric_{k}"] = v
            rows.append(row)
        return pd.DataFrame(rows)
