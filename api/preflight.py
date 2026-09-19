"""Read-only defense-demo preflight checks with safe, structured output."""

from __future__ import annotations

from pathlib import Path
import re
from typing import Any, Literal

from pydantic import BaseModel
from sqlalchemy import text


class PreflightCheck(BaseModel):
    key: str
    label: str
    status: Literal["pass", "warning", "fail"]
    critical: bool
    detail: str
    metadata: dict[str, Any] | None = None


class PreflightReport(BaseModel):
    status: Literal["ready", "degraded"]
    passed: int
    warnings: int
    failed: int
    checks: list[PreflightCheck]


def _mysql_snapshot(settings) -> dict[str, Any]:
    """Return only table counts and privilege categories, never row content."""
    from storage.mysql.client import get_connection

    identifier = re.compile(r"^[A-Za-z0-9_]+$")
    tables = tuple(settings.SQL_ALLOWED_TABLES)
    if not all(identifier.fullmatch(table) for table in tables):
        raise ValueError("invalid configured table identifier")
    counts: dict[str, int] = {}
    with get_connection() as connection:
        for table in tables:
            count = connection.execute(
                text(f"SELECT COUNT(*) AS row_count FROM `{table}`")
            ).scalar_one()
            counts[table] = int(count)
        grant_rows = connection.execute(text("SHOW GRANTS FOR CURRENT_USER()"))
        grants = " ".join(str(row[0]).upper() for row in grant_rows)
    granted_privileges = {
        privilege.strip()
        for clause in re.findall(r"GRANT\s+(.+?)\s+ON\s", grants)
        for privilege in clause.split(",")
    }
    write_privileges = {"INSERT", "UPDATE", "DELETE", "CREATE", "DROP", "ALTER"}
    least_privilege = (
        "ALL PRIVILEGES" not in granted_privileges
        and granted_privileges.isdisjoint(write_privileges)
    )
    return {"table_counts": counts, "least_privilege": least_privilege}


def _add(
    checks: list[PreflightCheck],
    key: str,
    label: str,
    status: Literal["pass", "warning", "fail"],
    critical: bool,
    detail: str,
    metadata: dict[str, Any] | None = None,
) -> None:
    checks.append(PreflightCheck(
        key=key,
        label=label,
        status=status,
        critical=critical,
        detail=detail,
        metadata=metadata,
    ))


def run_preflight(task_store, settings) -> PreflightReport:
    checks: list[PreflightCheck] = []

    try:
        snapshot = _mysql_snapshot(settings)
        counts = snapshot["table_counts"]
        populated = all(count > 0 for count in counts.values())
        _add(
            checks,
            "dataset",
            "演示数据库",
            "pass" if populated else "fail",
            True,
            f"核心表均已加载（{len(counts)} 张表，{sum(counts.values()):,} 行）"
            if populated
            else "存在空表，演示查询可能无结果",
            {"table_counts": counts},
        )
        _add(
            checks,
            "db_privileges",
            "数据库最小权限",
            "pass" if snapshot["least_privilege"] else "warning",
            False,
            "当前账号为只读权限"
            if snapshot["least_privilege"]
            else "本地账号权限较高；Docker 演示使用 SELECT-only 账号",
        )
    except Exception as exc:
        _add(
            checks,
            "dataset",
            "演示数据库",
            "fail",
            True,
            f"数据库预检失败（{type(exc).__name__}）",
        )
        _add(
            checks,
            "db_privileges",
            "数据库最小权限",
            "warning",
            False,
            "数据库不可用，暂无法核验账号权限",
        )

    llm_ready = bool(settings.LLM_API_KEY and "your_" not in settings.LLM_API_KEY)
    _add(
        checks,
        "llm",
        "LLM 配置",
        "pass" if llm_ready else "fail",
        True,
        "模型与 API Key 已配置" if llm_ready else "缺少有效的 LLM API Key",
        {"model": settings.LLM_MODEL} if llm_ready else None,
    )

    task_db_ready = task_store.health_check()
    _add(
        checks,
        "task_store",
        "任务持久化",
        "pass" if task_db_ready else "fail",
        True,
        "SQLite WAL 任务库可读写" if task_db_ready else "任务库快速检查失败",
    )

    execution_mode = settings.TASK_EXECUTION_MODE
    workers = task_store.list_workers(
        active_within_seconds=max(5, settings.TASK_WORKER_LEASE_SECONDS)
    )
    executor_ready = execution_mode == "embedded" or bool(workers)
    _add(
        checks,
        "executor",
        "任务执行器",
        "pass" if executor_ready else "fail",
        True,
        "内嵌执行器已启用"
        if execution_mode == "embedded"
        else f"外部 Worker 在线：{len(workers)}",
        {"mode": execution_mode, "active_workers": len(workers)},
    )

    frontend = Path(settings.BASE_DIR) / "frontend" / "index.html"
    _add(
        checks,
        "frontend",
        "答辩前端",
        "pass" if frontend.is_file() else "fail",
        True,
        "运行态势与分析页面已就位" if frontend.is_file() else "前端入口文件缺失",
    )

    final_results = Path(settings.BASE_DIR) / "evaluation" / "final_results"
    artifacts_ready = all(
        (final_results / name).is_file()
        for name in ("final_metrics.json", "source_manifest.json")
    )
    _add(
        checks,
        "evaluation",
        "论文指标证据",
        "pass" if artifacts_ready else "warning",
        False,
        "冻结指标与来源清单完整"
        if artifacts_ready
        else "冻结指标或来源清单缺失",
    )

    vector_path = Path(settings.CHROMA_PERSIST_DIR)
    _add(
        checks,
        "vector_store",
        "Schema 向量库",
        "pass" if vector_path.exists() else "warning",
        False,
        "本地向量库目录存在"
        if vector_path.exists()
        else "首次查询时可能需要构建向量索引",
    )

    failed = sum(check.status == "fail" for check in checks)
    warnings = sum(check.status == "warning" for check in checks)
    passed = sum(check.status == "pass" for check in checks)
    critical_failed = any(
        check.critical and check.status == "fail" for check in checks
    )
    return PreflightReport(
        status="degraded" if critical_failed else "ready",
        passed=passed,
        warnings=warnings,
        failed=failed,
        checks=checks,
    )
