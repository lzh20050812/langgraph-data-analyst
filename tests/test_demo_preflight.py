import json
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

from api.main import app
from api.preflight import run_preflight
from api.task_store import TaskStore


client = TestClient(app)


def _settings(root: Path, **overrides):
    values = {
        "BASE_DIR": root,
        "SQL_ALLOWED_TABLES": ("customers", "orders"),
        "LLM_API_KEY": "configured-key",
        "LLM_MODEL": "demo-model",
        "TASK_EXECUTION_MODE": "embedded",
        "TASK_WORKER_LEASE_SECONDS": 60,
        "CHROMA_PERSIST_DIR": str(root / "data" / "chromadb"),
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _demo_files(root: Path):
    (root / "frontend").mkdir(parents=True)
    (root / "frontend" / "index.html").write_text("demo", encoding="utf-8")
    results = root / "evaluation" / "final_results"
    results.mkdir(parents=True)
    (results / "final_metrics.json").write_text("{}", encoding="utf-8")
    (results / "source_manifest.json").write_text("{}", encoding="utf-8")
    (root / "data" / "chromadb").mkdir(parents=True)


def test_preflight_reports_ready_with_noncritical_privilege_warning(
    monkeypatch, tmp_path: Path
):
    _demo_files(tmp_path)
    monkeypatch.setattr(
        "api.preflight._mysql_snapshot",
        lambda _: {
            "table_counts": {"customers": 100, "orders": 200},
            "least_privilege": False,
        },
    )
    report = run_preflight(TaskStore(tmp_path / "tasks.db"), _settings(tmp_path))

    assert report.status == "ready"
    assert report.failed == 0
    assert next(
        check for check in report.checks if check.key == "db_privileges"
    ).status == "warning"


def test_preflight_degrades_when_critical_demo_dependencies_fail(
    monkeypatch, tmp_path: Path
):
    monkeypatch.setattr(
        "api.preflight._mysql_snapshot",
        lambda _: {
            "table_counts": {"customers": 0, "orders": 200},
            "least_privilege": True,
        },
    )
    report = run_preflight(
        TaskStore(tmp_path / "tasks.db"),
        _settings(
            tmp_path,
            LLM_API_KEY="",
            TASK_EXECUTION_MODE="external",
        ),
    )

    assert report.status == "degraded"
    failed_keys = {
        check.key for check in report.checks if check.status == "fail"
    }
    assert {"dataset", "llm", "executor", "frontend"}.issubset(failed_keys)


def test_task_store_quick_check(tmp_path: Path):
    assert TaskStore(tmp_path / "healthy.db").health_check() is True


def test_preflight_endpoint_requires_admin(monkeypatch):
    settings = SimpleNamespace(
        API_AUTH_ENABLED=True,
        API_KEYS=(),
        API_PRINCIPALS_JSON=json.dumps([
            {"id": "alice", "role": "analyst", "key": "alice-key"},
            {"id": "root-admin", "role": "admin", "key": "admin-key"},
        ]),
    )
    monkeypatch.setattr("api.auth.get_settings", lambda: settings)
    monkeypatch.setattr(
        "api.main.run_preflight",
        lambda *_: {
            "status": "ready",
            "passed": 1,
            "warnings": 0,
            "failed": 0,
            "checks": [{
                "key": "demo",
                "label": "Demo",
                "status": "pass",
                "critical": True,
                "detail": "ready",
                "metadata": None,
            }],
        },
    )

    analyst = client.get(
        "/operations/preflight", headers={"X-API-Key": "alice-key"}
    )
    admin = client.get(
        "/operations/preflight", headers={"X-API-Key": "admin-key"}
    )
    assert analyst.status_code == 403
    assert admin.status_code == 200
    assert admin.json()["status"] == "ready"
