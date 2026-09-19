from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

from api.main import app
from api.task_runtime import SubmissionReceipt
from api.task_store import TaskStore


client = TestClient(app)


def test_shared_limit_is_atomic_across_store_instances(tmp_path: Path):
    path = tmp_path / "rate-limit.db"
    stores = [TaskStore(path) for _ in range(10)]

    with ThreadPoolExecutor(max_workers=10) as pool:
        decisions = list(pool.map(
            lambda store: store.consume_rate_limit(
                owner_id="alice",
                route_scope="analysis",
                limit=5,
                window_seconds=60,
                now=120.0,
            ),
            stores,
        ))

    assert sum(decision["allowed"] for decision in decisions) == 5
    assert max(decision["request_count"] for decision in decisions) == 5


def test_rate_limit_is_tenant_scoped_and_resets_by_window(tmp_path: Path):
    store = TaskStore(tmp_path / "tenant-rate-limit.db")
    first = store.consume_rate_limit(
        owner_id="alice", route_scope="analysis", limit=1,
        window_seconds=60, now=120.0,
    )
    blocked = store.consume_rate_limit(
        owner_id="alice", route_scope="analysis", limit=1,
        window_seconds=60, now=121.0,
    )
    other_tenant = store.consume_rate_limit(
        owner_id="bob", route_scope="analysis", limit=1,
        window_seconds=60, now=121.0,
    )
    next_window = store.consume_rate_limit(
        owner_id="alice", route_scope="analysis", limit=1,
        window_seconds=60, now=180.0,
    )

    assert first["allowed"] is True
    assert blocked["allowed"] is False
    assert other_tenant["allowed"] is True
    assert next_window["allowed"] is True


def test_task_api_returns_standard_rate_limit_headers_and_429(monkeypatch):
    decisions = iter([
        {
            "allowed": True, "limit": 1, "remaining": 0,
            "reset_at": 180, "retry_after": 30,
        },
        {
            "allowed": False, "limit": 1, "remaining": 0,
            "reset_at": 180, "retry_after": 29,
        },
    ])
    audits = []
    monkeypatch.setattr(
        "api.main.get_settings",
        lambda: SimpleNamespace(
            API_RATE_LIMIT_REQUESTS=1,
            API_RATE_LIMIT_WINDOW_SECONDS=60,
            AUDIT_RETENTION_SECONDS=3600,
            AUDIT_MAX_RECORDS=100,
        ),
    )
    monkeypatch.setattr(
        "api.main.task_store.consume_rate_limit", lambda **_: next(decisions)
    )
    monkeypatch.setattr(
        "api.main.task_runtime.submit_with_receipt",
        lambda *_, **__: SubmissionReceipt("rate-task", replayed=False),
    )
    monkeypatch.setattr(
        "api.main.task_store.append_audit", lambda **kwargs: audits.append(kwargs) or 1
    )
    monkeypatch.setattr("api.main.task_store.prune_audit", lambda *_: 0)

    accepted = client.post("/tasks", json={"query": "first"})
    rejected = client.post("/tasks", json={"query": "second"})

    assert accepted.status_code == 202
    assert accepted.headers["X-RateLimit-Limit"] == "1"
    assert accepted.headers["X-RateLimit-Remaining"] == "0"
    assert rejected.status_code == 429
    assert rejected.headers["Retry-After"] == "29"
    assert any(event["outcome"] == "rate_limited" for event in audits)


def test_rate_limit_inventory_requires_admin(monkeypatch):
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
        "api.main.task_store.list_rate_limits",
        lambda **_: [{
            "owner_id": "alice", "route_scope": "analysis",
            "window_start": 120, "request_count": 3, "updated_at": 121,
        }],
    )

    analyst = client.get(
        "/operations/rate-limits", headers={"X-API-Key": "alice-key"}
    )
    admin = client.get(
        "/operations/rate-limits", headers={"X-API-Key": "admin-key"}
    )
    assert analyst.status_code == 403
    assert admin.status_code == 200
    assert admin.json()["windows"][0]["remaining"] >= 0


def test_metrics_expose_rate_limit_rejections(monkeypatch):
    monkeypatch.setattr(
        "api.main.task_store.audit_outcome_count", lambda *_, **__: 4
    )
    response = client.get("/operations/metrics")
    assert response.status_code == 200
    assert "ai_analytics_rate_limit_rejections_retained 4" in response.text
