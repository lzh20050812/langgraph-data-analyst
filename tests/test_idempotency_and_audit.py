from concurrent.futures import ThreadPoolExecutor
import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

from api.main import app
from api.task_runtime import SubmissionReceipt, TaskRuntime
from api.task_store import IdempotencyConflict, TaskStore


client = TestClient(app)


def _identity_settings():
    return SimpleNamespace(
        API_AUTH_ENABLED=True,
        API_KEYS=(),
        API_PRINCIPALS_JSON=json.dumps([
            {"id": "alice", "role": "analyst", "key": "alice-key"},
            {"id": "root-admin", "role": "admin", "key": "admin-key"},
        ]),
    )


def test_concurrent_idempotent_submissions_create_one_task(tmp_path: Path):
    path = tmp_path / "idempotency.db"
    first = TaskRuntime(TaskStore(path), 1, 20, execution_mode="external")
    second = TaskRuntime(TaskStore(path), 1, 20, execution_mode="external")

    with ThreadPoolExecutor(max_workers=2) as pool:
        receipts = list(pool.map(
            lambda runtime: runtime.submit_with_receipt(
                "查询客户数",
                "sql_query",
                owner_id="alice",
                idempotency_key="request-001",
            ),
            (first, second),
        ))

    assert len({receipt.task_id for receipt in receipts}) == 1
    assert sorted(receipt.replayed for receipt in receipts) == [False, True]
    task_id = receipts[0].task_id
    assert len(first.store.events_after(task_id)) == 1
    assert first.store.queue_depth() == 1


def test_idempotency_key_reuse_with_different_payload_conflicts(tmp_path: Path):
    runtime = TaskRuntime(
        TaskStore(tmp_path / "conflict.db"), 1, 20, execution_mode="external"
    )
    runtime.submit("查询客户数", owner_id="alice", idempotency_key="same-key")

    try:
        runtime.submit("查询订单数", owner_id="alice", idempotency_key="same-key")
    except IdempotencyConflict:
        pass
    else:
        raise AssertionError("different payload must not reuse an idempotency key")


def test_idempotency_keys_are_scoped_per_tenant(tmp_path: Path):
    runtime = TaskRuntime(
        TaskStore(tmp_path / "tenants.db"), 1, 20, execution_mode="external"
    )
    alice = runtime.submit("query", owner_id="alice", idempotency_key="shared")
    bob = runtime.submit("query", owner_id="bob", idempotency_key="shared")
    assert alice != bob


def test_audit_store_filters_and_decodes_metadata(tmp_path: Path):
    store = TaskStore(tmp_path / "audit.db")
    store.append_audit(
        owner_id="alice",
        role="analyst",
        action="task.submit",
        resource_type="task",
        resource_id="task-1",
        request_id="request-1",
        outcome="accepted",
        metadata={"idempotency_key_present": True},
    )
    store.append_audit(
        owner_id="bob",
        role="analyst",
        action="task.cancel",
        resource_type="task",
        resource_id="task-2",
        request_id="request-2",
        outcome="cancelled",
    )

    events = store.list_audit(owner_id="alice")
    assert len(events) == 1
    assert events[0]["metadata"] == {"idempotency_key_present": True}
    assert store.audit_count() == 2
    assert store.audit_count(owner_id="alice") == 1


def test_submit_api_forwards_idempotency_key_and_audits(monkeypatch):
    captured = {}
    audits = []

    def submit(query, intent=None, owner_id="local", idempotency_key=None):
        captured.update(
            query=query,
            owner_id=owner_id,
            idempotency_key=idempotency_key,
        )
        return SubmissionReceipt("task-idempotent", replayed=True)

    monkeypatch.setattr("api.main.task_runtime.submit_with_receipt", submit)
    monkeypatch.setattr(
        "api.main.task_store.append_audit", lambda **kwargs: audits.append(kwargs) or 1
    )
    monkeypatch.setattr("api.main.task_store.prune_audit", lambda *_: 0)

    response = client.post(
        "/tasks",
        headers={"Idempotency-Key": "browser-request-1"},
        json={"query": "查询客户数"},
    )

    assert response.status_code == 202
    assert response.json()["idempotency_replayed"] is True
    assert captured["idempotency_key"] == "browser-request-1"
    assert audits[0]["outcome"] == "replayed"
    assert "browser-request-1" not in json.dumps(audits)


def test_submit_api_rejects_idempotency_payload_conflict(monkeypatch):
    monkeypatch.setattr(
        "api.main.task_runtime.submit_with_receipt",
        lambda *_, **__: (_ for _ in ()).throw(IdempotencyConflict("same-key")),
    )
    monkeypatch.setattr("api.main.task_store.append_audit", lambda **_: 1)
    monkeypatch.setattr("api.main.task_store.prune_audit", lambda *_: 0)

    response = client.post(
        "/tasks",
        headers={"Idempotency-Key": "same-key"},
        json={"query": "different query"},
    )
    assert response.status_code == 409


def test_audit_endpoint_requires_admin(monkeypatch):
    monkeypatch.setattr("api.auth.get_settings", _identity_settings)
    monkeypatch.setattr(
        "api.main.task_store.list_audit",
        lambda **_: [{"sequence": 7, "action": "task.submit"}],
    )

    analyst = client.get(
        "/operations/audit", headers={"X-API-Key": "alice-key"}
    )
    admin = client.get(
        "/operations/audit", headers={"X-API-Key": "admin-key"}
    )
    assert analyst.status_code == 403
    assert admin.status_code == 200
    assert admin.json()["next_after"] == 7


def test_legacy_database_migrates_idempotency_columns(tmp_path: Path):
    path = tmp_path / "legacy.db"
    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            CREATE TABLE tasks (
                task_id TEXT PRIMARY KEY, query TEXT NOT NULL, intent TEXT,
                status TEXT NOT NULL, result_json TEXT, error TEXT,
                created_at REAL NOT NULL, updated_at REAL NOT NULL
            )
            """
        )
    store = TaskStore(path)
    store.create("legacy-new", "query", None)
    task = store.get("legacy-new")
    assert "idempotency_key" in task
    assert "request_fingerprint" in task
