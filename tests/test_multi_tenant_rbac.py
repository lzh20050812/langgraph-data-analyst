import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

from fastapi.testclient import TestClient

from api.main import app
from api.result_store import result_store
from api.task_runtime import task_store
from api.task_store import TaskStore
from api.task_runtime import SubmissionReceipt


client = TestClient(app)


def _identity_settings():
    return SimpleNamespace(
        API_AUTH_ENABLED=True,
        API_KEYS=(),
        API_PRINCIPALS_JSON=json.dumps([
            {"id": "alice", "role": "analyst", "key": "alice-key"},
            {"id": "bob", "role": "analyst", "key": "bob-key"},
            {"id": "root-admin", "role": "admin", "key": "admin-key"},
        ]),
    )


def test_analysts_are_isolated_and_admin_can_read_all(monkeypatch):
    monkeypatch.setattr("api.auth.get_settings", _identity_settings)
    alice_task = "alice-" + uuid4().hex
    bob_task = "bob-" + uuid4().hex
    task_store.create(alice_task, "Alice query", None, owner_id="alice")
    task_store.create(bob_task, "Bob query", None, owner_id="bob")

    alice_headers = {"X-API-Key": "alice-key"}
    bob_headers = {"X-API-Key": "bob-key"}
    admin_headers = {"X-API-Key": "admin-key"}

    assert client.get(f"/tasks/{alice_task}", headers=alice_headers).status_code == 200
    assert client.get(f"/tasks/{bob_task}", headers=alice_headers).status_code == 404
    assert client.get(f"/tasks/{alice_task}", headers=bob_headers).status_code == 404
    assert client.get(f"/tasks/{bob_task}", headers=admin_headers).status_code == 200

    alice_list = client.get("/tasks?limit=100", headers=alice_headers).json()["tasks"]
    assert alice_task in {item["task_id"] for item in alice_list}
    assert bob_task not in {item["task_id"] for item in alice_list}
    task_store.delete(alice_task)
    task_store.delete(bob_task)


def test_submitted_task_is_bound_to_authenticated_subject(monkeypatch):
    monkeypatch.setattr("api.auth.get_settings", _identity_settings)
    monkeypatch.setattr("api.main._record_audit", lambda *_, **__: None)
    captured = {}

    def submit(query, intent=None, owner_id="local", idempotency_key=None):
        captured.update(
            query=query,
            intent=intent,
            owner_id=owner_id,
            idempotency_key=idempotency_key,
        )
        return SubmissionReceipt("owned-task", replayed=False)

    monkeypatch.setattr("api.main.task_runtime.submit_with_receipt", submit)
    response = client.post(
        "/tasks",
        headers={"X-API-Key": "alice-key"},
        json={"query": "Alice query", "intent": "sql_query"},
    )
    assert response.status_code == 202
    assert captured["owner_id"] == "alice"


def test_result_artifacts_are_owner_scoped(monkeypatch):
    monkeypatch.setattr("api.auth.get_settings", _identity_settings)
    request_id = "result-" + uuid4().hex
    result_store.put(request_id, {
        "intent": "sql_query",
        "report": "private report",
        "charts": [],
        "_owner_id": "alice",
    })

    assert client.get(
        f"/report/{request_id}", headers={"X-API-Key": "alice-key"}
    ).status_code == 200
    assert client.get(
        f"/report/{request_id}", headers={"X-API-Key": "bob-key"}
    ).status_code == 404
    assert client.get(
        f"/report/{request_id}", headers={"X-API-Key": "admin-key"}
    ).status_code == 200


def test_identity_and_prometheus_metrics_are_tenant_scoped(monkeypatch):
    monkeypatch.setattr("api.auth.get_settings", _identity_settings)
    task_id = "metric-alice-" + uuid4().hex
    task_store.create(task_id, "Metric query", None, owner_id="alice")
    task_store.set_status(task_id, "running")
    task_store.set_status(task_id, "completed", result={"success": True})

    headers = {"X-API-Key": "alice-key"}
    identity = client.get("/auth/me", headers=headers)
    assert identity.status_code == 200
    assert identity.json()["subject"] == "alice"
    assert identity.json()["role"] == "analyst"

    metrics = client.get("/operations/metrics", headers=headers)
    assert metrics.status_code == 200
    assert "ai_analytics_tasks_total" in metrics.text
    assert "ai_analytics_task_duration_seconds" in metrics.text
    assert metrics.headers["content-type"].startswith("text/plain")
    task_store.delete(task_id)


def test_openapi_declares_api_key_security():
    schema = client.get("/openapi.json").json()
    scheme = schema["components"]["securitySchemes"]["ApiKeyAuth"]
    assert scheme == {"type": "apiKey", "in": "header", "name": "X-API-Key"}
    assert schema["paths"]["/tasks"]["post"]["security"] == [{"ApiKeyAuth": []}]


def test_structured_log_records_identity_but_never_key(monkeypatch):
    monkeypatch.setattr("api.auth.get_settings", _identity_settings)
    messages = []
    monkeypatch.setattr("api.main.logger.info", messages.append)

    client.get("/tasks/not-found", headers={"X-API-Key": "alice-key"})

    records = [json.loads(message) for message in messages if message.startswith("{")]
    request_log = next(item for item in records if item.get("event") == "http_request")
    assert request_log["principal"] == "alice"
    assert request_log["role"] == "analyst"
    assert "alice-key" not in messages[0]


def test_worker_inventory_requires_admin(monkeypatch):
    monkeypatch.setattr("api.auth.get_settings", _identity_settings)
    worker_id = "worker-test-" + uuid4().hex
    task_store.heartbeat_worker(worker_id, "idle")

    analyst = client.get(
        "/operations/workers", headers={"X-API-Key": "alice-key"}
    )
    admin = client.get(
        "/operations/workers", headers={"X-API-Key": "admin-key"}
    )
    assert analyst.status_code == 403
    assert admin.status_code == 200
    assert worker_id in {item["worker_id"] for item in admin.json()["workers"]}
    task_store.delete_worker(worker_id)


def test_existing_sqlite_store_is_migrated_with_local_owner(tmp_path: Path):
    path = tmp_path / "legacy-tasks.db"
    with sqlite3.connect(path) as connection:
        connection.execute(
            """
            CREATE TABLE tasks (
                task_id TEXT PRIMARY KEY,
                query TEXT NOT NULL,
                intent TEXT,
                status TEXT NOT NULL,
                result_json TEXT,
                error TEXT,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL
            )
            """
        )
        connection.execute(
            """
            INSERT INTO tasks(task_id, query, status, created_at, updated_at)
            VALUES ('legacy', 'query', 'completed', 1, 1)
            """
        )

    migrated = TaskStore(path)
    assert migrated.get("legacy")["owner_id"] == "local"
