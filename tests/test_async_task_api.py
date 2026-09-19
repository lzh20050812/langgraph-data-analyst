import asyncio
from uuid import uuid4

from fastapi.testclient import TestClient

from api.main import app
from api.main import stream_task_events
from api.task_runtime import task_store


client = TestClient(app)


def test_durable_task_status_checkpoint_and_sse_endpoints():
    task_id = "test-" + uuid4().hex
    task_store.create(task_id, "查询客户数", "sql_query")
    task_store.set_status(task_id, "running")
    task_store.append_event(task_id, {"type": "node_started", "node": "planner"})
    task_store.save_checkpoint(task_id, "planner", {"current_step": "intent_parsed"})
    task_store.set_status(
        task_id,
        "completed",
        result={"request_id": task_id, "success": True, "intent": "sql_query"},
    )
    task_store.append_event(task_id, {"type": "task_completed", "success": True})

    status = client.get(f"/tasks/{task_id}")
    assert status.status_code == 200
    assert status.json()["result"]["success"] is True

    checkpoint = client.get(f"/tasks/{task_id}/checkpoint")
    assert checkpoint.status_code == 200
    assert checkpoint.json()["node"] == "planner"

    events = client.get(f"/tasks/{task_id}/events")
    assert events.status_code == 200
    assert "event: node_started" in events.text
    assert "event: task_completed" in events.text

    listing = client.get("/tasks?limit=10")
    assert listing.status_code == 200
    assert any(item["task_id"] == task_id for item in listing.json()["tasks"])

    summary = client.get("/operations/summary")
    assert summary.status_code == 200
    assert summary.json()["total_tasks"] >= 1
    task_store.delete(task_id)


def test_task_cancel_endpoint_records_request(monkeypatch):
    monkeypatch.setattr("api.main._record_audit", lambda *_, **__: None)
    task_id = "cancel-api-" + uuid4().hex
    task_store.create(task_id, "长任务", "analysis")
    task_store.set_status(task_id, "running")

    response = client.delete(f"/tasks/{task_id}")
    assert response.status_code == 202
    assert response.json()["status"] == "cancelling"
    assert task_store.get(task_id)["status"] == "cancelling"

    task_store.set_status(task_id, "cancelled")
    task_store.delete(task_id)


def test_unknown_task_endpoints_return_404():
    unknown = "missing-" + uuid4().hex
    assert client.get(f"/tasks/{unknown}").status_code == 404
    assert client.get(f"/tasks/{unknown}/events").status_code == 404


def test_sse_reconnect_resumes_after_last_event_without_duplicates():
    task_id = "sse-reconnect-" + uuid4().hex
    task_store.create(task_id, "query", None)
    task_store.set_status(task_id, "running")
    first = task_store.append_event(task_id, {"type": "node_started", "node": "planner"})
    task_store.append_event(task_id, {"type": "node_completed", "node": "planner"})
    task_store.set_status(task_id, "completed", result={"success": True})

    response = client.get(
        f"/tasks/{task_id}/events",
        headers={"Last-Event-ID": str(first)},
    )
    assert response.status_code == 200
    assert "event: node_started" not in response.text
    assert "event: node_completed" in response.text
    assert task_store.get(task_id)["status"] == "completed"
    task_store.delete(task_id)


def test_sse_client_disconnect_does_not_cancel_background_task(monkeypatch):
    task_id = "sse-disconnect-" + uuid4().hex
    task_store.create(task_id, "query", None)
    monkeypatch.setattr("api.main._task_for_principal", lambda *_: task_store.get(task_id))

    class DisconnectedRequest:
        headers = {}

        async def is_disconnected(self):
            return True

    async def consume():
        response = await stream_task_events(task_id, DisconnectedRequest())
        return [chunk async for chunk in response.body_iterator]

    assert asyncio.run(consume()) == []
    assert task_store.get(task_id)["status"] == "queued"
    task_store.delete(task_id)
