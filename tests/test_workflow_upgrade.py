from pathlib import Path
from time import monotonic, sleep

import pytest

from agents.contracts import EvidenceBundle, TaskPlan
from agents.observability import traced_node
from agents.run_context import WorkflowCancelled, run_hooks
from agents.sql_safety import enforce_row_limit, validate_read_only_sql
from agents.state import create_initial_state
from api.task_runtime import TaskQueueFull, TaskRuntime
from api.task_store import TaskStore


def test_strong_contracts_reject_inconsistent_payloads():
    with pytest.raises(ValueError):
        TaskPlan(intent="sql_query", analysis_tools=["rfm"])

    with pytest.raises(ValueError):
        EvidenceBundle(
            user_query="客户数",
            task_plan={},
            rows=[{"count": 1}],
            row_count=0,
        )


def test_node_tracing_emits_events_and_checkpoint():
    events = []
    checkpoints = []
    state = create_initial_state("客户数", run_id="run-1")

    def node(value):
        value["current_step"] = "done"
        return value

    with run_hooks(
        event_sink=events.append,
        checkpoint_sink=lambda name, value: checkpoints.append((name, value)),
    ):
        result = traced_node("demo_node", node)(state)

    assert [event["type"] for event in events] == [
        "node_started",
        "node_completed",
    ]
    assert result["execution_trace"][0]["node"] == "demo_node"
    assert checkpoints[0][0] == "demo_node"


def test_node_cancellation_keeps_last_successful_checkpoint():
    state = create_initial_state("客户数", run_id="cancel-1")
    checks = iter([False, True])
    checkpoints = []

    def node(value):
        value["current_step"] = "done"
        return value

    with run_hooks(
        cancellation_check=lambda: next(checks),
        checkpoint_sink=lambda name, value: checkpoints.append((name, value)),
    ):
        with pytest.raises(WorkflowCancelled):
            traced_node("cancel_demo", node)(state)

    assert checkpoints[0][0] == "cancel_demo"
    assert state["execution_trace"][0]["status"] == "completed"


def test_task_store_is_durable_and_marks_interrupted_runs(tmp_path: Path):
    path = tmp_path / "tasks.db"
    store = TaskStore(path)
    store.create("task-1", "客户数", None)
    store.set_status("task-1", "running")
    store.append_event("task-1", {"type": "node_started", "node": "planner"})
    store.save_checkpoint("task-1", "planner", {"current_step": "planned"})

    reopened = TaskStore(path)
    assert reopened.interrupt_stale(0) == 1
    assert reopened.get("task-1")["status"] == "interrupted"
    assert reopened.events_after("task-1")[0]["node"] == "planner"
    assert reopened.latest_checkpoint("task-1")["state"]["current_step"] == "planned"


def test_task_store_lists_summarizes_and_prunes_terminal_records(tmp_path: Path):
    store = TaskStore(tmp_path / "lifecycle.db")
    store.create("old", "旧任务", None)
    store.set_status("old", "running")
    store.set_status("old", "completed", result={"success": True})
    sleep(0.01)
    store.create("new", "新任务", None)
    store.set_status("new", "running")
    store.set_status("new", "completed", result={"success": True})

    assert store.list_recent(limit=1)[0]["task_id"] == "new"
    assert store.summary()["status_counts"]["completed"] == 2
    assert store.prune(retention_seconds=3600, max_records=1) == 1
    assert store.get("old") is None
    assert store.get("new") is not None


def test_task_runtime_persists_public_result(monkeypatch, tmp_path: Path):
    def fake_run_query(query, requested_intent=None, run_id=None):
        return {
            "run_id": run_id,
            "intent": requested_intent or "sql_query",
            "query_result": [{"count": 1}],
            "error": None,
            "execution_trace": [],
        }

    monkeypatch.setattr("agents.planner.run_query", fake_run_query)
    store = TaskStore(tmp_path / "runtime.db")
    runtime = TaskRuntime(store, max_workers=1)
    task_id = runtime.submit("客户数")

    deadline = monotonic() + 3
    while monotonic() < deadline:
        task = store.get(task_id)
        if task["status"] in {"completed", "failed"}:
            break
        sleep(0.01)

    assert task["status"] == "completed"
    assert task["result"]["query_result"] == [{"count": 1}]
    assert store.events_after(task_id)[-1]["type"] == "task_completed"


def test_running_task_cannot_be_retried(tmp_path: Path):
    store = TaskStore(tmp_path / "retry.db")
    runtime = TaskRuntime(store, max_workers=1)
    store.create("running-task", "客户数", None)
    store.set_status("running-task", "running")
    with pytest.raises(RuntimeError):
        runtime.retry("running-task")


def test_queued_task_can_be_cancelled_before_execution(tmp_path: Path):
    store = TaskStore(tmp_path / "cancel.db")
    runtime = TaskRuntime(store, max_workers=1)
    store.create("queued-task", "客户数", None)
    assert runtime.cancel("queued-task") == "cancelled"
    runtime._execute("queued-task", "客户数", None)
    assert store.get("queued-task")["status"] == "cancelled"
    assert store.events_after("queued-task")[-1]["type"] == "task_cancelled"


def test_background_queue_is_bounded(tmp_path: Path):
    store = TaskStore(tmp_path / "bounded.db")
    runtime = TaskRuntime(store, max_workers=1, max_queued_tasks=0)
    assert runtime._capacity.acquire(blocking=False)
    try:
        with pytest.raises(TaskQueueFull):
            runtime.submit("客户数")
    finally:
        runtime._capacity.release()


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT SLEEP(10)",
        "SELECT * FROM mysql.user",
        "SELECT * FROM customers INTO OUTFILE '/tmp/customers.csv'",
        "WITH x AS (SELECT 1) DELETE FROM customers",
    ],
)
def test_ast_guard_rejects_side_effects_and_unauthorized_tables(sql):
    safe, _ = validate_read_only_sql(sql, allowed_tables={"customers", "orders"})
    assert safe is False


def test_ast_guard_accepts_ctes_and_enforces_result_limit():
    sql = "WITH totals AS (SELECT customer_id FROM orders) SELECT * FROM totals"
    safe, reason = validate_read_only_sql(sql, allowed_tables={"orders"})
    assert safe, reason
    assert "LIMIT 100" in enforce_row_limit(sql, 100).upper()
    assert "LIMIT 10" in enforce_row_limit("SELECT * FROM orders LIMIT 10", 100).upper()
