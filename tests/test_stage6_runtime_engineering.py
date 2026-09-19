from pathlib import Path

import pytest

from agents.runtime_policy import (
    RunBudget,
    WorkflowBudgetExceeded,
    classify_failure,
)
from api.event_safety import sanitize_event
from api.task_runtime import TaskRuntime, UnsafeCheckpointResume
from api.task_state import InvalidTaskTransition
from api.task_store import TaskStore


def test_illegal_terminal_transition_is_rejected(tmp_path: Path):
    store = TaskStore(tmp_path / "states.db")
    store.create("task", "query", None)
    with pytest.raises(InvalidTaskTransition):
        store.set_status("task", "completed")
    assert store.get("task")["status"] == "queued"


def test_restart_retry_records_parent_mode_and_duplicate_risks(tmp_path: Path):
    store = TaskStore(tmp_path / "retry.db")
    runtime = TaskRuntime(store, max_workers=1, execution_mode="external")
    store.create("original", "query", "sql_query", owner_id="tenant")
    store.set_status("original", "running")
    assert store.finalize("original", "failed", {"type": "task_failed"})

    retried = runtime.retry("original", mode="restart")
    task = store.get(retried)
    assert task["parent_task_id"] == "original"
    assert task["retry_mode"] == "restart"
    assert "llm_generation_may_be_billed_again" in task["retry_risks"]

    with pytest.raises(UnsafeCheckpointResume):
        runtime.retry("original", mode="checkpoint")


def test_stale_worker_token_cannot_write_after_lease_handoff(tmp_path: Path):
    store = TaskStore(tmp_path / "ownership.db")
    store.create("task", "query", None)
    first = store.claim_next("worker-a", lease_seconds=0.001)
    import time
    time.sleep(0.01)
    assert store.recover_expired_leases(max_attempts=2)["requeued"] == 1
    second = store.claim_next("worker-b", lease_seconds=30)

    assert store.append_event("task", {"type": "late"}, first["execution_token"]) is None
    assert store.save_checkpoint("task", "late", {}, first["execution_token"]) is None
    assert not store.finalize(
        "task", "completed", {"type": "late"}, execution_token=first["execution_token"]
    )
    assert store.get("task")["execution_token"] == second["execution_token"]


def test_embedded_execution_is_also_fenced(tmp_path: Path):
    store = TaskStore(tmp_path / "embedded-fence.db")
    store.create("task", "query", None)
    token = store.start_embedded("task")
    assert token
    assert not store.finalize("task", "completed", {"type": "unowned"})
    assert store.finalize(
        "task", "completed", {"type": "owned"}, execution_token=token
    )


def test_budget_limits_calls_context_tokens_cost_and_duration():
    calls = RunBudget(1, 10, 5, 30, 1)
    calls.before_llm_call([{"content": "short"}])
    with pytest.raises(WorkflowBudgetExceeded, match="call"):
        calls.before_llm_call([{"content": "short"}])

    context = RunBudget(2, 3, 100, 30, 1)
    with pytest.raises(WorkflowBudgetExceeded, match="context"):
        context.before_llm_call([{"content": "long"}])

    usage = RunBudget(2, 100, 5, 30, 1)
    usage.before_llm_call([{"content": "ok"}])
    with pytest.raises(WorkflowBudgetExceeded, match="token"):
        usage.record_usage(4, 3, 7)

    cost = RunBudget(2, 100, 100, 30, 0.000001, 1, 1)
    cost.before_llm_call([{"content": "ok"}])
    with pytest.raises(WorkflowBudgetExceeded, match="cost"):
        cost.record_usage(1, 1, 2)

    duration = RunBudget(2, 100, 100, 0, 1)
    with pytest.raises(WorkflowBudgetExceeded, match="duration"):
        duration.check_duration()


@pytest.mark.parametrize(
    ("error", "category", "retryable"),
    [
        (TimeoutError(), "timeout", True),
        (ConnectionError(), "network", True),
        (WorkflowBudgetExceeded(), "budget", False),
        (ValueError(), "internal", False),
    ],
)
def test_failures_are_classified(error, category, retryable):
    result = classify_failure(error)
    assert result["category"] == category
    assert result["retryable"] is retryable


def test_trace_events_are_redacted_and_bounded(tmp_path: Path):
    store = TaskStore(tmp_path / "redaction.db")
    store.create("task", "query", None)
    store.append_event("task", {
        "type": "diagnostic",
        "api_key": "secret",
        "prompt": "private",
        "detail": "x" * 600,
    })
    event = store.events_after("task")[0]
    assert event["api_key"] == "[REDACTED]"
    assert event["prompt"] == "[REDACTED]"
    assert len(event["detail"]) <= 501
    assert sanitize_event({"authorization": "bearer"})["authorization"] == "[REDACTED]"


def test_usage_summary_is_owner_scoped(tmp_path: Path):
    store = TaskStore(tmp_path / "usage.db")
    store.create("a", "query", None, owner_id="a")
    store.create("b", "query", None, owner_id="b")
    store.append_event("a", {"type": "llm_call_completed", "model": "m", "prompt_tokens": 4, "completion_tokens": 2, "total_tokens": 6, "duration_ms": 10})
    store.append_event("b", {"type": "llm_call_completed", "model": "m", "total_tokens": 99})
    summary = store.usage_summary(owner_id="a")
    assert summary["llm_calls"] == 1
    assert summary["total_tokens"] == 6
    assert summary["models"] == {"m": 1}
