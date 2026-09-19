from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from time import sleep

from api.task_runtime import TaskRuntime
from api.task_store import TaskStore
from api.task_worker import TaskWorker


def test_only_one_worker_can_atomically_claim_a_task(tmp_path: Path):
    path = tmp_path / "claims.db"
    seed = TaskStore(path)
    seed.create("task-1", "query", None)
    first = TaskStore(path)
    second = TaskStore(path)

    with ThreadPoolExecutor(max_workers=2) as pool:
        claims = list(pool.map(
            lambda args: args[0].claim_next(args[1], lease_seconds=30),
            [(first, "worker-a"), (second, "worker-b")],
        ))

    claimed = [task for task in claims if task is not None]
    assert len(claimed) == 1
    assert claimed[0]["attempts"] == 1
    assert claimed[0]["worker_id"] in {"worker-a", "worker-b"}


def test_expired_lease_requeues_then_exhausts_attempt_budget(tmp_path: Path):
    store = TaskStore(tmp_path / "leases.db")
    store.create("task-1", "query", None)
    assert store.claim_next("worker-a", lease_seconds=0.01)["attempts"] == 1
    sleep(0.02)
    assert store.recover_expired_leases(max_attempts=2)["requeued"] == 1
    assert store.get("task-1")["status"] == "queued"

    assert store.claim_next("worker-b", lease_seconds=0.01)["attempts"] == 2
    sleep(0.02)
    assert store.recover_expired_leases(max_attempts=2)["interrupted"] == 1
    assert store.get("task-1")["status"] == "interrupted"


def test_expired_cancelling_lease_finishes_as_cancelled(tmp_path: Path):
    store = TaskStore(tmp_path / "cancel-lease.db")
    store.create("task-1", "query", None)
    store.claim_next("worker-a", lease_seconds=0.01)
    store.set_status("task-1", "cancelling")
    sleep(0.02)
    assert store.recover_expired_leases(max_attempts=2)["cancelled"] == 1
    assert store.get("task-1")["status"] == "cancelled"


def test_external_mode_submit_only_enqueues(tmp_path: Path):
    store = TaskStore(tmp_path / "external.db")
    runtime = TaskRuntime(
        store,
        max_workers=1,
        max_queued_tasks=2,
        execution_mode="external",
    )
    task_id = runtime.submit("query", owner_id="tenant-a")
    task = store.get(task_id)
    assert task["status"] == "queued"
    assert task["owner_id"] == "tenant-a"
    assert task["attempts"] == 0


def test_worker_run_once_claims_and_executes(monkeypatch, tmp_path: Path):
    store = TaskStore(tmp_path / "worker.db")
    store.create("task-1", "query", "sql_query")
    executed = []

    class Runtime:
        def execute_claimed(self, task):
            executed.append(task)
            store.finalize(
                task["task_id"],
                "completed",
                {"type": "task_completed"},
                result={"success": True},
                execution_token=task["execution_token"],
            )

    worker = TaskWorker(
        store,
        Runtime(),
        worker_id="worker-a",
        poll_seconds=0.01,
        lease_seconds=30,
        max_attempts=2,
    )
    assert worker.run_once() is True
    assert executed[0]["worker_id"] == "worker-a"
    assert store.get("task-1")["status"] == "completed"
    assert store.list_workers()[0]["status"] == "idle"


def test_docker_compose_separates_api_and_worker_processes():
    compose = (
        Path(__file__).resolve().parents[1] / "docker" / "docker-compose.yml"
    ).read_text(encoding="utf-8")
    assert "worker:" in compose
    assert "TASK_EXECUTION_MODE: external" in compose
    assert "command: python -m api.task_worker" in compose
