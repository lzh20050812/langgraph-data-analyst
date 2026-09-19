"""Background execution runtime for durable analysis tasks."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
import json
import logging
from threading import BoundedSemaphore, Event, Thread
from typing import Any
from uuid import uuid4

from agents.run_context import WorkflowCancelled, run_hooks
from agents.runtime_policy import RunBudget, classify_failure
from api.task_store import IdempotencyConflict, TaskStore
from api.task_store import TERMINAL_STATUSES
from config.settings import get_settings


logger = logging.getLogger(__name__)


class TaskQueueFull(RuntimeError):
    pass


class UnsafeCheckpointResume(RuntimeError):
    """Checkpoint continuation is rejected until nodes are idempotent."""


def _complete_conversation(
    task_id: str, result: dict[str, Any] | None, error: str | None
) -> None:
    try:
        from api.portal_store import portal_store
        portal_store.complete_task(task_id, result, error)
    except Exception:
        logger.exception(
            "Failed to persist conversation completion", extra={"task_id": task_id}
        )


@dataclass(frozen=True)
class SubmissionReceipt:
    task_id: str
    replayed: bool


def request_fingerprint(query: str, intent: str | None) -> str:
    payload = json.dumps(
        {"query": query, "intent": intent},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return sha256(payload.encode("utf-8")).hexdigest()


def public_result(state: dict[str, Any]) -> dict[str, Any]:
    rows = list(state.get("query_result") or [])
    total_rows = len(rows)
    evidence = deepcopy(state.get("evidence"))
    if isinstance(evidence, dict) and isinstance(evidence.get("rows"), list):
        evidence["rows"] = evidence["rows"][:100]
        evidence["rows_truncated"] = evidence.get("row_count", 0) > 100
    return {
        "request_id": state.get("run_id"),
        "result_mode": "live_agent",
        "success": state.get("error") is None,
        "intent": state.get("intent", ""),
        "sql": state.get("sql"),
        "query_result": rows[:100],
        "query_result_total_rows": total_rows,
        "query_result_truncated": total_rows > 100,
        "governance_result": state.get("governance_result"),
        "analysis_result": state.get("analysis_result"),
        "prediction_result": state.get("prediction_result"),
        "evidence": evidence,
        "report": state.get("report"),
        "charts": state.get("charts"),
        "error": state.get("error"),
        "messages": state.get("messages"),
        "execution_trace": state.get("execution_trace") or [],
    }


def checkpoint_snapshot(state: dict[str, Any]) -> dict[str, Any]:
    """Keep checkpoints useful without duplicating unbounded result payloads."""
    snapshot = deepcopy(state)
    if isinstance(snapshot.get("query_result"), list):
        snapshot["query_result"] = snapshot["query_result"][:100]
    evidence = snapshot.get("evidence")
    if isinstance(evidence, dict) and isinstance(evidence.get("rows"), list):
        evidence["rows"] = evidence["rows"][:100]
        evidence["rows_checkpoint_truncated"] = evidence.get("row_count", 0) > 100
    return snapshot


class TaskRuntime:
    def __init__(
        self,
        store: TaskStore,
        max_workers: int,
        max_queued_tasks: int = 20,
        execution_mode: str = "embedded",
        lease_seconds: float = 60,
    ) -> None:
        if max_workers < 1 or max_queued_tasks < 0:
            raise ValueError("worker and queue limits must be non-negative")
        self.store = store
        if execution_mode not in {"embedded", "external"}:
            raise ValueError("execution_mode must be embedded or external")
        self.execution_mode = execution_mode
        self.max_queued_tasks = max_queued_tasks
        self.lease_seconds = lease_seconds
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="analysis-task"
        )
        self._capacity = BoundedSemaphore(max_workers + max_queued_tasks)

    def submit(
        self,
        query: str,
        intent: str | None = None,
        owner_id: str = "local",
        idempotency_key: str | None = None,
        conversation_id: str | None = None,
        user_message_id: int | None = None,
        parent_task_id: str | None = None,
        retry_mode: str | None = None,
        retry_risks: list[str] | None = None,
    ) -> str:
        return self.submit_with_receipt(
            query,
            intent,
            owner_id=owner_id,
            idempotency_key=idempotency_key,
            conversation_id=conversation_id,
            user_message_id=user_message_id,
            parent_task_id=parent_task_id,
            retry_mode=retry_mode,
            retry_risks=retry_risks,
        ).task_id

    def submit_with_receipt(
        self,
        query: str,
        intent: str | None = None,
        *,
        owner_id: str = "local",
        idempotency_key: str | None = None,
        conversation_id: str | None = None,
        user_message_id: int | None = None,
        parent_task_id: str | None = None,
        retry_mode: str | None = None,
        retry_risks: list[str] | None = None,
    ) -> SubmissionReceipt:
        fingerprint = request_fingerprint(query, intent) if idempotency_key else None
        if idempotency_key:
            existing = self.store.get_by_idempotency(owner_id, idempotency_key)
            if existing is not None:
                if existing.get("request_fingerprint") != fingerprint:
                    raise IdempotencyConflict(idempotency_key)
                return SubmissionReceipt(existing["task_id"], replayed=True)

        capacity_acquired = False
        if self.execution_mode == "embedded":
            capacity_acquired = self._capacity.acquire(blocking=False)
            if not capacity_acquired:
                raise TaskQueueFull("analysis task queue is full")
        elif self.store.queue_depth() >= self.max_queued_tasks:
            raise TaskQueueFull("analysis task queue is full")
        task_id = uuid4().hex
        try:
            settings = get_settings()
            self.store.prune(
                settings.TASK_RETENTION_SECONDS, settings.TASK_MAX_RECORDS
            )
            task_id, created = self.store.create_idempotent(
                task_id,
                query,
                intent,
                owner_id=owner_id,
                idempotency_key=idempotency_key,
                request_fingerprint=fingerprint,
                parent_task_id=parent_task_id,
                retry_mode=retry_mode,
                retry_risks=retry_risks,
            )
            if not created:
                if capacity_acquired:
                    self._capacity.release()
                return SubmissionReceipt(task_id, replayed=True)
            if conversation_id and user_message_id is not None:
                from api.portal_store import portal_store
                portal_store.link_task(task_id, conversation_id, user_message_id)
            self.store.append_event(task_id, {
                "type": "task_queued",
                "trace_id": task_id,
                "retry_mode": retry_mode,
                "parent_task_id": parent_task_id,
                "duplicate_execution_risks": retry_risks or [],
            })
            if self.execution_mode == "embedded":
                future = self._executor.submit(self._execute, task_id, query, intent)
                future.add_done_callback(lambda _: self._capacity.release())
        except Exception:
            if capacity_acquired:
                self._capacity.release()
            raise
        return SubmissionReceipt(task_id, replayed=False)

    def retry(self, task_id: str, mode: str = "restart") -> str:
        task = self.store.get(task_id)
        if task is None:
            raise KeyError(task_id)
        if task["status"] not in TERMINAL_STATUSES:
            raise RuntimeError("only terminal tasks can be retried")
        if mode == "checkpoint":
            raise UnsafeCheckpointResume(
                "checkpoint continuation is disabled because SQL and LLM nodes are not idempotent"
            )
        if mode != "restart":
            raise ValueError("retry mode must be restart or checkpoint")
        retry_risks = [
            "llm_generation_may_be_billed_again",
            "read_only_sql_may_execute_again",
            "downstream_analysis_may_repeat",
        ]
        return self.submit(
            task["query"],
            task.get("intent"),
            owner_id=task.get("owner_id", "local"),
            parent_task_id=task_id,
            retry_mode="restart",
            retry_risks=retry_risks,
        )

    def resume_waiting(
        self, task_id: str, query: str, intent: str | None = None
    ) -> str:
        task = self.store.get(task_id)
        if task is None:
            raise KeyError(task_id)
        if task["status"] != "waiting_clarification":
            raise RuntimeError("task is not waiting for clarification")
        capacity_acquired = False
        if self.execution_mode == "embedded":
            capacity_acquired = self._capacity.acquire(blocking=False)
            if not capacity_acquired:
                raise TaskQueueFull("analysis task queue is full")
        try:
            if not self.store.resume_waiting(task_id, query, intent):
                raise RuntimeError("clarification task changed concurrently")
            if self.execution_mode == "embedded":
                future = self._executor.submit(self._execute, task_id, query, intent)
                future.add_done_callback(lambda _: self._capacity.release())
        except Exception:
            if capacity_acquired:
                self._capacity.release()
            raise
        return task_id

    def cancel(self, task_id: str) -> str:
        return self.store.request_cancel(task_id)

    def execute_claimed(self, task: dict[str, Any]) -> None:
        worker_id = task.get("worker_id")
        if task.get("status") != "running" or not worker_id:
            raise ValueError("execute_claimed requires a leased running task")
        self._execute(
            task["task_id"],
            task["query"],
            task.get("intent"),
            worker_id=worker_id,
        )

    def _execute(
        self,
        task_id: str,
        query: str,
        intent: str | None,
        worker_id: str | None = None,
    ) -> None:
        queued_task = self.store.get(task_id)
        if queued_task is None:
            return
        execution_token = queued_task.get("execution_token") if worker_id else None
        if queued_task["status"] == "cancelled":
            return
        if queued_task["status"] == "cancelling":
            self.store.finalize(
                task_id,
                "cancelled",
                {"type": "task_cancelled"},
                execution_token=execution_token,
            )
            return
        if worker_id is None:
            execution_token = self.store.start_embedded(task_id)
            if execution_token is None:
                return
        if self.store.append_event(
            task_id, {"type": "task_started", "trace_id": task_id}, execution_token
        ) is None:
            return

        lease_stop = Event()
        ownership_lost = Event()
        lease_thread = None
        if worker_id:
            def renew_lease_loop():
                interval = max(0.1, self.lease_seconds / 3)
                while not lease_stop.wait(interval):
                    renewed = self.store.renew_lease(
                        task_id, worker_id, self.lease_seconds, execution_token
                    )
                    if not renewed:
                        ownership_lost.set()
                        return
                    self.store.heartbeat_worker(worker_id, "busy", task_id)

            self.store.heartbeat_worker(worker_id, "busy", task_id)
            lease_thread = Thread(
                target=renew_lease_loop,
                name=f"lease-{worker_id}",
                daemon=True,
            )
            lease_thread.start()

        def event_sink(event: dict[str, Any]) -> None:
            sequence = self.store.append_event(task_id, event, execution_token)
            if sequence is None:
                ownership_lost.set()
                raise WorkflowCancelled("workflow execution ownership lost")
            if worker_id:
                if not self.store.renew_lease(
                    task_id, worker_id, self.lease_seconds, execution_token
                ):
                    ownership_lost.set()
                    raise WorkflowCancelled("workflow execution ownership lost")

        def checkpoint_sink(node: str, state: dict[str, Any]) -> None:
            sequence = self.store.save_checkpoint(
                task_id, node, checkpoint_snapshot(state), execution_token
            )
            if sequence is None:
                ownership_lost.set()
                raise WorkflowCancelled("workflow execution ownership lost")
            if worker_id:
                if not self.store.renew_lease(
                    task_id, worker_id, self.lease_seconds, execution_token
                ):
                    ownership_lost.set()
                    raise WorkflowCancelled("workflow execution ownership lost")

        try:
            from agents.planner import run_query

            settings = get_settings()
            budget = RunBudget(
                max_calls=settings.TASK_MAX_LLM_CALLS,
                max_context_chars=settings.TASK_MAX_CONTEXT_CHARS,
                max_total_tokens=settings.TASK_MAX_TOTAL_TOKENS,
                max_duration_seconds=settings.TASK_MAX_DURATION_SECONDS,
                max_cost_usd=settings.TASK_MAX_ESTIMATED_COST_USD,
                input_cost_per_million=settings.LLM_INPUT_COST_PER_MILLION,
                output_cost_per_million=settings.LLM_OUTPUT_COST_PER_MILLION,
            )

            with run_hooks(
                event_sink=event_sink,
                checkpoint_sink=checkpoint_sink,
                cancellation_check=lambda: (
                    ownership_lost.is_set() or
                    (self.store.get(task_id) or {}).get("status") == "cancelling"
                ),
                owner_id=queued_task.get("owner_id", "local"),
                budget=budget,
            ):
                state = run_query(query, requested_intent=intent, run_id=task_id)
            result = public_result(state)
            status = "completed" if result["success"] else "failed"
            committed = self.store.finalize(
                task_id,
                status,
                {"type": f"task_{status}", "success": result["success"]},
                result=result,
                error=result.get("error"),
                execution_token=execution_token,
            )
            if committed:
                _complete_conversation(task_id, result, result.get("error"))
        except WorkflowCancelled:
            if not ownership_lost.is_set() and self.store.finalize(
                task_id, "cancelled", {"type": "task_cancelled"},
                execution_token=execution_token,
            ):
                _complete_conversation(task_id, None, "任务已取消")
        except Exception as exc:
            logger.exception("Background analysis task failed", extra={"task_id": task_id})
            failure = classify_failure(exc)
            committed = self.store.finalize(
                task_id,
                "failed",
                {"type": "task_failed", **failure},
                error=f"analysis task failed ({failure['category']}); inspect server logs",
                execution_token=execution_token,
            )
            if committed:
                _complete_conversation(task_id, None, "分析任务执行失败")
        finally:
            if lease_thread is not None:
                lease_stop.set()
                lease_thread.join(timeout=1)
                self.store.heartbeat_worker(worker_id, "idle", None)


_settings = get_settings()
task_store = TaskStore(_settings.TASK_DB_PATH)
if _settings.TASK_EXECUTION_MODE == "embedded":
    task_store.interrupt_stale(_settings.TASK_STALE_SECONDS)
task_runtime = TaskRuntime(
    task_store,
    _settings.MAX_CONCURRENT_QUERIES,
    _settings.MAX_QUEUED_TASKS,
    execution_mode=_settings.TASK_EXECUTION_MODE,
    lease_seconds=_settings.TASK_WORKER_LEASE_SECONDS,
)
