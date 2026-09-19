"""Standalone durable-task worker with leases and crash recovery."""

from __future__ import annotations

import argparse
import logging
import os
import signal
import socket
from threading import Event
from uuid import uuid4

from api.task_runtime import TaskRuntime
from api.task_store import TaskStore
from config.settings import get_settings


logger = logging.getLogger(__name__)


class TaskWorker:
    def __init__(
        self,
        store: TaskStore,
        runtime: TaskRuntime,
        *,
        worker_id: str,
        poll_seconds: float,
        lease_seconds: float,
        max_attempts: int,
    ) -> None:
        self.store = store
        self.runtime = runtime
        self.worker_id = worker_id
        self.poll_seconds = poll_seconds
        self.lease_seconds = lease_seconds
        self.max_attempts = max_attempts
        self.stop_event = Event()

    def run_once(self) -> bool:
        self.store.recover_expired_leases(self.max_attempts)
        self.store.heartbeat_worker(self.worker_id, "idle", None)
        task = self.store.claim_next(self.worker_id, self.lease_seconds)
        if task is None:
            return False
        logger.info(
            "Worker %s claimed task %s attempt %s",
            self.worker_id,
            task["task_id"],
            task["attempts"],
        )
        self.runtime.execute_claimed(task)
        return True

    def run_forever(self) -> None:
        logger.info("Task worker started: %s", self.worker_id)
        try:
            while not self.stop_event.is_set():
                processed = self.run_once()
                if not processed:
                    self.stop_event.wait(self.poll_seconds)
        finally:
            self.store.heartbeat_worker(self.worker_id, "stopped", None)
            logger.info("Task worker stopped: %s", self.worker_id)

    def stop(self, *_args) -> None:
        self.stop_event.set()


def build_worker(worker_id: str | None = None) -> TaskWorker:
    settings = get_settings()
    resolved_id = worker_id or (
        f"{socket.gethostname()}-{os.getpid()}-{uuid4().hex[:8]}"
    )
    store = TaskStore(settings.TASK_DB_PATH)
    runtime = TaskRuntime(
        store,
        max_workers=1,
        max_queued_tasks=settings.MAX_QUEUED_TASKS,
        execution_mode="external",
        lease_seconds=settings.TASK_WORKER_LEASE_SECONDS,
    )
    return TaskWorker(
        store,
        runtime,
        worker_id=resolved_id,
        poll_seconds=settings.TASK_WORKER_POLL_SECONDS,
        lease_seconds=settings.TASK_WORKER_LEASE_SECONDS,
        max_attempts=settings.TASK_WORKER_MAX_ATTEMPTS,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the AI analytics task worker")
    parser.add_argument("--worker-id")
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    worker = build_worker(args.worker_id)
    signal.signal(signal.SIGINT, worker.stop)
    signal.signal(signal.SIGTERM, worker.stop)
    if args.once:
        worker.run_once()
    else:
        worker.run_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
