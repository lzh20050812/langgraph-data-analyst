"""Durable SQLite storage for asynchronous analysis tasks and checkpoints."""

from __future__ import annotations

import json
from math import ceil
import sqlite3
from pathlib import Path
from threading import RLock
from time import time
from typing import Any
from uuid import uuid4

from api.event_safety import sanitize_event
from api.task_state import TERMINAL_STATUSES, ensure_transition


class IdempotencyConflict(RuntimeError):
    """An idempotency key was reused for a different request payload."""


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str, separators=(",", ":"))


class TaskStore:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._lock, self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA synchronous=NORMAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS tasks (
                    task_id TEXT PRIMARY KEY,
                    owner_id TEXT NOT NULL DEFAULT 'local',
                    idempotency_key TEXT,
                    request_fingerprint TEXT,
                    worker_id TEXT,
                    execution_token TEXT,
                    lease_expires_at REAL,
                    attempts INTEGER NOT NULL DEFAULT 0,
                    parent_task_id TEXT,
                    retry_mode TEXT,
                    retry_risks_json TEXT,
                    trace_id TEXT,
                    query TEXT NOT NULL,
                    intent TEXT,
                    status TEXT NOT NULL,
                    result_json TEXT,
                    error TEXT,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS task_events (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT NOT NULL,
                    event_json TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    FOREIGN KEY(task_id) REFERENCES tasks(task_id)
                );
                CREATE INDEX IF NOT EXISTS idx_task_events_task_sequence
                    ON task_events(task_id, sequence);
                CREATE TABLE IF NOT EXISTS task_checkpoints (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT NOT NULL,
                    node TEXT NOT NULL,
                    state_json TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    FOREIGN KEY(task_id) REFERENCES tasks(task_id)
                );
                CREATE INDEX IF NOT EXISTS idx_task_checkpoints_task_sequence
                    ON task_checkpoints(task_id, sequence);
                CREATE TABLE IF NOT EXISTS worker_heartbeats (
                    worker_id TEXT PRIMARY KEY,
                    status TEXT NOT NULL,
                    current_task_id TEXT,
                    started_at REAL NOT NULL,
                    last_seen_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS audit_events (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    owner_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    action TEXT NOT NULL,
                    resource_type TEXT NOT NULL,
                    resource_id TEXT,
                    request_id TEXT,
                    outcome TEXT NOT NULL,
                    metadata_json TEXT,
                    created_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_audit_owner_sequence
                    ON audit_events(owner_id, sequence DESC);
                CREATE TABLE IF NOT EXISTS rate_limit_windows (
                    owner_id TEXT NOT NULL,
                    route_scope TEXT NOT NULL,
                    window_start INTEGER NOT NULL,
                    request_count INTEGER NOT NULL,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY(owner_id, route_scope, window_start)
                );
                CREATE INDEX IF NOT EXISTS idx_rate_limit_window
                    ON rate_limit_windows(window_start);
                """
            )
            columns = {
                row["name"] for row in connection.execute("PRAGMA table_info(tasks)")
            }
            if "owner_id" not in columns:
                connection.execute(
                    "ALTER TABLE tasks ADD COLUMN owner_id TEXT NOT NULL DEFAULT 'local'"
                )
            if "idempotency_key" not in columns:
                connection.execute("ALTER TABLE tasks ADD COLUMN idempotency_key TEXT")
            if "request_fingerprint" not in columns:
                connection.execute(
                    "ALTER TABLE tasks ADD COLUMN request_fingerprint TEXT"
                )
            if "worker_id" not in columns:
                connection.execute("ALTER TABLE tasks ADD COLUMN worker_id TEXT")
            if "execution_token" not in columns:
                connection.execute("ALTER TABLE tasks ADD COLUMN execution_token TEXT")
            if "lease_expires_at" not in columns:
                connection.execute("ALTER TABLE tasks ADD COLUMN lease_expires_at REAL")
            if "attempts" not in columns:
                connection.execute(
                    "ALTER TABLE tasks ADD COLUMN attempts INTEGER NOT NULL DEFAULT 0"
                )
            if "parent_task_id" not in columns:
                connection.execute("ALTER TABLE tasks ADD COLUMN parent_task_id TEXT")
            if "retry_mode" not in columns:
                connection.execute("ALTER TABLE tasks ADD COLUMN retry_mode TEXT")
            if "retry_risks_json" not in columns:
                connection.execute("ALTER TABLE tasks ADD COLUMN retry_risks_json TEXT")
            if "trace_id" not in columns:
                connection.execute("ALTER TABLE tasks ADD COLUMN trace_id TEXT")
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_tasks_owner_created "
                "ON tasks(owner_id, created_at DESC)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_tasks_queue "
                "ON tasks(status, created_at)"
            )
            connection.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_tasks_owner_idempotency "
                "ON tasks(owner_id, idempotency_key) "
                "WHERE idempotency_key IS NOT NULL"
            )

    def health_check(self) -> bool:
        """Verify the local task database without exposing stored content."""
        try:
            with self._lock, self._connect() as connection:
                row = connection.execute("PRAGMA quick_check").fetchone()
            return row is not None and str(row[0]).lower() == "ok"
        except Exception:
            return False

    def create(
        self,
        task_id: str,
        query: str,
        intent: str | None,
        owner_id: str = "local",
    ) -> None:
        self.create_idempotent(task_id, query, intent, owner_id=owner_id)

    def create_idempotent(
        self,
        task_id: str,
        query: str,
        intent: str | None,
        *,
        owner_id: str = "local",
        idempotency_key: str | None = None,
        request_fingerprint: str | None = None,
        parent_task_id: str | None = None,
        retry_mode: str | None = None,
        retry_risks: list[str] | None = None,
        trace_id: str | None = None,
    ) -> tuple[str, bool]:
        """Create once per owner/key, returning ``(task_id, created)``."""
        if idempotency_key and not request_fingerprint:
            raise ValueError("request_fingerprint is required with idempotency_key")
        now = time()
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            if idempotency_key:
                existing = connection.execute(
                    """
                    SELECT task_id, request_fingerprint FROM tasks
                    WHERE owner_id=? AND idempotency_key=?
                    """,
                    (owner_id, idempotency_key),
                ).fetchone()
                if existing is not None:
                    if existing["request_fingerprint"] != request_fingerprint:
                        raise IdempotencyConflict(idempotency_key)
                    return str(existing["task_id"]), False
            connection.execute(
                """
                INSERT INTO tasks(
                    task_id, owner_id, idempotency_key, request_fingerprint,
                    parent_task_id, retry_mode, retry_risks_json, trace_id,
                    query, intent, status, created_at, updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'queued', ?, ?)
                """,
                (
                    task_id,
                    owner_id,
                    idempotency_key,
                    request_fingerprint,
                    parent_task_id,
                    retry_mode,
                    _json_dumps(retry_risks) if retry_risks else None,
                    trace_id or task_id,
                    query,
                    intent,
                    now,
                    now,
                ),
            )
        return task_id, True

    def get_by_idempotency(
        self, owner_id: str, idempotency_key: str
    ) -> dict[str, Any] | None:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT task_id FROM tasks WHERE owner_id=? AND idempotency_key=?",
                (owner_id, idempotency_key),
            ).fetchone()
        return self.get(str(row["task_id"])) if row is not None else None

    def append_audit(
        self,
        *,
        owner_id: str,
        role: str,
        action: str,
        resource_type: str,
        resource_id: str | None,
        request_id: str | None,
        outcome: str,
        metadata: dict[str, Any] | None = None,
    ) -> int:
        """Persist a bounded, secret-free security and mutation audit record."""
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO audit_events(
                    owner_id, role, action, resource_type, resource_id,
                    request_id, outcome, metadata_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    owner_id,
                    role,
                    action,
                    resource_type,
                    resource_id,
                    request_id,
                    outcome,
                    _json_dumps(metadata) if metadata else None,
                    time(),
                ),
            )
            return int(cursor.lastrowid)

    def list_audit(
        self,
        *,
        limit: int = 50,
        after: int = 0,
        owner_id: str | None = None,
    ) -> list[dict[str, Any]]:
        limit = max(1, min(limit, 200))
        clauses = ["sequence>?"]
        params: list[Any] = [max(0, after)]
        if owner_id is not None:
            clauses.append("owner_id=?")
            params.append(owner_id)
        params.append(limit)
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                """
                SELECT sequence, owner_id, role, action, resource_type,
                       resource_id, request_id, outcome, metadata_json, created_at
                FROM audit_events WHERE """
                + " AND ".join(clauses)
                + " ORDER BY sequence LIMIT ?",
                params,
            ).fetchall()
        events = []
        for row in rows:
            event = dict(row)
            metadata_json = event.pop("metadata_json")
            event["metadata"] = json.loads(metadata_json) if metadata_json else None
            events.append(event)
        return events

    def audit_count(self, owner_id: str | None = None) -> int:
        with self._lock, self._connect() as connection:
            if owner_id is None:
                row = connection.execute(
                    "SELECT COUNT(*) AS count FROM audit_events"
                ).fetchone()
            else:
                row = connection.execute(
                    "SELECT COUNT(*) AS count FROM audit_events WHERE owner_id=?",
                    (owner_id,),
                ).fetchone()
        return int(row["count"])

    def audit_outcome_count(
        self, outcome: str, owner_id: str | None = None
    ) -> int:
        with self._lock, self._connect() as connection:
            if owner_id is None:
                row = connection.execute(
                    "SELECT COUNT(*) AS count FROM audit_events WHERE outcome=?",
                    (outcome,),
                ).fetchone()
            else:
                row = connection.execute(
                    """
                    SELECT COUNT(*) AS count FROM audit_events
                    WHERE outcome=? AND owner_id=?
                    """,
                    (outcome, owner_id),
                ).fetchone()
        return int(row["count"])

    def consume_rate_limit(
        self,
        *,
        owner_id: str,
        route_scope: str,
        limit: int,
        window_seconds: int,
        now: float | None = None,
    ) -> dict[str, Any]:
        """Atomically consume one shared fixed-window request allowance."""
        if limit < 1 or window_seconds < 1:
            raise ValueError("rate limit and window_seconds must be positive")
        current_time = time() if now is None else float(now)
        window_start = int(current_time // window_seconds) * window_seconds
        reset_at = window_start + window_seconds
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                "DELETE FROM rate_limit_windows WHERE window_start<?",
                (window_start - window_seconds,),
            )
            row = connection.execute(
                """
                SELECT request_count FROM rate_limit_windows
                WHERE owner_id=? AND route_scope=? AND window_start=?
                """,
                (owner_id, route_scope, window_start),
            ).fetchone()
            current_count = int(row["request_count"]) if row else 0
            allowed = current_count < limit
            consumed_count = current_count
            if allowed:
                consumed_count += 1
                connection.execute(
                    """
                    INSERT INTO rate_limit_windows(
                        owner_id, route_scope, window_start,
                        request_count, updated_at
                    ) VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(owner_id, route_scope, window_start)
                    DO UPDATE SET request_count=excluded.request_count,
                                  updated_at=excluded.updated_at
                    """,
                    (
                        owner_id,
                        route_scope,
                        window_start,
                        consumed_count,
                        current_time,
                    ),
                )
        return {
            "allowed": allowed,
            "limit": limit,
            "remaining": max(0, limit - consumed_count),
            "reset_at": reset_at,
            "retry_after": max(1, ceil(reset_at - current_time)),
            "request_count": consumed_count,
        }

    def list_rate_limits(
        self, *, current_window_start: int, limit: int = 200
    ) -> list[dict[str, Any]]:
        limit = max(1, min(limit, 500))
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                """
                SELECT owner_id, route_scope, window_start,
                       request_count, updated_at
                FROM rate_limit_windows WHERE window_start>=?
                ORDER BY request_count DESC, owner_id, route_scope LIMIT ?
                """,
                (current_window_start, limit),
            ).fetchall()
        return [dict(row) for row in rows]

    def prune_audit(self, retention_seconds: float, max_records: int) -> int:
        if retention_seconds <= 0 or max_records < 1:
            raise ValueError("retention_seconds and max_records must be positive")
        cutoff = time() - retention_seconds
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                """
                DELETE FROM audit_events
                WHERE created_at<? OR sequence NOT IN (
                    SELECT sequence FROM audit_events
                    ORDER BY sequence DESC LIMIT ?
                )
                """,
                (cutoff, max_records),
            )
            return cursor.rowcount

    def set_status(
        self,
        task_id: str,
        status: str,
        *,
        result: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT status FROM tasks WHERE task_id=?", (task_id,)
            ).fetchone()
            if row is None:
                raise KeyError(task_id)
            ensure_transition(str(row["status"]), status)
            connection.execute(
                """
                UPDATE tasks
                SET status=?, result_json=?, error=?, updated_at=?
                WHERE task_id=?
                """,
                (
                    status,
                    _json_dumps(result) if result is not None else None,
                    error,
                    time(),
                    task_id,
                ),
            )

    def request_cancel(self, task_id: str) -> str:
        """Atomically cancel queued work or request cancellation of its owner."""
        now = time()
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT status FROM tasks WHERE task_id=?", (task_id,)
            ).fetchone()
            if row is None:
                raise KeyError(task_id)
            current = str(row["status"])
            if current in TERMINAL_STATUSES:
                return current
            if current == "cancelling":
                return current
            target = (
                "cancelled"
                if current in {"queued", "waiting_clarification"}
                else "cancelling"
            )
            cursor = connection.execute(
                """
                UPDATE tasks SET status=?, updated_at=?
                WHERE task_id=? AND status=?
                """,
                (target, now, task_id, current),
            )
            if cursor.rowcount != 1:
                return str(connection.execute(
                    "SELECT status FROM tasks WHERE task_id=?", (task_id,)
                ).fetchone()["status"])
            event_type = "task_cancelled" if target == "cancelled" else "cancellation_requested"
            connection.execute(
                "INSERT INTO task_events(task_id, event_json, created_at) VALUES (?, ?, ?)",
                (task_id, _json_dumps({"type": event_type}), now),
            )
            return target

    def resume_waiting(self, task_id: str, query: str, intent: str | None) -> bool:
        """Atomically resume a clarification-paused task into the queue."""
        now = time()
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            cursor = connection.execute(
                """
                UPDATE tasks
                SET query=?, intent=?, status='queued', error=NULL, updated_at=?
                WHERE task_id=? AND status='waiting_clarification'
                """,
                (query, intent, now, task_id),
            )
            if cursor.rowcount != 1:
                return False
            connection.execute(
                "INSERT INTO task_events(task_id, event_json, created_at) VALUES (?, ?, ?)",
                (task_id, _json_dumps({"type": "task_resumed"}), now),
            )
            return True

    def finalize(
        self,
        task_id: str,
        status: str,
        event: dict[str, Any],
        *,
        result: dict[str, Any] | None = None,
        error: str | None = None,
        execution_token: str | None = None,
    ) -> bool:
        """Atomically persist the terminal event and matching task status."""
        if status not in TERMINAL_STATUSES:
            raise ValueError("finalize requires a terminal status")
        now = time()
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            current = connection.execute(
                "SELECT status, execution_token FROM tasks WHERE task_id=?", (task_id,)
            ).fetchone()
            if current is None:
                return False
            if current["execution_token"] is not None and current["execution_token"] != execution_token:
                return False
            if str(current["status"]) in TERMINAL_STATUSES:
                return False
            if str(current["status"]) == "cancelling" and status != "cancelled":
                return False
            ensure_transition(str(current["status"]), status)
            token_clause = " AND execution_token=?" if execution_token else ""
            params = [status, _json_dumps(result) if result is not None else None,
                      error, now, task_id]
            if execution_token:
                params.append(execution_token)
            cursor = connection.execute(
                "UPDATE tasks SET status=?, result_json=?, error=?, updated_at=?, "
                "worker_id=NULL, lease_expires_at=NULL, execution_token=NULL "
                f"WHERE task_id=?{token_clause}", params,
            )
            if cursor.rowcount != 1:
                return False
            connection.execute(
                "INSERT INTO task_events(task_id, event_json, created_at) VALUES (?, ?, ?)",
                (task_id, _json_dumps(sanitize_event(event)), now),
            )
            return True

    def start_embedded(self, task_id: str) -> str | None:
        """Atomically start in-process work with the same fencing as workers."""
        token = uuid4().hex
        now = time()
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            cursor = connection.execute(
                """
                UPDATE tasks SET status='running', execution_token=?,
                    attempts=attempts+1, updated_at=?
                WHERE task_id=? AND status='queued'
                """,
                (token, now, task_id),
            )
            return token if cursor.rowcount == 1 else None

    def claim_next(self, worker_id: str, lease_seconds: float) -> dict[str, Any] | None:
        """Atomically claim the oldest queued task for one worker."""
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        now = time()
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT task_id FROM tasks
                WHERE status='queued'
                ORDER BY created_at, task_id LIMIT 1
                """
            ).fetchone()
            if row is None:
                return None
            task_id = row["task_id"]
            execution_token = uuid4().hex
            cursor = connection.execute(
                """
                UPDATE tasks
                SET status='running', worker_id=?, execution_token=?, lease_expires_at=?,
                    attempts=attempts+1, updated_at=?
                WHERE task_id=? AND status='queued'
                """,
                (worker_id, execution_token, now + lease_seconds, now, task_id),
            )
            if cursor.rowcount != 1:
                return None
            connection.execute(
                """
                INSERT INTO task_events(task_id, event_json, created_at)
                VALUES (?, ?, ?)
                """,
                (
                    task_id,
                    _json_dumps(sanitize_event({"type": "task_claimed", "worker_id": worker_id})),
                    now,
                ),
            )
        return self.get(task_id)

    def renew_lease(
        self, task_id: str, worker_id: str, lease_seconds: float,
        execution_token: str | None = None,
    ) -> bool:
        now = time()
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE tasks SET lease_expires_at=?, updated_at=?
                WHERE task_id=? AND worker_id=?
                  AND (? IS NULL OR execution_token=?)
                  AND status IN ('running','cancelling')
                """,
                (now + lease_seconds, now, task_id, worker_id,
                 execution_token, execution_token),
            )
            return cursor.rowcount == 1

    def recover_expired_leases(self, max_attempts: int) -> dict[str, int]:
        """Requeue abandoned work or terminate it after the retry budget."""
        if max_attempts < 1:
            raise ValueError("max_attempts must be positive")
        now = time()
        recovered = {"requeued": 0, "interrupted": 0, "cancelled": 0}
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            rows = connection.execute(
                """
                SELECT task_id, status, attempts FROM tasks
                WHERE status IN ('running','cancelling')
                  AND lease_expires_at IS NOT NULL AND lease_expires_at<=?
                """,
                (now,),
            ).fetchall()
            for row in rows:
                task_id = row["task_id"]
                if row["status"] == "cancelling":
                    status, event_type, counter = "cancelled", "task_cancelled", "cancelled"
                elif int(row["attempts"]) >= max_attempts:
                    status, event_type, counter = (
                        "interrupted",
                        "task_interrupted",
                        "interrupted",
                    )
                else:
                    status, event_type, counter = "queued", "task_requeued", "requeued"
                connection.execute(
                    """
                    UPDATE tasks SET status=?, worker_id=NULL, execution_token=NULL, lease_expires_at=NULL,
                        error=?, updated_at=? WHERE task_id=?
                    """,
                    (
                        status,
                        "worker lease expired" if status == "interrupted" else None,
                        now,
                        task_id,
                    ),
                )
                connection.execute(
                    """
                    INSERT INTO task_events(task_id, event_json, created_at)
                    VALUES (?, ?, ?)
                    """,
                    (task_id, _json_dumps(sanitize_event({"type": event_type})), now),
                )
                recovered[counter] += 1
        return recovered

    def heartbeat_worker(
        self, worker_id: str, status: str, current_task_id: str | None = None
    ) -> None:
        now = time()
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO worker_heartbeats(
                    worker_id, status, current_task_id, started_at, last_seen_at
                ) VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(worker_id) DO UPDATE SET
                    status=excluded.status,
                    current_task_id=excluded.current_task_id,
                    last_seen_at=excluded.last_seen_at
                """,
                (worker_id, status, current_task_id, now, now),
            )

    def list_workers(self, active_within_seconds: float = 60) -> list[dict[str, Any]]:
        cutoff = time() - active_within_seconds
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                """
                SELECT worker_id, status, current_task_id, started_at, last_seen_at
                FROM worker_heartbeats WHERE last_seen_at>=?
                ORDER BY worker_id
                """,
                (cutoff,),
            ).fetchall()
        return [dict(row) for row in rows]

    def delete_worker(self, worker_id: str) -> bool:
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM worker_heartbeats WHERE worker_id=?", (worker_id,)
            )
            return cursor.rowcount > 0

    def queue_depth(self) -> int:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT COUNT(*) AS count FROM tasks WHERE status='queued'"
            ).fetchone()
        return int(row["count"])

    def append_event(
        self, task_id: str, event: dict[str, Any],
        execution_token: str | None = None,
    ) -> int | None:
        with self._lock, self._connect() as connection:
            if execution_token is not None:
                connection.execute("BEGIN IMMEDIATE")
                owned = connection.execute(
                    "SELECT 1 FROM tasks WHERE task_id=? AND execution_token=? "
                    "AND status IN ('running','cancelling')",
                    (task_id, execution_token),
                ).fetchone()
                if owned is None:
                    return None
            cursor = connection.execute(
                """
                INSERT INTO task_events(task_id, event_json, created_at)
                VALUES (?, ?, ?)
                """,
                (task_id, _json_dumps(sanitize_event(event)), time()),
            )
            connection.execute(
                "UPDATE tasks SET updated_at=? WHERE task_id=?", (time(), task_id)
            )
            return int(cursor.lastrowid)

    def save_checkpoint(
        self, task_id: str, node: str, state: dict[str, Any],
        execution_token: str | None = None,
    ) -> int | None:
        with self._lock, self._connect() as connection:
            if execution_token is not None:
                connection.execute("BEGIN IMMEDIATE")
                owned = connection.execute(
                    "SELECT 1 FROM tasks WHERE task_id=? AND execution_token=? "
                    "AND status IN ('running','cancelling')",
                    (task_id, execution_token),
                ).fetchone()
                if owned is None:
                    return None
            cursor = connection.execute(
                """
                INSERT INTO task_checkpoints(task_id, node, state_json, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (task_id, node, _json_dumps(state), time()),
            )
            connection.execute(
                "UPDATE tasks SET updated_at=? WHERE task_id=?", (time(), task_id)
            )
            return int(cursor.lastrowid)

    def interrupt_stale(self, stale_after_seconds: float) -> int:
        """Mark abandoned queued/running work while leaving active workers alone."""
        if stale_after_seconds < 0:
            raise ValueError("stale_after_seconds cannot be negative")
        now = time()
        cutoff = now - stale_after_seconds
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            rows = connection.execute(
                "SELECT task_id FROM tasks WHERE status IN ('queued','running') AND updated_at<=?",
                (cutoff,),
            ).fetchall()
            for row in rows:
                connection.execute(
                    """UPDATE tasks SET status='interrupted', worker_id=NULL,
                    execution_token=NULL, lease_expires_at=NULL,
                    error='task heartbeat expired; restart required', updated_at=?
                    WHERE task_id=? AND status IN ('queued','running')""",
                    (now, row["task_id"]),
                )
                connection.execute(
                    "INSERT INTO task_events(task_id,event_json,created_at) VALUES(?,?,?)",
                    (row["task_id"], _json_dumps({"type": "task_interrupted", "reason": "stale_runtime"}), now),
                )
            return len(rows)

    def get(self, task_id: str) -> dict[str, Any] | None:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM tasks WHERE task_id=?", (task_id,)
            ).fetchone()
        if row is None:
            return None
        item = dict(row)
        result_json = item.pop("result_json")
        retry_risks_json = item.pop("retry_risks_json", None)
        item["result"] = json.loads(result_json) if result_json else None
        item["retry_risks"] = json.loads(retry_risks_json) if retry_risks_json else []
        return item

    def list_recent(
        self,
        *,
        limit: int = 20,
        status: str | None = None,
        owner_id: str | None = None,
    ) -> list[dict[str, Any]]:
        limit = max(1, min(limit, 100))
        sql = (
            "SELECT task_id, owner_id, parent_task_id, retry_mode, trace_id, "
            "query, intent, status, error, created_at, updated_at "
            "FROM tasks"
        )
        params: list[Any] = []
        clauses = []
        if owner_id is not None:
            clauses.append("owner_id=?")
            params.append(owner_id)
        if status:
            clauses.append("status=?")
            params.append(status)
        if clauses:
            sql += " WHERE " + " AND ".join(clauses)
        sql += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)
        with self._lock, self._connect() as connection:
            rows = connection.execute(sql, params).fetchall()
        return [dict(row) for row in rows]

    def summary(self, owner_id: str | None = None) -> dict[str, Any]:
        with self._lock, self._connect() as connection:
            owner_clause = " WHERE owner_id=?" if owner_id is not None else ""
            owner_params = (owner_id,) if owner_id is not None else ()
            status_rows = connection.execute(
                "SELECT status, COUNT(*) AS count FROM tasks"
                + owner_clause
                + " GROUP BY status",
                owner_params,
            ).fetchall()
            timing_clause = " AND owner_id=?" if owner_id is not None else ""
            timing = connection.execute(
                """
                SELECT AVG(updated_at-created_at) AS avg_seconds,
                       MAX(updated_at-created_at) AS max_seconds
                FROM tasks WHERE status='completed'
                """ + timing_clause,
                owner_params,
            ).fetchone()
            if owner_id is None:
                event_count = connection.execute(
                    "SELECT COUNT(*) AS count FROM task_events"
                ).fetchone()["count"]
                checkpoint_count = connection.execute(
                    "SELECT COUNT(*) AS count FROM task_checkpoints"
                ).fetchone()["count"]
            else:
                event_count = connection.execute(
                    """
                    SELECT COUNT(*) AS count FROM task_events e
                    JOIN tasks t ON t.task_id=e.task_id WHERE t.owner_id=?
                    """,
                    owner_params,
                ).fetchone()["count"]
                checkpoint_count = connection.execute(
                    """
                    SELECT COUNT(*) AS count FROM task_checkpoints c
                    JOIN tasks t ON t.task_id=c.task_id WHERE t.owner_id=?
                    """,
                    owner_params,
                ).fetchone()["count"]
        counts = {row["status"]: int(row["count"]) for row in status_rows}
        return {
            "status_counts": counts,
            "total_tasks": sum(counts.values()),
            "active_tasks": counts.get("running", 0) + counts.get("cancelling", 0),
            "queued_tasks": counts.get("queued", 0),
            "completed_avg_seconds": round(float(timing["avg_seconds"] or 0), 3),
            "completed_max_seconds": round(float(timing["max_seconds"] or 0), 3),
            "event_count": int(event_count),
            "checkpoint_count": int(checkpoint_count),
        }

    def usage_summary(self, owner_id: str | None = None) -> dict[str, Any]:
        """Aggregate redacted LLM trace events for the operations page."""
        sql = "SELECT e.event_json FROM task_events e JOIN tasks t ON t.task_id=e.task_id"
        params: tuple[Any, ...] = ()
        if owner_id is not None:
            sql += " WHERE t.owner_id=?"
            params = (owner_id,)
        with self._lock, self._connect() as connection:
            rows = connection.execute(sql, params).fetchall()
        totals: dict[str, Any] = {
            "llm_calls": 0,
            "llm_failures": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "llm_duration_ms": 0.0,
            "estimated_cost_usd": 0.0,
            "models": {},
            "failure_categories": {},
        }
        for row in rows:
            event = json.loads(row["event_json"])
            event_type = event.get("type")
            if event_type == "llm_call_completed":
                totals["llm_calls"] += 1
                totals["prompt_tokens"] += int(event.get("prompt_tokens") or 0)
                totals["completion_tokens"] += int(event.get("completion_tokens") or 0)
                totals["total_tokens"] += int(event.get("total_tokens") or 0)
                totals["llm_duration_ms"] += float(event.get("duration_ms") or 0)
                model = str(event.get("model") or "unknown")
                totals["models"][model] = totals["models"].get(model, 0) + 1
                budget = event.get("task_budget") or {}
                # The last cumulative budget value per task cannot be summed here;
                # cost is calculated per call from explicitly configured prices.
                if isinstance(budget, dict):
                    totals["estimated_cost_usd"] += float(event.get("estimated_call_cost_usd") or 0)
            elif event_type in {"llm_call_failed", "task_failed"}:
                totals["llm_failures"] += 1
                category = str(event.get("category") or "provider")
                totals["failure_categories"][category] = totals["failure_categories"].get(category, 0) + 1
        totals["llm_duration_ms"] = round(totals["llm_duration_ms"], 2)
        totals["estimated_cost_usd"] = round(totals["estimated_cost_usd"], 8)
        return totals

    def prune(self, retention_seconds: float, max_records: int) -> int:
        if retention_seconds <= 0 or max_records < 1:
            raise ValueError("retention_seconds and max_records must be positive")
        cutoff = time() - retention_seconds
        with self._lock, self._connect() as connection:
            old_rows = connection.execute(
                """
                SELECT task_id FROM tasks
                WHERE status IN ('completed','failed','cancelled','interrupted')
                  AND updated_at<?
                """,
                (cutoff,),
            ).fetchall()
            overflow_rows = connection.execute(
                """
                SELECT task_id FROM tasks
                WHERE status IN ('completed','failed','cancelled','interrupted')
                ORDER BY updated_at DESC LIMIT -1 OFFSET ?
                """,
                (max_records,),
            ).fetchall()
            task_ids = {row["task_id"] for row in (*old_rows, *overflow_rows)}
            if not task_ids:
                return 0
            placeholders = ",".join("?" for _ in task_ids)
            values = tuple(task_ids)
            connection.execute(
                f"DELETE FROM task_events WHERE task_id IN ({placeholders})", values
            )
            connection.execute(
                f"DELETE FROM task_checkpoints WHERE task_id IN ({placeholders})", values
            )
            connection.execute(
                f"DELETE FROM tasks WHERE task_id IN ({placeholders})", values
            )
            return len(task_ids)

    def delete(self, task_id: str) -> bool:
        """Delete one exact task and its dependent runtime artifacts."""
        with self._lock, self._connect() as connection:
            connection.execute("DELETE FROM task_events WHERE task_id=?", (task_id,))
            connection.execute(
                "DELETE FROM task_checkpoints WHERE task_id=?", (task_id,)
            )
            cursor = connection.execute(
                "DELETE FROM tasks WHERE task_id=?", (task_id,)
            )
            return cursor.rowcount > 0

    def events_after(self, task_id: str, sequence: int = 0) -> list[dict[str, Any]]:
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                """
                SELECT sequence, event_json, created_at
                FROM task_events
                WHERE task_id=? AND sequence>?
                ORDER BY sequence
                """,
                (task_id, sequence),
            ).fetchall()
        return [
            {
                "sequence": int(row["sequence"]),
                "created_at": row["created_at"],
                **json.loads(row["event_json"]),
            }
            for row in rows
        ]

    def latest_checkpoint(self, task_id: str) -> dict[str, Any] | None:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                """
                SELECT sequence, node, state_json, created_at
                FROM task_checkpoints
                WHERE task_id=?
                ORDER BY sequence DESC LIMIT 1
                """,
                (task_id,),
            ).fetchone()
        if row is None:
            return None
        return {
            "sequence": int(row["sequence"]),
            "node": row["node"],
            "created_at": row["created_at"],
            "state": json.loads(row["state_json"]),
        }
