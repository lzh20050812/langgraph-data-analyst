"""Local user, browser-session, and conversation persistence."""

from __future__ import annotations

from hashlib import scrypt, sha256
import json
import secrets
import sqlite3
from pathlib import Path
from threading import RLock
from time import time
from typing import Any
from uuid import uuid4

from api.auth import Principal


def _token_hash(token: str) -> str:
    return sha256(token.encode("utf-8")).hexdigest()


def _password_hash(password: str, salt: bytes) -> str:
    return scrypt(
        password.encode("utf-8"), salt=salt, n=2**14, r=8, p=1, dklen=64
    ).hex()


def _validate_password(password: str) -> None:
    if (
        len(password) < 8
        or not any(character.islower() for character in password)
        or not any(character.isupper() for character in password)
        or not any(character.isdigit() for character in password)
    ):
        raise ValueError(
            "password must contain at least 8 characters with upper-case, "
            "lower-case, and numeric characters"
        )


class PortalStore:
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
            connection.execute("PRAGMA foreign_keys=ON")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    username TEXT NOT NULL UNIQUE COLLATE NOCASE,
                    password_hash TEXT NOT NULL,
                    password_salt TEXT NOT NULL,
                    display_name TEXT NOT NULL,
                    role TEXT NOT NULL CHECK(role IN ('analyst','admin')),
                    enabled INTEGER NOT NULL DEFAULT 1,
                    created_at REAL NOT NULL,
                    last_login_at REAL
                );
                CREATE TABLE IF NOT EXISTS user_sessions (
                    session_hash TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    last_seen_at REAL NOT NULL,
                    expires_at REAL NOT NULL,
                    FOREIGN KEY(user_id) REFERENCES users(user_id) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_sessions_user
                    ON user_sessions(user_id, expires_at);
                CREATE TABLE IF NOT EXISTS conversations (
                    conversation_id TEXT PRIMARY KEY,
                    owner_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_conversations_owner_updated
                    ON conversations(owner_id, updated_at DESC);
                CREATE TABLE IF NOT EXISTS conversation_messages (
                    message_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT NOT NULL,
                    role TEXT NOT NULL CHECK(role IN ('user','assistant','system')),
                    content TEXT NOT NULL,
                    task_id TEXT,
                    metadata_json TEXT,
                    created_at REAL NOT NULL,
                    FOREIGN KEY(conversation_id) REFERENCES conversations(conversation_id)
                        ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_messages_conversation
                    ON conversation_messages(conversation_id, message_id);
                CREATE TABLE IF NOT EXISTS conversation_tasks (
                    task_id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    user_message_id INTEGER NOT NULL,
                    completed INTEGER NOT NULL DEFAULT 0,
                    FOREIGN KEY(conversation_id) REFERENCES conversations(conversation_id)
                        ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS conversation_context_changes (
                    change_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT NOT NULL,
                    revision INTEGER NOT NULL,
                    source_message_id INTEGER,
                    request_json TEXT NOT NULL,
                    changes_json TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    FOREIGN KEY(conversation_id) REFERENCES conversations(conversation_id)
                        ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_context_changes_conversation
                    ON conversation_context_changes(conversation_id, revision);
                """
            )
            columns = {
                row["name"]
                for row in connection.execute("PRAGMA table_info(conversations)")
            }
            if "analysis_context_json" not in columns:
                connection.execute(
                    "ALTER TABLE conversations ADD COLUMN analysis_context_json TEXT"
                )
            if "pending_clarification_json" not in columns:
                connection.execute(
                    "ALTER TABLE conversations ADD COLUMN pending_clarification_json TEXT"
                )
            if "context_revision" not in columns:
                connection.execute(
                    "ALTER TABLE conversations ADD COLUMN context_revision INTEGER NOT NULL DEFAULT 0"
                )
            if "pending_task_id" not in columns:
                connection.execute(
                    "ALTER TABLE conversations ADD COLUMN pending_task_id TEXT"
                )
            if "pending_clarification_at" not in columns:
                connection.execute(
                    "ALTER TABLE conversations ADD COLUMN pending_clarification_at REAL"
                )

    def bootstrap_admin(self, username: str, password: str) -> bool:
        with self._lock, self._connect() as connection:
            count = int(connection.execute(
                "SELECT COUNT(*) FROM users"
            ).fetchone()[0])
        if count:
            return False
        self.create_user(username, password, "系统管理员", "admin")
        return True

    def create_user(
        self,
        username: str,
        password: str,
        display_name: str,
        role: str = "analyst",
    ) -> dict[str, Any]:
        username = username.strip()
        display_name = display_name.strip()
        if not (3 <= len(username) <= 64) or not username.replace("_", "").isalnum():
            raise ValueError("username must be 3-64 letters, digits, or underscores")
        _validate_password(password)
        if not display_name or len(display_name) > 80:
            raise ValueError("display_name must contain 1-80 characters")
        if role not in {"analyst", "admin"}:
            raise ValueError("role must be analyst or admin")
        salt = secrets.token_bytes(16)
        user_id = uuid4().hex
        now = time()
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO users(
                    user_id, username, password_hash, password_salt,
                    display_name, role, enabled, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, 1, ?)
                """,
                (
                    user_id,
                    username,
                    _password_hash(password, salt),
                    salt.hex(),
                    display_name,
                    role,
                    now,
                ),
            )
        return self.get_user(user_id)

    def authenticate(self, username: str, password: str) -> dict[str, Any] | None:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM users WHERE username=? COLLATE NOCASE",
                (username.strip(),),
            ).fetchone()
            if row is None:
                # Equalize the expensive password-hash path for unknown users.
                _password_hash(password, bytes(16))
                return None
            expected = str(row["password_hash"])
            actual = _password_hash(password, bytes.fromhex(row["password_salt"]))
            if not secrets.compare_digest(actual, expected) or not row["enabled"]:
                return None
            connection.execute(
                "UPDATE users SET last_login_at=? WHERE user_id=?",
                (time(), row["user_id"]),
            )
        return self.get_user(str(row["user_id"]))

    def create_session(self, user_id: str, ttl_seconds: int) -> str:
        if ttl_seconds < 60:
            raise ValueError("session TTL must be at least 60 seconds")
        token = secrets.token_urlsafe(32)
        now = time()
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO user_sessions(
                    session_hash, user_id, created_at, last_seen_at, expires_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (_token_hash(token), user_id, now, now, now + ttl_seconds),
            )
            connection.execute("DELETE FROM user_sessions WHERE expires_at<=?", (now,))
        return token

    def principal_for_session(self, token: str | None) -> Principal | None:
        if not token:
            return None
        now = time()
        with self._lock, self._connect() as connection:
            row = connection.execute(
                """
                SELECT u.user_id, u.role, u.enabled
                FROM user_sessions s JOIN users u ON u.user_id=s.user_id
                WHERE s.session_hash=? AND s.expires_at>?
                """,
                (_token_hash(token), now),
            ).fetchone()
            if row is None or not row["enabled"]:
                return None
            connection.execute(
                "UPDATE user_sessions SET last_seen_at=? WHERE session_hash=?",
                (now, _token_hash(token)),
            )
        return Principal(subject=str(row["user_id"]), role=str(row["role"]))

    def revoke_session(self, token: str | None) -> bool:
        if not token:
            return False
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                "DELETE FROM user_sessions WHERE session_hash=?", (_token_hash(token),)
            )
        return cursor.rowcount > 0

    @staticmethod
    def _public_user(row: sqlite3.Row | None) -> dict[str, Any] | None:
        if row is None:
            return None
        item = dict(row)
        item.pop("password_hash", None)
        item.pop("password_salt", None)
        item["enabled"] = bool(item["enabled"])
        return item

    def get_user(self, user_id: str) -> dict[str, Any] | None:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM users WHERE user_id=?", (user_id,)
            ).fetchone()
        return self._public_user(row)

    def list_users(self) -> list[dict[str, Any]]:
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM users ORDER BY created_at, username"
            ).fetchall()
        return [self._public_user(row) for row in rows]

    def update_user(
        self,
        user_id: str,
        *,
        display_name: str | None = None,
        role: str | None = None,
        enabled: bool | None = None,
        password: str | None = None,
    ) -> dict[str, Any] | None:
        assignments: list[str] = []
        values: list[Any] = []
        if display_name is not None:
            if not display_name.strip() or len(display_name.strip()) > 80:
                raise ValueError("display_name must contain 1-80 characters")
            assignments.append("display_name=?")
            values.append(display_name.strip())
        if role is not None:
            if role not in {"analyst", "admin"}:
                raise ValueError("role must be analyst or admin")
            assignments.append("role=?")
            values.append(role)
        if enabled is not None:
            assignments.append("enabled=?")
            values.append(int(enabled))
        if password is not None:
            _validate_password(password)
            salt = secrets.token_bytes(16)
            assignments.extend(["password_hash=?", "password_salt=?"])
            values.extend([_password_hash(password, salt), salt.hex()])
        if not assignments:
            return self.get_user(user_id)
        values.append(user_id)
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                f"UPDATE users SET {', '.join(assignments)} WHERE user_id=?", values
            )
            if enabled is False or password is not None:
                connection.execute(
                    "DELETE FROM user_sessions WHERE user_id=?", (user_id,)
                )
        return self.get_user(user_id) if cursor.rowcount else None

    def create_conversation(self, owner_id: str, title: str) -> dict[str, Any]:
        conversation_id = uuid4().hex
        now = time()
        title = title.strip()[:100] or "新对话"
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO conversations(
                    conversation_id, owner_id, title, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?)
                """,
                (conversation_id, owner_id, title, now, now),
            )
        return self.get_conversation(conversation_id)

    def get_conversation(self, conversation_id: str) -> dict[str, Any] | None:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM conversations WHERE conversation_id=?",
                (conversation_id,),
            ).fetchone()
        if row is None:
            return None
        item = dict(row)
        raw_context = item.pop("analysis_context_json", None)
        raw_pending = item.pop("pending_clarification_json", None)
        item["analysis_context"] = json.loads(raw_context) if raw_context else None
        item["pending_clarification"] = json.loads(raw_pending) if raw_pending else None
        return item

    def list_conversations(self, owner_id: str, limit: int = 50) -> list[dict[str, Any]]:
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM conversations WHERE owner_id=?
                ORDER BY updated_at DESC LIMIT ?
                """,
                (owner_id, max(1, min(limit, 100))),
            ).fetchall()
        items = []
        for row in rows:
            item = dict(row)
            raw_context = item.pop("analysis_context_json", None)
            raw_pending = item.pop("pending_clarification_json", None)
            item["analysis_context"] = json.loads(raw_context) if raw_context else None
            item["pending_clarification"] = json.loads(raw_pending) if raw_pending else None
            items.append(item)
        return items

    def save_analysis_context(
        self,
        conversation_id: str,
        request: dict[str, Any],
        changes: list[dict[str, Any]],
        *,
        source_message_id: int | None,
        pending: bool = False,
        pending_task_id: str | None = None,
    ) -> dict[str, Any]:
        """Persist the effective request and its auditable field changes."""
        now = time()
        with self._lock, self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT context_revision FROM conversations WHERE conversation_id=?",
                (conversation_id,),
            ).fetchone()
            if row is None:
                raise KeyError(conversation_id)
            revision = int(row["context_revision"]) + 1
            if pending:
                connection.execute(
                    """
                    UPDATE conversations
                    SET pending_clarification_json=?, pending_task_id=?,
                        pending_clarification_at=?, context_revision=?, updated_at=?
                    WHERE conversation_id=?
                    """,
                    (
                        json.dumps(request, ensure_ascii=False), pending_task_id,
                        now, revision, now, conversation_id,
                    ),
                )
            else:
                connection.execute(
                    """
                    UPDATE conversations
                    SET analysis_context_json=?, pending_clarification_json=NULL,
                        pending_task_id=NULL, pending_clarification_at=NULL,
                        context_revision=?, updated_at=?
                    WHERE conversation_id=?
                    """,
                    (json.dumps(request, ensure_ascii=False), revision, now, conversation_id),
                )
            connection.execute(
                """
                INSERT INTO conversation_context_changes(
                    conversation_id, revision, source_message_id,
                    request_json, changes_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    conversation_id, revision, source_message_id,
                    json.dumps(request, ensure_ascii=False),
                    json.dumps(changes, ensure_ascii=False), now,
                ),
            )
        return self.get_conversation(conversation_id)

    def clear_pending_clarification(self, conversation_id: str) -> None:
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                UPDATE conversations
                SET pending_clarification_json=NULL, pending_task_id=NULL,
                    pending_clarification_at=NULL, updated_at=?
                WHERE conversation_id=?
                """,
                (time(), conversation_id),
            )

    def clear_pending_for_task(self, task_id: str) -> bool:
        """Detach a cancelled clarification task from its conversation."""
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                """
                UPDATE conversations
                SET pending_clarification_json=NULL, pending_task_id=NULL,
                    pending_clarification_at=NULL, updated_at=?
                WHERE pending_task_id=?
                """,
                (time(), task_id),
            )
        return cursor.rowcount > 0

    def close_task_link(self, task_id: str) -> bool:
        """Mark a linked task handled when no assistant result message is expected."""
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                "UPDATE conversation_tasks SET completed=1 WHERE task_id=?",
                (task_id,),
            )
        return cursor.rowcount > 0

    def list_context_changes(
        self, conversation_id: str, limit: int = 100
    ) -> list[dict[str, Any]]:
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM conversation_context_changes
                WHERE conversation_id=? ORDER BY revision DESC LIMIT ?
                """,
                (conversation_id, max(1, min(limit, 500))),
            ).fetchall()
        result = []
        for row in reversed(rows):
            item = dict(row)
            item["request"] = json.loads(item.pop("request_json"))
            item["changes"] = json.loads(item.pop("changes_json"))
            result.append(item)
        return result

    def update_conversation(self, conversation_id: str, title: str) -> dict[str, Any] | None:
        title = title.strip()[:100]
        if not title:
            raise ValueError("title must contain 1-100 characters")
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                "UPDATE conversations SET title=?, updated_at=? WHERE conversation_id=?",
                (title, time(), conversation_id),
            )
        return self.get_conversation(conversation_id) if cursor.rowcount else None

    def delete_conversation(self, conversation_id: str) -> bool:
        with self._lock, self._connect() as connection:
            connection.execute("PRAGMA foreign_keys=ON")
            cursor = connection.execute(
                "DELETE FROM conversations WHERE conversation_id=?", (conversation_id,)
            )
        return cursor.rowcount > 0

    def add_message(
        self,
        conversation_id: str,
        role: str,
        content: str,
        *,
        task_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        now = time()
        with self._lock, self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO conversation_messages(
                    conversation_id, role, content, task_id, metadata_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    conversation_id,
                    role,
                    content,
                    task_id,
                    json.dumps(metadata, ensure_ascii=False) if metadata else None,
                    now,
                ),
            )
            connection.execute(
                "UPDATE conversations SET updated_at=? WHERE conversation_id=?",
                (now, conversation_id),
            )
        return self.get_message(int(cursor.lastrowid))

    def get_message(self, message_id: int) -> dict[str, Any] | None:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM conversation_messages WHERE message_id=?",
                (message_id,),
            ).fetchone()
        if row is None:
            return None
        item = dict(row)
        raw = item.pop("metadata_json")
        item["metadata"] = json.loads(raw) if raw else None
        return item

    def list_messages(self, conversation_id: str, limit: int = 200) -> list[dict[str, Any]]:
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM conversation_messages WHERE conversation_id=?
                ORDER BY message_id LIMIT ?
                """,
                (conversation_id, max(1, min(limit, 500))),
            ).fetchall()
        messages = []
        for row in rows:
            item = dict(row)
            raw = item.pop("metadata_json")
            item["metadata"] = json.loads(raw) if raw else None
            messages.append(item)
        return messages

    def list_recent_messages(
        self, conversation_id: str, limit: int = 12
    ) -> list[dict[str, Any]]:
        """Return the latest messages while preserving chronological order."""
        with self._lock, self._connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM conversation_messages WHERE conversation_id=?
                ORDER BY message_id DESC LIMIT ?
                """,
                (conversation_id, max(1, min(limit, 500))),
            ).fetchall()
        messages = []
        for row in reversed(rows):
            item = dict(row)
            raw = item.pop("metadata_json")
            item["metadata"] = json.loads(raw) if raw else None
            messages.append(item)
        return messages

    def link_task(
        self, task_id: str, conversation_id: str, user_message_id: int
    ) -> None:
        with self._lock, self._connect() as connection:
            connection.execute(
                """
                INSERT INTO conversation_tasks(task_id, conversation_id, user_message_id)
                VALUES (?, ?, ?)
                """,
                (task_id, conversation_id, user_message_id),
            )

    def complete_task(self, task_id: str, result: dict[str, Any] | None, error: str | None) -> bool:
        with self._lock, self._connect() as connection:
            row = connection.execute(
                """
                SELECT conversation_id FROM conversation_tasks
                WHERE task_id=? AND completed=0
                """,
                (task_id,),
            ).fetchone()
            if row is None:
                return False
            content = (
                str((result or {}).get("report") or "分析已完成，请查看结构化结果。")
                if not error
                else f"分析失败：{error}"
            )
            now = time()
            connection.execute(
                """
                INSERT INTO conversation_messages(
                    conversation_id, role, content, task_id, metadata_json, created_at
                ) VALUES (?, 'assistant', ?, ?, ?, ?)
                """,
                (
                    row["conversation_id"],
                    content,
                    task_id,
                    json.dumps({"success": not bool(error)}, ensure_ascii=False),
                    now,
                ),
            )
            connection.execute(
                "UPDATE conversation_tasks SET completed=1 WHERE task_id=?",
                (task_id,),
            )
            connection.execute(
                "UPDATE conversations SET updated_at=? WHERE conversation_id=?",
                (now, row["conversation_id"]),
            )
        return True


from config.settings import get_settings

_settings = get_settings()
portal_store = PortalStore(_settings.TASK_DB_PATH)
if getattr(_settings, "WEB_LOGIN_ENABLED", True):
    portal_store.bootstrap_admin(
        getattr(_settings, "INITIAL_ADMIN_USERNAME", "admin"),
        getattr(_settings, "INITIAL_ADMIN_PASSWORD", "Admin123!"),
    )
