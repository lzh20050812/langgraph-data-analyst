"""Bound and redact durable trace events before persistence."""

from __future__ import annotations

from typing import Any


_SENSITIVE_KEYS = {"secret", "password", "authorization", "cookie", "prompt", "query", "messages"}


def _is_sensitive(key: str) -> bool:
    lowered = key.lower()
    return (
        lowered in _SENSITIVE_KEYS
        or lowered.endswith("_key")
        or lowered.endswith("_token")
        or lowered.endswith("_secret")
        or lowered.endswith("_password")
    )


def sanitize_event(value: Any, *, key: str = "", depth: int = 0) -> Any:
    if _is_sensitive(key):
        return "[REDACTED]"
    if depth >= 4:
        return "[TRUNCATED]"
    if isinstance(value, dict):
        return {
            str(item_key)[:64]: sanitize_event(item_value, key=str(item_key), depth=depth + 1)
            for item_key, item_value in list(value.items())[:40]
        }
    if isinstance(value, (list, tuple)):
        return [sanitize_event(item, key=key, depth=depth + 1) for item in value[:40]]
    if isinstance(value, str):
        return value if len(value) <= 500 else value[:500] + "…"
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return str(value)[:500]
