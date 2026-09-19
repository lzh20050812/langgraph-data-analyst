"""Optional API-key authentication for analysis and result endpoints."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from hmac import compare_digest
import json
import re

from config.settings import get_settings


PROTECTED_PREFIXES = (
    "/query",
    "/tasks",
    "/charts",
    "/report",
    "/operations",
    "/auth",
    "/admin",
    "/conversations",
    "/data",
    "/demo",
    "/metrics",
    "/knowledge",
)
PUBLIC_AUTH_PATHS = {"/auth/login", "/auth/config"}


@dataclass(frozen=True)
class Principal:
    subject: str
    role: str

    @property
    def is_admin(self) -> bool:
        return self.role == "admin"


LOCAL_PRINCIPAL = Principal(subject="local", role="admin")


def configured_credentials(settings=None) -> list[tuple[Principal, str]]:
    settings = settings or get_settings()
    records: list[tuple[Principal, str]] = []
    raw = getattr(settings, "API_PRINCIPALS_JSON", "")
    if raw:
        payload = json.loads(raw)
        if not isinstance(payload, list):
            raise ValueError("API_PRINCIPALS_JSON must contain a JSON array")
        for item in payload:
            if not isinstance(item, dict):
                raise ValueError("API principal entries must be objects")
            subject = str(item.get("id", "")).strip()
            role = str(item.get("role", "analyst")).strip().lower()
            key = str(item.get("key", ""))
            if not re.fullmatch(r"[A-Za-z0-9_.-]{1,64}", subject):
                raise ValueError("API principal id contains unsupported characters")
            if role not in {"analyst", "admin"} or not key:
                raise ValueError("API principal requires a key and analyst/admin role")
            records.append((Principal(subject=subject, role=role), key))

    # Backward compatibility: legacy keys receive isolated analyst identities.
    for key in getattr(settings, "API_KEYS", ()):
        fingerprint = sha256(key.encode("utf-8")).hexdigest()[:12]
        records.append((Principal(subject=f"legacy-{fingerprint}", role="analyst"), key))
    return records


def requires_auth(path: str) -> bool:
    if path in PUBLIC_AUTH_PATHS:
        return False
    return any(path == prefix or path.startswith(prefix + "/") for prefix in PROTECTED_PREFIXES)


def authenticate_api_key(candidate: str | None) -> Principal | None:
    settings = get_settings()
    if not settings.API_AUTH_ENABLED:
        # Disabling an authentication mechanism must not turn arbitrary input
        # into an administrator identity. Local access is resolved explicitly
        # by the HTTP middleware when every login mechanism is disabled.
        return None
    if not candidate:
        return None
    for principal, expected in configured_credentials():
        if compare_digest(candidate, expected):
            return principal
    return None


def validate_api_key(candidate: str | None) -> bool:
    """Compatibility helper retained for direct security checks and tests."""
    return authenticate_api_key(candidate) is not None
