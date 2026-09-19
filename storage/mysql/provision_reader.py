"""Provision the least-privilege MySQL account used by the API service."""

from __future__ import annotations

import os
import re

from sqlalchemy import text

from config.settings import get_settings
from storage.mysql.client import get_engine


_IDENTIFIER = re.compile(r"^[A-Za-z0-9_]+$")


def _safe_identifier(value: str, label: str) -> str:
    if not _IDENTIFIER.fullmatch(value):
        raise ValueError(f"{label} must contain only letters, digits, and underscores")
    return value


def _quote_mysql_string(value: str) -> str:
    return value.replace("\\", "\\\\").replace("'", "''")


def provision_reader() -> None:
    """Create or update one SELECT-only account for the application."""
    settings = get_settings()
    username = _safe_identifier(
        os.getenv("MYSQL_APP_USER", "analytics_reader"), "MYSQL_APP_USER"
    )
    database = _safe_identifier(settings.MYSQL_DATABASE, "MYSQL_DATABASE")
    password = os.getenv("MYSQL_APP_PASSWORD", "analytics_read_password")
    if len(password) < 12:
        raise ValueError("MYSQL_APP_PASSWORD must contain at least 12 characters")

    account = f"'{username}'@'%'"
    quoted_password = _quote_mysql_string(password)
    statements = [
        f"CREATE USER IF NOT EXISTS {account} IDENTIFIED BY '{quoted_password}'",
        f"ALTER USER {account} IDENTIFIED BY '{quoted_password}'",
        f"REVOKE ALL PRIVILEGES, GRANT OPTION FROM {account}",
        f"GRANT SELECT ON `{database}`.* TO {account}",
    ]

    with get_engine().begin() as connection:
        for statement in statements:
            connection.execute(text(statement))

    print(f"[OK] Provisioned SELECT-only MySQL account: {username}")


if __name__ == "__main__":
    provision_reader()
