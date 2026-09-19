from types import SimpleNamespace

import pytest

from api.data_catalog import _allowed_table, preview


def test_data_catalog_rejects_non_allowlisted_and_malformed_tables():
    settings = SimpleNamespace(SQL_ALLOWED_TABLES=("orders",))
    assert _allowed_table("orders", settings) == "orders"
    with pytest.raises(KeyError):
        _allowed_table("users", settings)
    with pytest.raises(KeyError):
        _allowed_table("orders; DROP TABLE orders", settings)


def test_data_preview_is_bounded_to_fifty_rows(monkeypatch):
    captured = {}

    class Result:
        def keys(self):
            return ["id"]

        def fetchall(self):
            return [(1,), (2,)]

    class Connection:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return None

        def execute(self, _statement, params):
            captured.update(params)
            return Result()

    monkeypatch.setattr("api.data_catalog.get_connection", Connection)
    result = preview(
        "orders", SimpleNamespace(SQL_ALLOWED_TABLES=("orders",)), limit=999
    )
    assert captured["limit"] == 50
    assert result["count"] == 2
    assert result["columns"] == ["id"]
