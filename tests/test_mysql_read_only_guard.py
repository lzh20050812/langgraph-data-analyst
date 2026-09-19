import pytest
from contextlib import contextmanager

from storage.mysql import client


def test_rejects_write_cte_before_opening_connection():
    with pytest.raises(ValueError, match="只读"):
        client.execute_sql("WITH x AS (SELECT 1) DELETE FROM customers")


def test_rejects_multiple_statements_before_opening_connection():
    with pytest.raises(ValueError, match="只读"):
        client.execute_sql("SELECT 1; DROP TABLE customers")


def test_execute_sql_passes_values_as_bound_parameters(monkeypatch):
    calls = []

    class Result:
        def fetchmany(self, _size):
            return [(7,)]

        def keys(self):
            return ["customer_count"]

    class Connection:
        def execute(self, statement, parameters):
            calls.append((str(statement), parameters))
            return Result()

    @contextmanager
    def connection():
        yield Connection()

    monkeypatch.setattr(client, "get_connection", connection)
    rows = client.execute_sql(
        "SELECT COUNT(*) AS customer_count FROM customers WHERE country=:country",
        {"country": "United States' OR 1=1 --"},
    )
    assert rows == [{"customer_count": 7}]
    assert ":country" in calls[0][0]
    assert calls[0][1] == {"country": "United States' OR 1=1 --"}
